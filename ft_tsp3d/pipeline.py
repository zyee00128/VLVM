#!/usr/bin/env python3
"""Stage3 微调管道 · 训练循环（§11.2 #3）。与推理框架完全分开：只经「模型契约」交互。

模型侧接口契约（预测 schema 详见 `losses.py` 头部；桥接细节见 `model_bridge.py`）
----------------------------------------------------------------------
* **路径 A（直接契约）**：`model.forward_train(inputs) -> pred_dict`（独立实现模型 /
  `StubModel`）。`inputs = {"point_clouds": [Tensor(N,6)], "text": [str]}`，B=1 单样本调用。
  `pred_dict` 键：`boxes` (K,6) / `scores` (K,) 必填；`p_exist`（I_Gate 开时）。
  S1/S2 解耦分支的扩展键（`prop_boxes` / `match_logits` / `tgp_*` 等）随该分支移出，
  见 `ft_s12/`。
* **路径 B（`BeaUTyDETR` 结构化输出）**：训练循环调 `model(inputs)`（训练模式；自动附加
  `inputs["return_outputs"]=True`）→ 取返回的 `losses["ft_outputs"]` →
  `model_bridge.adapt_ft_outputs` 映射到同一契约（模型侧自带五项损失为保底，训练不使用）。
* 训练期**严禁硬截断 / 短路**（数据流无条件贯穿全网络）；门控乘子 `Conf = P_exist.detach()×s_k`
  由训练侧（`losses.compute_record_losses`）结算，模型侧按契约输出裸 `p_exist` 与 `scores`。
* 推理入口（`forward_test` / 服务包装）属模型侧职责，训练侧不调用。

训练口径（§7）
--------------
* 每步 = 1 个 view（`steps_per_epoch = len(dataset)`，§7 的 24.6 k/epoch）；同一视图的正/负
  records 在同一训练步（同一梯度窗口）内计算，点云只从数据集读取一次、张量复用。
* 显存安全：视图内**逐 record 前向 + 即时反传**（`run_view_streamed`；按分项 W_k 精确缩放，
  梯度与原「整视图聚合后一次反传」严格等价，逐 record 反传前有键集合校验）。
  峰值显存 ≈ 单 record 图（~13 GiB 实测）；原路径需留整视图所有 record 的图（最坏 >21 GiB）。
* 总 3 epoch：第 1 epoch 全组线性 warmup（1e-6 → 各组目标 LR）；第 2 起全组同步 cosine
  退火（eta_min = 1e-6）；AdamW(β=0.9/0.999, ε=1e-8, wd 见参数组) + 全局 grad clip 0.1。
* 冻结层 BN 策略：`model.train()` 后所有「参数全冻结」子模块置 eval（`init_from_pretrain`）。

用法
----
```
# 干跑（stub 模型 + 合成数据；验证训练循环/调度/ckpt 逻辑）
python ft_tsp3d/pipeline.py --dry-run --epochs 2 --stub-steps 8

# 正式：权重/日志分流 + 档位命名（--weight-dir / --log-dir / --tag）
python ft_tsp3d/pipeline.py --losses gate \
    --ckpt data/tsp3d_models/tsp3d_scanrefer.pth \
    --out-dir ft_tsp3d/ft_log --weight-dir data/tsp3d_ft --log-dir ft_tsp3d/ft_log \
    --tag gate --dump-gate ft_tsp3d/ft_log/gate_gate_val.npz
```
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_loader import LoaderConfig, LocalViewDataset  # noqa: E402
from gd_teacher import GateTeacherConfig  # noqa: E402
from model_bridge import adapt_ft_outputs  # noqa: E402
from losses import (  # noqa: E402
    LossConfig,
    aggregate_records,
    compute_record_losses,
    total_of,
)

# 全量损失上的逐模块消融（M0–M4）与 S1/S2 损失/开关装配已随解耦分支移出，
# 见 `ft_s12/`（该目录不被本管道导入）。
ETA_MIN = 1e-6
WARMUP_START = 1e-6


# ---------------------------------------------------------------------------
# 调度 / 工具
# ---------------------------------------------------------------------------


def lr_at(step: int, warmup_steps: int, total_steps: int, base_lr: float,
          eta_min: float = ETA_MIN) -> float:
    """§7 调度：第 1 epoch 线性 warmup（绝对值 1e-6 → base），其后 cosine 至 eta_min。"""
    if step < warmup_steps:
        frac = step / max(1, warmup_steps)
        return WARMUP_START + (base_lr - WARMUP_START) * frac
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(max(prog, 0.0), 1.0)
    return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * prog))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, step: int,
                     warmup_steps: int, total_steps: int) -> Dict[str, float]:
    lrs: Dict[str, float] = {}
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step, warmup_steps, total_steps, float(g.get("base_lr", g["lr"])))
        lrs[str(g.get("name", "group"))] = g["lr"]
    return lrs


def forward_train(model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """按契约获取单 record 训练前向输出（→ `losses.py` 契约 pred）。

    两条对接路径（细节见 `model_bridge.py` 头部）：
      1. `model.forward_train(inputs)` 直接返回契约 pred（`StubModel` / 独立模型）；
         若返回含 `ft_outputs` 的 losses dict 亦按路径 2 处理；
      2. `BeaUTyDETR`：`model(inputs)` 训练分支（`inputs["return_outputs"]=True`）
         返回 losses dict，取 `ft_outputs` 后经 `adapt_ft_outputs` 映射。
    """
    fn = getattr(model, "forward_train", None)
    if callable(fn):
        out = fn(inputs)
    else:
        # BeaUTyDETR 无 forward_train：nn.Module.__call__ → forward 训练分支
        out = model(inputs)
    if isinstance(out, dict) and "ft_outputs" in out:
        out = out["ft_outputs"]
    return adapt_ft_outputs(out)


def set_stochastic_eval(model: nn.Module) -> int:
    """把统计/随机性层切到 eval（配合 `model.train()` 打开训练分支时使用）。

    覆盖：torch BatchNorm 系、MinkowskiBatchNorm、`nn.Dropout`、`nn.MultiheadAttention`
    （dropout 权重）。返回切换的模块数。与 `freeze_all_eval_mode`（冻结层评估）互补。
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.MultiheadAttention)) or "BatchNorm" in type(m).__name__:
            m.eval()
            n += 1
    return n


def move_sample_points(sample: Dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(sample["points"][:, :6]),
                           dtype=torch.float32, device=device)


def run_view(
    model: nn.Module,
    sample: Dict[str, Any],
    loss_cfg: LossConfig,
    device: torch.device,
    training: bool,
    gate_buf: Optional[Dict[str, list]] = None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """一个 view = 一次梯度/评估单元：逐 record（B=1）前向，按 weight 聚合损失。

    返回 (聚合后的分项 dict, record 数)；`gate_buf` 非空时收集 (p_exist, label, weight)。
    """
    pts_t = move_sample_points(sample, device)
    parts_list: List[Dict[str, torch.Tensor]] = []
    weights: List[float] = []
    for rec in sample["records"]:
        inputs = {"point_clouds": [pts_t], "text": [rec["query"]], "return_outputs": True}
        pred = forward_train(model, inputs)
        if gate_buf is not None:
            pe = pred.get("p_exist")
            if pe is not None:
                pe_f = float(pe.detach().float().reshape(-1)[0].cpu())
                gate_buf["p_exist"].append(pe_f)
                gate_buf["y"].append(int(rec["label"]))
                gate_buf["weight"].append(float(rec["weight"]))
                gate_buf["scene"].append(str(sample["scene"]))
                gate_buf["view"].append(int(sample["view"]))
        parts = compute_record_losses(pred, rec, loss_cfg)
        if parts:
            parts_list.append(parts)
            weights.append(float(rec["weight"]))
    agg = aggregate_records(parts_list, weights)
    return agg, len(sample["records"])


def _gate_switch(model: nn.Module, loss_cfg: LossConfig) -> bool:
    """门控输出是否可用：优先读模型头实际开关，回退损失推导。

    流式反传用它预判 `gate` / `gate_dist` 分项是否存在。S1/S2（含 prop/refine 位
    与分项键预测）随解耦分支移出，见 `ft_s12/ablation.py`。
    """
    inner = getattr(model, "head", model)
    gate = getattr(inner, "enable_gate", None)
    return bool(loss_cfg.gate or loss_cfg.distill) if gate is None else bool(gate)


def _predicted_loss_keys(rec: Dict[str, Any], cfg: LossConfig, has_gate: bool) -> set:
    """预测单 record 的损失分项键集合（镜像 `losses.compute_record_losses` 的门控）。

    仅用于流式反传预计算归一化系数 W_k；每条 record 反传前会与实际键严格校验
    （不一致会立即报错，不会静默改变训练语义）。
    """
    keys = {"cls_base"}
    if int(rec.get("label", 0)) == 0:
        keys.add("bg")
    if rec.get("gt") is not None:
        keys |= {"l1_base", "giou_base"}
    if cfg.gate and has_gate:
        keys.add("gate")
    if cfg.distill and has_gate:
        teacher = rec.get("teacher")
        if isinstance(teacher, dict) and bool(teacher.get("valid")):
            keys.add("gate_dist")
    return keys


def run_view_streamed(
    model: nn.Module,
    sample: Dict[str, Any],
    loss_cfg: LossConfig,
    device: torch.device,
) -> Tuple[Dict[str, float], int, int]:
    """逐 record 流式反传（显存安全）；与原整视图聚合 **梯度严格等价**。

    原路径把视图内所有 record 的梯度图全部留存、按 key 加权平均后一次 backward
    （total = Σ_k agg[k]，agg[k] = Σ_i (w_i/W_k)·v_i^k）；最坏视图 4 条 record 同时
    留存实测 >21 GiB（24GB 卡 OOM）。这里先由 record 元数据预测每个分项 k 的
    W_k，逐 record 前向+按 s_{i,k}=w_i/W_k 缩放后 **立即反传**，线性缩放合并后
    总梯度与上式完全一致；峰值显存 ≈ 单 record 图（实测 ~13 GiB）。

    返回 (加权平均后的分项标量字典, record 数, 非有限分项 record 数)。
    """
    records = sample["records"]
    has_gate = _gate_switch(model, loss_cfg)
    pred_keys = [_predicted_loss_keys(rec, loss_cfg, has_gate) for rec in records]
    w_sum: Dict[str, float] = {}
    for keys, rec in zip(pred_keys, records):
        w = float(rec.get("weight", 1.0))
        for k in keys:
            w_sum[k] = w_sum.get(k, 0.0) + w
    pts_t = move_sample_points(sample, device)
    agg: Dict[str, float] = {}
    n_bad = 0
    for rec, keys in zip(records, pred_keys):
        inputs = {"point_clouds": [pts_t], "text": [rec["query"]], "return_outputs": True}
        pred = forward_train(model, inputs)
        parts = compute_record_losses(pred, rec, loss_cfg)
        actual = set(parts)
        if actual != keys:
            raise RuntimeError(
                "流式反传键集合校验失败（预测 %s != 实际 %s；scene=%s view=%s query=%r）。"
                "归一化系数不可信，已中止本步以避免静默改变训练语义。"
                % (sorted(keys), sorted(actual), sample.get("scene"),
                   sample.get("view"), rec.get("query")))
        w = float(rec.get("weight", 1.0))
        scaled = None
        for k, v in parts.items():
            if not bool(torch.isfinite(v)):
                n_bad += 1
                continue
            s = w / max(w_sum.get(k, w), 1e-6)
            agg[k] = agg.get(k, 0.0) + s * float(v.detach())
            t = v * s
            scaled = t if scaled is None else scaled + t
        if scaled is not None and scaled.requires_grad and w != 0.0:
            scaled.backward()
    return agg, len(records), n_bad


# ---------------------------------------------------------------------------
# 产物落点 / ckpt
# ---------------------------------------------------------------------------


def resolve_output_dirs(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    """产物落点：权重目录 / 日志目录 / 命名前缀（档位 tag）。

    默认（不传新参数）与旧行为一致：全部写 `--out-dir`、无前缀；
    `--weight-dir` / `--log-dir` 可把权重与日志分流，文件名变为 `<tag>_<原名>`。
    """
    out_dir = Path(args.out_dir)
    weight_dir = Path(args.weight_dir) if args.weight_dir else out_dir
    log_dir = Path(args.log_dir) if args.log_dir else out_dir
    return weight_dir, log_dir, str(args.tag or "")


def named_path(base: Path, tag: str, name: str) -> Path:
    """档位命名：`<base>/<tag>_<name>`（tag 为空时保持原名）。"""
    return base / (f"{tag}_{name}" if tag else name)


def save_ckpt(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
              epoch: int, global_step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "module", model)
    ft_config = dict(getattr(raw_model, "ft_config", {}) or {})
    torch.save({
        "model": model.state_dict(),          # 保持 'model' 键（与在线 tsp3d.py 加载口径一致）
        "optimizer": optimizer.state_dict(),
        "epoch": epoch, "global_step": global_step,
        "args": vars(args),
        # 在线构造模型必须先知道门控头等可选分支是否实际启用，不能仅凭
        # state_dict 键名反推。
        "ft_config": ft_config,
    }, str(path))


def load_ckpt(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"epoch": int(ckpt.get("epoch", 0)), "global_step": int(ckpt.get("global_step", 0)),
            "missing": list(missing), "unexpected": list(unexpected)}


# ---------------------------------------------------------------------------
# stub（干跑：验证循环 / 调度 / ckpt；与真实模型无关）
# ---------------------------------------------------------------------------


class StubModel(nn.Module):
    """契约形状的可训练桩：输出 boxes/scores/p_exist（K 固定）。"""

    def __init__(self, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.zeros(6 * k + k + 1))

    def forward_train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ARG002
        k = self.k
        boxes = torch.tanh(self.w[: 6 * k]).reshape(k, 6) * 0.5
        boxes = boxes + 0.0 * inputs["point_clouds"][0].sum()  # 保持批次相关性（形状无关）
        scores = torch.sigmoid(self.w[6 * k: 7 * k])
        p_exist = torch.sigmoid(self.w[-1])
        return {"boxes": boxes, "scores": scores, "p_exist": p_exist}


def make_fake_view(seed: int = 0, n_pts: int = 256) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    pts = (rng.normal(size=(n_pts, 6)).astype(np.float32))
    pts[:, :3] *= 2.0
    gt = np.array([1.0, 0.0, 0.4, 1.0, 0.6, 0.8], dtype=np.float32)
    recs = [{"query": "couch", "label": 1, "weight": 1.0, "gt": gt,
             "entities": None, "teacher": None, "poses": []}]
    for q in ("chair", "bed", "tv"):
        recs.append({"query": q, "label": 0, "weight": 1.0, "gt": None,
                     "entities": None, "teacher": None, "poses": []})
    return {"scene": "stub", "view": 0, "n_points": n_pts, "points": pts, "records": recs}


# ---------------------------------------------------------------------------
# 训练/验证主体
# ---------------------------------------------------------------------------


def build_loss_cfg(args: argparse.Namespace) -> LossConfig:
    """`--losses` → `LossConfig`（base 恒定生效）。

    可选开关：`gate` / `distill`；`s1` / `s2` / `tgp` 档位随 S1/S2 解耦分支移出
    （见 `ft_s12/`），此处不再接受。
    """
    flags = {"gate": False, "distill": False}
    if args.losses:
        want = {s.strip() for s in args.losses.split(",") if s.strip()}
        alias = {"gate": "gate", "distill": "distill", "base": ""}
        for w in want:
            key = alias.get(w, w)
            if key in flags:
                flags[key] = True
    return LossConfig(**flags)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    from init_from_pretrain import (
        apply_freezes, build_model, freeze_all_eval_mode, load_pretrained, split_param_groups,
    )

    out_dir = Path(args.out_dir)
    weight_dir, log_dir, tag = resolve_output_dirs(args)
    for _d in (out_dir, weight_dir, log_dir):
        _d.mkdir(parents=True, exist_ok=True)
    log_path = named_path(log_dir, tag, "train.log")
    loss_cfg = build_loss_cfg(args)

    # ---- 数据 / 模型 ----
    dataset: Optional[LocalViewDataset] = None
    if args.dry_run:
        steps_per_epoch = int(args.stub_steps)
    else:
        loader_cfg = LoaderConfig(
            data_dir=args.data_dir, split=args.split, inst_dir=args.inst_dir,
            use_sidecar=not args.no_sidecar, scenes=args.scenes, limit_views=args.limit_views,
            teacher=GateTeacherConfig(), cache_scenes=args.cache_scenes,
        )
        dataset = LocalViewDataset(loader_cfg)
        if dataset.warnings:
            for w in dataset.warnings[:5]:
                print("[data][warn]", w)
        print("[data]", json.dumps(dataset.summarize(), ensure_ascii=False))
        steps_per_epoch = len(dataset)

    if args.dry_run:
        model = StubModel()
        frozen_n = 0
        groups, _summary = split_param_groups(model, args.lr1, args.lr2, args.lr3, args.wd)
    else:
        # 模型开关 ← 损失开关（单变量一致性：损失需要哪条分支，模型开哪条）——
        # S1/S2 的 enable_prop / enable_refine 已随解耦分支移出（见 `ft_s12/`），
        # 此处只装配门控头。
        model = build_model(ft_config={
            "enable_gate": bool(loss_cfg.gate or loss_cfg.distill),
        })
        if args.ckpt:
            rep = load_pretrained(model, args.ckpt)
            print(f"[pretrain] keys={rep['ckpt_keys']} missing={rep['n_missing']} "
                  f"unexpected={rep['n_unexpected']}")
        frozen = apply_freezes(model)
        frozen_n = len(frozen)
        groups, _summary = split_param_groups(
            model, args.lr1, args.lr2, args.lr3, args.wd,
            g2_extra=[s.strip() for s in args.g2_extra.split(",") if s.strip()],
        )
    model.to(device)

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    start_epoch = 0
    global_step = 0
    if args.resume:
        info = load_ckpt(Path(args.resume), model, optimizer)
        start_epoch = info["epoch"] + 1
        global_step = info["global_step"]
        print(f"[resume] from {args.resume}: epoch={info['epoch']} step={global_step}")

    total_steps = int(args.epochs) * steps_per_epoch
    print(f"[train] steps/epoch={steps_per_epoch} total={total_steps} device={device} "
          f"losses={ {k: v for k, v in vars(loss_cfg).items() if isinstance(v, bool)} } frozen={frozen_n}")
    print(f"[train] 实时进度：tail -f {log_path}   "
          f"（每 {args.log_every} 步一行：全局进度 / 速度 / ETA / 损失分项+total / LR）", flush=True)

    rng = np.random.default_rng(args.seed + start_epoch)
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        if not args.dry_run:
            freeze_all_eval_mode(model)

        order = np.arange(steps_per_epoch)
        rng.shuffle(order)
        t0 = time.time()
        epoch_parts: Dict[str, List[float]] = {}
        n_bad = 0
        for it in range(steps_per_epoch):
            sample = (make_fake_view(seed=epoch * 100000 + it) if args.dry_run
                      else dataset[int(order[it])])
            if args.dry_run:
                # stub 合成数据量小：保持原「整视图一次反传」路径
                agg, _nrec = run_view(model, sample, loss_cfg, device, training=True)
                total = total_of(agg)
                if total is None or not torch.isfinite(total):
                    n_bad += 1
                else:
                    set_optimizer_lr(optimizer, global_step, steps_per_epoch, total_steps)
                    optimizer.zero_grad(set_to_none=True)
                    if total.requires_grad:
                        total.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], args.clip)
                        optimizer.step()
                log_parts = ({k: float(v.detach().float().cpu()) for k, v in agg.items()}
                             if agg else {})
                if total is not None and torch.isfinite(total):
                    log_parts["total"] = float(total.detach().float().cpu())
            else:
                # 逐 record 流式反传：峰值显存 ≈ 单 record 图；梯度与原聚合严格等价
                optimizer.zero_grad(set_to_none=True)
                agg, _nrec, n_bad_rec = run_view_streamed(model, sample, loss_cfg, device)
                total = float(sum(agg.values())) if agg else None
                if n_bad_rec or total is None or not math.isfinite(total):
                    n_bad += 1
                    optimizer.zero_grad(set_to_none=True)  # 坏步不更新（与原语义一致）
                else:
                    set_optimizer_lr(optimizer, global_step, steps_per_epoch, total_steps)
                    # 与原语义对齐：仅当确实存在可反传的梯度时才 clip+step
                    has_grad = any(p.grad is not None for p in model.parameters()
                                   if p.requires_grad)
                    if has_grad:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], args.clip)
                        optimizer.step()
                log_parts = dict(agg) if agg else {}
                if total is not None and math.isfinite(total):
                    log_parts["total"] = total
            for k, v in log_parts.items():
                epoch_parts.setdefault(k, []).append(float(v))
            global_step += 1
            if (it + 1) % max(1, args.log_every) == 0:
                msg = {k: round(float(np.mean(vs)), 4) for k, vs in epoch_parts.items()}
                lr_now = [round(g["lr"], 8) for g in optimizer.param_groups]
                spd = (time.time() - t0) / (it + 1)
                remain = ((steps_per_epoch - (it + 1))
                          + (int(args.epochs) - epoch - 1) * steps_per_epoch)
                eta_h = remain * spd / 3600.0
                pct = 100.0 * global_step / max(1, total_steps)
                # 进度行（flush=True：终端/tee 实时可见，不受 stdout 缓冲影响）
                print(f"[ep{epoch + 1}/{int(args.epochs)} step {it + 1}/{steps_per_epoch} | "
                      f"全局 {global_step}/{total_steps} ({pct:.1f}%) | {spd:.2f}s/step | "
                      f"本 epoch 已用 {(time.time() - t0) / 60.0:.1f}min | ETA {eta_h:.1f}h] "
                      f"loss={msg} lr={lr_now}", flush=True)
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({"epoch": epoch, "step": global_step, "parts": msg,
                                         "lr": lr_now, "n_bad": n_bad}) + "\n")
        if n_bad:
            print(f"[ep{epoch}] 警告：{n_bad} 步损失缺失/非有限（已跳过 optimizer.step）")

        # ---- 收尾：teacher 统计 + 保存 + val ----
        if dataset is not None:
            print("[teacher]", json.dumps(dataset.teacher_stats.summary(), ensure_ascii=False))
            dataset.reset_teacher_stats()
        save_ckpt(named_path(weight_dir, tag, "ckpt_last.pth"), model, optimizer, epoch, global_step, args)
        save_ckpt(named_path(weight_dir, tag, f"ckpt_epoch{epoch + 1}.pth"), model, optimizer, epoch, global_step, args)

        if not args.dry_run and args.val_every > 0 and ((epoch + 1) % args.val_every == 0
                                                        or epoch + 1 == int(args.epochs)):
            run_val(model, args, loss_cfg, device, out_dir)

    print(f"[train] done. weights -> {weight_dir} ({tag + '_' if tag else ''}ckpt_*.pth)"
          f"; logs -> {log_dir}")


def run_val(model: nn.Module, args: argparse.Namespace, loss_cfg: LossConfig,
            device: torch.device, out_dir: Path) -> None:
    """val：软前向（训练契约）+ P_exist 收集（供 τ 校准）；不改模型状态权重。"""
    loader_cfg = LoaderConfig(
        data_dir=args.data_dir, split=args.val_split, inst_dir=args.inst_dir,
        use_sidecar=not args.no_sidecar, scenes=args.val_scenes, limit_views=args.limit_views,
        cache_scenes=args.cache_scenes,
    )
    if not Path(loader_cfg.data_dir, loader_cfg.split).is_dir():
        print(f"[val] skip: {loader_cfg.split} 目录不存在")
        return
    ds = LocalViewDataset(loader_cfg)
    was_training = model.training
    # 软前向走「训练契约」：BeaUTyDETR 需 training 分支（`model(inputs)` 才给 ft_outputs），
    # 但统计/随机层保持 eval ⇒ 不更新 BN 统计、不激活 dropout，val 对模型状态零副作用。
    model.train(True)
    set_stochastic_eval(model)
    gate_buf: Dict[str, list] = {"p_exist": [], "y": [], "weight": [], "scene": [], "view": []}
    parts_all: Dict[str, List[float]] = {}
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            agg, _ = run_view(model, sample, loss_cfg, device, training=False, gate_buf=gate_buf)
            for k, v in agg.items():
                parts_all.setdefault(k, []).append(float(v.detach().float().cpu()))
    model.train(was_training)
    summary = {k: round(float(np.mean(vs)), 4) for k, vs in parts_all.items()}
    print(f"[val] loss={summary} views={len(ds)} gate_pts={len(gate_buf['p_exist'])}")
    _weight_dir, log_dir, tag = resolve_output_dirs(args)
    with open(named_path(log_dir, tag, "val.log"), "a") as fh:
        fh.write(json.dumps({"parts": summary, "views": len(ds)}) + "\n")
    if gate_buf["p_exist"]:
        dst = (Path(args.dump_gate) if args.dump_gate
               else named_path(log_dir, tag, "gate_val.npz"))
        np.savez_compressed(
            dst,
            p_exist=np.asarray(gate_buf["p_exist"], dtype=np.float32),
            y=np.asarray(gate_buf["y"], dtype=np.int8),
            weight=np.asarray(gate_buf["weight"], dtype=np.float32),
            scene=np.asarray(gate_buf["scene"], dtype="<U64"),
            view=np.asarray(gate_buf["view"], dtype=np.int32),
        )
        print(f"[val] gate dump -> {dst}")


def run_stub_training(steps: int = 8, epochs: int = 1) -> None:
    """smoke_test 入口：干跑 N 步（stub 模型 + 合成数据）。"""
    args = parse_args(["--dry-run", "--epochs", str(epochs), "--stub-steps", str(steps),
                       "--out-dir", "/tmp/ft_stub_run"])
    train(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    # 数据
    ap.add_argument("--data-dir", default=str(_REPO / "data" / "ft_datasets" / "local_view"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--inst-dir", default=None)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--val-scenes", default=None)
    ap.add_argument("--limit-views", type=int, default=0)
    ap.add_argument("--no-sidecar", action="store_true")
    ap.add_argument("--cache-scenes", type=int, default=3)
    # 模型 / ckpt
    ap.add_argument("--ckpt", default=str(_REPO / "data" / "tsp3d_models" / "tsp3d_scanrefer.pth"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--g2-extra", default="", help="逗号分隔：额外并入组 2 的参名前缀")
    # 优化
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--lr2", type=float, default=5e-5)
    ap.add_argument("--lr3", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    # 损失（M0–M4 消融预置与 s1/s2/tgp 档位随 S1/S2 解耦分支移出，见 ft_s12/）
    ap.add_argument("--losses", default="", help="逗号分隔：gate,distill（默认仅 base）")
    # 运行 / 产物落点（--weight-dir / --log-dir / --tag 可分流并按档位命名）
    ap.add_argument("--out-dir", default=str(_REPO / "ft_runs" / "default"))
    ap.add_argument("--weight-dir", default=None, help="权重目录（默认 = --out-dir）")
    ap.add_argument("--log-dir", default=None, help="日志/校准数据目录（默认 = --out-dir）")
    ap.add_argument("--tag", default="", help="档位名：产物文件名前缀（<tag>_ckpt_last.pth / <tag>_train.log …）")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--dump-gate", default=None)
    ap.add_argument("--dry-run", action="store_true", help="stub 模型 + 合成数据（自检循环/调度/ckpt）")
    ap.add_argument("--stub-steps", type=int, default=8, help="dry-run 每 epoch 步数")
    args, _ = ap.parse_known_args(argv)
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
