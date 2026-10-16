#!/usr/bin/env python3
"""微调管道，与推理框架完全分开。

训练口径
--------------
* 每步 = 1 个 view（`steps_per_epoch = len(dataset)`）；
  同一视图的正/负 records 在同一训练步内计算，点云只从数据集读取一次、张量复用。
* 显存安全：视图内**逐 record 前向 + 即时反传**（`run_view_streamed`；按分项 W_k 精确缩放，
  梯度与原「整视图聚合后一次反传」严格等价，逐 record 反传前有键集合校验;
  只走 train 路径，不走 val：val 不做 backward，根本不保留图，没有显存压力）。
  峰值显存 ≈ 单 record 图（~13 GiB 实测）；原路径需留整视图所有 record 的图（最坏 >21 GiB）。
* 常规训练默认 3 epoch：第 1 epoch 全组线性 warmup（1e-6 → 各组目标 LR）；第 2 起全组
  同步 cosine 退火（eta_min = 1e-6）。tail resume 模式保持已有参数组的 checkpoint 末端 LR。
* AdamW(β=0.9/0.999, ε=1e-8, wd 见参数组) + 全局 grad clip 0.1。
* 冻结层 BN 策略：`model.train()` 后所有「参数全冻结」子模块置 eval（`init_from_pretrain`）。

用法
----
```
# 干跑（stub 模型 + 合成数据；验证训练循环/调度/ckpt 逻辑）
python ft_tsp3d/pipeline.py --dry-run --epochs 2 --stub-steps 8

# 正式：权重/日志分流 + 档位命名（--weight-dir / --log-dir / --tag）
python ft_tsp3d/pipeline.py --losses gate --gate-only --unfreeze-profile gate \
    --ckpt data/vlvm_v1/vlvm_v1.pth \
    --out-dir ft_tsp3d/ft_log --weight-dir data/tsp3d_ft --log-dir ft_tsp3d/ft_log \
    --tag gate_only_original
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
from model_bridge import adapt_ft_outputs  # noqa: E402
from losses import (  # noqa: E402
    LossConfig,
    aggregate_records,
    compute_gate_only_losses,
    compute_record_losses,
    compute_view_rank_losses,
    delivery_tensors,
    select_view_cross_pairs,
    total_of,
)

ETA_MIN = 1e-6
WARMUP_START = 1e-6


# Tools
def lr_at(step: int, warmup_steps: int, total_steps: int, base_lr: float,
          eta_min: float = ETA_MIN) -> float:
    """第 1 epoch 线性 warmup（绝对值 1e-6 → base_lr），其后 cosine 至 eta_min。"""
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


def set_tail_resume_lr(
    optimizer: torch.optim.Optimizer,
    local_step: int,
    warmup_steps: int,
    total_new_steps: int,
    core_lrs: Dict[str, float],
) -> Dict[str, float]:
    """续训时保持 checkpoint 中各已有参数组的末端学习率。"""
    lrs: Dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name", "group"))
        group["lr"] = float(core_lrs.get(name, group["lr"]))
        lrs[name] = float(group["lr"])
    return lrs


def clip_trainable_gradients(model: nn.Module, max_norm: float) -> None:
    """对当前训练档位的全部可训练参数做一次全局梯度裁剪。"""
    params = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def build_epoch_mix(
    rng: np.random.Generator,
    n_primary: int,
    n_replay: int,
    replay_ratio: float,
    replay_to_primary: Optional[Sequence[int]] = None,
) -> List[Tuple[str, int]]:
    """
    默认保持旧的加法混合：原始数据完整遍历，再加入 replay。
    提供 replay_to_primary 时启用源视图配对替换：总步数固定为 n_primary，
    每个 replay view 替换其对应的原始 source view。
    """
    if n_primary <= 0:
        raise ValueError("primary dataset must contain at least one view")
    if not 0.0 <= replay_ratio < 1.0:
        raise ValueError("replay_ratio must be in [0, 1)")
    if replay_ratio > 0.0 and n_replay <= 0:
        raise ValueError("replay dataset must contain at least one view")

    if replay_to_primary is not None:
        if len(replay_to_primary) != n_replay:
            raise ValueError(
                "replay_to_primary size must match replay dataset: "
                f"mapping={len(replay_to_primary)} replay={n_replay}")
        mapped = [int(index) for index in replay_to_primary]
        if any(index < 0 or index >= n_primary for index in mapped):
            raise ValueError("replay_to_primary contains an out-of-range primary index")
        if len(set(mapped)) != len(mapped):
            raise ValueError("replay views must map one-to-one to unique primary source views")
        if replay_ratio == 0.0:
            primary_order = np.arange(n_primary)
            rng.shuffle(primary_order)
            return [("original", int(index)) for index in primary_order]
        
        # 先固定 primary source 顺序，再独立选择要替换的位置。这样 F0 与
        # replace-source F1 在同 seed 下保持完全相同的 source 顺序。
        primary_order = np.arange(n_primary)
        rng.shuffle(primary_order)
        n_replay_steps = max(1, int(round(n_primary * replay_ratio)))
        if n_replay_steps > n_replay:
            raise ValueError(
                f"not enough unique replay source views: need={n_replay_steps} have={n_replay}")
        replay_order = np.arange(n_replay)
        rng.shuffle(replay_order)
        selected_replay = [int(index) for index in replay_order[:n_replay_steps]]
        replay_by_primary = {mapped[index]: index for index in selected_replay}
        plan = [
            (("replay", replay_by_primary[int(index)])
             if int(index) in replay_by_primary else ("original", int(index)))
            for index in primary_order
        ]
        if len(plan) != n_primary:
            raise RuntimeError(
                f"source replacement changed epoch size: plan={len(plan)} primary={n_primary}")
        return plan

    primary_order = np.arange(n_primary)
    rng.shuffle(primary_order)
    plan: List[Tuple[str, int]] = [("original", int(i)) for i in primary_order]
    if replay_ratio == 0.0:
        return plan

    n_replay_steps = max(1, int(round(n_primary * replay_ratio / (1.0 - replay_ratio))))
    replay_indices: List[int] = []
    while len(replay_indices) < n_replay_steps:
        replay_order = np.arange(n_replay)
        rng.shuffle(replay_order)
        take = min(n_replay_steps - len(replay_indices), n_replay)
        replay_indices.extend(int(i) for i in replay_order[:take])
    plan.extend(("replay", i) for i in replay_indices)
    rng.shuffle(plan)
    return plan


def build_replay_source_map(
    primary: LocalViewDataset,
    replay: LocalViewDataset,
) -> List[int]:
    """把每个 replay view 映射到原始数据中的唯一 source view。"""
    primary_by_sample: Dict[str, int] = {}
    for global_index, scene, _view_index, records in primary.iter_view_metadata():
        if not records:
            raise ValueError(f"empty primary view metadata: scene={scene} index={global_index}")
        raw_view = int(records[0]["view"])
        sample_id = f"{scene}/v{raw_view:06d}"
        if sample_id in primary_by_sample:
            raise ValueError(f"duplicate primary sample id: {sample_id}")
        primary_by_sample[sample_id] = global_index

    replay_to_primary: List[int] = []
    used_sources: set[int] = set()
    for replay_index, scene, _view_index, records in replay.iter_view_metadata():
        source_ids = {
            str(record.get("hard_negative", {}).get("source_sample_id", ""))
            for record in records
        }
        source_ids.discard("")
        if len(source_ids) != 1:
            raise ValueError(
                "replay view must contain exactly one source_sample_id: "
                f"scene={scene} replay_index={replay_index} sources={sorted(source_ids)}")
        source_id = next(iter(source_ids))
        primary_index = primary_by_sample.get(source_id)
        if primary_index is None:
            raise ValueError(f"replay source is absent from primary dataset: {source_id}")
        if primary_index in used_sources:
            raise ValueError(f"multiple replay views map to the same primary source: {source_id}")
        used_sources.add(primary_index)
        replay_to_primary.append(primary_index)

    if len(replay_to_primary) != len(replay):
        raise RuntimeError(
            f"replay source mapping size mismatch: mapped={len(replay_to_primary)} replay={len(replay)}")
    return replay_to_primary


def forward_train(model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取单 record 训练前向输出（pred→ `losses.py`）。
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
    """
    把 BN/Dropout/注意力 dropout 等统计与随机层强制切到 eval，
    同时保留模型的 training 分支（配合 `model.train()` 打开训练分支时使用）。

    返回（切换的模块数）。与 `freeze_all_eval_mode`（冻结层评估）互补。
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
    gate_only: bool = False,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    一个 view = 一次梯度/评估单元：逐 record（B=1）前向，按 weight 聚合损失。

    返回 (聚合后的分项 dict, record 数)；`gate_buf` 非空时收集 (p_exist, label, weight)。
    """
    records = sample["records"]
    pts_t = move_sample_points(sample, device)

    parts_list: List[Dict[str, torch.Tensor]] = []
    weights: List[float] = []
    rank_preds: List[Dict[str, Any]] = []
    for rec in records:
        inputs = {
            "point_clouds": [pts_t], "text": [rec["query"]], "return_outputs": True,
            "gate_only": bool(gate_only),
        }
        pred = forward_train(model, inputs)
        if loss_cfg.rank:
            rank_preds.append(pred)
        if gate_buf is not None:
            pe = pred.get("p_exist")
            if pe is not None:
                pe_f = float(pe.detach().float().reshape(-1)[0].cpu())
                gate_buf["p_exist"].append(pe_f)
                gate_buf["y"].append(int(rec["label"]))
                gate_buf["weight"].append(float(rec["weight"]))
                gate_buf["scene"].append(str(sample["scene"]))
                gate_buf["view"].append(int(sample["view"]))
        parts = (compute_gate_only_losses(pred, rec, loss_cfg)
                 if gate_only else compute_record_losses(pred, rec, loss_cfg))
        if parts:
            parts_list.append(parts)
            weights.append(float(rec["weight"]))
    agg = aggregate_records(parts_list, weights)
    if loss_cfg.rank:
        agg.update(compute_view_rank_losses(rank_preds, sample["records"], loss_cfg))
    return agg, len(sample["records"])


def _gate_switch(model: nn.Module, loss_cfg: LossConfig) -> bool:
    """
    门控输出是否可用：优先读模型头实际开关，回退损失推导。
    流式反传用它预判 `gate` 分项是否存在。
    """
    inner = getattr(model, "head", model)
    gate = getattr(inner, "enable_gate", None)
    return bool(loss_cfg.gate) if gate is None else bool(gate)


def _predicted_loss_keys(
    rec: Dict[str, Any], cfg: LossConfig, has_gate: bool, gate_only: bool = False,
) -> set:
    """
    预测单 record 的损失分项键集合（镜像 `losses.compute_record_losses` 的门控）。
    仅用于流式反传预计算归一化系数 W_k；
    每条 record 反传前会与实际键严格校验（不一致会立即报错，不会静默改变训练语义）。
    """
    if gate_only:
        keys = {"gate_main"} if cfg.gate and has_gate else set()
        if keys and int(rec.get("label", 0)) == 1:
            keys.add("gate_recall")
        return keys

    keys = {"cls_base"}
    if int(rec.get("label", 0)) == 0:
        keys.add("bg")
    if rec.get("gt") is not None:
        keys |= {"l1_base", "giou_base"}
    return keys


def run_view_streamed(
    model: nn.Module,
    sample: Dict[str, Any],
    loss_cfg: LossConfig,
    device: torch.device,
    gate_only: bool = False,
) -> Tuple[Dict[str, float], int, int]:
    """
    逐 record 流式反传（显存安全）；与原整视图聚合 **梯度严格等价**。

    原路径把视图内所有 record 的梯度图全部留存、按 key 加权平均后一次 backward
    （total = Σ_k agg[k]，agg[k] = Σ_i (w_i/W_k)·v_i^k）；
    最坏视图 4 条 record 同时留存实测 >21 GiB（24GB 卡 OOM）。
    
    这里先由 record 元数据预测每个分项 k 的 W_k，
    逐 record 前向+按 s_{i,k}=w_i/W_k 缩放后 **立即反传**，
    线性缩放合并后总梯度与上式完全一致；
    峰值显存 ≈ 单 record 图（实测 ~13 GiB）。

    返回 (加权平均后的分项标量字典, record 数, 非有限分项 record 数)。
    """
    records = sample["records"]
    has_gate = _gate_switch(model, loss_cfg)
    pred_keys = [
        _predicted_loss_keys(rec, loss_cfg, has_gate, gate_only=gate_only)
        for rec in records
    ]
    
    w_sum: Dict[str, float] = {}
    for keys, rec in zip(pred_keys, records):
        w = float(rec.get("weight", 1.0))
        for k in keys:
            w_sum[k] = w_sum.get(k, 0.0) + w
    pts_t = move_sample_points(sample, device)
    agg: Dict[str, float] = {}
    n_bad = 0
    rank_preds: List[Dict[str, torch.Tensor]] = []
    rank_rng: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []

    def capture_rng() -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        return cpu_state, cuda_state

    def restore_rng(state: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> None:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)

    rank_enabled = bool(loss_cfg.rank and not gate_only)
    for rec, keys in zip(records, pred_keys):
        if rank_enabled:
            rank_rng.append(capture_rng())
        inputs = {
            "point_clouds": [pts_t], "text": [rec["query"]], "return_outputs": True,
            "gate_only": bool(gate_only),
        }
        pred = forward_train(model, inputs)

        if rank_enabled:
            final_boxes, final_logits = delivery_tensors(pred, loss_cfg.eps)
            rank_preds.append({
                "boxes": final_boxes.detach(),
                "scores": final_logits.detach().sigmoid(),
                "logits": final_logits.detach(),
            })
        
        parts = (compute_gate_only_losses(pred, rec, loss_cfg)
                 if gate_only else compute_record_losses(pred, rec, loss_cfg))
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

    # View 级排序需要跨 record 找 hardest negative。该梯度与一次性保留整 view 计算图严格相同。
    if rank_enabled:
        final_rng = capture_rng()
        cross_pairs = select_view_cross_pairs(rank_preds, records, loss_cfg)
        rank_values = compute_view_rank_losses(rank_preds, records, loss_cfg)
        if any(not bool(torch.isfinite(v)) for v in rank_values.values()):
            n_bad += 1
        else:
            agg.update({k: float(v.detach()) for k, v in rank_values.items()})
            coeffs: Dict[int, Dict[int, float]] = {}

            def add_coeffs(pairs: Sequence[Tuple[int, int, int, int]],
                           weight: float) -> None:
                if not pairs:
                    return
                cached = [delivery_tensors(pred, loss_cfg.eps)[1] for pred in rank_preds]
                scale = float(weight) / len(pairs)
                for pos_rec, pos_idx, neg_rec, neg_idx in pairs:
                    delta = (float(loss_cfg.rank_margin) - cached[pos_rec][pos_idx]
                             + cached[neg_rec][neg_idx])
                    grad = scale * float(torch.sigmoid(delta))
                    pos_map = coeffs.setdefault(pos_rec, {})
                    neg_map = coeffs.setdefault(neg_rec, {})
                    pos_map[pos_idx] = pos_map.get(pos_idx, 0.0) - grad
                    neg_map[neg_idx] = neg_map.get(neg_idx, 0.0) + grad

            add_coeffs(cross_pairs, loss_cfg.lam_rank)

            try:
                for rec_idx in sorted(coeffs):
                    restore_rng(rank_rng[rec_idx])
                    rec = records[rec_idx]
                    replay_pred = forward_train(model, {
                        "point_clouds": [pts_t], "text": [rec["query"]],
                        "return_outputs": True,
                    })
                    linearized = None
                    logits = delivery_tensors(replay_pred, loss_cfg.eps)[1]
                    for candidate_idx, coeff in coeffs[rec_idx].items():
                        if candidate_idx >= logits.numel():
                            raise RuntimeError(
                                "rank replay candidate count changed: "
                                f"record={rec_idx} idx={candidate_idx} n={logits.numel()}")
                        term = logits[candidate_idx] * coeff
                        linearized = term if linearized is None else linearized + term
                    if linearized is not None and linearized.requires_grad:
                        linearized.backward()
            finally:
                restore_rng(final_rng)
    return agg, len(records), n_bad


# ckpt
def resolve_output_dirs(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    """
    产物落点：权重目录 / 日志目录 / 命名前缀（档位 tag）。
    """
    out_dir = Path(args.out_dir)
    weight_dir = Path(args.weight_dir) if args.weight_dir else out_dir
    log_dir = Path(args.log_dir) if args.log_dir else out_dir
    return weight_dir, log_dir, str(args.tag or "")


def named_path(base: Path, tag: str, name: str) -> Path:
    """档位命名：`<base>/<tag>_<name>`。"""
    return base / (f"{tag}_{name}" if tag else name)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON record, recreating an externally removed log directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_ckpt(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
              epoch: int, global_step: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "module", model)
    ft_config = dict(getattr(raw_model, "ft_config", {}) or {})
    torch.save({
        "model": model.state_dict(),          # 保持 'model' 键（与 tsp3d.py 加载口径一致）
        "optimizer": optimizer.state_dict(),
        "epoch": epoch, "global_step": global_step,
        "args": vars(args),
        # 在线构造模型必须先知道门控头等可选分支是否实际启用，不能仅凭 state_dict 键名反推。
        "ft_config": ft_config,
    }, str(path))


def load_ckpt(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    restored_groups: List[str] = []
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        restored_groups = [str(group.get("name", "")) for group in optimizer.param_groups]
    
    return {"epoch": int(ckpt.get("epoch", 0)), "global_step": int(ckpt.get("global_step", 0)),
            "missing": list(missing), "unexpected": list(unexpected),
            "optimizer_restored_groups": restored_groups}


# stub：只服务 --dry-run 自检
class StubModel(nn.Module):
    """
    把候选数 K 存成实例属性，供 forward_train 切分参数向量；
    输出 boxes/scores/p_exist，不是真实模型。
    """

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
    recs = [{"query": "couch", "label": 1, "weight": 1.0, "gt": gt, "poses": []}]
    for q in ("chair", "bed", "tv"):
        recs.append({"query": q, "label": 0, "weight": 1.0, "gt": None, "poses": []})
    return {"scene": "stub", "view": 0, "n_points": n_pts, "points": pts, "records": recs}


# Train / Val
def build_loss_cfg(args: argparse.Namespace) -> LossConfig:
    """
    `--losses` → `LossConfig`（base 恒定生效）。

    可选开关：`gate` / `rank`。
    """
    flags = {"gate": False, "rank": False}
    if args.losses:
        want = {s.strip().lower() for s in args.losses.split(",") if s.strip()}
        alias = {
            "gate": "gate", "rank": "rank", "rank_cross": "rank", "base": "",
        }
        unsupported = sorted(w for w in want if w not in alias)
        if unsupported:
            raise ValueError(
                "当前主训练管线仅支持 base/rank/rank_cross/gate；未知损失: "
                + ", ".join(unsupported))
        for w in want:
            key = alias.get(w, w)
            if key in flags:
                flags[key] = True
    return LossConfig(
        **flags,
        lam_rank=float(getattr(args, "rank_weight", 1.0)),
        rank_margin=float(getattr(args, "rank_margin", 0.2)),
        rank_iou_threshold=float(getattr(args, "rank_iou_threshold", 0.25)),
        gate_gamma_neg=float(getattr(args, "gate_gamma_neg", 2.0)),
        gate_positive_margin=float(getattr(args, "gate_positive_margin", 0.70)),
        gate_recall_weight=float(getattr(args, "gate_recall_weight", 1.0)),
    )


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
    if loss_cfg.gate and not args.gate_only:
        raise ValueError("Gate loss only supports independent --gate-only training")
    if args.gate_only:
        if not loss_cfg.gate:
            raise ValueError("--gate-only requires --losses gate")
        if loss_cfg.rank:
            raise ValueError("--gate-only cannot be combined with rank loss")
        if args.unfreeze_profile != "gate":
            raise ValueError("--gate-only requires --unfreeze-profile gate")
        if args.replay_data_dir or float(args.replay_ratio) != 0.0:
            raise ValueError("Gate-only only supports original data; replay is disabled")

    # ---- data init ----
    dataset: Optional[LocalViewDataset] = None
    replay_dataset: Optional[LocalViewDataset] = None
    replay_to_primary: Optional[List[int]] = None
    if args.dry_run:
        steps_per_epoch = int(args.stub_steps)
    else:
        loader_cfg = LoaderConfig(
            data_dir=args.data_dir, split=args.split, scenes=args.scenes,
            limit_views=args.limit_views, cache_scenes=args.cache_scenes,
        )
        dataset = LocalViewDataset(loader_cfg)
        if dataset.warnings:
            for w in dataset.warnings[:5]:
                print("[data][warn]", w)
        data_summary = dataset.summarize()
        print("[data]", json.dumps(data_summary, ensure_ascii=False))
        if loss_cfg.gate:
            n_pos = int(data_summary["pos"])
            n_neg = int(data_summary["neg"])
            loss_cfg.gate_alpha = n_neg / max(1, n_pos + n_neg)
            print("[gate-config]", json.dumps({
                "alpha": loss_cfg.gate_alpha,
                "gamma_neg": loss_cfg.gate_gamma_neg,
                "positive_margin": loss_cfg.gate_positive_margin,
                "recall_weight": loss_cfg.gate_recall_weight,
            }, ensure_ascii=False))
        replay_ratio = float(args.replay_ratio)
        if bool(args.replay_data_dir) != (replay_ratio > 0.0):
            raise ValueError(
                "--replay-data-dir 与正数 --replay-ratio 必须同时提供；"
                "replay-ratio 表示 replay 占全部优化步的比例")
        
        if args.replay_data_dir:
            replay_cfg = LoaderConfig(
                data_dir=args.replay_data_dir, split=args.replay_split,
                scenes=args.scenes, limit_views=args.limit_views,
                cache_scenes=args.cache_scenes,
            )
            replay_dataset = LocalViewDataset(replay_cfg)
            if replay_dataset.warnings:
                for w in replay_dataset.warnings[:5]:
                    print("[replay-data][warn]", w)
            print("[replay-data]", json.dumps(replay_dataset.summarize(), ensure_ascii=False))
            if args.replay_mode == "replace-source":
                replay_to_primary = build_replay_source_map(dataset, replay_dataset)
                print("[replay-map]", json.dumps({
                    "mode": args.replay_mode,
                    "mapped_replay_views": len(replay_to_primary),
                    "unique_primary_sources": len(set(replay_to_primary)),
                }, ensure_ascii=False))

        preview_rng = np.random.default_rng(args.seed)
        preview_plan = build_epoch_mix(
            preview_rng, len(dataset), len(replay_dataset) if replay_dataset else 0,
            replay_ratio, replay_to_primary)
        n_original_steps = sum(source == "original" for source, _ in preview_plan)
        n_replay_steps = sum(source == "replay" for source, _ in preview_plan)
        steps_per_epoch = len(preview_plan)
        print("[data-mix]", json.dumps({
            "mode": args.replay_mode,
            "original_steps": n_original_steps,
            "replay_steps": n_replay_steps,
            "steps_per_epoch": steps_per_epoch,
            "effective_replay_ratio": n_replay_steps / steps_per_epoch,
        }, ensure_ascii=False))

    # ---- model init ----
    if args.dry_run:
        model = StubModel()
        frozen_n = 0
        groups, param_group_names = split_param_groups(
            model, args.lr1, args.lr2, args.lr3, args.wd)
    else:
        # 模型开关 ← 损失开关（单变量一致性：损失需要哪条分支，模型开哪条）
        model = build_model(ft_config={
            "enable_gate": bool(loss_cfg.gate),
            "gate_init_seed": int(args.gate_init_seed),
        })
        if args.ckpt:
            rep = load_pretrained(model, args.ckpt)
            print(f"[pretrain] keys={rep['ckpt_keys']} missing={rep['n_missing']} "
                  f"unexpected={rep['n_unexpected']}")
        frozen = apply_freezes(model, args.unfreeze_profile)
        frozen_n = len(frozen)
        groups, param_group_names = split_param_groups(
            model, args.lr1, args.lr2, args.lr3, args.wd,
            g2_extra=[s.strip() for s in args.g2_extra.split(",") if s.strip()],
        )
    model.to(device)

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable_names or not groups:
        raise RuntimeError(
            f"no trainable parameters for unfreeze profile {args.unfreeze_profile!r}")
    audit = {
        "unfreeze_profile": args.unfreeze_profile,
        "trainable_params": int(trainable_params),
        "trainable_tensors": len(trainable_names),
        "frozen_tensors": int(frozen_n),
        "trainable_names": trainable_names,
        "optimizer_groups": {
            key: {
                "n_tensors": len(names),
                "n_params": int(sum(dict(model.named_parameters())[n].numel() for n in names)),
            }
            for key, names in param_group_names.items()
        },
        "loss_config": vars(loss_cfg),
        "gate_only": bool(args.gate_only),
        "data": {
            "primary": str(args.data_dir),
            "replay": str(args.replay_data_dir or ""),
            "replay_ratio": float(args.replay_ratio),
            "replay_mode": str(args.replay_mode),
        },
    }
    audit_path = named_path(log_dir, tag, "trainable.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(
        f"[trainable] profile={args.unfreeze_profile} "
        f"params={trainable_params} tensors={len(trainable_names)} "
        f"audit={audit_path}",
        flush=True,
    )


    # ---- training setup and epoch planning ----
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    start_epoch = 0
    global_step = 0
    resume_global_step = 0
    resume_core_lrs: Dict[str, float] = {}
    if args.resume:
        info = load_ckpt(Path(args.resume), model, optimizer)
        start_epoch = info["epoch"] + 1
        global_step = info["global_step"]
        resume_global_step = global_step
        resume_core_lrs = {
            str(group.get("name", "group")): float(group["lr"])
            for group in optimizer.param_groups
            if not str(group.get("name", "")).startswith("gate_")
        }
        print(
            f"[resume] from {args.resume}: epoch={info['epoch']} step={global_step} "
            f"optimizer_groups={info['optimizer_restored_groups']} "
            f"lr_mode={args.resume_lr_mode}")

    total_steps = int(args.epochs) * steps_per_epoch
    tail_start_epoch = (
        int(args.resume_tail_start_epoch)
        if int(args.resume_tail_start_epoch) >= 0 else start_epoch
    )
    if tail_start_epoch > start_epoch:
        raise ValueError(
            f"resume-tail-start-epoch={tail_start_epoch} cannot exceed start_epoch={start_epoch}")
    if args.resume and args.resume_lr_mode == "tail" and int(args.resume_tail_start_epoch) >= 0:
        resume_global_step = tail_start_epoch * steps_per_epoch
    total_new_steps = max(1, (int(args.epochs) - tail_start_epoch) * steps_per_epoch)

    def update_optimizer_lr() -> Dict[str, float]:
        if args.resume and args.resume_lr_mode == "tail":
            return set_tail_resume_lr(
                optimizer,
                local_step=global_step - resume_global_step,
                warmup_steps=steps_per_epoch,
                total_new_steps=total_new_steps,
                core_lrs=resume_core_lrs,
            )
        return set_optimizer_lr(optimizer, global_step, steps_per_epoch, total_steps)
    print(f"[train] steps/epoch={steps_per_epoch} total={total_steps} device={device} "
          f"losses={ {k: v for k, v in vars(loss_cfg).items() if isinstance(v, bool)} } "
          f"profile={args.unfreeze_profile} frozen={frozen_n}")
    print(f"[train] 实时进度：tail -f {log_path}   "
          f"（每 {args.log_every} 步一行：全局进度 / 速度 / ETA / 损失分项+total / LR）", flush=True)

    rng = np.random.default_rng(args.seed + start_epoch)
    for epoch in range(start_epoch, int(args.epochs)):
        model.train()
        if not args.dry_run:
            freeze_all_eval_mode(model)

        # 独立 Gate 每个 epoch 使用固定 seed 流，原始数据顺序可复现。
        plan_rng = np.random.default_rng(args.seed + epoch) if args.gate_only else rng
        epoch_plan = ([("stub", i) for i in range(steps_per_epoch)] if args.dry_run
                      else build_epoch_mix(
                          plan_rng, len(dataset), len(replay_dataset) if replay_dataset else 0,
                          float(args.replay_ratio), replay_to_primary))
        t0 = time.time()
        epoch_parts: Dict[str, List[float]] = {}
        source_steps = {"original": 0, "replay": 0}
        n_bad = 0
        for it in range(steps_per_epoch):
            if args.dry_run:
                sample = make_fake_view(seed=epoch * 100000 + it)
            else:
                source, sample_idx = epoch_plan[it]
                active_dataset = replay_dataset if source == "replay" else dataset
                if active_dataset is None:
                    raise RuntimeError(f"dataset missing for source {source}")
                sample = active_dataset[sample_idx]
                sample["data_source"] = source
                source_steps[source] += 1
            
            if args.dry_run:
                # stub 合成数据量小：保持原「整视图一次反传」路径
                agg, _nrec = run_view(
                    model, sample, loss_cfg, device, training=True,
                    gate_only=bool(args.gate_only),
                )
                total = total_of(agg)
                if total is None or not torch.isfinite(total):
                    n_bad += 1
                else:
                    update_optimizer_lr()
                    optimizer.zero_grad(set_to_none=True)
                    if total.requires_grad:
                        total.backward()
                        clip_trainable_gradients(model, args.clip)
                        optimizer.step()
                log_parts = ({k: float(v.detach().float().cpu()) for k, v in agg.items()}
                             if agg else {})
                if total is not None and torch.isfinite(total):
                    log_parts["total"] = float(total.detach().float().cpu())
            else:
                # 逐 record 流式反传：峰值显存 ≈ 单 record 图；梯度与原聚合严格等价
                optimizer.zero_grad(set_to_none=True)
                agg, _nrec, n_bad_rec = run_view_streamed(
                    model, sample, loss_cfg, device,
                    gate_only=bool(args.gate_only),
                )
                total = float(sum(agg.values())) if agg else None
                if n_bad_rec or total is None or not math.isfinite(total):
                    n_bad += 1
                    optimizer.zero_grad(set_to_none=True)  # 坏步不更新（与原语义一致）
                else:
                    update_optimizer_lr()
                    # 与原语义对齐：仅当确实存在可反传的梯度时才 clip+step
                    has_grad = any(p.grad is not None for p in model.parameters()
                                   if p.requires_grad)
                    if has_grad:
                        clip_trainable_gradients(model, args.clip)
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
                # 进度行
                print(f"[ep{epoch + 1}/{int(args.epochs)} step {it + 1}/{steps_per_epoch} | "
                      f"全局 {global_step}/{total_steps} ({pct:.1f}%) | {spd:.2f}s/step | "
                      f"本 epoch 已用 {(time.time() - t0) / 60.0:.1f}min | ETA {eta_h:.1f}h] "
                      f"loss={msg} lr={lr_now}", flush=True)
                append_jsonl(log_path, {
                    "epoch": epoch, "step": global_step, "parts": msg,
                    "lr": lr_now, "n_bad": n_bad,
                })
                # 日志目录若被外部清理，恢复参数审计，避免训练完成后产物不完整。
                if not audit_path.is_file():
                    audit_path.parent.mkdir(parents=True, exist_ok=True)
                    audit_path.write_text(
                        json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        if n_bad:
            print(f"[ep{epoch}] 警告：{n_bad} 步损失缺失/非有限（已跳过 optimizer.step）")
        if not args.dry_run:
            print("[data-mix][epoch]", json.dumps({
                "epoch": epoch + 1, **source_steps,
                "effective_replay_ratio": source_steps["replay"] / steps_per_epoch,
            }, ensure_ascii=False))

        # ---- summarize + save + val ----
        save_ckpt(named_path(weight_dir, tag, "ckpt_last.pth"), model, optimizer, epoch, global_step, args)
        save_ckpt(named_path(weight_dir, tag, f"ckpt_epoch{epoch + 1}.pth"), model, optimizer, epoch, global_step, args)

        if not args.dry_run and args.val_every > 0 and ((epoch + 1) % args.val_every == 0
                                                        or epoch + 1 == int(args.epochs)):
            run_val(model, args, loss_cfg, device, out_dir, epoch=epoch + 1)

    print(f"[train] done. weights -> {weight_dir} ({tag + '_' if tag else ''}ckpt_*.pth)"
          f"; logs -> {log_dir}")


def run_val(model: nn.Module, args: argparse.Namespace, loss_cfg: LossConfig,
            device: torch.device, out_dir: Path, epoch: Optional[int] = None) -> None:
    """
    软前向（`run_view`） + P_exist 收集（供 τ 校准）；不改模型状态权重。
    """
    loader_cfg = LoaderConfig(
        data_dir=args.data_dir, split=args.val_split, scenes=args.val_scenes,
        limit_views=args.limit_views, cache_scenes=args.cache_scenes,
    )
    if not Path(loader_cfg.data_dir, loader_cfg.split).is_dir():
        print(f"[val] skip: {loader_cfg.split} 目录不存在")
        return
    
    ds = LocalViewDataset(loader_cfg)
    was_training = model.training
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
    append_jsonl(named_path(log_dir, tag, "val.log"), {
        "epoch": epoch, "parts": summary, "views": len(ds),
    })
    if gate_buf["p_exist"]:
        if args.dump_gate:
            requested = Path(args.dump_gate)
            dst = (requested.with_name(f"{requested.stem}_epoch{epoch}{requested.suffix}")
                   if epoch is not None else requested)
        else:
            name = f"gate_val_epoch{epoch}.npz" if epoch is not None else "gate_val.npz"
            dst = named_path(log_dir, tag, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
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


# CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    # 数据
    ap.add_argument("--data-dir", default=str(_REPO / "data" / "ft_datasets" / "local_view"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--val-scenes", default=None)
    ap.add_argument("--limit-views", type=int, default=0)
    ap.add_argument("--cache-scenes", type=int, default=3)
    ap.add_argument("--replay-data-dir", default=None,
                    help="hard-negative replay 数据根目录；需与 --replay-ratio 同时提供")
    ap.add_argument("--replay-split", default="train")
    ap.add_argument("--replay-ratio", type=float, default=0.0,
                    help="replay 占全部优化步的比例，范围 [0,1)")
    ap.add_argument("--replay-mode", choices=("additive", "replace-source"),
                    default="additive",
                    help="additive=额外加入 replay；replace-source=配对替换源视图并固定总步数")
    # 模型 / ckpt
    ap.add_argument("--ckpt", default=str(_REPO / "data" / "tsp3d_models" / "tsp3d_scanrefer.pth"))
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--resume-lr-mode", choices=("extend", "tail"), default="extend",
        help="extend=按新总 epoch 重算 cosine；tail=保持 checkpoint 末端 LR",
    )
    ap.add_argument(
        "--resume-tail-start-epoch", type=int, default=-1,
        help="tail 模式新增分支调度的绝对起始 epoch；用于从中间 checkpoint 继续时保持原 E4/E5 曲线",
    )
    ap.add_argument("--g2-extra", default="", help="逗号分隔：额外并入组 2 的参名前缀")
    ap.add_argument(
        "--unfreeze-profile", default="legacy",
        choices=("legacy", "gate", "u0", "u1", "u2", "u3"),
        help="可训练边界：legacy、gate-only 或逐层解冻 u0/u1/u2/u3",
    )
    # 优化
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--lr2", type=float, default=5e-5)
    ap.add_argument("--lr3", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    # 损失
    ap.add_argument("--losses", default="", help="rank，或独立 Gate 训练使用 gate")
    ap.add_argument("--rank-weight", type=float, default=1.0,
                    help="candidate pairwise ranking 损失权重")
    ap.add_argument("--rank-margin", type=float, default=0.2,
                    help="pairwise ranking 的 logit margin")
    ap.add_argument("--rank-iou-threshold", type=float, default=0.25,
                    help="matched candidate 作为 rank 正例的最低 3D IoU")
    ap.add_argument("--gate-gamma-neg", type=float, default=2.0)
    ap.add_argument("--gate-positive-margin", type=float, default=0.70)
    ap.add_argument("--gate-recall-weight", type=float, default=1.0)
    ap.add_argument("--gate-init-seed", type=int, default=0,
                    help="Presence Gate 独立初始化 seed（不推进核心路径 RNG）")
    ap.add_argument("--gate-only", action="store_true",
                    help="冻结核心并让训练前向/反向只消费 Gate 输出与 Gate 损失")
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
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(
            "[args] 说明：以下旧参数或未知参数不再由当前主流程处理，"
            "已忽略；训练继续，base 损失仍会正常生效："
            + json.dumps(unknown, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
