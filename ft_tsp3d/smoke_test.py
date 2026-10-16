#!/usr/bin/env python3
"""Stage3 微调 · 冒烟与验收测试。

覆盖（无需 GPU / 真实模型即可跑核心部分）：
  * 匹配器：唯一匹配 / 重复提议判负 / 1:1；
  * 损失边界：空 K、全剪枝空返回、零权重、单正样本、负样本 bg 项；
  * 区分度损失：IoU 正例门、同 view hardest-negative、pairwise 梯度与流式重放等价；
  * 数值自检：ASL（`losses._self_test`）；
  * 调度器：warmup 起点 → 第 1 epoch 末 = 目标 LR → 末步 = eta_min；
  * 数据管道：读取真实 npz 校验 offsets/records/schema（数据存在时）；
  * 训练循环：`pipeline.run_stub_training`（stub 模型 + 合成数据，验证循环/调度/ckpt）。

需真实模型/数据（模型侧就绪后运行）：
  * `--overfit`：1–2 场景过拟合跑（调 `pipeline.train` 全链路）；
  * 验收点：可微软剪枝重建后 `keep_conv` / `com_cls` 确实收到梯度 —— `check_prune_gradients`。

用法：
```
python ft_tsp3d/smoke_test.py                 # 全部离线用例
python ft_tsp3d/smoke_test.py --overfit --scenes 00006-HkseAnWCgqk --limit-views 20
```
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import losses as L  # noqa: E402
import pipeline as P  # noqa: E402

RESULTS: List[Dict[str, Any]] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        RESULTS.append({"test": name, "ok": True})
        print(f"[PASS] {name}")
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"test": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)


# ---------------------------------------------------------------------------
# 组件用例
# ---------------------------------------------------------------------------


def test_matcher_unique() -> None:
    pred = {"boxes": torch.tensor([[0.0, 0, 0, 1.0, 1.0, 1.0],
                                   [0.0, 0, 0, 1.0, 1.0, 1.0],
                                   [5.0, 5, 5, 1.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.4, 0.9, 0.6])}
    tgt = {"label": 1, "weight": 1.0,
           "gt": np.array([0.0, 0, 0, 1.0, 1.0, 1.0], np.float32)}
    parts = L.compute_record_losses(pred, tgt, L.LossConfig())
    assert "cls_base" in parts and "l1_base" in parts and "giou_base" in parts
    # 正样本不得出现 bg；重复提议（槽 0）必须判负：其 cls 贡献 > 0 且 l1 不作用于它
    assert "bg" not in parts
    # 匹配项必须是槽 1（IoU=1）：l1=0（槽 1 与 GT 完全重合）
    assert float(parts["l1_base"]) < 1e-5, f"l1 should match box-1, got {float(parts['l1_base'])}"


def test_loss_edge_cases() -> None:
    # 空 K（全剪枝空返回）
    empty = {"boxes": torch.zeros((0, 6)), "scores": torch.zeros((0,))}
    tgt = {"label": 1, "weight": 1.0,
           "gt": np.array([0, 0, 0, 1, 1, 1], np.float32)}
    parts = L.compute_record_losses(empty, tgt, L.LossConfig(gate=False))
    agg = L.aggregate_records([parts], [1.0])
    assert all(torch.isfinite(v) for v in agg.values())
    # 零权重聚合不炸
    parts2 = L.compute_record_losses({"boxes": torch.zeros((2, 6)), "scores": torch.full((2,), 0.9)},
                                     {"label": 0, "weight": 0.0, "gt": None}, L.LossConfig())
    agg0 = L.aggregate_records([parts2], [0.0])
    assert all(torch.isfinite(v) for v in agg0.values())
    # 负样本：bg 必须出现且 = mean(s^2)
    s = torch.tensor([0.6, 0.8])
    parts3 = L.compute_record_losses({"boxes": torch.zeros((2, 6)), "scores": s},
                                     {"label": 0, "weight": 1.0, "gt": None}, L.LossConfig())
    expect = float((s ** 2).mean())
    assert abs(float(parts3["bg"]) - expect) < 1e-6
    # Gate 损失只通过独立入口计算。
    pred_g = {"boxes": torch.zeros((2, 6)), "scores": torch.full((2,), 0.5),
              "p_exist": torch.tensor(0.4, requires_grad=True)}
    tgt_g = {"label": 1, "weight": 1.0,
             "gt": np.array([0, 0, 0, 1, 1, 1], np.float32)}
    pg = L.compute_gate_only_losses(pred_g, tgt_g, L.LossConfig(gate=True))
    assert {"gate_main", "gate_recall"} <= set(pg)
    (pg["gate_main"] + pg["gate_recall"]).backward()
    assert pred_g["p_exist"].grad is not None and abs(float(pred_g["p_exist"].grad)) > 0


def test_candidate_pairwise_rank() -> None:
    """R1v1：正 query Top-1 与同 view 最高分错误 query Top-1 配对。"""
    gt = np.array([0, 0, 0, 1, 1, 1], np.float32)
    pos_logits = torch.tensor([0.2, 0.8], requires_grad=True)
    neg_logits = torch.tensor([1.1, 0.5], requires_grad=True)
    preds = [
        {
            "boxes": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.float32),
            "scores": pos_logits[:1].sigmoid(), "logits": pos_logits[:1],
        },
        {
            "boxes": torch.tensor([[3, 3, 3, 1, 1, 1]], dtype=torch.float32),
            "scores": neg_logits[:1].sigmoid(), "logits": neg_logits[:1],
        },
    ]
    targets = [
        {"label": 1, "gt": gt},
        {"label": 0, "gt": None},
    ]
    cfg = L.LossConfig(rank=True, lam_rank=1.0, rank_margin=0.2,
                       rank_iou_threshold=0.25)
    pairs = L.select_view_cross_pairs(preds, targets, cfg)
    assert pairs == [(0, 0, 1, 0)], pairs
    loss = L.compute_view_rank_loss(preds, targets, cfg, pairs)
    expected = torch.nn.functional.softplus(torch.tensor(0.2 - 0.2 + 1.1))
    assert loss is not None and torch.allclose(loss, expected)
    parts = L.compute_view_rank_losses(preds, targets, cfg)
    assert set(parts) == {"rank_cross"}
    sum(parts.values()).backward()
    assert float(pos_logits.grad[0]) < 0.0 and float(neg_logits.grad[0]) > 0.0
    assert float(pos_logits.grad[1]) == 0.0 and float(neg_logits.grad[1]) == 0.0

    # 未达到 IoU 门槛时不得把错误框强制当正例。
    miss = dict(preds[0])
    miss["boxes"] = torch.tensor([[3, 3, 3, 1, 1, 1]], dtype=torch.float32)
    assert L.compute_view_rank_loss([miss, preds[1]], targets, cfg) is None
    assert L.compute_view_rank_loss(preds, targets, L.LossConfig(rank=False)) is None


def test_streamed_rank_gradient() -> None:
    """显存安全的 view 级重放梯度须与保留整 view 计算图完全一致。"""
    class RankStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[0.2, 0.8], [1.1, 0.5]]))
            self.register_buffer("boxes", torch.tensor([
                [[0, 0, 0, 1, 1, 1], [4, 4, 4, 1, 1, 1]],
                [[3, 3, 3, 1, 1, 1], [5, 5, 5, 1, 1, 1]],
            ], dtype=torch.float32))

        def forward_train(self, inputs):
            row = 0 if inputs["text"][0] == "target" else 1
            logits = self.logits[row]
            boxes = self.boxes[row]
            return {"boxes": boxes[:1], "scores": logits[:1].sigmoid(),
                    "logits": logits[:1]}

    gt = np.array([0, 0, 0, 1, 1, 1], np.float32)
    sample = {
        "scene": "stub", "view": 0,
        "points": np.zeros((4, 6), dtype=np.float32),
        "records": [
            {"query": "target", "label": 1, "weight": 1.0, "gt": gt},
            {"query": "absent", "label": 0, "weight": 1.0, "gt": None},
        ],
    }
    cfg = L.LossConfig(rank=True, rank_margin=0.2,
                       rank_iou_threshold=0.25)
    full = RankStub()
    streamed = RankStub()
    agg, _ = P.run_view(full, sample, cfg, torch.device("cpu"), training=True)
    total = L.total_of(agg)
    assert total is not None
    total.backward()
    streamed.zero_grad(set_to_none=True)
    agg_stream, _, n_bad = P.run_view_streamed(
        streamed, sample, cfg, torch.device("cpu"))
    assert n_bad == 0 and "rank_cross" in agg_stream
    assert abs(float(total.detach()) - sum(agg_stream.values())) < 1e-6
    assert torch.allclose(full.logits.grad, streamed.logits.grad, atol=1e-6), (
        full.logits.grad, streamed.logits.grad)


def test_gate_initialization_rng_isolation() -> None:
    """Gate construction must be reproducible without advancing CPU/CUDA RNG."""
    from vlfm.tsp3d_models.bdetr import _build_presence_gate_with_isolated_rng

    torch.manual_seed(12345)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    gate_a = _build_presence_gate_with_isolated_rng(128, 128, seed=7)
    assert torch.equal(cpu_before, torch.get_rng_state())
    if cuda_before is not None:
        assert torch.equal(cuda_before, torch.cuda.get_rng_state())

    torch.manual_seed(99999)
    gate_b = _build_presence_gate_with_isolated_rng(128, 128, seed=7)
    state_a = gate_a.state_dict()
    state_b = gate_b.state_dict()
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)


def test_loss_self_test() -> None:
    L._self_test()


def test_scheduler() -> None:
    base, ws, ts = 1e-4, 100, 300
    assert abs(P.lr_at(0, ws, ts, base) - 1e-6) < 1e-12
    assert abs(P.lr_at(ws, ws, ts, base) - base) < 1e-12          # warmup 峰值
    assert abs(P.lr_at(ts, ws, ts, base) - 1e-6) < 1e-12          # cosine 末端
    mid = P.lr_at(ws + (ts - ws) // 2, ws, ts, base)
    assert abs(mid - (1e-6 + (base - 1e-6) * 0.5)) < 1e-9
    assert P.lr_at(5, ws, ts, base) > P.lr_at(1, ws, ts, base)     # 单调 warmup


def test_tail_resume_scheduler() -> None:
    core = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([
        {"params": [core], "lr": 1e-6, "base_lr": 1e-4, "name": "g1_wd"},
    ])
    lr0 = P.set_tail_resume_lr(optimizer, 0, 10, 20, {"g1_wd": 1e-6})
    assert abs(lr0["g1_wd"] - 1e-6) < 1e-12
    lr1 = P.set_tail_resume_lr(optimizer, 10, 10, 20, {"g1_wd": 1e-6})
    assert abs(lr1["g1_wd"] - 1e-6) < 1e-12


def test_replay_mix_plan() -> None:
    """R2：replay 配对替换源视图，总训练步固定且计划可复现。"""
    replay_to_primary = list(range(10, 25))
    a = P.build_epoch_mix(
        np.random.default_rng(7), 90, 15, 0.10, replay_to_primary)
    b = P.build_epoch_mix(
        np.random.default_rng(7), 90, 15, 0.10, replay_to_primary)
    assert a == b
    original = [idx for source, idx in a if source == "original"]
    replay = [idx for source, idx in a if source == "replay"]
    assert len(a) == 90 and len(original) == 81 and len(replay) == 9
    assert len(set(replay)) == len(replay)
    replaced = {replay_to_primary[index] for index in replay}
    assert replaced.isdisjoint(original)
    assert sorted(set(original) | replaced) == list(range(90))
    assert abs(len(replay) / len(a) - 0.10) < 1e-12
    for epoch in range(3):
        baseline = P.build_epoch_mix(np.random.default_rng(7 + epoch), 90, 0, 0.0)
        mixed = P.build_epoch_mix(
            np.random.default_rng(7 + epoch), 90, 15, 0.10, replay_to_primary)
        baseline_sources = [index for _source, index in baseline]
        mixed_sources = [
            replay_to_primary[index] if source == "replay" else index
            for source, index in mixed
        ]
        assert mixed_sources == baseline_sources
    try:
        P.build_epoch_mix(np.random.default_rng(0), 1, 0, 0.10, [])
    except ValueError:
        pass
    else:
        raise AssertionError("空 replay 数据应被拒绝")


def test_gate_only_training() -> None:
    """Gate-only 前向/损失只能给 Presence Gate 产生梯度。"""
    class GateOnlyStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = torch.nn.Parameter(torch.tensor(2.0))
            self.presence_gate = torch.nn.Linear(1, 1)

        def forward_train(self, inputs):
            assert inputs.get("gate_only") is True
            logit = self.presence_gate.weight.reshape(()) + self.presence_gate.bias.reshape(())
            return {
                "boxes": torch.empty((0, 6)),
                "scores": torch.empty((0,)),
                "p_exist": torch.sigmoid(logit),
            }

    model = GateOnlyStub()
    sample = {
        "scene": "stub", "view": 0,
        "points": np.zeros((4, 6), dtype=np.float32),
        "records": [
            {"query": "chair", "label": 1, "weight": 1.0, "gt": None},
            {"query": "bed", "label": 0, "weight": 1.0, "gt": None},
        ],
    }
    agg, nrec, nbad = P.run_view_streamed(
        model, sample, L.LossConfig(gate=True), torch.device("cpu"), gate_only=True)
    assert nrec == 2 and nbad == 0
    assert set(agg) == {"gate_main", "gate_recall"}
    assert model.core.grad is None
    assert model.presence_gate.weight.grad is not None


def test_data_loader() -> None:
    from data_loader import LoaderConfig, LocalViewDataset

    root = _REPO / "data" / "ft_datasets" / "local_view"
    if not (root / "train").is_dir():
        raise RuntimeError("主数据集不存在（跳过：无 local_view/train）")
    n = len(sorted((root / "train").glob("*.npz")))
    if n == 0:
        raise RuntimeError("无 npz 文件")
    ds = LocalViewDataset(LoaderConfig(split="train", limit_views=3))
    s = ds[0]
    assert s["points"].dtype == np.float32 and s["points"].shape[1] == 6
    assert len(s["records"]) >= 1
    r0 = s["records"][0]
    assert {"query", "label", "weight", "gt", "poses"} <= set(r0)
    if r0["label"] == 1:
        assert r0["gt"] is not None and r0["gt"].shape == (6,)
    print(f"    dataset: scenes={len(ds.scene_ids)} views={len(ds.index)} "
          f"warnings={len(ds.warnings)}")


def test_stub_training() -> None:
    P.run_stub_training(steps=6, epochs=1)
    ckpt = Path("/tmp/ft_stub_run/ckpt_last.pth")
    assert ckpt.is_file(), "stub 训练未落 ckpt"
    state = torch.load(str(ckpt), map_location="cpu")
    assert "model" in state and "epoch" in state


def test_checkpoint_ft_config() -> None:
    """实际模型开关必须随 ckpt 保存；常驻新模块键不得参与模式猜测。"""
    from vlfm.vlm.tsp3d import (
        _checkpoint_ft_config,
        _checkpoint_state_dict,
        _load_tsp3d_checkpoint,
    )

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.ft_config = {
                "enable_gate": True,
                "num_proposals": 8,
                "gate_threshold": 0.3,
            }

    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ft_config.pth"
        args = argparse.Namespace(losses="gate")
        P.save_ckpt(path, model, optimizer, 0, 1, args)
        # 复用生产加载器（内含 torch < 1.13 的 weights_only 回退），两代 torch 均可跑。
        checkpoint = _load_tsp3d_checkpoint(str(path), "cpu")
        actual = _checkpoint_ft_config(checkpoint, _checkpoint_state_dict(checkpoint))
        assert actual == model.ft_config

    # 常驻推理模块键（bbox/cls conv）不参与模式猜测；只有门控头是条件实例化的。
    always_present = {"head.cls_conv.kernel": torch.zeros(1)}
    assert _checkpoint_ft_config({}, always_present) == {"enable_gate": False}
    with_gate = {"presence_gate.fuse.4.weight": torch.zeros(1)}
    assert _checkpoint_ft_config({}, with_gate) == {"enable_gate": True}


def test_attention_fastpath() -> None:
    """注意力快路径接线：need_weights=False + 全 False mask→None 不改数值输出。

    对应显存优化：原先 `nn.MultiheadAttention` 默认 need_weights=True 且 mask 非空
    （即使全 False）会强制慢路径并物化 [B×H, L, L] 注意力矩阵；大视图下单矩阵
    可达 GB 级。这里验证：① 空 mask 被改传 None；② 新旧调用输出浮点级一致。
    """
    from vlfm.tsp3d_models.trans_modules import (
        CrossAttentionLayer,
        PosTransformerEncoderLayerNoFFN,
        TransformerEncoderLayerNoFFN,
        _none_if_empty,
    )

    # ① 掩码助手行为
    assert _none_if_empty(None) is None
    empty = torch.zeros(1, 6, dtype=torch.bool)
    assert _none_if_empty(empty) is None
    nonempty = torch.tensor([[True, False, False, False, False, False]])
    assert _none_if_empty(nonempty) is nonempty

    # ② 输出一致性（CPU float64；dropout=0、eval 保证确定）
    S, B, D = 48, 1, 32
    x = torch.randn(S, B, D, dtype=torch.float64)
    pos = torch.randn(S, B, D, dtype=torch.float64)
    zeros = torch.zeros(B, S, dtype=torch.bool)

    layer = TransformerEncoderLayerNoFFN(d_model=D, nhead=4, dropout=0.0).double().eval()
    ref2 = layer.self_attn(x, x, x, attn_mask=None, key_padding_mask=zeros,
                           need_weights=True)[0]
    ref = layer.norm1(x + layer.dropout1(ref2))
    out_mask = layer(x, src_key_padding_mask=zeros)
    out_none = layer(x, src_key_padding_mask=None)
    assert torch.allclose(ref, out_mask, atol=1e-9), "快路径输出与旧调用不一致"
    assert torch.allclose(out_mask, out_none, atol=1e-12), "空 mask 未等价于 None"

    player = PosTransformerEncoderLayerNoFFN(d_model=D, nhead=4, dropout=0.0).double().eval()
    assert torch.allclose(player(x, pos, src_key_padding_mask=zeros),
                          player(x, pos, src_key_padding_mask=None), atol=1e-12)

    cross = CrossAttentionLayer(d_model=D, dropout=0.0, n_heads=4,
                                dim_feedforward=D).double().eval()
    v1, t1 = cross(x.transpose(0, 1), zeros, pos.transpose(0, 1), zeros,
                   pos.transpose(0, 1))
    v2, t2 = cross(x.transpose(0, 1), None, pos.transpose(0, 1), None,
                   pos.transpose(0, 1))
    assert torch.allclose(v1, v2, atol=1e-12) and torch.allclose(t1, t2, atol=1e-12)

    # ③ 梯度路径可用且一致（mask 空 vs None）
    xg = x.clone().requires_grad_(True)
    g1 = torch.autograd.grad(layer(xg, src_key_padding_mask=zeros).sum(), xg)[0]
    g2 = torch.autograd.grad(layer(xg, src_key_padding_mask=None).sum(), xg)[0]
    assert torch.allclose(g1, g2, atol=1e-10)


# ---------------------------------------------------------------------------
# 验收点（需模型侧就绪）
# ---------------------------------------------------------------------------


def check_prune_gradients(model: torch.nn.Module, sample: Dict[str, Any],
                          loss_cfg: Optional[L.LossConfig] = None) -> Dict[str, Any]:
    """验收：可微软剪枝重建后 `keep_conv` / `com_cls` 确实收到梯度（§11.3 #1）。"""
    cfg = loss_cfg or L.LossConfig(gate=False)
    model.train()
    model.zero_grad(set_to_none=True)
    agg, _ = P.run_view(model, sample, cfg, torch.device("cpu"), training=True)
    total = L.total_of(agg)
    assert total is not None, "无任何损失项"
    total.backward()
    report: Dict[str, Any] = {}
    for probe in ("head.keep_conv", "head.com_cls"):
        grads = []
        for name, p in model.named_parameters():
            if name.startswith(probe) and p.grad is not None:
                grads.append(float(p.grad.abs().sum()))
        report[probe] = {"n_params": len(grads), "grad_abs_sum": sum(grads)}
    return report


def overfit_run(args: argparse.Namespace) -> None:
    """1–2 场景过拟合跑（模型侧就绪后；直接走 pipeline 全链路）。"""
    argv = ["--data-dir", args.data_dir, "--split", "train",
            "--scenes", args.scenes or "", "--limit-views", str(args.limit_views or 20),
            "--epochs", str(args.epochs or 2), "--out-dir", args.out_dir or "/tmp/ft_overfit",
            "--losses", "gate", "--gate-only", "--unfreeze-profile", "gate"]
    P.train(P.parse_args(argv))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

OFFLINE_TESTS = [
    ("matcher_unique", test_matcher_unique),
    ("loss_edge_cases", test_loss_edge_cases),
    ("candidate_pairwise_rank", test_candidate_pairwise_rank),
    ("streamed_rank_gradient", test_streamed_rank_gradient),
    ("gate_initialization_rng_isolation", test_gate_initialization_rng_isolation),
    ("loss_self_test", test_loss_self_test),
    ("scheduler", test_scheduler),
    ("tail_resume_scheduler", test_tail_resume_scheduler),
    ("replay_mix_plan", test_replay_mix_plan),
    ("gate_only_training", test_gate_only_training),
    ("data_loader", test_data_loader),
    ("stub_training", test_stub_training),
    ("checkpoint_ft_config", test_checkpoint_ft_config),
    ("attention_fastpath", test_attention_fastpath),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests", default="", help="逗号分隔子集（默认全部离线用例）")
    ap.add_argument("--data-dir", default=str(_REPO / "data" / "ft_datasets" / "local_view"))
    ap.add_argument("--overfit", action="store_true", help="1–2 场景过拟合跑（需模型+数据）")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--limit-views", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.overfit:
        overfit_run(args)
        return

    want = {s.strip() for s in args.tests.split(",") if s.strip()}
    for name, fn in OFFLINE_TESTS:
        if want and name not in want:
            continue
        _check(name, fn)
    n_ok = sum(1 for r in RESULTS if r["ok"])
    print(f"\n== {n_ok}/{len(RESULTS)} passed ==")
    print(json.dumps(RESULTS, ensure_ascii=False, indent=1))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
