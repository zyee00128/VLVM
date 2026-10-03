#!/usr/bin/env python3
"""Stage3 微调 · 冒烟与验收测试（§11.3 #1）。

覆盖（无需 GPU / 真实模型即可跑核心部分）：
  * 匹配器：唯一匹配 / 重复提议判负 / 1:1；
  * TGP 体素掩码：几何包含 + margin；
  * 损失边界：空 K、全剪枝空返回、零权重、单正样本、负样本 bg 项；
  * 数值自检：ASL/Focal（`losses._self_test`）；
  * 调度器：warmup 起点 → 第 1 epoch 末 = 目标 LR → 末步 = eta_min；
  * 几何：canonical 变换刚性 / 视锥判定 / 表面点采样；
  * GD 教师：深度支撑过滤 / 弃权规则（no_ok_frame、gd_miss_gt）/ 软目标；
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
import math
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
from ft_geometry import (  # noqa: E402
    FrameGeometry,
    aabb_surface_points,
    box_visibility,
    frustum_mask_frame,
    pose_canonical_to_frame,
)
from gd_teacher import GateTeacherConfig, GateTeacherStats, soft_target, teacher_from_sidecar, view_smax  # noqa: E402

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
    tgt = {"label": 1, "weight": 1.0, "gt": np.array([0.0, 0, 0, 1.0, 1.0, 1.0], np.float32),
           "entities": None, "teacher": None}
    parts = L.compute_record_losses(pred, tgt, L.LossConfig())
    assert "cls_base" in parts and "l1_base" in parts and "giou_base" in parts
    # 正样本不得出现 bg；重复提议（槽 0）必须判负：其 cls 贡献 > 0 且 l1 不作用于它
    assert "bg" not in parts
    # 匹配项必须是槽 1（IoU=1）：l1=0（槽 1 与 GT 完全重合）
    assert float(parts["l1_base"]) < 1e-5, f"l1 should match box-1, got {float(parts['l1_base'])}"


def test_matcher_multi_entity() -> None:
    pred = {"boxes": torch.tensor([[0.0, 0, 0, 1.0, 1.0, 1.0], [5.0, 5, 5, 1.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.8, 0.7]),
            "prop_boxes": torch.tensor([[0.0, 0, 0, 1.0, 1.0, 1.0], [5.0, 5, 5, 1.0, 1.0, 1.0]]),
            "prop_obj_logits": torch.tensor([3.0, -3.0])}
    ents = np.array([[0.0, 0, 0, 1.0, 1.0, 1.0], [5.0, 5, 5, 1.0, 1.0, 1.0]], np.float32)
    tgt = {"label": 0, "weight": 1.0, "gt": None, "entities": ents, "teacher": None}
    parts = L.compute_record_losses(pred, tgt, L.LossConfig(prop=True))
    assert "s1_obj" in parts and "s1_box" in parts, parts.keys()
    assert float(parts["s1_box"]) < 1e-5
    # 空白视锥：entities 为空数组（表存在）⇒ 全部 y_obj=0，仍计算 obj 项
    tgt0 = dict(tgt, entities=np.zeros((0, 6), np.float32))
    parts0 = L.compute_record_losses(pred, tgt0, L.LossConfig(prop=True))
    assert "s1_obj" in parts0 and "s1_box" not in parts0


def test_tgp_mask() -> None:
    vox = torch.tensor([[0.0, 0, 0], [2.0, 2, 2], [0.4, 0, 0]])
    gt = torch.tensor([0.0, 0, 0, 1.0, 1.0, 1.0])
    m = L.build_tgp_mask(vox, gt, 0.05)
    assert m.tolist() == [1.0, 0.0, 1.0], m.tolist()
    # margin 边界：0.56 在 +0.05 之外
    vox2 = torch.tensor([[0.56, 0, 0]])
    assert float(L.build_tgp_mask(vox2, gt, 0.05)[0]) == 0.0
    # 无目标 ⇒ 全 0
    assert float(L.build_tgp_mask(vox, None, 0.05).sum()) == 0.0
    # 簇级 BCE 不崩
    logits = torch.tensor([2.0, -2.0, 1.0])
    cid = torch.tensor([0, 0, 1])
    v = L.tgp_cluster_bce(logits, cid, vox, None, 0.05)
    assert v is not None and torch.isfinite(v)


def test_loss_edge_cases() -> None:
    # 空 K（全剪枝空返回）
    empty = {"boxes": torch.zeros((0, 6)), "scores": torch.zeros((0,))}
    tgt = {"label": 1, "weight": 1.0, "gt": np.array([0, 0, 0, 1, 1, 1], np.float32),
           "entities": None, "teacher": None}
    parts = L.compute_record_losses(empty, tgt, L.LossConfig(gate=False))
    agg = L.aggregate_records([parts], [1.0])
    assert all(torch.isfinite(v) for v in agg.values())
    # 零权重聚合不炸
    parts2 = L.compute_record_losses({"boxes": torch.zeros((2, 6)), "scores": torch.full((2,), 0.9)},
                                     {"label": 0, "weight": 0.0, "gt": None, "entities": None,
                                      "teacher": None}, L.LossConfig())
    agg0 = L.aggregate_records([parts2], [0.0])
    assert all(torch.isfinite(v) for v in agg0.values())
    # 负样本：bg 必须出现且 = mean(s^2)
    s = torch.tensor([0.6, 0.8])
    parts3 = L.compute_record_losses({"boxes": torch.zeros((2, 6)), "scores": s},
                                     {"label": 0, "weight": 1.0, "gt": None, "entities": None,
                                      "teacher": None}, L.LossConfig())
    expect = float((s ** 2).mean())
    assert abs(float(parts3["bg"]) - expect) < 1e-6
    # 门控乘子：Conf = P.detach()*s（对 P 无梯度路径）
    pred_g = {"boxes": torch.zeros((2, 6)), "scores": torch.full((2,), 0.5),
              "p_exist": torch.tensor(0.4, requires_grad=True)}
    tgt_g = {"label": 1, "weight": 1.0, "gt": np.array([0, 0, 0, 1, 1, 1], np.float32),
             "entities": None, "teacher": None}
    pg = L.compute_record_losses(pred_g, tgt_g, L.LossConfig(gate=True))
    assert "gate" in pg
    (pg["cls_base"] + pg["gate"]).backward()
    assert pred_g["p_exist"].grad is not None and abs(float(pred_g["p_exist"].grad)) > 0
    # s 回传（经 detach 的乘子）：scores 无 grad（叶子常量）——此处只验证不崩


def test_distill_only() -> None:
    """distill-only 档必须产生 gate_dist，且不得隐式附加硬标签 gate loss。"""
    p_exist = torch.tensor(0.4, requires_grad=True)
    pred = {"boxes": torch.zeros((1, 6)), "scores": torch.full((1,), 0.5),
            "p_exist": p_exist}
    tgt = {"label": 1, "weight": 1.0, "gt": np.array([0, 0, 0, 1, 1, 1], np.float32),
           "entities": None, "teacher": {"valid": True, "y": 0.8}}
    parts = L.compute_record_losses(pred, tgt, L.LossConfig(distill=True))
    assert "gate_dist" in parts and "gate" not in parts, parts.keys()
    parts["gate_dist"].backward()
    assert p_exist.grad is not None and abs(float(p_exist.grad)) > 0


def test_arch_ablation_order() -> None:
    """M0 全损失开启；M1-M4 各自移除一个模块，与 .sh 的单项增加相反。"""
    assert P.ABLATIONS["M0"] == {
        "gate": True, "prop": True, "refine": True, "tgp": True, "distill": True}
    assert not P.ABLATIONS["M1"]["gate"]
    assert not P.ABLATIONS["M2"]["prop"]
    assert not P.ABLATIONS["M3"]["refine"] and not P.ABLATIONS["M3"]["tgp"]
    assert not P.ABLATIONS["M4"]["distill"]
    for key, removed in (("M1", {"gate"}), ("M2", {"prop"}),
                         ("M3", {"refine", "tgp"}), ("M4", {"distill"})):
        off = {name for name, enabled in P.ABLATIONS[key].items() if not enabled}
        assert off == removed, (key, off)


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


def test_geometry() -> None:
    geo = FrameGeometry()
    q = {"x": 1.0, "y": 2.0, "floor": 0.0, "yaw_deg": 30.0}
    f = {"x": 2.0, "y": 1.0, "floor": 0.1, "yaw_deg": 90.0}
    pts = np.random.RandomState(0).randn(50, 3)
    a = pose_canonical_to_frame(pts, q, q)
    assert np.allclose(a, pts, atol=1e-9)                           # 同帧恒等
    b = pose_canonical_to_frame(pts, q, f)
    # 刚性：xy 内距离不因刚体变换改变
    d_out = np.sqrt(np.sum((b[:, :2] - b[0, :2]) ** 2, axis=1))
    d_ref = np.sqrt(np.sum((pts[:, :2] - pts[0, :2]) ** 2, axis=1))
    assert np.allclose(d_out, d_ref, atol=1e-9)
    # 视锥：正前方 2 m 可见；背后不可见
    front = np.array([[2.0, 0.0, 0.88]])
    back = np.array([[-2.0, 0.0, 0.88]])
    m = frustum_mask_frame(np.vstack([front, back]), geo)
    assert bool(m[0]) and not bool(m[1])
    # 表面点采样
    surf = aabb_surface_points(np.zeros(3), np.array([1.0, 1.0, 1.0]), 0.2, 500)
    assert len(surf) > 10 and np.all(surf <= 0.5001) and np.all(surf >= -0.5001)
    # box_visibility：正面前方 2m 的盒子任一帧可见
    info = box_visibility(np.array([2.0, 0.0, 0.5, 1.0, 1.0, 1.0]), q, [q], geo)
    assert info["any_visible"]


def test_gd_teacher() -> None:
    cfg = GateTeacherConfig()
    frames = [
        {"ok": True, "dets": [
            {"c": "couch", "b": [0, 0, 0.5, 0.5], "s": 0.8, "n": 100, "z": 2.0},
            {"c": "couch", "b": [0, 0, 0.5, 0.5], "s": 0.9, "n": 10, "z": 2.0},     # n<40 过滤
            {"c": "couch", "b": [0, 0, 0.5, 0.5], "s": 0.95, "n": 100, "z": 6.0},   # z 越界过滤
            {"c": "chair", "b": [0, 0, 0.5, 0.5], "s": 0.99, "n": 100, "z": 1.0},   # 非目标类
        ]},
        {"ok": False, "dets": [{"c": "couch", "b": [0, 0, 0.5, 0.5], "s": 0.99, "n": 100, "z": 1.0}]},
    ]
    info = view_smax(frames, "couch", cfg)
    assert abs(info["s_max"] - 0.8) < 1e-9 and info["any_ok"] and info["n_dets"] == 1
    # 软目标
    y = soft_target(0.8, cfg)
    expect = 1.0 / (1.0 + math.exp(-math.log(0.8 / 0.2) / 1.5))
    assert abs(y - expect) < 1e-9
    # S=0 ⇒ clamp(ε=1e-6) 下 Ŷ ≈ 1e-4（量级可忽略，§8 · 5「Ŷ → 0」）
    assert soft_target(0.0, cfg) < 1e-3
    # 弃权：全帧失败
    stats = GateTeacherStats()
    t = teacher_from_sidecar([{"ok": False, "dets": []}], "couch", 0, False, cfg, stats)
    assert not t["valid"] and t["reason"] == "no_ok_frame"
    # 弃权：正样本 GD 漏检（教师 0 且 GT 可见）
    t2 = teacher_from_sidecar([{"ok": True, "dets": []}], "couch", 1, True, cfg, GateTeacherStats())
    assert not t2["valid"] and t2["reason"] == "gd_miss_gt"
    # 有效：正样本 GD 漏检但 GT 不可见（弱可见场景不弃权——实际数据中已被生成规则剔除）
    t3 = teacher_from_sidecar([{"ok": True, "dets": []}], "couch", 1, False, cfg, GateTeacherStats())
    assert t3["valid"] and t3["s_max"] == 0.0
    # 负样本误检保留（有意召回侧偏置）
    t4 = teacher_from_sidecar(frames, "couch", 0, False, cfg, GateTeacherStats())
    assert t4["valid"] and t4["s_max"] > 0


def test_data_loader() -> None:
    from data_loader import LoaderConfig, LocalViewDataset

    root = _REPO / "data" / "ft_datasets" / "local_view"
    if not (root / "train").is_dir():
        raise RuntimeError("主数据集不存在（跳过：无 local_view/train）")
    n = len(sorted((root / "train").glob("*.npz")))
    if n == 0:
        raise RuntimeError("无 npz 文件")
    ds = LocalViewDataset(LoaderConfig(split="train", limit_views=3, use_sidecar=True))
    s = ds[0]
    assert s["points"].dtype == np.float32 and s["points"].shape[1] == 6
    assert len(s["records"]) >= 1
    r0 = s["records"][0]
    assert {"query", "label", "weight", "gt", "teacher", "entities"} <= set(r0)
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
                "enable_prop": False,
                "enable_refine": False,
                "num_proposals": 8,
                "gate_threshold": 0.3,
            }

    model = Toy()
    optimizer = torch.optim.AdamW(model.parameters())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ft_config.pth"
        args = argparse.Namespace(losses="distill", ablation=None)
        P.save_ckpt(path, model, optimizer, 0, 1, args)
        # 复用生产加载器（内含 torch < 1.13 的 weights_only 回退），两代 torch 均可跑。
        checkpoint = _load_tsp3d_checkpoint(str(path), "cpu")
        actual = _checkpoint_ft_config(checkpoint, _checkpoint_state_dict(checkpoint))
        assert actual == model.ft_config

    always_present = {
        "head.proposal_obj.kernel": torch.zeros(1),
        "head.entity_reg.weight": torch.zeros(1),
    }
    assert _checkpoint_ft_config({}, always_present) == {
        "enable_gate": False, "enable_prop": False, "enable_refine": False}
    assert _checkpoint_ft_config(
        {"args": {"losses": "distill"}}, always_present
    ) == {"enable_gate": True, "enable_prop": False, "enable_refine": False}
    # 旧 ckpt（只存 args、无 ft_config）：按消融名 M0 应还原为全开
    assert _checkpoint_ft_config(
        {"args": {"ablation": "M0"}}, always_present
    ) == {"enable_gate": True, "enable_prop": True, "enable_refine": True}


def test_s2_proposal_gradient() -> None:
    """S2 不启用 L_s1 时，最终 base loss仍须监督所选提议的 objectness。"""
    from vlfm.tsp3d_models.multilevel_head import TSPHead

    proposal = torch.tensor([-1.0, 0.5], requires_grad=True)
    match = torch.tensor([0.2, -0.3], requires_grad=True)
    final = TSPHead._combine_candidate_logits(proposal, match)
    torch.nn.functional.binary_cross_entropy_with_logits(
        final, torch.tensor([1.0, 0.0])
    ).backward()
    assert proposal.grad is not None and float(proposal.grad.abs().sum()) > 0
    assert match.grad is not None and float(match.grad.abs().sum()) > 0


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
    cfg = loss_cfg or L.LossConfig(gate=False, prop=True, refine=True, tgp=True)
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
            "--losses", "s1,s2,gate,tgp"]
    if args.no_sidecar:
        argv.append("--no-sidecar")
    P.train(P.parse_args(argv))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

OFFLINE_TESTS = [
    ("matcher_unique", test_matcher_unique),
    ("matcher_multi_entity", test_matcher_multi_entity),
    ("tgp_mask", test_tgp_mask),
    ("loss_edge_cases", test_loss_edge_cases),
    ("distill_only", test_distill_only),
    ("arch_ablation_order", test_arch_ablation_order),
    ("loss_self_test", test_loss_self_test),
    ("scheduler", test_scheduler),
    ("geometry", test_geometry),
    ("gd_teacher", test_gd_teacher),
    ("data_loader", test_data_loader),
    ("stub_training", test_stub_training),
    ("checkpoint_ft_config", test_checkpoint_ft_config),
    ("s2_proposal_gradient", test_s2_proposal_gradient),
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
    ap.add_argument("--no-sidecar", action="store_true")
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
