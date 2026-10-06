#!/usr/bin/env python3
"""Stage3 微调管道 · 损失与匹配（S3 §8 全量实现；训练侧独立，无 policy/habitat 依赖）。

预测契约（模型侧 `forward_train` 输出，单 view 单 record；全部 float32）
--------------------------------------------------------------------
```
pred = {
  "boxes":          (K, 6)      # 交付框 canonical (x, y, z, l, w, h)，l/w/h = Δx/Δy/Δz
  "scores":         (K,)        # 交付置信度 s_k（sigmoid 后概率）
  "p_exist":        scalar | None   # 门控概率 P_exist（I_Gate 开时必需）
}
```
模型输出框与数据层 GT、在线消费口径一致：**轴对齐 6 元组 (x, y, z, l, w, h)**（§9 定稿；
若模型侧内部回归序不同，由模型侧转齐后给出）。K = 最终输出候选数（单候选路径 K=1）。

S1/S2 解耦分支的 pred 扩展键（`prop_boxes` / `prop_obj_logits` / `match_logits` /
`refine_boxes` / `tgp_logits` / `tgp_cluster_id` / `tgp_voxel_xyz`）随该分支一并移出，
见 `ft_s12/losses_s12.py`。

目标契约（`data_loader` 输出）
------------------------------
```
target = {"query", "label": 0|1, "weight": float,
          "gt": None | (6,) f32,               # 正样本 canonical (x, y, z, l, w, h)
          "entities": None | (M, 6) f32,       # L_s1 标签面（S1/S2 已归档，暂不消费）
          "teacher": None | {"valid", "y", "s_max", "reason"}}  # L_gate_dist
```

损失总表
--------
* `L_base`（恒定）= `cls_base`（ASL, γ+=1.0 / γ−=4.0，输入 = P_exist.detach()×s_k 的门控乘子）
  + [正样本] l1_base + giou3d_base + `bg`（负视锥二次压制 Σs_k²/K）；
* `L_gate`（I_Gate）= 加权 Focal（α = n_neg/(n_pos+n_neg)，γ=2）；
* `L_gate_dist`（I_Distill）= 软-BCE（α=0.75），教师 Ŷ 严 detach；仅质检层通过样本参与。

`L_s1`（I_Prop）与 `L_s2`（I_Refine，含 I_TGP）随 S1/S2 解耦分支一并移出，
见 `ft_s12/losses_s12.py`；接回步骤见 `ft_s12/__init__.py`。

归约：各分项 = 批内加权均值（Σ w·ℓ / Σ w，w = record.weight）；空批 / 零正样本 / 全零权重
均有守卫，缺失分支返回 0 标量并跳过。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from axis_aligned_iou_loss import axis_aligned_bbox_overlaps_3d  # noqa: E402

# ---------------------------------------------------------------------------
# 损失基元
# ---------------------------------------------------------------------------


def soft_asl(p: torch.Tensor, y: torch.Tensor, gamma_pos: float = 1.0,
             gamma_neg: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    """Asymmetric Loss（soft-binary 形式，`y` 可为 0/1 或软值）。

    ℓ = −y·(1−p)^{γ+}·log p − (1−y)·p^{γ−}·log(1−p)
    """
    p = p.clamp(eps, 1.0 - eps)
    pos = -((1.0 - p).pow(gamma_pos)) * torch.log(p)
    neg = -(p.pow(gamma_neg)) * torch.log(1.0 - p)
    return y * pos + (1.0 - y) * neg


def soft_focal(p: torch.Tensor, y: torch.Tensor, alpha: float,
               gamma: float, eps: float = 1e-6) -> torch.Tensor:
    """加权 Focal（soft-binary 形式；对硬标签与标准 FL 等价）。

    ℓ = −α·y·(1−p)^γ·log p − (1−α)·(1−y)·p^γ·log(1−p)
    """
    p = p.clamp(eps, 1.0 - eps)
    pos = -alpha * (1.0 - p).pow(gamma) * torch.log(p)
    neg = -(1.0 - alpha) * p.pow(gamma) * torch.log(1.0 - p)
    return y * pos + (1.0 - y) * neg


def soft_bce_alpha(p: torch.Tensor, y: torch.Tensor, alpha: float,
                   eps: float = 1e-6) -> torch.Tensor:
    """加权软 BCE（§8 · 5：L_gate_dist 的 α-配平形式；p = sigmoid 概率）。"""
    p = p.clamp(eps, 1.0 - eps)
    return -(alpha * y * torch.log(p) + (1.0 - alpha) * (1.0 - y) * torch.log(1.0 - p))


def corners_from_cxyzwh(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 6) center+size -> (..., 6) corner (x1, y1, z1, x2, y2, z2)。"""
    c, s = boxes[..., :3], boxes[..., 3:]
    return torch.cat([c - s / 2.0, c + s / 2.0], dim=-1)


def iou3d_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """(N, 6) × (M, 6) -> (N, M) 轴对齐 3D IoU。"""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    ca = corners_from_cxyzwh(boxes_a)[None]
    cb = corners_from_cxyzwh(boxes_b)[None]
    return axis_aligned_bbox_overlaps_3d(ca, cb, mode="iou", is_aligned=False)[0]


def giou3d_aligned(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """(N, 6) × (N, 6) -> (N,) 逐对 GIoU（轴对齐）。"""
    if boxes_a.numel() == 0:
        return boxes_a.new_zeros((0,))
    ca = corners_from_cxyzwh(boxes_a)[None]
    cb = corners_from_cxyzwh(boxes_b)[None]
    return axis_aligned_bbox_overlaps_3d(ca, cb, mode="giou", is_aligned=True)[0]


def hungarian_match(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """匈牙利 1:1 匹配（成本 = 1 − IoU3D）；返回 (row, col)。"""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    r, c = linear_sum_assignment(cost)
    return r.astype(np.int64), c.astype(np.int64)


def l1_box(pred6: torch.Tensor, gt6: torch.Tensor) -> torch.Tensor:
    """中心位置 L1 + 尺寸对数尺度 L1（轴对齐，无 yaw；§8）。"""
    dc = (pred6[..., :3] - gt6[..., :3]).abs().sum(dim=-1)
    dpred = pred6[..., 3:].clamp(min=1e-6)
    dgt = gt6[..., 3:].clamp(min=1e-6)
    dd = (torch.log(dpred) - torch.log(dgt)).abs().sum(dim=-1)
    return dc + dd


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """消融开关（§8 头部）+ 系数（首轮基准全 1.0；蒸馏外层 λ_distill 只控启停）。"""

    # ---- 开关（base 恒定生效，不参与消融）----
    gate: bool = False           # I_Gate：L_gate
    distill: bool = False        # I_Distill：L_gate_dist
    # ---- base（§8 · 1）----
    lam_l1: float = 1.0
    lam_giou: float = 1.0
    lam_bg: float = 1.0
    asl_gamma_pos: float = 1.0
    asl_gamma_neg: float = 4.0
    # ---- gate（§8 · 2）----
    gate_alpha: float = 0.75     # 设计配比 1:3；生成后按 stats.json 实测复核
    gate_gamma: float = 2.0
    # ---- distill（§8 · 5）----
    distill_alpha: float = 0.75
    # ---- misc ----
    eps: float = 1e-6
    # S1/S2（I_Prop / I_Refine / I_TGP）开关与系数已随解耦分支移出，
    # 见 `ft_s12/losses_s12.S12LossConfig`；接回时把那些字段并回本类。


# ---------------------------------------------------------------------------
# 单 record 损失
# ---------------------------------------------------------------------------


def _as_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def compute_record_losses(pred: Dict[str, Any], target: Dict[str, Any],
                          cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """单 record（一次前向）的全部损失分项（未加权标量张量）。

    返回 dict（缺分支的项自动跳过）；调用方按 record.weight 聚合。
    """
    out: Dict[str, torch.Tensor] = {}
    boxes = pred["boxes"]
    device = boxes.device
    scores = pred.get("scores")
    label = int(target.get("label", 0))
    gt_np = target.get("gt")

    # ---- 交付置信度：Conf_k = P_exist.detach() × s_k（§8 · 2 梯度边界）----
    p_exist = pred.get("p_exist")
    # Gate 头在 gate 或 distill 档都会实例化。是否存在门控输出与是否启用硬标签
    # L_gate 必须分开判断，否则 distill-only 档会因 cfg.gate=False 而静默跳过
    # L_gate_dist。只要模型含 Gate，交付置信度都遵循 P_exist.detach() * s_k。
    has_gate = p_exist is not None
    deliver = (p_exist.detach() * scores) if has_gate else scores

    # ---- 与真目标的匈牙利匹配（成本 = 1 − IoU3D；1:1，重复提议判负）----
    k_star: Optional[int] = None
    if gt_np is not None and boxes.shape[0] > 0:
        gt_t = _as_tensor(gt_np, device)
        with torch.no_grad():
            iou = iou3d_matrix(boxes.detach(), gt_t.reshape(1, 6))  # (K, 1)
            cost = (1.0 - iou).cpu().numpy()
        rows, _cols = hungarian_match(cost)
        if rows.size:
            k_star = int(rows[0])

    # ---- L_cls_base（ASL，独占；终端置信度）----
    if scores is not None and deliver.numel() > 0:
        y_cls = torch.zeros_like(deliver)
        if k_star is not None:
            y_cls[k_star] = 1.0
        out["cls_base"] = soft_asl(deliver, y_cls, cfg.asl_gamma_pos, cfg.asl_gamma_neg, cfg.eps).mean()
        # ---- L_bg（负视锥二次压制；base 恒常组件）----
        if label == 0:
            out["bg"] = cfg.lam_bg * (deliver.pow(2).mean())

    # ---- L_l1_base / L_giou3d_base（仅正样本匹配项）----
    if k_star is not None and gt_np is not None:
        gt_t = _as_tensor(gt_np, device)
        b = boxes[k_star]
        out["l1_base"] = cfg.lam_l1 * l1_box(b, gt_t)
        out["giou_base"] = cfg.lam_giou * (1.0 - giou3d_aligned(b.reshape(1, 6), gt_t.reshape(1, 6))[0])

    # ---- L_gate（加权 Focal，终端召回守卫）----
    if cfg.gate and has_gate:
        y_ex = torch.as_tensor(float(label), device=device)
        out["gate"] = soft_focal(p_exist, y_ex, cfg.gate_alpha, cfg.gate_gamma, cfg.eps)

    # ---- L_gate_dist（软-BCE；教师质检层通过样本才参与）----
    teacher = target.get("teacher")
    if cfg.distill and has_gate and isinstance(teacher, dict) and bool(teacher.get("valid")):
        y_soft = torch.as_tensor(float(teacher["y"]), device=device)
        out["gate_dist"] = soft_bce_alpha(p_exist, y_soft, cfg.distill_alpha, cfg.eps)

    return out


# ---------------------------------------------------------------------------
# 视图/批次聚合
# ---------------------------------------------------------------------------


def aggregate_records(
    loss_dicts: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """按 record.weight 加权平均（Σ w·ℓ / Σ w）；仅对实际出现的分项聚合。"""
    keys: List[str] = []
    for d in loss_dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    agg: Dict[str, torch.Tensor] = {}
    for k in keys:
        vals = [d[k] for d in loss_dicts if k in d]
        ws = [float(w) for d, w in zip(loss_dicts, weights) if k in d]
        if not vals:
            continue
        wsum = max(sum(ws), eps)
        agg[k] = sum(w * v for w, v in zip(ws, vals)) / wsum
    return agg


def total_of(parts: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """分项求和；无任何项时返回 None（调用方守卫：requires_grad 检查）。"""
    if not parts:
        return None
    total = None
    for v in parts.values():
        total = v if total is None else total + v
    return total


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def _self_test() -> None:
    # 数值对照（§9 的算例）
    p = torch.tensor([0.7, 0.9])
    y0 = torch.zeros(2)
    asl_neg = soft_asl(p, y0, gamma_neg=4.0)
    print("ASL 负样本 p=0.7/0.9:", [round(float(v), 4) for v in asl_neg])  # ~0.10 / ~1.6x
    fl = soft_focal(p, torch.ones(2), alpha=0.5, gamma=2.0)
    print("Focal 正样本 p=0.7/0.9:", [round(float(v), 4) for v in fl])

    # 匹配器：K=3，目标在槽 1（IoU 最高）
    pred = {
        "boxes": torch.tensor([[0.0, 0, 0, 1.0, 1.0, 1.0],
                               [0.0, 0, 0, 1.0, 1.0, 1.0],   # 与 GT 完全重合
                               [3.0, 3, 3, 1.0, 1.0, 1.0]]),
        "scores": torch.tensor([0.6, 0.9, 0.7]),
    }
    target = {"label": 1, "weight": 1.0,
              "gt": np.array([0.0, 0, 0, 1.0, 1.0, 1.0], dtype=np.float32),
              "entities": np.zeros((0, 6), dtype=np.float32), "teacher": None}
    parts = compute_record_losses(pred, target, LossConfig())
    print("base parts:", {k: round(float(v), 4) for k, v in parts.items()})

    # 空批守卫
    empty = compute_record_losses({"boxes": torch.zeros((0, 6)), "scores": torch.zeros((0,))},
                                  {"label": 0, "weight": 1.0, "gt": None, "entities": None,
                                   "teacher": None}, LossConfig())
    print("empty parts:", {k: round(float(v), 4) for k, v in empty.items()})


if __name__ == "__main__":
    _self_test()
