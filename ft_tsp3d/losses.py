#!/usr/bin/env python3
"""Stage3 微调管道 · 损失与匹配（S3 §8 全量实现；训练侧独立，无 policy/habitat 依赖）。

预测契约（模型侧 `forward_train` 输出，单 view 单 record；全部 float32）
--------------------------------------------------------------------
```
pred = {
  "boxes":          (K, 6)      # 交付框 canonical (x, y, z, l, w, h)，l/w/h = Δx/Δy/Δz
  "scores":         (K,)        # 交付置信度 s_k（sigmoid 后概率）
  "p_exist":        scalar | None   # 门控概率 P_exist（I_Gate 开时必需）
  # ---- M2+ 解耦分支（可选）----
  "prop_boxes":     (K1, 6) | None, "prop_obj_logits": (K1,) | None        # Stage 1 纯几何提议
  "match_logits":   (K,) | None, "refine_boxes": (K, 6) | None             # Stage 2 实体精炼
  "tgp_logits":     (V,) | None, "tgp_cluster_id": (V,) | None,            # 实体级 TGP（逐体素）
  "tgp_voxel_xyz":  (V, 3) | None   # 体素坐标（canonical 系 = coordinates × voxel_size）
}
```
模型输出框与数据层 GT、在线消费口径一致：**轴对齐 6 元组 (x, y, z, l, w, h)**（§9 定稿；
若模型侧内部回归序不同，由模型侧转齐后给出）。K/ K1 槽位数 = 最终输出候选数
（M0/M1 档 K=1；M2+ 档 = Stage 1 提议数）。

目标契约（`data_loader` 输出）
------------------------------
```
target = {"query", "label": 0|1, "weight": float,
          "gt": None | (6,) f32,               # 正样本 canonical (x, y, z, l, w, h)
          "entities": None | (M, 6) f32,       # L_s1：视锥内可见实体（None = 实例表缺失）
          "teacher": None | {"valid", "y", "s_max", "reason"}}  # L_gate_dist
```

损失总表（§8）
--------------
* `L_base`（恒定）= `cls_base`（ASL, γ+=1.0 / γ−=4.0，输入 = P_exist.detach()×s_k 的门控乘子）
  + [正样本] l1_base + giou3d_base + `bg`（负视锥二次压制 Σs_k²/K）；
* `L_gate`（I_Gate）= 加权 Focal（α = n_neg/(n_pos+n_neg)，γ=2）；
* `L_s1`（I_Prop）= obj Focal(α=0.5, γ=2，目标 = 视锥内可见实体匈牙利匹配) + [有实体] box1 + giou1；
* `L_s2`（I_Refine）= match Focal(α=0.25, γ=2) + [正样本] box2 + giou2 + [I_TGP] tgp BCE（簇级等权，
  体素级几何掩码 box_margin=0.05；无目标视图掩码全 0）；
* `L_gate_dist`（I_Distill）= 软-BCE（α=0.75），教师 Ŷ 严 detach；仅质检层通过样本参与。

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
import torch.nn.functional as F
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


def build_tgp_mask(voxel_xyz: torch.Tensor, gt_cxyzwh: Optional[torch.Tensor],
                   margin: float = 0.05) -> torch.Tensor:
    """体素级几何掩码（§8 · 4）：体素 ∈ 目标 GT AABB（含 margin）⇒ 1；无目标 ⇒ 全 0。"""
    v = voxel_xyz
    if gt_cxyzwh is None:
        return v.new_zeros((v.shape[0],))
    lo = gt_cxyzwh[:3] - gt_cxyzwh[3:] / 2.0 - float(margin)
    hi = gt_cxyzwh[:3] + gt_cxyzwh[3:] / 2.0 + float(margin)
    inside = ((v >= lo[None, :]) & (v <= hi[None, :])).all(dim=-1)
    return inside.to(v.dtype)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """消融开关（§8 头部）+ 系数（首轮基准全 1.0；蒸馏外层 λ_distill 只控启停）。"""

    # ---- 开关（base 恒定生效，不参与消融）----
    gate: bool = False           # I_Gate：L_gate
    prop: bool = False           # I_Prop：L_s1（Stage 1 纯几何提议）
    refine: bool = False         # I_Refine：L_s2（实体精炼/匹配）
    tgp: bool = True             # I_TGP：L_s2 内的 TGP 剪枝项（仅 M3+ 开）
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
    # ---- s1（§8 · 3）----
    s1_alpha: float = 0.5
    s1_gamma: float = 2.0
    lam_box1: float = 1.0
    lam_giou1: float = 1.0
    # ---- s2（§8 · 4）----
    s2_alpha: float = 0.25
    s2_gamma: float = 2.0
    lam_box2: float = 1.0
    lam_giou2: float = 1.0
    lam_tgp: float = 1.0
    # ---- distill（§8 · 5）----
    distill_alpha: float = 0.75
    # ---- misc ----
    box_margin: float = 0.05     # TGP 掩码含边（与数据层 target_pts 同一判定）
    eps: float = 1e-6


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

    # ---- L_s1（纯几何提议：obj Focal + 匹配实体的 box/giou）----
    if cfg.prop and pred.get("prop_boxes") is not None and pred.get("prop_obj_logits") is not None:
        prop_boxes = pred["prop_boxes"]
        p_obj = torch.sigmoid(pred["prop_obj_logits"])
        ents_np = target.get("entities")
        if ents_np is not None:
            ents = _as_tensor(ents_np, device).reshape(-1, 6)
            k1 = prop_boxes.shape[0]
            y_obj = torch.zeros(k1, device=device)
            if ents.shape[0] > 0 and k1 > 0:
                with torch.no_grad():
                    iou = iou3d_matrix(prop_boxes.detach(), ents)
                    cost = (1.0 - iou).cpu().numpy()
                rows, cols = hungarian_match(cost)
                if rows.size:
                    y_obj[torch.as_tensor(rows, device=device, dtype=torch.long)] = 1.0
                    out["s1_box"] = cfg.lam_box1 * l1_box(prop_boxes[rows], ents[cols]).mean()
                    out["s1_giou"] = cfg.lam_giou1 * (1.0 - giou3d_aligned(prop_boxes[rows], ents[cols])).mean()
            out["s1_obj"] = soft_focal(p_obj, y_obj, cfg.s1_alpha, cfg.s1_gamma, cfg.eps).mean()

    # ---- L_s2（实体精炼与匹配）----
    if cfg.refine and pred.get("match_logits") is not None:
        k = boxes.shape[0]
        y_m = torch.zeros(k, device=device)
        if k_star is not None:
            y_m[k_star] = 1.0
        p_match = torch.sigmoid(pred["match_logits"])
        out["s2_match"] = soft_focal(p_match, y_m, cfg.s2_alpha, cfg.s2_gamma, cfg.eps).mean()

        if k_star is not None and gt_np is not None and pred.get("refine_boxes") is not None:
            gt_t = _as_tensor(gt_np, device)
            rb = pred["refine_boxes"][k_star]
            out["s2_box"] = cfg.lam_box2 * l1_box(rb, gt_t)
            out["s2_giou"] = cfg.lam_giou2 * (1.0 - giou3d_aligned(rb.reshape(1, 6), gt_t.reshape(1, 6))[0])

        # ---- L_tgp_s2（BCE；簇级等权；体素级几何掩码）----
        if (cfg.tgp and pred.get("tgp_logits") is not None
                and pred.get("tgp_cluster_id") is not None and pred.get("tgp_voxel_xyz") is not None):
            tgp_loss = tgp_cluster_bce(
                pred["tgp_logits"], pred["tgp_cluster_id"], pred["tgp_voxel_xyz"],
                _as_tensor(gt_np, device) if gt_np is not None else None, cfg.box_margin)
            if tgp_loss is not None:
                out["tgp"] = cfg.lam_tgp * tgp_loss

    # ---- L_gate_dist（软-BCE；教师质检层通过样本才参与）----
    teacher = target.get("teacher")
    if cfg.distill and has_gate and isinstance(teacher, dict) and bool(teacher.get("valid")):
        y_soft = torch.as_tensor(float(teacher["y"]), device=device)
        out["gate_dist"] = soft_bce_alpha(p_exist, y_soft, cfg.distill_alpha, cfg.eps)

    return out


def tgp_cluster_bce(logits: torch.Tensor, cluster_id: torch.Tensor,
                    voxel_xyz: torch.Tensor, gt: Optional[torch.Tensor],
                    margin: float) -> Optional[torch.Tensor]:
    """簇级等权 TGP BCE：先簇内平均、再跨簇平均（§8 · 4）。"""
    if logits.numel() == 0 or cluster_id.numel() == 0:
        return None
    kmax = int(cluster_id.max().item()) + 1
    parts: List[torch.Tensor] = []
    for k in range(kmax):
        m = cluster_id == k
        if not bool(m.any()):
            continue
        mask = build_tgp_mask(voxel_xyz[m], gt, margin)
        parts.append(F.binary_cross_entropy_with_logits(logits[m].float(), mask.float()))
    if not parts:
        return None
    return torch.stack(parts).mean()


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

    # TGP 掩码
    vox = torch.tensor([[0.0, 0, 0], [2.0, 2, 2], [0.4, 0, 0]])
    gt = torch.tensor([0.0, 0, 0, 1.0, 1.0, 1.0])
    print("tgp mask:", build_tgp_mask(vox, gt, 0.05).tolist())  # [1, 0, 1]

    # 空批守卫
    empty = compute_record_losses({"boxes": torch.zeros((0, 6)), "scores": torch.zeros((0,))},
                                  {"label": 0, "weight": 1.0, "gt": None, "entities": None,
                                   "teacher": None}, LossConfig())
    print("empty parts:", {k: round(float(v), 4) for k, v in empty.items()})


if __name__ == "__main__":
    _self_test()
