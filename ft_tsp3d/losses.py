#!/usr/bin/env python3
"""微调管道 · 损失与匹配（训练侧独立，无 policy/habitat 依赖）。

预测契约（模型侧 `forward_train` 输出，单 view 单 record；全部 float32）
--------------------------------------------------------------------
```
pred = {
  "boxes":          (K, 6)      # 交付框 canonical (x, y, z, l, w, h)，l/w/h = Δx/Δy/Δz
  "scores":         (K,)        # 交付置信度 s_k（sigmoid 后概率）
  "p_exist":        scalar | None   # 门控概率 P_exist（I_Gate 开时必需）
}
```
模型输出框与数据层 GT、在线消费口径一致：
轴对齐 6 元组 (x, y, z, l, w, h)（若模型侧内部回归序不同，由模型侧转齐后给出）。
K = 最终输出候选数（单候选路径 K=1）。

目标契约（`data_loader` 输出）
------------------------------
```
target = {"query", "label": 0|1, "weight": float,
          "gt": None | (6,) f32}               # 正样本 canonical (x, y, z, l, w, h)
```

损失总表
--------
* `L_base`（恒定）= `cls_base`（ASL, γ+=1.0 / γ−=4.0，输入 = 核心 raw s_k）
  + [正样本](l1_base + giou3d_base) + `bg`（负视锥二次压制 Σs_k²/K）；
* `L_cross`（I_Rank，view 级）= 正 query 的交付 Top-1 与同 view 错误 query
  的最高分 Top-1 做 `softplus(m - z_pos + z_hard_neg)`；
* `L_gate` 仅用于冻结核心的独立 Gate 训练：正样本完整 BCE + 负样本 focal + 正样本召回下界。

归约：各分项 = 批内加权均值（Σ w·ℓ / Σ w，w = record.weight）；
空批 / 零正样本 / 全零权重均有守卫，缺失分支返回 0 标量并跳过。

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


# Loss Init
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

def hungarian_match(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """匈牙利 1:1 匹配（成本 = 1 − IoU3D）；返回 (row, col)。"""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    r, c = linear_sum_assignment(cost)
    return r.astype(np.int64), c.astype(np.int64)

# base / rank / gate Loss
def soft_asl(p: torch.Tensor, y: torch.Tensor, gamma_pos: float = 1.0,
             gamma_neg: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Asymmetric Loss（soft-binary 形式，`y` 可为 0/1 或软值）。

    ℓ = −y·(1−p)^{γ+}·log p − (1−y)·p^{γ−}·log(1−p)
    """
    p = p.clamp(eps, 1.0 - eps)
    pos = -((1.0 - p).pow(gamma_pos)) * torch.log(p)
    neg = -(p.pow(gamma_neg)) * torch.log(1.0 - p)
    return y * pos + (1.0 - y) * neg

def l1_box(pred6: torch.Tensor, gt6: torch.Tensor) -> torch.Tensor:
    """中心位置 L1 + 尺寸对数尺度 L1（轴对齐，无 yaw）。"""
    dc = (pred6[..., :3] - gt6[..., :3]).abs().sum(dim=-1)
    dpred = pred6[..., 3:].clamp(min=1e-6)
    dgt = gt6[..., 3:].clamp(min=1e-6)
    dd = (torch.log(dpred) - torch.log(dgt)).abs().sum(dim=-1)
    return dc + dd

def giou3d_aligned(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """(N, 6) × (N, 6) -> (N,) 逐对 GIoU（轴对齐）。"""
    if boxes_a.numel() == 0:
        return boxes_a.new_zeros((0,))
    ca = corners_from_cxyzwh(boxes_a)[None]
    cb = corners_from_cxyzwh(boxes_b)[None]
    return axis_aligned_bbox_overlaps_3d(ca, cb, mode="giou", is_aligned=True)[0]

def pairwise_rank_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor,
                       margin: float = 0.2) -> torch.Tensor:
    """候选 pairwise logistic ranking：softplus(m - z_pos + z_neg)。"""
    return F.softplus(float(margin) - pos_logits + neg_logits)

def presence_gate_terms(
    p: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    gamma_neg: float = 2.0,
    positive_margin: float = 0.70,
    recall_weight: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Presence gate：正 BCE、负 focal、正样本 margin 召回守卫。"""
    p = p.clamp(eps, 1.0 - eps)
    y = y.to(device=p.device, dtype=p.dtype)
    main = (
        -float(alpha) * y * torch.log(p)
        -(1.0 - float(alpha)) * (1.0 - y) * p.pow(float(gamma_neg))
        * torch.log(1.0 - p)
    )
    out = {"gate_main": main}
    if bool((y > 0.5).item()):
        out["gate_recall"] = float(recall_weight) * torch.relu(
            p.new_tensor(float(positive_margin)) - p).pow(2)
    return out


# Config Init
@dataclass
class LossConfig:
    """消融开关 + 系数（首轮基准全 1.0）。"""

    # ---- 开关（base 恒定生效）----
    gate: bool = False           # I_Gate：L_gate
    rank: bool = False           # I_Rank：跨 query Top-1 ranking
    # ---- base ----
    lam_l1: float = 1.0
    lam_giou: float = 1.0
    lam_bg: float = 1.0
    asl_gamma_pos: float = 1.0
    asl_gamma_neg: float = 4.0
    # ---- gate ----
    gate_alpha: float = 0.75     # 设计配比 1:3；生成后按 stats.json 实测复核
    gate_gamma_neg: float = 2.0
    gate_positive_margin: float = 0.70
    gate_recall_weight: float = 1.0
    # ---- candidate ranking ----
    lam_rank: float = 1.0
    rank_margin: float = 0.2
    rank_iou_threshold: float = 0.25
    # ---- misc ----
    eps: float = 1e-6


# Loss per record
def _as_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

def delivery_tensors(pred: Dict[str, Any], eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """读取在线交付 Top-1 `(boxes, raw_logits)`。"""
    boxes = pred.get("boxes")
    if boxes is None:
        raise RuntimeError("delivery candidate contract missing boxes")
    logits = pred.get("logits")
    if logits is None:
        scores = pred.get("scores")
        if scores is None:
            raise RuntimeError("delivery candidate contract missing scores/logits")
        logits = torch.logit(scores.clamp(float(eps), 1.0 - float(eps)))
    boxes = boxes.reshape(-1, 6)
    logits = logits.reshape(-1)
    if boxes.shape[0] != logits.shape[0]:
        raise RuntimeError(
            f"delivery candidate count mismatch: boxes={boxes.shape[0]} logits={logits.shape[0]}")
    return boxes, logits

def select_view_cross_pairs(
    preds: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    cfg: LossConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    R1v1 跨 query 排序：只消费各 query 的交付 Top-1。

    正 query 的 Top-1 达到 IoU 门后，与同 view 其他错误 query 的最高 raw-logit Top-1 配对。
    选择过程严格 detach，仅排序标量保留梯度。
    """
    if len(preds) != len(targets):
        raise ValueError(f"pred/target length mismatch: {len(preds)} != {len(targets)}")

    candidates: List[Tuple[torch.Tensor, torch.Tensor]] = []
    positive_keys: List[Tuple[int, int]] = []
    for ri, (pred, target) in enumerate(zip(preds, targets)):
        boxes, logits = delivery_tensors(pred, cfg.eps)
        candidates.append((boxes, logits))
        if int(target.get("label", 0)) != 1 or target.get("gt") is None or boxes.shape[0] == 0:
            continue
        gt = _as_tensor(target["gt"], boxes.device, boxes.dtype).reshape(1, 6)
        with torch.no_grad():
            best_iou = iou3d_matrix(boxes[:1].detach(), gt).reshape(-1)[0]
        if float(best_iou) >= float(cfg.rank_iou_threshold):
            positive_keys.append((ri, 0))
    if not positive_keys:
        return []
    
    positive_set = set(positive_keys)
    wrong: List[Tuple[float, int, int]] = []
    for ri, (_boxes, logits) in enumerate(candidates):
        if logits.numel() and (ri, 0) not in positive_set:
            wrong.append((float(logits[0].detach()), ri, 0))
    if not wrong:
        return []
    
    _score, neg_record, neg_idx = max(wrong, key=lambda item: item[0])
    return [(pos_record, pos_idx, neg_record, neg_idx)
            for pos_record, pos_idx in positive_keys]

def _loss_from_pairs(
    preds: Sequence[Dict[str, Any]],
    pairs: Sequence[Tuple[int, int, int, int]],
    cfg: LossConfig,
) -> Optional[torch.Tensor]:
    if not pairs:
        return None
    tensors = [delivery_tensors(pred, cfg.eps)[1] for pred in preds]
    terms = [
        pairwise_rank_loss(tensors[pr][pi], tensors[nr][ni], cfg.rank_margin)
        for pr, pi, nr, ni in pairs
    ]
    return float(cfg.lam_rank) * torch.stack(terms).mean()

def compute_view_rank_losses(
    preds: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """计算 R1v1 跨 query Top-1 排序损失。"""
    out: Dict[str, torch.Tensor] = {}
    if cfg.rank:
        cross = _loss_from_pairs(
            preds, select_view_cross_pairs(preds, targets, cfg), cfg)
        if cross is not None:
            out["rank_cross"] = cross
    return out


def compute_record_losses(pred: Dict[str, Any], target: Dict[str, Any],
                          cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """
    单 record 的全部损失分项（未加权标量张量）。
    返回 dict（缺分支的项自动跳过）；调用方按 record.weight 聚合。
    """
    out: Dict[str, torch.Tensor] = {}
    boxes = pred["boxes"]
    device = boxes.device
    scores = pred.get("scores")
    label = int(target.get("label", 0))
    gt_np = target.get("gt")
    deliver = scores

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

    # ---- L_cls_base（ASL，独占）----
    if scores is not None and deliver.numel() > 0:
        y_cls = torch.zeros_like(deliver)
        if k_star is not None:
            y_cls[k_star] = 1.0
        out["cls_base"] = soft_asl(deliver, y_cls, cfg.asl_gamma_pos, cfg.asl_gamma_neg, cfg.eps).mean()
        # ---- L_bg（负视锥二次压制）----
        if label == 0:
            out["bg"] = cfg.lam_bg * (deliver.pow(2).mean())

    # ---- L_l1_base / L_giou3d_base（仅正样本匹配项）----
    if k_star is not None and gt_np is not None:
        gt_t = _as_tensor(gt_np, device)
        b = boxes[k_star]
        out["l1_base"] = cfg.lam_l1 * l1_box(b, gt_t)
        out["giou_base"] = cfg.lam_giou * (1.0 - giou3d_aligned(b.reshape(1, 6), gt_t.reshape(1, 6))[0])

    return out


def compute_gate_only_losses(pred: Dict[str, Any], target: Dict[str, Any],
                             cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """原始数据独立监督 Presence Gate，不触碰 base/rank 核心路径。"""
    if not cfg.gate or pred.get("p_exist") is None:
        return {}

    p_exist = pred["p_exist"]
    y_ex = torch.as_tensor(float(target.get("label", 0)), device=p_exist.device)
    return presence_gate_terms(
        p_exist, y_ex, cfg.gate_alpha, cfg.gate_gamma_neg,
        cfg.gate_positive_margin, cfg.gate_recall_weight, cfg.eps)


# Summary / Aggregate (used by dry-run patch / run_view)
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


def compute_view_rank_loss(
    preds: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    cfg: LossConfig,
    pairs: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> Optional[torch.Tensor]:
    """计算 view 级排序损失；没有有效正负 pair 时返回 None。"""
    if not cfg.rank:
        return None
    selected = list(pairs) if pairs is not None else select_view_cross_pairs(preds, targets, cfg)
    return _loss_from_pairs(preds, selected, cfg)

def _self_test() -> None:
    # 数值对照
    p = torch.tensor([0.7, 0.9])
    y0 = torch.zeros(2)
    asl_neg = soft_asl(p, y0, gamma_neg=4.0)
    print("ASL 负样本 p=0.7/0.9:", [round(float(v), 4) for v in asl_neg])  # ~0.10 / ~1.6x
    # 匹配器：K=3，目标在槽 1（IoU 最高）
    pred = {
        "boxes": torch.tensor([[0.0, 0, 0, 1.0, 1.0, 1.0],
                               [0.0, 0, 0, 1.0, 1.0, 1.0],   # 与 GT 完全重合
                               [3.0, 3, 3, 1.0, 1.0, 1.0]]),
        "scores": torch.tensor([0.6, 0.9, 0.7]),
    }
    target = {"label": 1, "weight": 1.0,
              "gt": np.array([0.0, 0, 0, 1.0, 1.0, 1.0], dtype=np.float32)}
    parts = compute_record_losses(pred, target, LossConfig())
    print("base parts:", {k: round(float(v), 4) for k, v in parts.items()})

    # 空批守卫
    empty = compute_record_losses({"boxes": torch.zeros((0, 6)), "scores": torch.zeros((0,))},
                                  {"label": 0, "weight": 1.0, "gt": None}, LossConfig())
    print("empty parts:", {k: round(float(v), 4) for k, v in empty.items()})

if __name__ == "__main__":
    _self_test()
