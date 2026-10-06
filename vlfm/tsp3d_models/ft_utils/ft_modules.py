from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _none_if_empty(mask):
    """全 False 布尔 mask 不含信息：改传 None 以启用高效注意力。"""
    if mask is None or not torch.is_tensor(mask):
        return mask
    if mask.dtype == torch.bool and not bool(mask.any()):
        return None
    return mask


def masked_mean(x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean-pool ``x`` over its sequence dimension.

    ``padding_mask`` follows ``MultiheadAttention`` semantics (``True`` means padding). 
    An all-padding row is handled safely and returns zeros.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [B,L,C], got {tuple(x.shape)}")
    if padding_mask is None:
        return x.mean(dim=1)
    keep = (~padding_mask).to(dtype=x.dtype).unsqueeze(-1)
    denom = keep.sum(dim=1).clamp_min(1.0)
    return (x * keep).sum(dim=1) / denom


class PresenceGate(nn.Module):
    """
    Global scene/text existence gate.

    The sparse tensor is pooled per batch item using its batch column.  
    Text tokens are mean pooled after masking and projected into the same feature space.  
    The output is a logit; callers should use ``sigmoid`` for a probability and 
    can choose their own post-training threshold.
    """

    def __init__(self, visual_dim: int = 128, text_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(visual_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.text = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        # A single pooled visual token queries the text sequence.
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=max(1, min(8, hidden_dim // 16)),
            batch_first=True, dropout=0.0)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pool_sparse(x, batch_size: int) -> torch.Tensor:
        feats = x.F
        batch = x.C[:, 0].long()
        pooled = feats.new_zeros((batch_size, feats.shape[1]))
        counts = feats.new_zeros((batch_size, 1))
        pooled.index_add_(0, batch, feats)
        counts.index_add_(0, batch, feats.new_ones((feats.shape[0], 1)))
        return pooled / counts.clamp_min(1.0)

    def forward(self, sparse_features, text_features, text_padding_mask=None) -> torch.Tensor:
        batch_size = int(text_features.shape[0])
        visual = self._pool_sparse(sparse_features, batch_size)
        visual_h = self.visual(visual)
        text_h = self.text(text_features)
        query = visual_h.unsqueeze(1)
        attn_mask = text_padding_mask
        if attn_mask is not None and bool(attn_mask.all(dim=1).any()):
            # MultiheadAttention produces NaNs for an all-masked row; 
            # retain a harmless zeroed token as the fallback key/value instead.
            attn_mask = attn_mask.clone()
            empty_rows = attn_mask.all(dim=1)
            attn_mask[empty_rows, 0] = False
        attended, _ = self.cross_attn(
            query, text_h, text_h, key_padding_mask=_none_if_empty(attn_mask),
            need_weights=False)
        return self.fuse(torch.cat([visual_h, masked_mean(text_h, text_padding_mask), attended.squeeze(1)], dim=-1)).squeeze(-1)


def center_dim_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(cx,cy,cz,dx,dy,dz)`` boxes to ``(x1,y1,z1,x2,y2,z2)``."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 6)
    half = boxes[..., 3:6].clamp_min(1e-4) / 2.0
    return torch.cat([boxes[..., :3] - half, boxes[..., :3] + half], dim=-1)


def focal_binary_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25,
                      gamma: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    """Numerically stable sigmoid focal loss for arbitrary-shaped tensors."""
    targets = targets.to(dtype=logits.dtype)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    at = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    out = at * (1.0 - pt).pow(gamma) * bce
    if reduction == "sum":
        return out.sum()
    if reduction == "none":
        return out
    return out.mean() if out.numel() else logits.sum() * 0.0


def asymmetric_binary_loss(logits: torch.Tensor, targets: torch.Tensor,
                           gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                           reduction: str = "mean") -> torch.Tensor:
    """Minimal ASL implementation used by the final candidate confidence."""
    targets = targets.to(dtype=logits.dtype)
    p = torch.sigmoid(logits)
    pos = targets * (1.0 - p).pow(gamma_pos) * torch.log(p.clamp_min(1e-6))
    neg = (1.0 - targets) * p.pow(gamma_neg) * torch.log((1.0 - p).clamp_min(1e-6))
    out = -(pos + neg)
    if reduction == "sum":
        return out.sum()
    if reduction == "none":
        return out
    return out.mean() if out.numel() else logits.sum() * 0.0
