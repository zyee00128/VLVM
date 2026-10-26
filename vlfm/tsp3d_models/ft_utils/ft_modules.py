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
    def _pool_sparse(x, batch_size: int, detach: bool = False) -> torch.Tensor:
        feats = x.F.detach() if detach else x.F
        batch = x.C[:, 0].long()
        pooled = feats.new_zeros((batch_size, feats.shape[1]))
        counts = feats.new_zeros((batch_size, 1))
        pooled.index_add_(0, batch, feats)
        counts.index_add_(0, batch, feats.new_ones((feats.shape[0], 1)))
        return pooled / counts.clamp_min(1.0)

    def forward(self, sparse_features, text_features, text_padding_mask=None,
                detach_inputs: bool = False) -> torch.Tensor:
        batch_size = int(text_features.shape[0])
        visual = self._pool_sparse(sparse_features, batch_size, detach=detach_inputs)
        if detach_inputs:
            text_features = text_features.detach()
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
