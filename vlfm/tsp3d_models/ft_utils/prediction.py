from typing import Any

import torch


def batch_size_of(sparse) -> int:
    """稀疏张量的批大小（batch 列最大编号 + 1）。"""
    return int(sparse.C[:, 0].max().item()) + 1 if len(sparse) else 0


def empty_prediction(device, dtype=torch.float32, box_type=None):
    """空检测的三元组 `(boxes, scores, labels)`；`box_type` 给出时包装为盒对象。"""
    empty = torch.empty((0, 6), device=device, dtype=dtype)
    if box_type is not None:
        empty = box_type(empty, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
    return (
        empty,
        torch.empty((0,), device=device, dtype=dtype),
        torch.empty((0,), device=device, dtype=torch.long),
    )


def to_box_tensor(value, device, dtype):
    """Normalise common mmdet3d/list/tensor box containers to ``[N,6]``."""
    if value is None:
        return torch.empty((0, 6), device=device, dtype=dtype)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "tensor"):
        value = value.tensor
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.numel() == 0:
        return value.reshape(0, 6)
    value = value.reshape(-1, value.shape[-1])
    if value.shape[-1] >= 7:  # drop yaw when an old 7-D box is supplied
        value = torch.cat([value[:, :3], value[:, 3:6]], dim=-1)
    if value.shape[-1] != 6:
        raise ValueError(f"expected boxes with 6 dims, got {tuple(value.shape)}")
    # Accept either center/dim or corner boxes.  
    # Negative dimensions are a reliable signal that the input is malformed; 
    # otherwise preserve the documented center/dim training contract.
    return value


def item_at(value: Any, index: int, batch_size: int):
    """逐样本容器取第 `index` 项；非容器原样返回；越界返回 None。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    if torch.is_tensor(value) and value.shape[0] == batch_size:
        if value.ndim > 1 or batch_size > 1:
            return value[index]
    return value
