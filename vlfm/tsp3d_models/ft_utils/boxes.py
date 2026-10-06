import torch


def box_iou(boxes1, boxes2):
    """
    Axis-aligned 3D IoU，return `[N, M]`; return 0 when either is empty.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    
    lo = torch.maximum(boxes1[:, None, :3], boxes2[None, :, :3])
    hi = torch.minimum(boxes1[:, None, 3:], boxes2[None, :, 3:])
    inter = (hi - lo).clamp_min(0).prod(dim=-1)
    vol1 = (boxes1[:, 3:] - boxes1[:, :3]).clamp_min(0).prod(dim=-1)[:, None]
    vol2 = (boxes2[:, 3:] - boxes2[:, :3]).clamp_min(0).prod(dim=-1)[None, :]
    return inter / (vol1 + vol2 - inter).clamp_min(1e-6)


def box_giou(boxes1, boxes2):
    """Axis-aligned 3D GIoU for the model-side fallback loss."""
    iou = box_iou(boxes1, boxes2)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou
    lo = torch.minimum(boxes1[:, None, :3], boxes2[None, :, :3])
    hi = torch.maximum(boxes1[:, None, 3:], boxes2[None, :, 3:])
    enclosing = (hi - lo).clamp_min(0).prod(dim=-1).clamp_min(1e-6)
    vol1 = (boxes1[:, 3:] - boxes1[:, :3]).clamp_min(0).prod(dim=-1)[:, None]
    vol2 = (boxes2[:, 3:] - boxes2[:, :3]).clamp_min(0).prod(dim=-1)[None, :]
    inter_lo = torch.maximum(boxes1[:, None, :3], boxes2[None, :, :3])
    inter_hi = torch.minimum(boxes1[:, None, 3:], boxes2[None, :, 3:])
    inter = (inter_hi - inter_lo).clamp_min(0).prod(dim=-1)
    union = vol1 + vol2 - inter
    return iou - (enclosing - union) / enclosing
