"""base / gate 微调所需工具（模型侧）。

归属
----
* `ft_modules.py`  —— 掩码/池化工具与门控头 `PresenceGate`；
* `prediction.py`  —— 预测契约工具：批大小、空预测、盒容器归一、逐样本取值；
* `boxes.py`       —— 轴对齐 3D 盒的 IoU / GIoU。

`TSPHead`（推理 + 结构化训练输出）与 `BeaUTyDETR`（gate 头装配）从这里取用；
本包只包含当前 base / gate 路径依赖。
"""

from . import boxes, ft_modules, prediction
from .boxes import box_giou, box_iou
from .ft_modules import (
    PresenceGate,
    _none_if_empty,
    masked_mean,
)
from .prediction import batch_size_of, empty_prediction, item_at, to_box_tensor

__all__ = [
    "boxes",
    "ft_modules",
    "prediction",
    "PresenceGate",
    "masked_mean",
    "box_iou",
    "box_giou",
    "batch_size_of",
    "empty_prediction",
    "to_box_tensor",
    "item_at",
]
_binary_loss,
    masked_mean,
)
from .prediction import batch_size_of, empty_prediction, item_at, to_box_tensor

__all__ = [
    "boxes",
    "ft_modules",
    "prediction",
    "PresenceGate",
    "masked_mean",
    "center_dim_to_corners",
    "focal_binary_loss",
    "box_iou",
    "box_giou",
    "batch_size_of",
    "empty_prediction",
    "to_box_tensor",
    "item_at",
]
