"""base / gate 微调所需工具（模型侧）。

归属
----
* `ft_modules.py`  —— 掩码/池化工具、门控头 `PresenceGate`、盒格式转换及归档分支
  仍使用的 `focal_binary_loss`；
* `prediction.py`  —— 预测契约工具：批大小、空预测、盒容器归一、逐样本取值；
* `boxes.py`       —— 轴对齐 3D 盒的 IoU / GIoU。

`TSPHead`（推理 + 结构化训练输出）与 `BeaUTyDETR`（gate 头装配）从这里取用；
本包不导入任何 S1 / S2 / Distill 代码。

解耦归档
--------
* `vlfm/tsp3d_models/ft_s12/` —— 已移出的 S1/S2（提议 / 实体精修）实现；
* `vlfm/tsp3d_models/ft_distill/` —— 已移出的 Distill（GD 教师软标签蒸馏）。
"""

from . import boxes, ft_modules, prediction
from .boxes import box_giou, box_iou
from .ft_modules import (
    PresenceGate,
    _none_if_empty,
    center_dim_to_corners,
    focal_binary_loss,
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
