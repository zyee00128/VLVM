"""base / gate / distill 微调所需工具（模型侧）。

归属
----
* `ft_modules.py`  —— 掩码/池化工具、门控头 `PresenceGate`、盒格式转换与模型侧
  保底损失（`focal_binary_loss` / `asymmetric_binary_loss`）；
* `prediction.py`  —— 预测契约工具：批大小、空预测、盒容器归一、逐样本取值；
* `boxes.py`       —— 轴对齐 3D 盒的 IoU / GIoU。

`TSPHead`（推理 + base 训练路径）与 `BeaUTyDETR`（gate 头装配）从这里取用；
本包不导入任何 S1/S2 代码。

S1/S2 归档
----------
`vlfm/tsp3d_models/ft_s12/` 存放已移出的 S1/S2（提议 / 实体精修）实现，
模型与本包都不导入；重新接线前不参与任何路径，接口说明见其 `__init__.py`
与 `s12_decoupled.py` 头部。
"""

from . import boxes, ft_modules, prediction
from .boxes import box_giou, box_iou
from .ft_modules import (
    PresenceGate,
    _none_if_empty,
    asymmetric_binary_loss,
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
    "asymmetric_binary_loss",
    "box_iou",
    "box_giou",
    "batch_size_of",
    "empty_prediction",
    "to_box_tensor",
    "item_at",
]
