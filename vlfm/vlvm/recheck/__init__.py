"""Frontier-policy helpers: recheck queue and explore mode switching.

| 模块 | 职责 |
| :--- | :--- |
| ``expl_recheck`` | ``expl recheck``：explore frontier 耗尽后的巡回，explore 侧唯一受控 STOP |
| ``explore_mode`` | 纯决策逻辑：按纯 ValueMap 分数 ``S`` 在 semantic / nearest frontier 之间切换 |
| ``explore_runtime`` | 运行期 mixin：explore 状态初始化、stuck 检测、STOP 抑制与 recheck 兜底 |
"""

from .expl_recheck import ExplRecheckMixin
from .explore_mode import (
    MODE_NEAREST,
    MODE_SEMANTIC,
    ExploreModeConfig,
    classify,
    decision_row,
    nearest_index,
)
from .explore_runtime import ExploreRuntimeMixin

__all__ = [
    "ExplRecheckMixin",
    "ExploreRuntimeMixin",
    "ExploreModeConfig",
    "MODE_NEAREST",
    "MODE_SEMANTIC",
    "classify",
    "decision_row",
    "nearest_index",
]
