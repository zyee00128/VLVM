from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class SoftFrontierBiasConfig:
    """M5 超参数配置"""
    enable: bool = True
    bias_weight: float = 0.60              # 软前沿引力增益基础权重 w_bias
    gaussian_sigma: float = 1.50           # 高斯核空间标准差 sigma (米)，控制引力扩散范围
    max_influence_radius: float = 4.50     # 目标引力的最大作用半径 (米)，超出则衰减为 0
    normalize_output: bool = False         # 是否将最终打分归一化到 [0, 1]

@dataclass
class SoftTargetItem:
    """输入给引力偏置引擎的 Soft 目标结构"""
    target_id: int
    target_class: str
    coord: np.ndarray                      # 2D [x, y] 或 3D [x, y, z] 坐标
    confidence: float = 1.0                # S-penalty 调制后的置信度 c'
    is_suspicious: bool = True             # 是否处于 Soft/Suspicious 状态

class SoftFrontierBiasEngine:
    """M5：软前沿引力偏置引擎。"""

    def __init__(self, config: Optional[SoftFrontierBiasConfig] = None):
        self.config = config or SoftFrontierBiasConfig()

    def compute_bias_heatmap(
        self,
        frontiers_xy: np.ndarray,
        soft_targets: List[Union[SoftTargetItem, Dict[str, Any], np.ndarray]],
    ) -> np.ndarray:
        """
        计算所有 Frontier 候选点处由 Soft 目标群体叠加的总高斯引力偏置值。

        Args:
            frontiers_xy: shape (M, 2) 或 (M, >=2) 的 Frontier 候选点坐标
            soft_targets: Soft 弱目标列表（支持 SoftTargetItem / Dict / ndarray）

        Returns:
            bias_scores: shape (M,) 浮点数组，每个 Frontier 获得的空间偏置得分
        """
        if not self.config.enable or len(frontiers_xy) == 0 or len(soft_targets) == 0:
            return np.zeros(len(frontiers_xy), dtype=np.float32)

        f_pts = np.asarray(frontiers_xy)[:, :2]  # (M, 2)
        total_bias = np.zeros(len(f_pts), dtype=np.float32)

        two_sigma_sq = 2.0 * (self.config.gaussian_sigma ** 2)

        for item in soft_targets:
            # 提取坐标与置信度
            if isinstance(item, SoftTargetItem):
                t_coord = np.asarray(item.coord)[:2]
                conf = float(item.confidence)
            elif isinstance(item, dict):
                coord = item.get("centroid", item.get("coord", np.zeros(2)))
                t_coord = np.asarray(coord)[:2]
                conf = float(item.get("confidence", item.get("score", 1.0)))
            elif isinstance(item, np.ndarray):
                t_coord = item[:2]
                conf = 1.0
            else:
                continue

            # 向量化计算所有 Frontier 到该 Soft 目标的欧式距离
            dists = np.linalg.norm(f_pts - t_coord, axis=1)  # (M,)
            # 在最大影响半径内的点计算高斯热度
            valid_mask = dists <= self.config.max_influence_radius
            if not np.any(valid_mask):
                continue

            # Gaussian Kernel: w_bias * c' * exp(- dist^2 / (2 * sigma^2))
            gaussian_val = np.exp(- (dists[valid_mask] ** 2) / two_sigma_sq)
            bias_val = self.config.bias_weight * conf * gaussian_val

            total_bias[valid_mask] += bias_val.astype(np.float32)

        return total_bias

    def apply_bias_and_rank(
        self,
        frontiers: np.ndarray,
        base_values: Optional[Union[List[float], np.ndarray]] = None,
        soft_targets: Optional[List[Union[SoftTargetItem, Dict[str, Any], np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        向前沿价值场注入软引力偏置，并重新排序前沿候选点。

        Args:
            frontiers: shape (M, 2) 或 (M, >=2) 前沿坐标列表
            base_values: shape (M,) 基础语义/几何价值评分 (如 S + lambda*H1)
            soft_targets: 当前记忆库中所有处于 Soft 态的目标候选

        Returns:
            sorted_frontiers: shape (M, 2) 重新加权排序后的前沿点 (降序)
            final_scores: shape (M,) 对应的综合评分 (降序)
            diagnostics: 诊断指标
        """
        if len(frontiers) == 0:
            return np.empty((0, 2)), np.empty((0,), dtype=np.float32), {"num_frontiers": 0}

        f_arr = np.asarray(frontiers)
        M = len(f_arr)

        if base_values is None:
            base_arr = np.zeros(M, dtype=np.float32)
        else:
            base_arr = np.asarray(base_values, dtype=np.float32)

        soft_list = soft_targets or []

        # 1. 计算高斯引力偏置
        bias_arr = self.compute_bias_heatmap(f_arr, soft_list)
        # 2. 融合基础价值与引力偏置
        final_scores = base_arr + bias_arr
        if self.config.normalize_output and len(final_scores) > 0:
            s_min, s_max = np.min(final_scores), np.max(final_scores)
            if s_max > s_min:
                final_scores = (final_scores - s_min) / (s_max - s_min)
        # 3. 按综合评分降序排序
        sorted_indices = np.argsort(-final_scores)
        sorted_frontiers = f_arr[sorted_indices]
        sorted_scores = final_scores[sorted_indices]

        diagnostics = {
            "num_frontiers": M,
            "num_soft_targets": len(soft_list),
            "max_bias_added": float(np.max(bias_arr)) if len(bias_arr) > 0 else 0.0,
            "mean_bias_added": float(np.mean(bias_arr)) if len(bias_arr) > 0 else 0.0,
            "best_frontier_score": float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0,
        }

        return sorted_frontiers, sorted_scores, diagnostics

