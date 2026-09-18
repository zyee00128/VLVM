from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Basic geometry / 3D box helpers
def decompose_box_8corners(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover OBB center / orthonormal axes / half-extents from 8 corners via SVD.
    Order-agnostic; robust to any 3DVG box output.
    """
    center = np.mean(corners, axis=0)
    centered = corners - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt  # (3, 3) orthonormal axes; one unit vector per row
    projections = np.abs(centered @ axes.T)  # (8, 3)
    half_extents = np.max(projections, axis=0)
    return center, axes, half_extents

def points_in_box_3d(points: np.ndarray, box_corners: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """Vectorized membership test of points inside a 3D OBB (with margin, m)."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    pts_xyz = points[:, :3]
    center, axes, half_extents = decompose_box_8corners(box_corners)

    rel_pts = pts_xyz - center  # (N, 3)
    proj = np.abs(rel_pts @ axes.T)  # (N, 3)

    in_box_mask = np.all(proj <= (half_extents + margin), axis=1)
    return in_box_mask

def voxel_downsample(points: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """Fast voxel-grid downsample."""
    if len(points) == 0:
        return points
    coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]

def count_box_occupancy(
    box_corners: np.ndarray,
    obstacle_map_3d: Any,
) -> Tuple[int, int, int]:
    """Count occupied / explored voxels and total cells in a box AABB."""
    min_xyz = np.min(box_corners, axis=0)
    max_xyz = np.max(box_corners, axis=0)
    bbox_aabb = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
    ])
    grid_idx = obstacle_map_3d._xyz_to_grid_index(bbox_aabb)
    px_min = max(0, min(grid_idx[0, 0], grid_idx[1, 0]))
    px_max = min(obstacle_map_3d.size - 1, max(grid_idx[0, 0], grid_idx[1, 0]))
    py_min = max(0, min(grid_idx[0, 1], grid_idx[1, 1]))
    py_max = min(obstacle_map_3d.size - 1, max(grid_idx[0, 1], grid_idx[1, 1]))
    cz_min = max(0, min(grid_idx[0, 2], grid_idx[1, 2]))
    cz_max = min(obstacle_map_3d._height_size - 1, max(grid_idx[0, 2], grid_idx[1, 2]))
    if px_min > px_max or py_min > py_max or cz_min > cz_max:
        return 0, 0, 0
    occ_sub = obstacle_map_3d._map[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]
    exp_sub = obstacle_map_3d.explored_area[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]
    return int(np.sum(occ_sub)), int(np.sum(exp_sub)), int(occ_sub.size)

# Box-occupancy consistency gate
@dataclass
class BoxOccupancyConfig:
    enable: bool = False                   # master switch
    min_occupied_voxels: int = 8           # min occupied voxels inside the box AABB
    min_occupancy_ratio: float = 0.001     # min occupied/(box voxel count) ratio
    check_explored_free: bool = True       # reject when the box is almost all free (ray-penetration)
    free_ratio_rejection_thresh: float = 0.90  # free ratio above which the box is judged air
class BoxOccupancyGate:
    """Box-occupancy consistency gate against the accumulated 3D occupancy grid."""

    def __init__(self, cfg: Optional[BoxOccupancyConfig] = None):
        self.cfg = cfg or BoxOccupancyConfig()

    def evaluate(
        self,
        box_corners: np.ndarray,
        obstacle_map_3d: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not self.cfg.enable:
            return True, {"passed": True, "reason": "gate_off",
                          "num_occ": 0, "num_explored": 0, "occ_ratio": 0.0}
        center, _, _ = decompose_box_8corners(box_corners)

        min_xyz = np.min(box_corners, axis=0)
        max_xyz = np.max(box_corners, axis=0)

        bbox_corners_aabb = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ])
        grid_idx = obstacle_map_3d._xyz_to_grid_index(bbox_corners_aabb)

        px_min = max(0, min(grid_idx[0, 0], grid_idx[1, 0]))
        px_max = min(obstacle_map_3d.size - 1, max(grid_idx[0, 0], grid_idx[1, 0]))
        py_min = max(0, min(grid_idx[0, 1], grid_idx[1, 1]))
        py_max = min(obstacle_map_3d.size - 1, max(grid_idx[0, 1], grid_idx[1, 1]))
        cz_min = max(0, min(grid_idx[0, 2], grid_idx[1, 2]))
        cz_max = min(obstacle_map_3d._height_size - 1, max(grid_idx[0, 2], grid_idx[1, 2]))

        if px_min > px_max or py_min > py_max or cz_min > cz_max:
            return False, {"reason": "out_of_bounds", "num_occ": 0}

        occ_subgrid = obstacle_map_3d._map[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]
        explored_subgrid = obstacle_map_3d.explored_area[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]

        num_occ = int(np.sum(occ_subgrid))
        num_explored = int(np.sum(explored_subgrid))
        total_sub_voxels = occ_subgrid.size

        # Rule A: hard occupied-voxel floor
        if num_occ < self.cfg.min_occupied_voxels:
            return False, {
                "passed": False,
                "reason": f"insufficient_occ_voxels ({num_occ} < {self.cfg.min_occupied_voxels})",
                "num_occ": num_occ,
                "num_explored": num_explored,
            }

        # Rule B: ray-penetration anti-hallucination
        if self.cfg.check_explored_free and num_explored > (self.cfg.min_occupied_voxels * 2):
            free_in_box = num_explored - num_occ
            free_ratio = free_in_box / float(num_explored)
            if free_ratio > self.cfg.free_ratio_rejection_thresh and num_occ < (self.cfg.min_occupied_voxels * 2):
                return False, {
                    "passed": False,
                    "reason": f"air_penetration_detected (free_ratio={free_ratio:.2f})",
                    "num_occ": num_occ,
                    "free_ratio": free_ratio,
                }

        # Rule C: occupancy-volume ratio
        occ_ratio = num_occ / float(total_sub_voxels)
        if occ_ratio < self.cfg.min_occupancy_ratio:
            return False, {
                "passed": False,
                "reason": f"low_occupancy_ratio ({occ_ratio:.4f} < {self.cfg.min_occupancy_ratio})",
                "num_occ": num_occ,
                "occ_ratio": occ_ratio,
            }

        return True, {
            "passed": True,
            "reason": "pass",
            "num_occ": num_occ,
            "num_explored": num_explored,
            "occ_ratio": occ_ratio,
        }



# Unified admission engine (occupancy-consistency gate)
class TargetGeometricGatingEngine:
    """Unified geometric admission engine: occupancy-consistency gate only.

    历史注记（results/VLVM-V8_2.md 最终基座判定）：原 M3「盒内点数×帧数」密度门实测
    与关闭时逐集零差异（occ 门同阈值前置硬拒 ⇒ 其 occ 捷径恒真），已按预注册处置整套
    删除；引擎只保留 occ 一致性门。
    """

    def __init__(self, occ_cfg: Optional[BoxOccupancyConfig] = None):
        self.occ_gate = BoxOccupancyGate(occ_cfg)

    def reset(self) -> None:
        """Kept for API symmetry with the callers (`policy._reset`); stateless gate."""
        return

    @property
    def enabled(self) -> bool:
        """True if the occupancy-consistency gate is on."""
        return bool(self.occ_gate.cfg.enable)

    def process_detection(
        self,
        target_class: str,
        box_corners: np.ndarray,
        current_frame_pcd: np.ndarray,
        obstacle_map_3d: Any,
        confidence: float,
        step: int,
        robot_xyz: Optional[np.ndarray] = None,
    ) -> Tuple[str, Optional[Any], Dict[str, Any]]:
        """Run one detection through the occupancy-consistency gate.

        Returns status 'REJECTED' | 'HARD_CONFIRMED' — 'REJECTED' = empty-air box
        (no occupied-voxel support) which the caller drops before the S-penalty gate.
        The remaining arguments are kept for call-site compatibility.
        """
        diag_summary: Dict[str, Any] = {}

        # occupancy-consistency check against the accumulated 3D grid
        occ_passed, occ_diag = self.occ_gate.evaluate(box_corners, obstacle_map_3d)
        diag_summary["occ_diag"] = occ_diag

        if not occ_passed:
            return "REJECTED", None, {
                "status": "REJECTED",
                "rejected_at": "occupancy_consistency_gate",
                "diagnostics": diag_summary,
            }

        return "HARD_CONFIRMED", None, {
            "status": "HARD_CONFIRMED",
            "is_suspicious": False,
            "diagnostics": diag_summary,
        }
