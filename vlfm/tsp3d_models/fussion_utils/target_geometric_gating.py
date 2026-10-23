from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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
            return True, {"passed": True, "geom_status": "gate_off", "reason": "gate_off",
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
            # S5-S1: the accumulated occupancy grid has no evidence for this
            # box.  This is unknown geometry, not negative geometry, so keep
            # the candidate for the downstream S-penalty/fallback path.
            return True, {
                "passed": True,
                "geom_status": "geom_unknown",
                "reason": "out_of_bounds",
                "num_occ": 0,
                "num_explored": 0,
                "occ_ratio": None,
            }

        occ_subgrid = obstacle_map_3d._map[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]
        explored_subgrid = obstacle_map_3d.explored_area[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]

        num_occ = int(np.sum(occ_subgrid))
        num_explored = int(np.sum(explored_subgrid))
        total_sub_voxels = occ_subgrid.size

        # Rule A: hard occupied-voxel floor
        if num_occ < self.cfg.min_occupied_voxels:
            return False, {
                "passed": False,
                "geom_status": "geom_reject",
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
                    "geom_status": "geom_reject",
                    "reason": f"air_penetration_detected (free_ratio={free_ratio:.2f})",
                    "num_occ": num_occ,
                    "free_ratio": free_ratio,
                }

        # Rule C: occupancy-volume ratio
        occ_ratio = num_occ / float(total_sub_voxels)
        if occ_ratio < self.cfg.min_occupancy_ratio:
            return False, {
                "passed": False,
                "geom_status": "geom_reject",
                "reason": f"low_occupancy_ratio ({occ_ratio:.4f} < {self.cfg.min_occupancy_ratio})",
                "num_occ": num_occ,
                "occ_ratio": occ_ratio,
            }

        return True, {
            "passed": True,
            "geom_status": "geom_confirmed",
            "reason": "pass",
            "num_occ": num_occ,
            "num_explored": num_explored,
            "occ_ratio": occ_ratio,
        }



# Unified admission engine (occupancy-consistency gate)
class TargetGeometricGatingEngine:
    """
    Unified geometric admission engine: occupancy-consistency gate only.
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
        """
        Run one detection through the occupancy-consistency gate.

        Returns status 'REJECTED' | 'GEOM_UNKNOWN' | 'HARD_CONFIRMED'
        'REJECTED' = empty-air box (no occupied-voxel support) 
        which the caller drops before the S-penalty gate.
        The remaining arguments are kept for call-site compatibility.
        """
        diag_summary: Dict[str, Any] = {}

        # occupancy-consistency check against the accumulated 3D grid
        occ_passed, occ_diag = self.occ_gate.evaluate(box_corners, obstacle_map_3d)
        diag_summary["occ_diag"] = occ_diag

        geom_status = str(occ_diag.get("geom_status", "geom_confirmed"))
        if not occ_passed:
            return "REJECTED", None, {
                "status": "REJECTED",
                "rejected_at": "occupancy_consistency_gate",
                "diagnostics": diag_summary,
            }

        status = "GEOM_UNKNOWN" if geom_status == "geom_unknown" else "HARD_CONFIRMED"
        return status, None, {
            "status": status,
            "is_suspicious": False,
            "diagnostics": diag_summary,
        }

# Detection-level admission wrapper
def apply_geom_gate(
    detections,
    pcd_world: np.ndarray,
    engine: TargetGeometricGatingEngine,
    obstacle_map_3d: Any,
    step: int,
    robot_xyz: Optional[np.ndarray] = None,
    per_det_meta: Optional[List[Any]] = None,
    meta_out: Optional[List[Any]] = None,
):
    """Geometric admission on the current candidates.

    REJECTED (empty-air) detections are dropped before the S-penalty gate.
    Returns the same container, filtered in place when anything is rejected.
    """
    n = detections.num_detections
    if n == 0:
        return detections
    if per_det_meta is not None:
        assert len(per_det_meta) == n, "per_det_meta length mismatch"
    boxes_np = detections.boxes.detach().cpu().numpy()
    logits_np = detections.logits.detach().cpu().numpy()
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        status, _cluster, diag = engine.process_detection(
            target_class=detections.phrases[i],
            box_corners=boxes_np[i],
            current_frame_pcd=pcd_world,
            obstacle_map_3d=obstacle_map_3d,
            confidence=float(logits_np[i]),
            step=step,
            robot_xyz=robot_xyz,
        )
        keep[i] = status != "REJECTED"
        if per_det_meta is not None and isinstance(per_det_meta[i], dict):
            occ_diag = ((diag or {}).get("diagnostics") or {}).get("occ_diag") or {}
            per_det_meta[i].update(
                {
                    "occ_passed": bool(keep[i]),
                    "occ_status": str((diag or {}).get("status", status)),
                    "occ_rejected_at": (diag or {}).get("rejected_at"),
                    "occ_reason": occ_diag.get("reason"),
                    "geom_status": occ_diag.get("geom_status"),
                    "occ_num_occ": occ_diag.get("num_occ"),
                    "occ_num_explored": occ_diag.get("num_explored"),
                    "occ_ratio": occ_diag.get("occ_ratio"),
                    "occ_free_ratio": occ_diag.get("free_ratio"),
                }
            )
        if keep[i] and meta_out is not None:
            meta_out.append(per_det_meta[i] if per_det_meta is not None else None)
    if not keep.all():
        detections.filter_by_mask(keep)
    return detections
