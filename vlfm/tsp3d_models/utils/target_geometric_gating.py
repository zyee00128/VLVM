from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
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


# Density accumulation gate (in-box point cloud over time)
@dataclass
class TargetPointCluster:
    """Spatio-temporal in-box point pool and cross-frame stats of one candidate."""
    cluster_id: int
    target_class: str
    centroid: np.ndarray                     # (3,) EMA centroid
    points: np.ndarray                       # (K, 3) accumulated surface depth points
    obs_count: int = 1                       # observed frames
    first_seen_step: int = 0
    last_seen_step: int = 0
    is_confirmed: bool = False               # crossed the hard admission threshold
    is_suspicious: bool = False              # latest observation was suspicious
    max_confidence: float = 0.0              # highest detection confidence seen
    view_angles: List[float] = field(default_factory=list)  # observation azimuths (rad), for the multi-view span test
@dataclass
class BoxPointDensityConfig:
    """Density-accumulation admission gate (in-box physical depth points over time)."""
    enable: bool = False                    # master switch (off = detections pass straight through)
    min_points_confirm: int = 150           # hard-confirm accumulated in-box points (VLFM min_points 100-150)
    min_frames_confirm: int = 2             # hard-confirm distinct observation frames
    cluster_merge_dist: float = 0.50        # cross-frame cluster association radius (m)
    use_occupancy_support: bool = True      # enable the occupancy shortcut
    occ_min_voxels: int = 8                 # occupancy voxel count floor (same as the occupancy gate)
    occ_min_ratio: float = 0.001            # occupancy/(box voxels) floor
    point_downsample_voxel: float = 0.02    # voxel resolution of the pooled points (2 cm)
    max_points_per_cluster: int = 2000      # per-cluster point-pool cap
    cluster_decay_steps: int = 50           # drop unconfirmed clusters idle longer than this

    # multi-view span: confirm on the density path only when the observations cover an
    # azimuth span >= min_view_span_deg (0 = disabled). Frontal hallucinations repeat
    # at ~0 span and cannot confirm; occupancy support still bypasses.
    min_view_span_deg: float = 0.0

def _circular_span_deg(angles_rad: np.ndarray) -> float:
    """Minimal circular arc (deg) covering all azimuths (0 if <2 samples)."""
    if angles_rad is None or len(angles_rad) < 2:
        return 0.0
    a = np.sort(np.asarray(angles_rad, dtype=np.float64))
    gaps = np.diff(a)
    gaps = np.append(gaps, a[0] + 2.0 * np.pi - a[-1])
    return float(np.degrees(2.0 * np.pi - gaps.max()))


class BoxPointDensityAccumulator:
    """Spatio-temporal in-box point accumulator with density clustering gating."""

    def __init__(self, cfg: Optional[BoxPointDensityConfig] = None):
        self.cfg = cfg or BoxPointDensityConfig()
        self._clusters: List[TargetPointCluster] = []
        self._next_cluster_id = 0

    def reset(self) -> None:
        self._clusters.clear()
        self._next_cluster_id = 0

    def update_and_evaluate(
        self,
        target_class: str,
        box_corners: np.ndarray,
        current_frame_pcd: np.ndarray,
        obstacle_map_3d: Optional[Any] = None,
        confidence: float = 0.0,
        step: int = 0,
        is_suspicious: bool = False,
        robot_xyz: Optional[np.ndarray] = None,
    ) -> Tuple[bool, TargetPointCluster, Dict[str, Any]]:
        """Merge this frame's in-box points into a cluster and evaluate admission.

        HARD = occupancy-support shortcut OR
        (density path: enough accumulated points AND
        enough frames AND, when enabled, multi-view span).
        """
        in_mask = points_in_box_3d(current_frame_pcd, box_corners)
        pts_in_box = current_frame_pcd[in_mask, :3]
        pts_in_box = voxel_downsample(pts_in_box, self.cfg.point_downsample_voxel)

        box_center, _, _ = decompose_box_8corners(box_corners)
        # observation azimuth for the multi-view span test
        view_angle: Optional[float] = None
        if robot_xyz is not None and len(robot_xyz) >= 2:
            view_angle = float(np.arctan2(box_center[1] - robot_xyz[1], box_center[0] - robot_xyz[0]))

        matched_cluster: Optional[TargetPointCluster] = None
        min_dist = float("inf")

        for cl in self._clusters:
            if cl.target_class != target_class:
                continue
            dist = float(np.linalg.norm(cl.centroid - box_center))
            if dist < self.cfg.cluster_merge_dist and dist < min_dist:
                min_dist = dist
                matched_cluster = cl

        if matched_cluster is not None:
            if len(pts_in_box) > 0:
                merged_pts = np.vstack([matched_cluster.points, pts_in_box])
                matched_cluster.points = voxel_downsample(merged_pts, self.cfg.point_downsample_voxel)
                if len(matched_cluster.points) > self.cfg.max_points_per_cluster:
                    idx = np.random.choice(len(matched_cluster.points), self.cfg.max_points_per_cluster, replace=False)
                    matched_cluster.points = matched_cluster.points[idx]

            if view_angle is not None:
                matched_cluster.view_angles.append(view_angle)
                if len(matched_cluster.view_angles) > 64:
                    matched_cluster.view_angles = matched_cluster.view_angles[-64:]
            matched_cluster.centroid = 0.8 * matched_cluster.centroid + 0.2 * box_center
            matched_cluster.obs_count += 1
            matched_cluster.last_seen_step = step
            matched_cluster.max_confidence = max(matched_cluster.max_confidence, confidence)
            matched_cluster.is_suspicious = is_suspicious
            cluster = matched_cluster
        else:
            cluster = TargetPointCluster(
                cluster_id=self._next_cluster_id,
                target_class=target_class,
                centroid=box_center.copy(),
                points=pts_in_box.copy(),
                obs_count=1,
                first_seen_step=step,
                last_seen_step=step,
                is_confirmed=False,
                is_suspicious=is_suspicious,
                max_confidence=confidence,
                view_angles=[view_angle] if view_angle is not None else [],
            )
            self._next_cluster_id += 1
            self._clusters.append(cluster)

        # Admission: HARD = occupancy-support shortcut OR density accumulation
        eff_min_points = int(self.cfg.min_points_confirm)
        total_points = len(cluster.points)
        meets_points = total_points >= eff_min_points
        meets_frames = cluster.obs_count >= self.cfg.min_frames_confirm
        # multi-view span over the recorded observation azimuths
        view_span_deg = _circular_span_deg(np.asarray(cluster.view_angles, dtype=np.float64))
        meets_views = (self.cfg.min_view_span_deg <= 0.0) or (view_span_deg >= self.cfg.min_view_span_deg)
        occ_support = False
        num_occ = 0
        num_explored = 0
        total_voxels = 0
        if obstacle_map_3d is not None and self.cfg.use_occupancy_support:
            num_occ, num_explored, total_voxels = count_box_occupancy(box_corners, obstacle_map_3d)
            occ_support = (num_occ >= self.cfg.occ_min_voxels) and \
                          (float(num_occ) / max(total_voxels, 1) >= self.cfg.occ_min_ratio)

        confirmed = occ_support or (meets_points and meets_frames and meets_views and (not is_suspicious))
        cluster.is_confirmed = confirmed

        diagnostics = {
            "cluster_id": cluster.cluster_id,
            "total_points": total_points,
            "eff_min_points": eff_min_points,
            "obs_count": cluster.obs_count,
            "meets_points": meets_points,
            "meets_frames": meets_frames,
            "view_span_deg": round(view_span_deg, 2),
            "meets_views": meets_views,
            "occ_support": occ_support,
            "num_occ": num_occ,
            "num_explored": num_explored,
            "is_suspicious": is_suspicious,
            "is_confirmed": cluster.is_confirmed,
            "confirm_source": "occupancy" if (occ_support and not (meets_points and meets_frames and meets_views)) else ("density" if (meets_points and meets_frames and meets_views and not is_suspicious) else "none"),
        }

        self._prune_stale_clusters(step)
        return cluster.is_confirmed, cluster, diagnostics

    def _prune_stale_clusters(self, current_step: int) -> None:
        self._clusters = [
            cl for cl in self._clusters
            if cl.is_confirmed or (current_step - cl.last_seen_step <= self.cfg.cluster_decay_steps)
        ]


# Unified admission engine (occ-consistency gate -> density gate)
class TargetGeometricGatingEngine:
    """Unified geometric admission engine: occupancy-consistency gate -> density gate."""

    def __init__(
        self,
        occ_cfg: Optional[BoxOccupancyConfig] = None,
        density_cfg: Optional[BoxPointDensityConfig] = None,
    ):
        self.occ_gate = BoxOccupancyGate(occ_cfg)
        self.density_acc = BoxPointDensityAccumulator(density_cfg)

    def reset(self) -> None:
        self.density_acc.reset()

    @property
    def enabled(self) -> bool:
        """True if any geometric gate (M8 occupancy / M3 density) is on."""
        return bool(
            self.occ_gate.cfg.enable
            or self.density_acc.cfg.enable
        )

    def process_detection(
        self,
        target_class: str,
        box_corners: np.ndarray,
        current_frame_pcd: np.ndarray,
        obstacle_map_3d: Any,
        confidence: float,
        step: int,
        robot_xyz: Optional[np.ndarray] = None,
    ) -> Tuple[str, Optional[TargetPointCluster], Dict[str, Any]]:
        """Run one detection through the occupancy gate, then the density gate.
        Returns status 'REJECTED' | 'SOFT_ACCUMULATING' | 'HARD_CONFIRMED'.
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

        # density accumulation: HARD/SOFT split
        if self.density_acc.cfg.enable:
            density_passed, cluster, density_diag = self.density_acc.update_and_evaluate(
                target_class=target_class,
                box_corners=box_corners,
                current_frame_pcd=current_frame_pcd,
                obstacle_map_3d=obstacle_map_3d,
                confidence=confidence,
                step=step,
                is_suspicious=False,
                robot_xyz=robot_xyz,
            )
            diag_summary["density_diag"] = density_diag
            status = "HARD_CONFIRMED" if density_passed else "SOFT_ACCUMULATING"
            cluster_id = cluster.cluster_id
            cluster_points = len(cluster.points)
            obs_count = cluster.obs_count
        else:
            # density off -> pass straight through
            status = "HARD_CONFIRMED"
            cluster = None
            cluster_id = None
            cluster_points = 0
            obs_count = 0

        return status, cluster, {
            "status": status,
            "is_suspicious": False,
            "cluster_id": cluster_id,
            "cluster_points": cluster_points,
            "obs_count": obs_count,
            "diagnostics": diag_summary,
        }
