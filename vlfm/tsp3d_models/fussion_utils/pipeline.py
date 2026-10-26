from typing import Optional

import numpy as np

from .sampling import cap_point_count, distance_adaptive_sample
from .world_map import WorldLocalMap


def _to_camera_canonical(
    pts_world: np.ndarray, robot_xyz: np.ndarray, robot_yaw: float
) -> np.ndarray:
    """World -> camera-canonical (xy translation + yaw rotation, z kept)."""
    pts = np.asarray(pts_world, dtype=np.float32).copy()
    if len(pts) == 0:
        return pts
    pts[:, :2] -= np.asarray(robot_xyz)[:2]
    cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
    xy = pts[:, :2].copy()
    pts[:, 0] = cos_yaw * xy[:, 0] + sin_yaw * xy[:, 1]
    pts[:, 1] = -sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]
    return pts


class TSP3DInputPreprocessor:
    """TSP3D input: world-frame local map or raw current frame -> send-side post-processing."""

    def __init__(
        self,
        fusion_style: str = "world",
        *,
        map_voxel_size: float = 0.02,
        map_radius: Optional[float] = 6.0,
        min_view_disp: float = 0.15,
        min_view_yaw: float = np.deg2rad(15.0),
        max_map_voxels: int = 400000,
        map_max_frames: Optional[int] = 8,
        map_near_refresh_radius: Optional[float] = None,
        map_near_refresh_value: bool = False,
        max_points: int = 200000,
        cap_style: str = "random",
        near_dist: float = 1.5,
        mid_dist: float = 3.0,
        near_voxel: float = 0.01,
        mid_voxel: float = 0.02,
        far_voxel: float = 0.05,
        use_distance_sampling: bool = False,
    ) -> None:
        if fusion_style not in ("world", "none"):
            raise ValueError("fusion_style MUST BE 'world' or 'none'")
        self._fusion_style = fusion_style
        self._map: Optional[WorldLocalMap] = None
        self._last_frame: Optional[np.ndarray] = None
        if fusion_style == "world":
            self._map = WorldLocalMap(
                voxel_size=map_voxel_size,
                radius=map_radius,
                min_view_disp=min_view_disp,
                min_view_yaw=min_view_yaw,
                max_voxels=max_map_voxels,
                max_frames=map_max_frames,
                near_refresh_radius=map_near_refresh_radius,
                near_refresh_value=map_near_refresh_value,
            )
        self._max_points = max_points
        self._cap_style = cap_style
        self._use_distance_sampling = use_distance_sampling
        self._near_dist = near_dist
        self._mid_dist = mid_dist
        self._near_voxel = near_voxel
        self._mid_voxel = mid_voxel
        self._far_voxel = far_voxel

    def reset(self) -> None:
        if self._map is not None:
            self._map.reset()
        self._last_frame = None

    def update(
        self,
        frame_pcd_world: np.ndarray,
        robot_xyz: np.ndarray,
        robot_yaw: float,
    ) -> None:
        if self._fusion_style == "none":
            self._last_frame = frame_pcd_world
            return
        assert self._map is not None
        pose = np.array([robot_xyz[0], robot_xyz[1], robot_yaw], dtype=np.float32)
        self._map.update(frame_pcd_world, pose)

    def prepare(
        self,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        camera_height: float,
    ) -> np.ndarray:
        if self._fusion_style == "none":
            if self._last_frame is None or len(self._last_frame) == 0:
                return np.empty((0, 6), dtype=np.float32)
            pts = _to_camera_canonical(self._last_frame, robot_xyz, robot_yaw)
        else:
            assert self._map is not None
            pts = self._map.to_camera_canonical(robot_xyz, robot_yaw)
        if len(pts) == 0:
            return np.empty((0, 6), dtype=np.float32)

        camera_pos_local = np.array([0.0, 0.0, camera_height], dtype=np.float32)
        if self._fusion_style != "none" and self._use_distance_sampling:
            pts = distance_adaptive_sample(
                pts,
                camera_pos_local,
                near_dist=self._near_dist,
                mid_dist=self._mid_dist,
                near_voxel=self._near_voxel,
                mid_voxel=self._mid_voxel,
                far_voxel=self._far_voxel,
            )
        return cap_point_count(pts, camera_pos_local, self._max_points, self._cap_style)
