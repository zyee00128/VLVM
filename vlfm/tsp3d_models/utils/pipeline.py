"""Unified TSP3D input preprocessor (VLVM 4.3c): camera/world fusion + send-side post-processing."""
from typing import Optional
import numpy as np

from .local_map import WorldLocalMap
from .sliding_window import TemporalPcdWindow
from .sampling import cap_point_count, distance_adaptive_sample, voxelize_world


class TSP3DInputPreprocessor:
    """End-to-end input adaptation: fusion (camera window or world map) -> unified
    send-side post-processing (send voxelization / distance sampling / point cap)."""

    def __init__(
        self,
        fusion_style: str = "camera",
        # camera route
        window_size: int = 8,
        fuse_voxel_size: float = 0.02,
        cam_radius: Optional[float] = None,
        cam_min_view_disp: float = 0.0,
        cam_min_view_yaw: float = 0.0,
        # world route
        map_voxel_size: float = 0.02,
        map_radius: float = 6.0,
        min_view_disp: float = 0.15,
        min_view_yaw: float = np.deg2rad(15.0),
        max_map_voxels: int = 400000,
        map_max_frames: Optional[int] = 8,
        # send-side post-processing (shared by both routes)
        send_voxel_size: Optional[float] = None,  # D2
        max_points: int = 200000,
        cap_style: str = "random",                # D1: "random" / "near_first"
        near_dist: float = 1.5,
        mid_dist: float = 3.0,
        near_voxel: float = 0.01,
        mid_voxel: float = 0.02,
        far_voxel: float = 0.05,
        use_distance_sampling: bool = False,      # D3
    ) -> None:
        if fusion_style not in ("camera", "world"):
            raise ValueError("fusion_style MUST BE 'camera' or 'world'")
        self._fusion_style = fusion_style
        if fusion_style == "world":
            self._map = WorldLocalMap(
                voxel_size=map_voxel_size,
                radius=map_radius,
                min_view_disp=min_view_disp,
                min_view_yaw=min_view_yaw,
                max_voxels=max_map_voxels,
                max_frames=map_max_frames,
            )
            self._window: Optional[TemporalPcdWindow] = None
        else:
            self._window = TemporalPcdWindow(
                window_size=window_size,
                fuse_voxel_size=fuse_voxel_size,
                radius=cam_radius,
                min_view_disp=cam_min_view_disp,
                min_view_yaw=cam_min_view_yaw,
            )
            self._map: Optional[WorldLocalMap] = None

        self._send_voxel_size = send_voxel_size
        self._max_points = max_points
        self._cap_style = cap_style

        self._use_distance_sampling = use_distance_sampling
        self._near_dist = near_dist
        self._mid_dist = mid_dist
        self._near_voxel = near_voxel
        self._mid_voxel = mid_voxel
        self._far_voxel = far_voxel

    def reset(self) -> None:
        if self._fusion_style == "world":
            self._map.reset()
        else:
            self._window.reset()

    @property
    def num_map_voxels(self) -> int:
        if self._fusion_style == "world":
            return self._map.num_voxels
        return self._window.num_frames

    def update(
        self,
        frame_pcd_world: np.ndarray,
        robot_xyz: np.ndarray,
        robot_yaw: float,
    ) -> None:
        pose = np.array([robot_xyz[0], robot_xyz[1], robot_yaw], dtype=np.float32)
        if self._fusion_style == "world":
            self._map.update(frame_pcd_world, pose)
        else:
            self._window.update(frame_pcd_world, pose)

    def prepare(
        self,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        camera_height: float,
    ) -> np.ndarray:
        if self._fusion_style == "world":
            pts = self._map.to_camera_canonical(robot_xyz, robot_yaw)
        else:
            pts = self._window.fuse_to_camera_canonical(robot_xyz, robot_yaw)
        if len(pts) == 0:
            return np.empty((0, 6), dtype=np.float32)

        if self._send_voxel_size is not None:
            pts = voxelize_world(pts, self._send_voxel_size)

        camera_pos_local = np.array([0.0, 0.0, camera_height], dtype=np.float32)

        if self._use_distance_sampling:
            pts = distance_adaptive_sample(
                pts,
                camera_pos_local,
                near_dist=self._near_dist,
                mid_dist=self._mid_dist,
                near_voxel=self._near_voxel,
                mid_voxel=self._mid_voxel,
                far_voxel=self._far_voxel,
            )

        pts = cap_point_count(pts, camera_pos_local, self._max_points, self._cap_style)
        return pts
