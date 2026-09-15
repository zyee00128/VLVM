from typing import Optional
import numpy as np

from .world_map import WorldLocalMap
from .sampling import cap_point_count, distance_adaptive_sample

class TSP3DInputPreprocessor:
    """
    End-to-end input adaptation: world-frame incremental map accumulation ->
    unified send-side post-processing (distance sampling / point cap).
    """

    def __init__(
        self,
        fusion_style: str = "world",
        # world route
        map_voxel_size: float = 0.02,
        map_radius: float = 6.0,
        min_view_disp: float = 0.15,
        min_view_yaw: float = np.deg2rad(15.0),
        max_map_voxels: int = 400000,
        map_max_frames: Optional[int] = 8,
        map_near_refresh_radius: Optional[float] = None,
        map_near_refresh_value: bool = False,
        # send-side post-processing
        max_points: int = 200000,
        cap_style: str = "random",  # "random" / "near_first"
        near_dist: float = 1.5,
        mid_dist: float = 3.0,
        near_voxel: float = 0.01,
        mid_voxel: float = 0.02,
        far_voxel: float = 0.05,
        use_distance_sampling: bool = False,
    ) -> None:
        if fusion_style != "world":
            raise ValueError("fusion_style MUST BE 'world' (the final route)")
        self._fusion_style = fusion_style
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
        self._map.reset()

    @property
    def num_map_voxels(self) -> int:
        return self._map.num_voxels

    def update(
        self,
        frame_pcd_world: np.ndarray,
        robot_xyz: np.ndarray,
        robot_yaw: float,
    ) -> None:
        pose = np.array([robot_xyz[0], robot_xyz[1], robot_yaw], dtype=np.float32)
        self._map.update(frame_pcd_world, pose)

    def prepare(
        self,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        camera_height: float,
    ) -> np.ndarray:
        pts = self._map.to_camera_canonical(robot_xyz, robot_yaw)
        if len(pts) == 0:
            return np.empty((0, 6), dtype=np.float32)

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

