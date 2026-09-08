from typing import Optional
import numpy as np

from .world_map import WorldLocalMap
from .sliding_window import TemporalPcdWindow
from .panoramic import PanoramicFusion, _to_camera_canonical
from .sampling import cap_point_count, distance_adaptive_sample

class TSP3DInputPreprocessor:
    """
    End-to-end input adaptation: fusion (camera window or world map) -> 
    unified send-side post-processing (distance sampling / point cap).
    """

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
        map_near_refresh_radius: Optional[float] = None,
        map_near_refresh_value: bool = False,
        # send-side post-processing (shared by both routes)
        max_points: int = 200000,
        cap_style: str = "random",  # "random" / "near_first"
        near_dist: float = 1.5,
        mid_dist: float = 3.0,
        near_voxel: float = 0.01,
        mid_voxel: float = 0.02,
        far_voxel: float = 0.05,
        use_distance_sampling: bool = False,
        # panoramic route (fully independent, panoramic_* params only)
        panoramic_turn_steps: int = 6,
        panoramic_voxel_size: float = 0.02,
        panoramic_radius: Optional[float] = 6.0,
        panoramic_max_points: int = 200000,
        panoramic_min_move: float = 2.0,
    ) -> None:
        if fusion_style not in ("camera", "world", "panoramic", "none"):
            raise ValueError("fusion_style MUST BE 'camera', 'world', 'panoramic' or 'none'")
        self._fusion_style = fusion_style
        if fusion_style == "none":
            # No fusion at all: the raw current frame is sent straight to TSP3D, 
            # only converted to the camera-canonical frame of the query pose.
            self._panoramic: Optional[PanoramicFusion] = None
            self._map: Optional[WorldLocalMap] = None
            self._window: Optional[TemporalPcdWindow] = None
            self._last_frame: Optional[np.ndarray] = None
        elif fusion_style == "panoramic":
            self._panoramic: Optional[PanoramicFusion] = PanoramicFusion(
                turn_steps=panoramic_turn_steps,
                voxel_size=panoramic_voxel_size,
                radius=panoramic_radius,
                max_points=panoramic_max_points,
                min_move=panoramic_min_move,
            )
            self._map: Optional[WorldLocalMap] = None
            self._window: Optional[TemporalPcdWindow] = None
        elif fusion_style == "world":
            self._panoramic = None
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
            self._window: Optional[TemporalPcdWindow] = None
        else:
            self._panoramic = None
            self._window = TemporalPcdWindow(
                window_size=window_size,
                fuse_voxel_size=fuse_voxel_size,
                radius=cam_radius,
                min_view_disp=cam_min_view_disp,
                min_view_yaw=cam_min_view_yaw,
            )
            self._map: Optional[WorldLocalMap] = None

        self._max_points = max_points
        self._cap_style = cap_style

        self._use_distance_sampling = use_distance_sampling
        self._near_dist = near_dist
        self._mid_dist = mid_dist
        self._near_voxel = near_voxel
        self._mid_voxel = mid_voxel
        self._far_voxel = far_voxel

    def reset(self) -> None:
        if self._fusion_style == "none":
            self._last_frame = None
        elif self._fusion_style == "panoramic":
            self._panoramic.reset()
        elif self._fusion_style == "world":
            self._map.reset()
        else:
            self._window.reset()

    @property
    def num_map_voxels(self) -> int:
        if self._fusion_style in ("none", "panoramic"):
            return 0
        if self._fusion_style == "world":
            return self._map.num_voxels
        return self._window.num_frames

    def update(
        self,
        frame_pcd_world: np.ndarray,
        robot_xyz: np.ndarray,
        robot_yaw: float,
    ) -> None:
        if self._fusion_style == "panoramic":
            # Panoramic route is driven by the policy scan state machine 
            # via the dedicated methods below
            return
        if self._fusion_style == "none":
            # No fusion: keep only the raw current frame (single-view).
            self._last_frame = frame_pcd_world
            return
        
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
        if self._fusion_style == "none":
            # Single-view: raw current frame, only converted to the camera-canonical
            # frame of the query pose. No fusion, no crop, no sampling.
            if self._last_frame is None or len(self._last_frame) == 0:
                return np.empty((0, 6), dtype=np.float32)
            pts = _to_camera_canonical(self._last_frame, robot_xyz, robot_yaw)
        elif self._fusion_style == "world":
            pts = self._map.to_camera_canonical(robot_xyz, robot_yaw)
        else:
            pts = self._window.fuse_to_camera_canonical(robot_xyz, robot_yaw)
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

        pts = cap_point_count(pts, camera_pos_local, self._max_points, self._cap_style)

        return pts


    # Panoramic route (fusion_style == "panoramic" only)
    # Pure spatial fusion inside one decision point; no temporal history.
    @property
    def is_scanning(self) -> bool:
        return self._panoramic.is_scanning if self._panoramic is not None else False

    @property
    def scan_remaining(self) -> int:
        return self._panoramic.scan_remaining if self._panoramic is not None else 0

    def allow_scan(self, robot_xyz: np.ndarray) -> bool:
        if self._panoramic is None:
            return False
        return self._panoramic.allow_scan(robot_xyz)

    def set_scan_position(self, robot_xyz: np.ndarray, robot_yaw: float) -> None:
        if self._panoramic is not None:
            self._panoramic.set_scan_position(robot_xyz, robot_yaw)

    def begin_scan(self, frame_pcd_world: np.ndarray, robot_xyz: np.ndarray, robot_yaw: float) -> None:
        if self._panoramic is not None:
            self._panoramic.begin_scan(frame_pcd_world, robot_xyz, robot_yaw)

    def continue_scan(self, frame_pcd_world: np.ndarray) -> int:
        if self._panoramic is None:
            return 0
        return self._panoramic.continue_scan(frame_pcd_world)

    def finish_scan(self, robot_xyz: np.ndarray, robot_yaw: float, camera_height: float) -> np.ndarray:
        if self._panoramic is None:
            return np.empty((0, 6), dtype=np.float32)
        return self._panoramic.finish_scan(robot_xyz, robot_yaw, camera_height, self._cap_style)

    def raw_frame(self, frame_pcd_world: np.ndarray, robot_xyz: np.ndarray, robot_yaw: float, camera_height: float) -> np.ndarray:
        if self._panoramic is None:
            return np.empty((0, 6), dtype=np.float32)
        return self._panoramic.raw_frame(frame_pcd_world, robot_xyz, robot_yaw, camera_height, self._cap_style)
