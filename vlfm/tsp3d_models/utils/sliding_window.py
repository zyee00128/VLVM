from collections import deque
from typing import Optional
import numpy as np


def _angle_diff(a: float, b: float) -> float:
    """Absolute angular difference in radians, in [0, pi]. Same as WorldLocalMap."""
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


class TemporalPcdWindow:
    """Keeps the last N per-frame voxelized clouds; query-time concat + round dedup."""

    def __init__(
        self,
        window_size: int = 8,
        fuse_voxel_size: float = 0.02,
        radius: Optional[float] = None,
        min_view_disp: float = 0.0,
        min_view_yaw: float = 0.0,
    ) -> None:
        self._window: deque = deque(maxlen=max(int(window_size), 1))
        self._fuse_voxel_size = float(fuse_voxel_size)
        self._radius = float(radius) if radius is not None else None
        self._min_view_disp = float(min_view_disp)
        self._min_view_yaw = float(min_view_yaw)
        self._last_pose: Optional[np.ndarray] = None

        # B2 (stricter gating) and B3 (keyframe sampling) are KEPT as reference only.
        # To re-enable, uncomment the gating blocks below and wire params through
        # pipeline/policy (disp/yaw thresholds in meters/degrees).
        # self._b2_min_view_disp = None
        # self._b2_min_view_yaw = None
        # self._kf_disp_step = None
        # self._kf_yaw_step = None

    @property
    def num_frames(self) -> int:
        return len(self._window)

    def reset(self) -> None:
        self._window.clear()
        self._last_pose = None

    def _view_gate(self, robot_pose: np.ndarray) -> bool:
        """True = skip this frame (not enough new view information).

        B1 (base) gate is enabled; B2/B3 blocks below are kept commented.
        """
        if self._last_pose is None:
            return False
        disp = float(np.linalg.norm(np.asarray(robot_pose)[:2] - self._last_pose[:2]))
        yaw_chg = _angle_diff(float(robot_pose[2]), float(self._last_pose[2]))
        if disp < self._min_view_disp and yaw_chg < self._min_view_yaw:
            return True
        # # B2 stricter gating (higher thresholds)
        # if self._b2_min_view_disp is not None:
        #     if disp < self._b2_min_view_disp and yaw_chg < self._b2_min_view_yaw:
        #         return True
        # # B3 keyframe sampling (fixed step)
        # if self._kf_disp_step is not None:
        #     if disp < self._kf_disp_step and yaw_chg < self._kf_yaw_step:
        #         return True
        return False

    def update(
        self,
        pcd_world: np.ndarray,
        robot_pose: np.ndarray,
    ) -> None:
        if len(pcd_world) == 0:
            return
        if self._view_gate(robot_pose):
            return

        vox = np.floor(pcd_world[:, :3] / self._fuse_voxel_size).astype(np.int64)
        _, idx = np.unique(vox, axis=0, return_index=True)

        self._window.append(pcd_world[np.sort(idx)])
        self._last_pose = np.asarray(robot_pose, dtype=np.float32).copy()

    def fuse_to_camera_canonical(
        self,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        z_offset: float = 0.0,  # 当前发送点云的 z 基准与 TSP3D 训练分布一致，平移破坏对齐
    ) -> np.ndarray:
        if len(self._window) == 0:
            return np.empty((0, 6), dtype=np.float32)

        frames = list(self._window)
        fused_pts = np.concatenate(frames, axis=0)

        # Transform to current camera-canonical frame (xy translation + yaw rotation, z kept)
        pts = fused_pts.copy()
        pts[:, :2] -= np.asarray(robot_xyz)[:2]
        if z_offset != 0.0:
            pts[:, 2] += z_offset
        cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        xy = pts[:, :2]
        pts[:, 0] = cos_yaw * xy[:, 0] + sin_yaw * xy[:, 1]
        pts[:, 1] = -sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]

        # Round-voxelization dedup (first observation wins)
        voxel_coords = np.round(pts[:, :3] / self._fuse_voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        pts = pts[np.sort(unique_idx)]

        if self._radius is not None:
            dist = np.linalg.norm(pts[:, :2], axis=1)
            pts = pts[dist <= self._radius]

        return pts
