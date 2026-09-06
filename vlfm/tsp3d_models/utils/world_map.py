from collections import deque
from typing import Optional
import numpy as np

_KEY_BASE = 1 << 15
_KEY_STRIDE = 2 * _KEY_BASE


def _flat_key(vox: np.ndarray) -> np.ndarray:
    """Encode world-frame voxel integer coords (N,3) into collision-free int64 flat keys (1D)."""
    rel = vox.astype(np.int64)
    return (
        (rel[:, 0] + _KEY_BASE) * _KEY_STRIDE * _KEY_STRIDE
        + (rel[:, 1] + _KEY_BASE) * _KEY_STRIDE
        + (rel[:, 2] + _KEY_BASE)
    )


def _angle_diff(a: float, b: float) -> float:
    """Absolute angular difference in radians, in [0, pi]."""
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


class WorldLocalMap:
    """Incremental world-frame occupancy of fused points.

    Dedup happens at merge time via flat keys; voxels are removed by a
    fixed number of recent frames (``max_frames``) and a radius slide-out.
    """

    def __init__(
        self,
        voxel_size: float = 0.02,
        radius: float = 6.0,
        min_view_disp: float = 0.15,
        min_view_yaw: float = np.deg2rad(15.0),
        max_voxels: int = 400000,
        max_frames: Optional[int] = 8,
        near_refresh_radius: Optional[float] = None,
        near_refresh_value: bool = False,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.radius = float(radius) if radius is not None else None
        self.min_view_disp = float(min_view_disp)
        self.min_view_yaw = float(min_view_yaw)
        self.max_voxels = int(max_voxels)
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.near_refresh_radius = (
            float(near_refresh_radius) if near_refresh_radius is not None else None
        )
        self.near_refresh_value = bool(near_refresh_value)

        # B2 (stricter gating) and B3 (keyframe sampling) are KEPT as reference only.
        # To re-enable, uncomment the gating blocks below and wire params through
        # pipeline/policy (disp/yaw thresholds in meters/degrees).
        # self._b2_min_view_disp = None
        # self._b2_min_view_yaw = None
        # self._kf_disp_step = None
        # self._kf_yaw_step = None

        self._keys: np.ndarray = np.empty(0, dtype=np.int64)   # sorted flat keys
        self._points: np.ndarray = np.empty((0, 6), dtype=np.float32)  # (x,y,z,r,g,b)
        self._frame_keys: deque = deque()  # per-frame new keys, for frame-window slide-out
        self._last_pose: Optional[np.ndarray] = None  # [x, y, yaw]

    @property
    def num_voxels(self) -> int:
        return int(self._keys.shape[0])

    @property
    def is_empty(self) -> bool:
        return self._keys.shape[0] == 0

    def reset(self) -> None:
        self._keys = np.empty(0, dtype=np.int64)
        self._points = np.empty((0, 6), dtype=np.float32)
        self._frame_keys.clear()
        self._last_pose = None

    def _view_gate(self, robot_pose: np.ndarray) -> bool:
        """True = skip this frame (not enough new view information).
        """
        if self._last_pose is None:
            return False
        disp = float(np.linalg.norm(robot_pose[:2] - self._last_pose[:2]))
        yaw_chg = _angle_diff(float(robot_pose[2]), float(self._last_pose[2]))
        if disp < self.min_view_disp and yaw_chg < self.min_view_yaw:
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

    def _slide_radius(self, center_xy: np.ndarray) -> None:
        """Drop voxels beyond radius (skipped if radius is None); if still above
        the hard max_voxels cap, keep the nearest voxels by distance."""
        if self._keys.shape[0] == 0:
            return
        if self.radius is not None:
            dist = np.linalg.norm(self._points[:, :2] - np.asarray(center_xy)[:2], axis=1)
            keep = dist <= self.radius
            if not keep.all():
                self._keys = self._keys[keep]
                self._points = self._points[keep]
        if self._keys.shape[0] > self.max_voxels:
            dist = np.linalg.norm(self._points[:, :2] - np.asarray(center_xy)[:2], axis=1)
            keep = np.argsort(dist)[: self.max_voxels]
            self._keys = self._keys[keep]
            self._points = self._points[keep]

    def update(
        self,
        pcd_world: np.ndarray,
        robot_pose: np.ndarray,
    ) -> None:
        if len(pcd_world) == 0:
            return
        if self._view_gate(robot_pose):
            return

        # Fixed-grid voxelization (floor, same grid as window fusion)
        vox = np.floor(pcd_world[:, :3] / self.voxel_size).astype(np.int64)
        _, first_idx = np.unique(vox, axis=0, return_index=True)

        vox = vox[np.sort(first_idx)]
        pts = pcd_world[np.sort(first_idx)]
        new_keys = _flat_key(vox)
        self._last_pose = np.asarray(robot_pose, dtype=np.float32).copy()

        # Merge only unseen voxels
        refresh_keys = np.empty(0, dtype=np.int64)
        if self._keys.shape[0] == 0:
            order = np.argsort(new_keys, kind="stable")
            self._keys = new_keys[order]
            self._points = pts[order].astype(np.float32)
            new_only_keys = new_keys
        else:
            is_dup = np.isin(new_keys, self._keys)
            new_only_keys = new_keys[~is_dup]
            new_only_pts = pts[~is_dup]
            # Near-field refresh: re-observed near-field voxels are re-owned by the current frame.
            if self.max_frames is not None and self.near_refresh_radius is not None:
                dup_keys = new_keys[is_dup]
                if dup_keys.shape[0] > 0:
                    dup_pts = pts[is_dup]
                    dist = np.linalg.norm(
                        dup_pts[:, :2] - np.asarray(robot_pose)[:2], axis=1
                    )
                    refresh = dist <= self.near_refresh_radius
                    refresh_keys = dup_keys[refresh]
                    if refresh_keys.shape[0] > 0:
                        for i in range(len(self._frame_keys)):
                            fk = self._frame_keys[i]
                            if len(fk) > 0:
                                rm = np.isin(fk, refresh_keys)
                                if rm.any():
                                    self._frame_keys[i] = fk[~rm]
                        # Value-layer refresh: overwrite the stored point with the current observation.
                        if self.near_refresh_value:
                            idx = np.searchsorted(self._keys, refresh_keys)
                            self._points[idx] = dup_pts[refresh]
            if new_only_keys.shape[0] > 0:
                all_keys = np.concatenate([self._keys, new_only_keys])
                all_pts = np.concatenate([self._points, new_only_pts], axis=0)
                order = np.argsort(all_keys, kind="stable")
                self._keys = all_keys[order]
                self._points = all_pts[order]

        # Keep only the most recent max_frames frames
        if self.max_frames is not None:
            if refresh_keys.shape[0] > 0 and new_only_keys.shape[0] > 0:
                current_keys = np.concatenate([new_only_keys, refresh_keys])
            elif refresh_keys.shape[0] > 0:
                current_keys = refresh_keys
            else:
                current_keys = new_only_keys
            self._frame_keys.append(current_keys)
            while len(self._frame_keys) > self.max_frames:
                oldest = self._frame_keys.popleft()
                if len(oldest) > 0:
                    rm = np.isin(self._keys, oldest)
                    self._keys = self._keys[~rm]
                    self._points = self._points[~rm]

        # Bounded-radius slide-out centered at the robot
        self._slide_radius(robot_pose[:2])

    def to_camera_canonical(
        self,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        z_offset: float = 0.0,  # 当前发送点云的 z 基准与 TSP3D 训练分布一致，平移破坏对齐
    ) -> np.ndarray:
        if self._keys.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)

        pts = self._points.copy()
        pts[:, :2] -= np.asarray(robot_xyz)[:2]
        if z_offset != 0.0:
            pts[:, 2] += z_offset
        cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        xy = pts[:, :2]
        pts[:, 0] = cos_yaw * xy[:, 0] + sin_yaw * xy[:, 1]
        pts[:, 1] = -sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]
        return pts
