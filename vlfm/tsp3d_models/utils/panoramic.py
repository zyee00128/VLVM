from typing import List, Optional
import numpy as np


def _to_camera_canonical(pts_world: np.ndarray, robot_xyz: np.ndarray, robot_yaw: float) -> np.ndarray:
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


def _round_dedup(pts_local: np.ndarray, voxel_size: float) -> np.ndarray:
    """Round-voxelization dedup in the local frame (intra-scan stitching)."""
    if len(pts_local) == 0:
        return pts_local
    vc = np.round(pts_local[:, :3] / float(voxel_size)).astype(np.int64)
    _, idx = np.unique(vc, axis=0, return_index=True)
    return pts_local[np.sort(idx)]


def _cap_near_first(pts_local: np.ndarray, camera_pos_local: np.ndarray, max_points: int) -> np.ndarray:
    """Cap to max_points keeping the nearest points (deterministic near-field priority)."""
    if len(pts_local) <= max_points:
        return pts_local
    dist = np.linalg.norm(pts_local[:, :3] - np.asarray(camera_pos_local)[:3], axis=1)
    order = np.argsort(dist, kind="stable")
    return pts_local[order[:max_points]]


class PanoramicFusion:
    """360° spatial fusion at a frontier decision point (independent route)."""

    def __init__(
        self,
        turn_steps: int = 6,
        voxel_size: float = 0.02,
        radius: Optional[float] = 6.0,
        max_points: int = 200000,
        min_move: float = 2.0,
    ) -> None:
        self.turn_steps = max(int(turn_steps), 1)
        self.voxel_size = float(voxel_size)
        self.radius = float(radius) if radius is not None else None
        self.max_points = int(max_points)
        self.min_move = float(min_move)
        self.reset()

    def reset(self) -> None:
        self._scan_frames: List[np.ndarray] = []
        self._scan_remaining: int = 0
        self._last_scan_pos: Optional[np.ndarray] = None  # xy of the last scan / init pose
        self._last_scan_yaw: float = 0.0

    @property
    def is_scanning(self) -> bool:
        return len(self._scan_frames) > 0

    @property
    def scan_remaining(self) -> int:
        return self._scan_remaining

    # ======================================================================
    # Scan-decision helpers (frontier-arrival trigger lives in the policy)
    # ======================================================================
    def allow_scan(self, robot_xyz: np.ndarray) -> bool:
        """True if the robot has moved >= min_move from the last scan / init pose."""
        if self._last_scan_pos is None:
            return True
        robot_xy = np.asarray(robot_xyz)[:2]
        return float(np.linalg.norm(robot_xy - self._last_scan_pos)) >= self.min_move

    def set_scan_position(self, robot_xyz: np.ndarray, robot_yaw: float) -> None:
        """Record a completed scan at this pose (init is treated as a scan)."""
        self._last_scan_pos = np.asarray(robot_xyz)[:2].copy()
        self._last_scan_yaw = float(robot_yaw)

    # ======================================================================
    # Scan lifecycle (policy state machine drives the wide turns)
    # ======================================================================
    def begin_scan(self, frame_pcd_world: np.ndarray, robot_xyz: np.ndarray, robot_yaw: float) -> None:
        """Start a scan with the current frame as the first slice."""
        self._scan_frames = [frame_pcd_world]
        self._scan_remaining = max(self.turn_steps - 1, 0)
        self._last_scan_pos = np.asarray(robot_xyz)[:2].copy()
        self._last_scan_yaw = float(robot_yaw)

    def continue_scan(self, frame_pcd_world: np.ndarray) -> int:
        """Accumulate one more frame (one more wide in-place turn). Returns remaining."""
        self._scan_frames.append(frame_pcd_world)
        self._scan_remaining = max(self._scan_remaining - 1, 0)
        return self._scan_remaining

    def finish_scan(self, robot_xyz: np.ndarray, robot_yaw: float, camera_height: float) -> np.ndarray:
        """Stitch the accumulated frames into one local 360° cloud (send once)."""
        local = self._build(self._scan_frames, robot_xyz, robot_yaw, camera_height)
        self._scan_frames = []
        self._scan_remaining = 0
        return local

    # ======================================================================
    # Single-frame route (all non-scan steps)
    # ======================================================================
    def raw_frame(self, frame_pcd_world: np.ndarray, robot_xyz: np.ndarray, robot_yaw: float, camera_height: float) -> np.ndarray:
        """Convert the current frame to a camera-canonical cloud (sent as-is)."""
        return self._build([frame_pcd_world], robot_xyz, robot_yaw, camera_height)

    # ======================================================================
    # Core build
    # ======================================================================
    def _build(
        self,
        frames: List[np.ndarray],
        robot_xyz: np.ndarray,
        robot_yaw: float,
        camera_height: float,
    ) -> np.ndarray:
        frames = [f for f in frames if f is not None and len(f) > 0]
        if not frames:
            return np.empty((0, 6), dtype=np.float32)

        pts_world = np.concatenate(frames, axis=0).astype(np.float32)

        # Camera-canonical frame of the query instant (scan-end pose); z kept.
        local = _to_camera_canonical(pts_world, robot_xyz, robot_yaw)
        # Intra-scan stitching dedup + radius crop + cap.
        local = _round_dedup(local, self.voxel_size)
        if self.radius is not None:
            dist = np.linalg.norm(local[:, :2], axis=1)
            local = local[dist <= self.radius]
        if len(local) == 0:
            return np.empty((0, 6), dtype=np.float32)
        camera_pos_local = np.array([0.0, 0.0, camera_height], dtype=np.float32)
        local = _cap_near_first(local, camera_pos_local, self.max_points)
        return local
