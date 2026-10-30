"""Compressed side-channel episode logger for offline field calibration."""

from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
import zipfile
from typing import Any, Dict, Iterable, Optional

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert numpy / torch / scalar values to a CPU numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class FieldObservationLogger:
    """Write frame records to a temporary ZIP, then finalize with episode GT.

    Frames contain filtered normalized RGB-D, camera-to-episodic transform,
    camera intrinsics/depth range, robot pose, frontier sensor points, and map
    projection parameters. ``robot_xyz`` is the camera center; ``robot_xy`` is
    the ground-plane robot position. ``view_index`` disambiguates multi-view
    frames at the same step. The logger never mutates its inputs.
    """

    def __init__(self, root: str, map_spec: Dict[str, Any]):
        self.root = os.path.join(os.path.abspath(root), "field_observations")
        os.makedirs(self.root, exist_ok=True)
        fd, self._tmp_path = tempfile.mkstemp(prefix=".pending_field_", suffix=".zip", dir=self.root)
        os.close(fd)
        self._zip = zipfile.ZipFile(self._tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED)
        self._map_spec = _jsonable(map_spec)
        self._frame_count = 0
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def append_frame(
        self,
        *,
        step: int,
        target_object: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_to_episodic: np.ndarray,
        depth_range_intrinsics: Iterable[float],
        robot_xyz: np.ndarray,
        robot_yaw: float,
        frontiers_3d: np.ndarray,
        frontiers_2d: np.ndarray,
        pixel_origin: np.ndarray,
        view_index: int = 0,
        robot_xy: Optional[np.ndarray] = None,
    ) -> None:
        if self._closed:
            return
        idx = self._frame_count
        robot_arr = _as_numpy(robot_xyz).astype(np.float32).reshape(-1)
        robot_ground = (
            robot_arr[:2]
            if robot_xy is None
            else _as_numpy(robot_xy).astype(np.float32).reshape(-1)[:2]
        )
        record = {
            "step": int(step),
            "view_index": int(view_index),
            "target_object": str(target_object),
            "robot_xyz": robot_arr[:3],
            "robot_xy": robot_ground[:2],
            "robot_yaw_rad": np.float32(robot_yaw),
            "frontiers_3d": _as_numpy(frontiers_3d).astype(np.float32).reshape(-1, 3),
            "frontiers_2d": _as_numpy(frontiers_2d).astype(np.float32),
            "rgb": _as_numpy(rgb).astype(np.uint8),
            "depth_normalized_filtered": _as_numpy(depth).astype(np.float32),
            "camera_to_episodic": _as_numpy(camera_to_episodic).astype(np.float64),
            "depth_range_intrinsics": np.asarray(list(depth_range_intrinsics), dtype=np.float32),
            "map_pixel_origin": _as_numpy(pixel_origin).astype(np.int32),
        }
        buffer = io.BytesIO()
        # The outer ZIP already deflates; avoid a second compression pass here.
        np.savez(buffer, **record)
        self._zip.writestr(f"frames/{idx:06d}.npz", buffer.getvalue())
        self._frame_count += 1

    def finalize(
        self,
        *,
        scene_id: str,
        episode_id: int,
        target_object: str,
        infos: Dict[str, Any],
    ) -> Optional[str]:
        if self._closed:
            return None
        top_down = infos.get("top_down_map") or {}
        metadata = {
            "schema": "semantic_spatial_field_observations_v1",
            "scene_id": str(scene_id),
            "episode_id": int(episode_id),
            "target_object": str(target_object or infos.get("target_object", "unknown")),
            "frame_count": self._frame_count,
            "map_spec": self._map_spec,
            "depth_representation": "output of the policy depth filter; depth range and intrinsics are recorded per frame",
            "gt_map_keys": [
                "target_bboxes_mask", "upper_bound", "lower_bound",
                "grid_resolution", "tf_episodic_to_global",
            ],
        }
        self._zip.writestr("episode.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        gt = {}
        for key in metadata["gt_map_keys"]:
            if key in top_down:
                gt[key] = _as_numpy(top_down[key])
        if gt:
            buffer = io.BytesIO()
            np.savez(buffer, **gt)
            self._zip.writestr("ground_truth/top_down_map.npz", buffer.getvalue())
        self._zip.close()
        self._closed = True

        scene = os.path.basename(str(scene_id)).split(".")[0]
        scene_dir = os.path.join(self.root, scene)
        os.makedirs(scene_dir, exist_ok=True)
        destination = os.path.join(
            scene_dir, f"{int(episode_id)}_{uuid.uuid4().hex[:8]}.zip"
        )
        os.replace(self._tmp_path, destination)
        return destination

    def abort(self) -> None:
        """Close and remove an unfinished archive (used only on explicit cleanup)."""
        if not self._closed:
            try:
                self._zip.close()
            except Exception:
                pass
            self._closed = True
        try:
            os.remove(self._tmp_path)
        except OSError:
            pass
