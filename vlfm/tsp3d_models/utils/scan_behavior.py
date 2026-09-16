from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Union

import numpy as np

from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.detections import ObjectDetections
from vlfm.tsp3d_models.utils.panoramic import (
    _cap_point_count,
    _round_dedup,
    _to_camera_canonical,
)

if TYPE_CHECKING:  # pragma: no cover - annotation only (avoids an import cycle)
    from torch import Tensor
    from vlfm.policy.tsp3d_objectnav_policy import TSP3DObjectNavPolicy


def _act_panoramic(
    self: "TSP3DObjectNavPolicy",
    observations: Dict,
    pcd: np.ndarray,
    robot_xyz: np.ndarray,
    robot_yaw: float,
    rgb: np.ndarray,
    depth: np.ndarray,
    tf: np.ndarray,
    min_depth: float,
    max_depth: float,
    fx: float,
    fy: float,
    goal_3d: Union[None, np.ndarray],
) -> Tuple[ObjectDetections, str, "Tensor"]:
    """Panoramic route under a unified per-step decision (scan is an independent behavior, not a sub-state of explore).

    Each step follows the priority below:
      1. scan in progress   -> accumulate frame + wide-turn (no TSP3D);
                              at the last frame stitch 360° -> ONE query -> 
                              ``_after_scan_action`` (direction by c').
      2. initialize         -> 12 x 30° in-place turns; each step also sends
                              a single frame to TSP3D; init counts as a scan
                              of the start pose.
      3. decision layer (init done):
         - target in memory            -> navigate (single-frame input);
         - reached a NEW frontier      -> begin an independent 360° scan;
         - otherwise                   -> explore (single-frame input, move
                                          along the semantic value field).

    - Obstacle map is updated every step.
    - Mid-scan steps only accumulate the frame and turn with the scan-only wide-turn action (decoupled from the 30° nav turn).
    - Value-map updates are skipped during scans (BaseITMPolicy checks ``_scan_in_progress``).
    - Anti-hallucination fallback runs only on query steps.
    """
    pp = self._preprocessor
    self._obstacle_map3d.update_map(
        pcd=pcd, tf_camera_to_episodic=tf, depth=depth,
        min_depth=min_depth, max_depth=max_depth, fx=fx, fy=fy,
    )
    empty_det = ObjectDetections(
        boxes=[], logits=[], phrases=[], pcd_source=pcd, image_source=rgb,
        fx=fx, fy=fy, tf_camera_to_episodic=tf,
    )

    # mid-scan: accumulate this frame, keep turning
    if pp.is_scanning:
        # Record this scan frame's view 
        # so the whole 360° scan can refresh the semantic value field in ONE update at the scan end.
        self._scan_value_views.append(
            (rgb, depth, tf, min_depth, max_depth, float(self._camera_fov))
        )
        remaining = pp.continue_scan(pcd)
        if remaining <= 0:
            # Full turn done: stitch the 360° cloud and query TSP3D once,
            # then pick the next direction from this scan's candidates.
            pcd_local = pp.finish_scan(robot_xyz, robot_yaw, self._camera_height)
            # Refresh the semantic value field once with ALL scan frames before the single TSP3D query.
            _views = self._scan_value_views
            self._scan_value_views = []
            _value_updater = getattr(self, "_update_value_map", None)
            if callable(_value_updater) and _views:
                self._observations_cache["value_map_rgbd"] = _views
                _value_updater()
            det = self._query_and_process(
                pcd_local, robot_xyz, robot_yaw, pcd, rgb,
                tf, max_depth, fx, fy,
            )
            action = _after_scan_action(self, observations, robot_xyz, robot_yaw)
            # Scan finished -> next step resumes the unified decision flow.
            self._scan_in_progress = False
        else:
            det = empty_det
            action = self._scan_turn_action
            self._scan_in_progress = True
        mode = "scan"
        return det, mode, action

    # unified decision (scan not in progress)
    self._scan_in_progress = False
    if not self._done_initializing:
        # Cold-start guard against an immediate re-scan.
        pp.set_scan_position(robot_xyz, robot_yaw)
        pcd_local = pp.raw_frame(pcd, robot_xyz, robot_yaw, self._camera_height)
        det = self._query_and_process(
            pcd_local, robot_xyz, robot_yaw, pcd, rgb, tf, max_depth, fx, fy,
        )
        mode = "initialize"
        action = self._initialize()
    elif goal_3d is None:
        if _at_new_frontier(self, robot_xyz):
            # Reached a NEW frontier point -> independent 360° scan.
            # This step's view is the first slice of the scan.
            self._scan_value_views.append(
                (rgb, depth, tf, min_depth, max_depth, float(self._camera_fov))
            )
            pp.begin_scan(pcd, robot_xyz, robot_yaw)
            self._scan_in_progress = True
            mode = "scan"
            action = self._scan_turn_action
            return empty_det, mode, action
        # Travel step: send the raw single frame straight to TSP3D.
        pcd_local = pp.raw_frame(pcd, robot_xyz, robot_yaw, self._camera_height)
        det = self._query_and_process(
            pcd_local, robot_xyz, robot_yaw, pcd, rgb, tf, max_depth, fx, fy,
        )
        mode = "explore"
        action = self._explore(observations)
    else:
        mode = "navigate"
        print(f"[TSP3D Mode] Target '{self._target_object}' located at {goal_3d}. Navigating.")
        pcd_local = pp.raw_frame(pcd, robot_xyz, robot_yaw, self._camera_height)
        det = self._query_and_process(
            pcd_local, robot_xyz, robot_yaw, pcd, rgb, tf, max_depth, fx, fy,
        )
        action = self._pointnav(goal_3d[:2], stop=True)
    return det, mode, action


def _allow_scan(self: "TSP3DObjectNavPolicy", robot_xyz: np.ndarray) -> bool:
    """Scan-novelty gate: the robot must be >= panoramic_min_move 
    from the last scan / init pose before another scan is allowed.

    - panoramic route: delegated to the PanoramicFusion module.
    - world+scan: policy-owned ``_scan_pos``.
    """
    if self._fusion_style == "panoramic":
        return self._preprocessor.allow_scan(robot_xyz)
    if self._scan_pos is None:
        return True
    robot_xy = np.asarray(robot_xyz)[:2]
    return float(np.linalg.norm(robot_xy - self._scan_pos)) >= self._panoramic_min_move


def _build_scan_cloud(
    self: "TSP3DObjectNavPolicy",
    robot_xyz: np.ndarray,
    robot_yaw: float,
    camera_height: float,
) -> np.ndarray:
    """
    Stitch the buffered scan slices into ONE camera-canonical 360° cloud, INDEPENDENT of the WorldLocalMap.
    Returns an empty cloud when no slices are buffered.
    """
    frames = [
        f for f in getattr(self, "_scan_pcd_frames", []) if f is not None and len(f) > 0
    ]
    if not frames:
        return np.empty((0, 6), dtype=np.float32)
    pts_world = np.concatenate(frames, axis=0).astype(np.float32)
    local = _to_camera_canonical(pts_world, robot_xyz, robot_yaw)
    local = _round_dedup(local, self._scan_voxel_size)
    if self._scan_radius is not None:
        dist = np.linalg.norm(local[:, :2], axis=1)
        local = local[dist <= self._scan_radius]
    if len(local) == 0:
        return np.empty((0, 6), dtype=np.float32)
    camera_pos_local = np.array([0.0, 0.0, camera_height], dtype=np.float32)
    return _cap_point_count(local, camera_pos_local, self._scan_max_points, style="near_first")


def _scan_gap_ok(self: "TSP3DObjectNavPolicy") -> bool:
    """
    Sparse-scan cooldown: a new scan is allowed 
    only after ``scan_min_gap`` env steps since the last scan ended.
    """
    gap = int(getattr(self, "_scan_min_gap", 1))
    now = int(getattr(self, "_num_steps", 0))
    last = int(getattr(self, "_last_scan_step", -10**9))
    return (now - last) >= gap


def _scan_opportunistic_ok(self: "TSP3DObjectNavPolicy") -> bool:
    """
    Opportunistic trigger: fire when explore has run 
    ``scan_opportunistic_after`` env steps without locking a nav goal.
    """
    gap = int(getattr(self, "_scan_opportunistic_after", 0))
    if gap <= 0:
        return False
    now = int(getattr(self, "_num_steps", 0))
    last_lock = int(getattr(self, "_last_lock_step", 0))
    return (now - last_lock) >= gap


def _start_scan_step(
    self: "TSP3DObjectNavPolicy",
    pcd: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    tf: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> Tuple[str, "Tensor"]:
    """Begin an independent 360° scan with the current frame as the first slice
    (slices are buffered on the policy and never fed into the WorldLocalMap)."""
    self._scan_pcd_frames = [pcd]
    self._scan_value_views.append(
        (rgb, depth, tf, min_depth, max_depth, float(self._camera_fov))
    )
    self._scan_remaining = max(self._panoramic_turn_steps - 1, 0)
    self._scan_in_progress = True
    return "scan", self._scan_turn_action


def _act_world_scan(
    self: "TSP3DObjectNavPolicy",
    observations: Dict,
    pcd: np.ndarray,
    robot_xyz: np.ndarray,
    robot_yaw: float,
    rgb: np.ndarray,
    depth: np.ndarray,
    tf: np.ndarray,
    min_depth: float,
    max_depth: float,
    fx: float,
    fy: float,
    goal_3d: Union[None, np.ndarray],
) -> Tuple[ObjectDetections, str, "Tensor"]:
    """world+scan mechanism: world temporal fusion + sparse spatial scans.

    Active when ``fusion_style="world"`` and ``enable_scan=True``.
      - travel / init / navigate keep the pure world behaviour;
      - a scan is a BYPASS independent of the WorldLocalMap: its slices are
        buffered in ``self._scan_pcd_frames`` and never fed into the map;
      - scans fire on frontier arrival or after a long no-lock explore stretch;
      - scan steps update the obstacle map and record the value view only
        (no semantic-field update / no TSP3D query / fallback frozen)."""
    pp = self._preprocessor
    self._obstacle_map3d.update_map(
        pcd=pcd, tf_camera_to_episodic=tf, depth=depth,
        min_depth=min_depth, max_depth=max_depth, fx=fx, fy=fy,
    )
    empty_det = ObjectDetections(
        boxes=[], logits=[], phrases=[], pcd_source=pcd, image_source=rgb,
        fx=fx, fy=fy, tf_camera_to_episodic=tf,
    )

    # mid-scan: buffer this frame for the independent 360° stitch, keep turning
    if self._scan_in_progress:
        self._scan_pcd_frames.append(pcd)  # bypass: never fed into the WorldLocalMap
        self._scan_value_views.append(
            (rgb, depth, tf, min_depth, max_depth, float(self._camera_fov))
        )
        self._scan_remaining -= 1
        if self._scan_remaining <= 0:
            # Full turn done: refresh the semantic field once with ALL scan frames,
            # stitch the buffered slices into one 360° cloud, then ONE query.
            _views = self._scan_value_views
            self._scan_value_views = []
            _value_updater = getattr(self, "_update_value_map", None)
            if callable(_value_updater) and _views:
                self._observations_cache["value_map_rgbd"] = _views
                _value_updater()
            pcd_local = _build_scan_cloud(self, robot_xyz, robot_yaw, self._camera_height)
            det = self._query_and_process(
                pcd_local, robot_xyz, robot_yaw, pcd, rgb,
                tf, max_depth, fx, fy,
            )
            self._scan_in_progress = False
            self._scan_pos = np.asarray(robot_xyz)[:2].copy()
            self._scan_pcd_frames = []
            self._last_scan_step = self._num_steps  # sparse-scan cooldown start
            action = self._explore(observations)
        else:
            det = empty_det
            action = self._scan_turn_action
        mode = "scan"
        return det, mode, action

    # unified decision (no scan running)
    if not self._done_initializing:
        # Cold-start guard against an immediate re-scan.
        self._scan_pos = np.asarray(robot_xyz)[:2].copy()
        pp.update(pcd, robot_xyz, robot_yaw)
        pcd_local = pp.prepare(robot_xyz, robot_yaw, camera_height=self._camera_height)
        det = self._query_and_process(
            pcd_local, robot_xyz, robot_yaw, pcd, rgb, tf, max_depth, fx, fy,
        )
        mode = "initialize"
        action = self._initialize()
    elif goal_3d is None:
        # Scan triggers (both still obey the sparse cooldown _scan_gap_ok):
        if ((self._frontier_trigger and _at_new_frontier(self, robot_xyz)) or _scan_opportunistic_ok(self)) and _scan_gap_ok(self):
            mode, action = _start_scan_step(self, pcd, rgb, depth, tf, min_depth, max_depth)
            return empty_det, mode, action
        # travel step: world-map query, move along the semantic value field.
        pp.update(pcd, robot_xyz, robot_yaw)
        pcd_local = pp.prepare(robot_xyz, robot_yaw, camera_height=self._camera_height)
        det = self._query_and_process(
            pcd_local, robot_xyz, robot_yaw, pcd, rgb, tf, max_depth, fx, fy,
        )
        mode = "explore"
        action = self._explore(observations)
    else:
        mode = "navigate"
        print(f"[TSP3D Mode] Target '{self._target_object}' located at {goal_3d}. Navigating.")
        self._last_lock_step = self._num_steps  # opportunistic trigger: reset on a nav-goal lock
        pp.update(pcd, robot_xyz, robot_yaw)
        pcd_local = pp.prepare(robot_xyz, robot_yaw, camera_height=self._camera_height)
        det = self._query_and_process(
            pcd_local, robot_xyz, robot_yaw, pcd, rgb, tf, max_depth, fx, fy,
        )
        action = self._pointnav(goal_3d[:2], stop=True)
    return det, mode, action


def _after_scan_action(
    self: "TSP3DObjectNavPolicy",
    observations: Dict,
    robot_xyz: np.ndarray,
    robot_yaw: float,
) -> "Tensor":
    """Direction right after a finished 360° scan, decided S-penalty style.

    Move toward the candidate with the HIGHEST c' so every advance direction best matches the target. 
    With no admitted candidate, fall back to the semantic value field.
    """
    pending = getattr(self, "_last_pending", None) or []
    if pending:
        best = max(pending, key=lambda p: float(p[0]))  # highest c'
        conf_amp, centroid, near_surface, _ = best
        goal = near_surface if near_surface is not None else centroid
        print(
            f"[ScanEnd] best c'={conf_amp:.3f} goal={np.round(np.asarray(goal)[:2], 2)} "
            "-> moving to target"
        )
        return self._pointnav(np.asarray(goal)[:2], stop=False)
    return self._explore(observations)


def _at_new_frontier(self: "TSP3DObjectNavPolicy", robot_xyz: np.ndarray) -> bool:
    """True when the robot has REACHED a NEW frontier point (scan trigger).
    
    ``panoramic_min_move`` still guards against re-scanning the same spot.
    - frontier tracking (main path): a scan fires when the frontier cluster
      the explore layer is pursuing disappears from ``frontier_sensor``.
    - distance fallback (no tracked cluster yet): any frontier within
      ``panoramic_arrive_dist`` -> scan, so a scan is never starved.
    """
    frontiers = self._observations_cache.get("frontier_sensor")
    if frontiers is None or len(frontiers) == 0:
        return False
    frontiers = np.asarray(frontiers, dtype=np.float32)
    if frontiers.ndim != 2 or frontiers.shape[1] < 2:
        return False
    # Empty-frontier sentinel returned by the sensor is zeros((1, 2)).
    if frontiers.shape == (1, 2) and float(np.abs(frontiers).max()) < 1e-6:
        return False
    if not _allow_scan(self, robot_xyz):
        return False

    robot_xy = np.asarray(robot_xyz)[:2]
    # Frontier-tracking main path: fire only when the pursued cluster is gone
    # from the current frontier list (reached + digested).
    last = np.asarray(getattr(self, "_last_frontier", np.zeros(2)), dtype=np.float32)
    if last.shape == (2,) and float(np.abs(last).max()) >= 1e-6:
        # Still present (exact or within 0.5 m) -> keep advancing.
        if closest_point_within_threshold(frontiers[:, :2], last, threshold=0.5) != -1:
            return False
        # The pursued cluster was digested -> genuinely reached a new boundary.
        return True
    # No tracked cluster yet (reset / navigation since last explore pick):
    # distance-gate fallback so a scan is never starved.
    dists = np.linalg.norm(frontiers[:, :2] - robot_xy, axis=1)
    return float(dists.min()) <= self._panoramic_arrive_dist
