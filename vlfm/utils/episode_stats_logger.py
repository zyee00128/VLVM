# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from frontier_exploration.utils.general_utils import xyz_to_habitat

from vlfm.utils.geometry_utils import transform_points
from vlfm.utils.habitat_visualizer import sim_xy_to_grid_xy
from vlfm.utils.log_saver import log_episode


def log_episode_stats(episode_id: int, scene_id: str, infos: Dict) -> str:
    """Log episode stats to the console.

    Args:
        episode_id: The episode ID.
        scene_id: The scene ID.
        infos: The info dict from the environment after update with policy info.
    """
    scene = os.path.basename(scene_id).split(".")[0]
    if infos["success"] == 1:
        failure_cause = "did_not_fail"
    else:
        failure_cause = determine_failure_cause(infos)
        print(f"Episode {episode_id} in scene {scene} failed due to '{failure_cause}'.")

    oracle_succ = compute_oracle_success(infos)
    print(
        f"[Oracle] Episode {episode_id} scene {scene} "
        f"target '{infos.get('target_object', 'unknown')}' "
        f"success={int(infos['success'])} oracle_success={oracle_succ}"
    )

    if "ZSOS_LOG_DIR" in os.environ:
        infos_no_map = infos.copy()
        infos_no_map.pop("top_down_map")

        data = {
            "failure_cause": failure_cause,
            **remove_numpy_arrays(infos_no_map),
        }

        log_episode(episode_id, scene, data)

    return failure_cause


def determine_failure_cause(infos: Dict) -> str:
    """Using the info and policy_info dicts, determine the cause of failure.

    Args:
        infos: The info dict from the environment after update with policy info.

    Returns:
        A string describing the cause of failure.
    """
    if infos["target_detected"]:
        if was_false_positive(infos):
            return "false_positive"
        else:
            if infos["stop_called"]:
                return "bad_stop_true_positive"
            else:
                return "timeout_true_positive"
    else:
        if was_target_seen(infos):
            return "false_negative"
        else:
            if infos["traveled_stairs"]:
                cause = "never_saw_target_traveled_stairs"
            else:
                cause = "never_saw_target_did_not_travel_stairs"
            if not infos["top_down_map"]["is_feasible"]:
                return cause + "_likely_infeasible"
            else:
                return cause + "_feasible"


def was_target_seen(infos: Dict[str, Any]) -> bool:
    target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]
    explored_area = infos["top_down_map"]["fog_of_war_mask"]
    # Dilate the target_bboxes_mask by 10 pixels to add a margin of error
    target_bboxes_mask = cv2.dilate(target_bboxes_mask, np.ones((10, 10)))
    target_explored = bool(np.any(np.logical_and(explored_area, target_bboxes_mask)))
    return target_explored


def compute_oracle_success(infos: Dict[str, Any]) -> int:
    """
    Oracle Success Rate for a single episode.

    Oracle = 1 if the target bbox was ever covered by the explored (fog-of-war) area during the episode. 
    fog_of_war is cumulative, so the final frame's state is equivalent to "ever seen". 
    Ground-truth signal, detector-agnostic (same signal used by ``determine_failure_cause`` for ``false_negative``).
    """
    try:
        return int(was_target_seen(infos))
    except Exception:
        return 0


def was_false_positive(infos: Dict[str, Any]) -> bool:
    """Return whether the point goal target is within a bounding box."""
    target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]

    nav_goal_episodic_xy = np.array(infos["nav_goal"])
    nav_goal_episodic_xyz = np.array([nav_goal_episodic_xy[0], nav_goal_episodic_xy[1], 0]).reshape(1, 3)

    upper_bound = infos["top_down_map"]["upper_bound"]
    lower_bound = infos["top_down_map"]["lower_bound"]
    grid_resolution = infos["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos["top_down_map"]["tf_episodic_to_global"]

    nav_goal_global_xyz = transform_points(tf_episodic_to_global, nav_goal_episodic_xyz)
    nav_goal_global_habitat = xyz_to_habitat(nav_goal_global_xyz)
    nav_goal_global_habitat_xy = nav_goal_global_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        nav_goal_global_habitat_xy,
        remove_duplicates=True,
    )

    try:
        return target_bboxes_mask[grid_xy[0, 0], grid_xy[0, 1]] == 0
    except IndexError:
        # If the point goal is outside the map, assume it is a false positive
        return True


def remove_numpy_arrays(d: Any) -> Dict:
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = remove_numpy_arrays(value)
        elif isinstance(value, np.ndarray):
            # Serialize small coordinate/feature arrays to lists 
            # to prevent 3D motion metrics from being lost in the logs
            if value.size <= 10:
                new_dict[key] = value.tolist()
        else:
            new_dict[key] = value

    return new_dict


def _point_in_target_bbox(
    infos: Dict[str, Any], point_episodic_xy, dilated_mask=None
) -> bool:
    """
    Whether a world (episodic) point falls inside the dilated target bbox mask.

    Uses the final-frame top_down_map's ``target_bboxes_mask`` (static GT), 
    the same coordinate transform as ``was_false_positive``. 
    ``dilated_mask`` may be precomputed to avoid re-dilating per call.
    """
    try:
        if dilated_mask is None:
            target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]
            dilated_mask = cv2.dilate(target_bboxes_mask, np.ones((10, 10)))
        upper_bound = infos["top_down_map"]["upper_bound"]
        lower_bound = infos["top_down_map"]["lower_bound"]
        grid_resolution = infos["top_down_map"]["grid_resolution"]
        tf_episodic_to_global = infos["top_down_map"]["tf_episodic_to_global"]

        point_episodic_xyz = np.array(
            [point_episodic_xy[0], point_episodic_xy[1], 0.0]
        ).reshape(1, 3)
        point_global_xyz = transform_points(tf_episodic_to_global, point_episodic_xyz)
        point_global_habitat = xyz_to_habitat(point_global_xyz)
        point_global_habitat_xy = point_global_habitat[:, [2, 0]]

        grid_xy = sim_xy_to_grid_xy(
            upper_bound,
            lower_bound,
            grid_resolution,
            point_global_habitat_xy,
            remove_duplicates=True,
        )
        return bool(dilated_mask[grid_xy[0, 0], grid_xy[0, 1]] != 0)
    except Exception:
        return False


def aggregate_detect_stats(
    infos: Dict[str, Any], detect_logs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Detection-level statistics for one episode (TSP3D detections vs GT target bbox).

    ``detect_logs`` entries: {conf, conf_amp, centroid, s, admitted} 
    (from apply_s_penalty(..., out_log=...)``).

    Returns a dict with counts and per-detection score arrays.
    """
    total = len(detect_logs)
    if total == 0:
        return {"total": 0, "tp": 0, "fp": 0, "scores": [], "conf_amps": [], "tp_flags": [], "admitted": [], "s_vals": [], "rejects": {}}

    try:
        dilated_mask = cv2.dilate(
            infos["top_down_map"]["target_bboxes_mask"], np.ones((10, 10))
        )
    except Exception:
        dilated_mask = None

    tp_flags = []
    for det in detect_logs:
        tp_flags.append(_point_in_target_bbox(infos, det["centroid"][:2], dilated_mask))
    tp = int(sum(tp_flags))
    rejects: Dict[str, int] = {}
    for det in detect_logs:
        reason = det.get("reject")
        if reason:
            rejects[str(reason)] = rejects.get(str(reason), 0) + 1
    return {
        "total": total,
        "tp": tp,
        "fp": total - tp,
        "scores": [float(d["conf"]) for d in detect_logs],
        "conf_amps": [float(d.get("conf_amp", d["conf"])) for d in detect_logs],
        "tp_flags": tp_flags,
        "admitted": [bool(d.get("admitted", False)) for d in detect_logs],
        "s_vals": [d.get("s", None) for d in detect_logs],
        "rejects": rejects,
    }


def _as_opt_float(value: Any) -> Optional[float]:
    """Best-effort float conversion for jsonl dumping (None stays None)."""
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def dump_detection_logs(
    path: str,
    *,
    scene_id: str,
    episode_id: int,
    target_object: str,
    infos: Dict[str, Any],
    detect_logs: List[Dict[str, Any]],
    cfg_snapshot: Dict[str, Any],
    success: int,
    oracle_success: int,
    failure_cause: str,
) -> int:
    """SP1 per-detection jsonl dump (append; one file spans the whole eval run).

    Layout: one ``{"kind": "ep"}`` header per episode (scene / episode / target /
    outcome / the S-penalty config in force), then one ``{"kind": "det"}`` line per
    detection carrying the raw S-penalty inputs and the decision branch. TP is decided
    here with the same dilated target-mask test as ``aggregate_detect_stats``, so
    ``ft_tsp3d/eval_tools/s_evidence.py`` can recompute S-only / fused statistics offline.

    Only ``t_S`` is searched and can be re-derived offline from the logged raw ``s``.
    ``s_min=0.1`` and query ``radius_m=0.5`` are fixed implementation constants.

    Returns the number of detection lines written.
    """
    scene = os.path.basename(str(scene_id)).split(".")[0]
    try:
        dilated_mask = cv2.dilate(
            infos["top_down_map"]["target_bboxes_mask"], np.ones((10, 10))
        )
    except Exception:
        dilated_mask = None

    target = str(target_object or "unknown")
    n_written = 0
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "ep",
                    "scene": scene,
                    "episode_id": int(episode_id),
                    "target_object": target,
                    "success": int(success),
                    "oracle_success": int(oracle_success),
                    "failure_cause": str(failure_cause),
                    "n_det": len(detect_logs),
                    "cfg": cfg_snapshot,
                }
            )
            + "\n"
        )
        for idx, det in enumerate(detect_logs):
            centroid = np.asarray(det.get("centroid", []), dtype=np.float64).reshape(-1)
            tp = _point_in_target_bbox(infos, centroid[:2], dilated_mask)
            fh.write(
                json.dumps(
                    {
                        "kind": "det",
                        "scene": scene,
                        "episode_id": int(episode_id),
                        "target_object": target,
                        "success": int(success),
                        "oracle_success": int(oracle_success),
                        "failure_cause": str(failure_cause),
                        "idx": int(idx),
                        "tp": bool(tp),
                        "step": det.get("step"),
                        "conf": _as_opt_float(det.get("conf")),
                        "pass_prop_045": bool(det.get("pass_prop_045", False)),
                        "pass_prop_050": bool(det.get("pass_prop_050", False)),
                        "pass_prop_055": bool(det.get("pass_prop_055", False)),
                        "pass_prop_060": bool(det.get("pass_prop_060", False)),
                        "sigma_prop_floor": _as_opt_float(det.get("sigma_prop_floor")),
                        "sigma_tar_admit": _as_opt_float(det.get("sigma_tar_admit")),
                        "conf_amp": _as_opt_float(det.get("conf_amp")),
                        "s": _as_opt_float(det.get("s")),
                        "s_pt": det.get("s_pt"),
                        "s_pt_source": det.get("s_pt_source"),
                        "w_s": _as_opt_float(det.get("w_s")),
                        "branch": det.get("branch"),
                        "dist": _as_opt_float(det.get("dist")),
                        "admitted": bool(det.get("admitted", False)),
                        "reject": det.get("reject"),
                        "use_surface": det.get("use_surface"),
                    }
                )
                + "\n"
            )
            n_written += 1
    return n_written


def _jsonable(value: Any) -> Any:
    """Convert numpy-heavy diagnostic records to JSON-safe builtin values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def dump_phase_g_logs(
    directory: str,
    *,
    scene_id: str,
    episode_id: int,
    target_object: str,
    infos: Dict[str, Any],
    candidate_logs: List[Dict[str, Any]],
    memory_event_logs: List[Dict[str, Any]],
    nav_goal_logs: List[Dict[str, Any]],
    cfg_snapshot: Dict[str, Any],
    episode_stats: Dict[str, Any],
    success: int,
    oracle_success: int,
    failure_cause: str,
) -> Dict[str, int]:
    """Write the four-layer Phase-G contract without changing policy behavior."""
    os.makedirs(directory, exist_ok=True)
    scene = os.path.basename(str(scene_id)).split(".")[0]
    episode = int(episode_id)
    prefix = f"{scene}:{episode}"
    try:
        dilated_mask = cv2.dilate(
            infos["top_down_map"]["target_bboxes_mask"], np.ones((10, 10))
        )
    except Exception:
        dilated_mask = None

    def _uid(local_id: Any, tag: str) -> Optional[str]:
        return None if local_id in (None, "") else f"{prefix}:{tag}:{local_id}"

    def _tp(record: Dict[str, Any]) -> bool:
        point = record.get("centroid")
        if point is None:
            point = record.get("goal")
        point_array = np.asarray([] if point is None else point)
        return _point_in_target_bbox(infos, point_array[:2], dilated_mask)

    common = {
        "scene": scene,
        "episode_id": episode,
        "target_object": str(target_object or "unknown"),
        "success": int(success),
        "oracle_success": int(oracle_success),
        "failure_cause": str(failure_cause),
    }

    outputs: Dict[str, List[Dict[str, Any]]] = {
        "candidates.jsonl": [],
        "memory_events.jsonl": [],
        "nav_goals.jsonl": [],
    }
    for record in candidate_logs:
        row = {**common, "kind": "candidate", **_jsonable(record)}
        row["candidate_uid"] = _uid(row.get("candidate_id"), "candidate")
        row["tp"] = bool(_tp(row))
        outputs["candidates.jsonl"].append(row)
    for record in memory_event_logs:
        row = {**common, "kind": "memory_event", **_jsonable(record)}
        row["candidate_uid"] = _uid(row.get("candidate_id"), "candidate")
        row["memory_uid"] = _uid(row.get("memory_id"), "memory")
        row["tp"] = bool(_tp(row))
        outputs["memory_events.jsonl"].append(row)
    labelled_nav: List[Dict[str, Any]] = []
    for record in nav_goal_logs:
        row = {**common, "kind": "nav_goal", **_jsonable(record)}
        row["candidate_uid"] = _uid(row.get("candidate_id"), "candidate")
        row["memory_uid"] = _uid(row.get("memory_id"), "memory")
        row["tp"] = bool(_tp(row))
        labelled_nav.append(row)
    outputs["nav_goals.jsonl"] = labelled_nav

    for filename, rows in outputs.items():
        if not rows:
            continue
        with open(os.path.join(directory, filename), "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    first_wrong = next((row for row in labelled_nav if not row.get("tp")), None)
    episode_row = {
        **common,
        "kind": "episode",
        "cfg": _jsonable(cfg_snapshot),
        "reward": _as_opt_float(episode_stats.get("reward")),
        "spl": _as_opt_float(episode_stats.get("spl")),
        "soft_spl": _as_opt_float(episode_stats.get("soft_spl")),
        "steps_count": _as_opt_float(episode_stats.get("steps_count")),
        "n_candidates": len(candidate_logs),
        "n_memory_events": len(memory_event_logs),
        "n_nav_goals": len(nav_goal_logs),
        "first_wrong_lock_step": None if first_wrong is None else first_wrong.get("step"),
        "first_wrong_memory_uid": None if first_wrong is None else first_wrong.get("memory_uid"),
        "wrong_stop_before_target_coverage": bool(
            str(failure_cause) == "false_positive" and not int(oracle_success)
        ),
    }
    with open(os.path.join(directory, "episodes.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(episode_row, ensure_ascii=False) + "\n")

    return {
        "candidates": len(candidate_logs),
        "memory_events": len(memory_event_logs),
        "nav_goals": len(nav_goal_logs),
        "episodes": 1,
    }
