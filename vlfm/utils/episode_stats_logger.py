# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List

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
    """Oracle Success Rate for a single episode.

    Oracle = 1 if the target bbox was ever covered by the explored (fog-of-war)
    area during the episode. fog_of_war is cumulative, so the final frame's state
    is equivalent to "ever seen". Ground-truth signal, detector-agnostic (same
    signal used by ``determine_failure_cause`` for ``false_negative``).
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
            # Serialize small coordinate/feature arrays to lists to prevent
            # 3D motion metrics from being lost in the logs
            if value.size <= 10:
                new_dict[key] = value.tolist()
        else:
            new_dict[key] = value

    return new_dict


def _point_in_target_bbox(
    infos: Dict[str, Any], point_episodic_xy, dilated_mask=None
) -> bool:
    """Whether a world (episodic) point falls inside the dilated target bbox mask.

    Uses the final-frame top_down_map's ``target_bboxes_mask`` (static GT), the
    same coordinate transform as ``was_false_positive``. ``dilated_mask`` may be
    precomputed to avoid re-dilating per call.
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


def aggregate_memory_stats(
    infos: Dict[str, Any], entries: Any = None
) -> Dict[str, Any]:
    """Per-entry tp / fp labels of the episode-end memory (A4 measurement #2, §3.7.10).

    Entries are read from the policy snapshot (`policy._memory_entries_snapshot()`, read by
    the trainer right after the episode ends) or, as a fallback, from
    ``infos["memory_entries"]``. Every entry is labelled against the static GT target bbox
    with the same transform as ``was_false_positive``; results are split by proposer tag
    (`src`) and by the suspicious subset, which is what A2 / A3 need to judge
    "deleted tp share" and "lock-right : lock-wrong".

    M-a (§五 量测): `gd_conf_*` slots carry the same splits for the raw GD sigmoid.
    Only entries with `gd_conf > 0` (entries that ever carried a GD vote) enter those
    means — a `tsp3d`-only entry writes no GD score and would otherwise dilute the
    average with zeros. `gd_conf_tp` vs `gd_conf_fp` is the separability read-out the
    confidence direction needs (a flat pair = GD score cannot rank tp over fp).

    3-5 shadow (§优化方向 3): `geo_ind_*` groups the `src=gd` entries by tp / fp and
    sums their geometric-independent vote count (`gd_geo_ind` from the memory manager)
    plus how many reach `>= 2` (`ge2`) — the read-out for whether that criterion can
    replace the raw `num_obs` lock count.
    """
    if entries is None:
        entries = infos.get("memory_entries")
    if isinstance(entries, dict):
        entries = list(entries.values())
    entries = list(entries or [])
    if not entries:
        return {}
    try:
        dilated_mask = cv2.dilate(
            infos["top_down_map"]["target_bboxes_mask"], np.ones((10, 10))
        )
    except Exception:
        dilated_mask = None

    def _gslot() -> Dict[str, float]:
        # n = entries, gd_conf_n = entries WITH a GD score, gd_conf = sum over those.
        # geo_ind = sum of the 3-5 geometric-independent vote count over the entries.
        return {"n": 0, "tp": 0, "num_obs": 0.0,
                "gd_conf": 0.0, "gd_conf_n": 0.0, "geo_ind": 0.0}

    out: Dict[str, Any] = {
        "total": 0, "tp": 0, "by_src": {},
        "suspicious": {"n": 0, "tp": 0},
        "gd_conf_tp": _gslot(), "gd_conf_fp": _gslot(),
        # 3-5 前提量测：`src=gd` 条目的几何独立票数 tp / fp 分组（`ge2` = ind>=2 的条目数）。
        "geo_ind_tp": {"n": 0, "ind": 0.0, "ge2": 0},
        "geo_ind_fp": {"n": 0, "ind": 0.0, "ge2": 0},
        # D2-4 前提量测：逐条目 `g_self`（簇密度 / 框面积占比）的 tp / fp 分离。
        "gself_tp": {"n": 0, "density": 0.0, "box_frac": 0.0},
        "gself_fp": {"n": 0, "density": 0.0, "box_frac": 0.0},
    }
    for entry in entries:
        try:
            cxy = np.array(
                [float(entry.get("x", 0.0)), float(entry.get("y", 0.0))], dtype=np.float64
            )
        except Exception:
            continue
        is_tp = _point_in_target_bbox(infos, cxy, dilated_mask)
        out["total"] += 1
        out["tp"] += int(is_tp)
        gconf = float(entry.get("gd_conf", 0.0) or 0.0)
        src = str(entry.get("src") or "tsp3d")
        slot = out["by_src"].setdefault(src, _gslot())
        slot["n"] += 1
        slot["tp"] += int(is_tp)
        slot["num_obs"] += float(entry.get("num_obs", 0.0))
        geo_ind = float(entry.get("gd_geo_ind", 0.0) or 0.0)
        slot["geo_ind"] += geo_ind
        if src == "gd":
            # 3-5 shadow: the criterion is a candidate replacement of `num_obs` for
            # `gd` single votes — group those entries by tp / fp.
            grp_geo = out["geo_ind_tp"] if is_tp else out["geo_ind_fp"]
            grp_geo["n"] += 1
            grp_geo["ind"] += geo_ind
            grp_geo["ge2"] += int(geo_ind >= 2)
        if gconf > 0.0:
            slot["gd_conf"] += gconf
            slot["gd_conf_n"] += 1
        # tp / fp split of the GD score itself (the M-a judgement gate).
        if gconf > 0.0:
            grp = out["gd_conf_tp"] if is_tp else out["gd_conf_fp"]
            grp["n"] += 1
            grp["gd_conf"] += gconf
        if entry.get("suspicious"):
            out["suspicious"]["n"] += 1
            out["suspicious"]["tp"] += int(is_tp)
        # D2-4 前提：带 GD 提案特征的条目按 tp / fp 分组累计（g_self 可分性）。
        if float(entry.get("gself_n", 0) or 0) > 0:
            grp = out["gself_tp"] if is_tp else out["gself_fp"]
            grp["n"] += 1
            grp["density"] += float(entry.get("gself_density", 0.0) or 0.0)
            grp["box_frac"] += float(entry.get("gself_box_frac", 0.0) or 0.0)
    for slot in out["by_src"].values():
        slot["num_obs"] /= max(1, slot["n"])
        slot["gd_conf"] /= max(1.0, slot["gd_conf_n"])
        slot["geo_ind"] /= max(1, slot["n"])
    for key in ("gd_conf_tp", "gd_conf_fp"):
        out[key]["gd_conf"] /= max(1.0, out[key]["n"])
    for key in ("gself_tp", "gself_fp"):
        out[key]["density"] /= max(1, out[key]["n"])
        out[key]["box_frac"] /= max(1, out[key]["n"])
    return out


def aggregate_detect_stats(
    infos: Dict[str, Any], detect_logs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Detection-level statistics for one episode (TSP3D detections vs GT target bbox).

    ``detect_logs`` entries: {conf, conf_amp, centroid, s, admitted} (from apply_s_penalty(..., out_log=...)``).

    Returns a dict with counts and per-detection score arrays.
    """
    total = len(detect_logs)
    if total == 0:
        return {"total": 0, "tp": 0, "fp": 0, "scores": [], "conf_amps": [], "tp_flags": [], "admitted": [], "s_vals": []}

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
    return {
        "total": total,
        "tp": tp,
        "fp": total - tp,
        "scores": [float(d["conf"]) for d in detect_logs],
        "conf_amps": [float(d.get("conf_amp", d["conf"])) for d in detect_logs],
        "tp_flags": tp_flags,
        "admitted": [bool(d.get("admitted", False)) for d in detect_logs],
        # 09-17 量测补强：S（语义场值，检测框近表面点；None/≤0 = 无覆盖 = 免罚）。
        "s_vals": [d.get("s", None) for d in detect_logs],
    }
