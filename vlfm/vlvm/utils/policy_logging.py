"""Policy logging helpers (detection / memory / nav-goal / recheck rows).

Pure bookkeeping: nothing here changes selection, actions or STOP.  Moved out
of the policy file to keep it readable.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class PolicyLoggingMixin:
    @staticmethod
    def _sigma_tier(score: float) -> int:
        """only for log compatibility."""
        if score >= 0.70:
            return 3
        if score >= 0.65:
            return 2
        if score >= 0.60:
            return 1
        return 0

    def _log_stop_shadow(
        self,
        *,
        chosen_idx: int,
        chosen_2d: np.ndarray,
        robot_xy: np.ndarray,
        dists_2d: np.ndarray,
        valid: Dict[str, List[Any]],
    ) -> None:
        """Append one read-only STOP shadow row for the selected goal.

        This only writes diagnostics; it never changes selection, action, or
        STOP eligibility.
        """
        idx = int(chosen_idx)
        chosen_2d = np.asarray(chosen_2d, dtype=np.float64).reshape(-1)[:2]
        robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
        chosen_state = str(valid["states"][idx])
        chosen_is_gd_tsp = bool(valid["gd_tsp"][idx])
        chosen_is_promoted = bool(valid["promoted_gd_only"][idx])
        chosen_is_gd_only = (
            bool(valid["gd_only"][idx]) or chosen_state == "gd"
        )
        if not (chosen_is_gd_only or chosen_state == "confirmed"):
            return

        goal_distance = float(np.linalg.norm(chosen_2d - robot_xy))
        obs = int(valid["obs"][idx])
        gd_hits = int(valid["gd_hits"][idx])
        row = {
            "event": "stop_shadow",
            "shadow_only": True,
            "step": int(self._num_steps),
            "memory_id": valid["memory_ids"][idx],
            "candidate_id": valid["candidate_ids"][idx],
            "target_class": str(self._target_object.split("|")[0]),
            "centroid": np.asarray(
                valid["centroids"][idx], dtype=np.float64
            ).reshape(-1)[:3].tolist(),
            "goal": np.asarray(chosen_2d, dtype=np.float64).reshape(-1)[:2].tolist(),
            "distance": float(dists_2d[idx]),
            "goal_distance": goal_distance,
            "lifecycle_state": chosen_state,
            "origin": valid["origins"][idx],
            "gd_only": bool(chosen_is_gd_only),
            "gd_tsp": bool(chosen_is_gd_tsp),
            "promoted_from_gd_only": bool(chosen_is_promoted),
            "sigma_c": float(valid["scores"][idx]),
            "sigma_band": int(valid["bands"][idx]),
            "gd_hits": gd_hits,
            "num_obs": obs,
            "stale_steps": int(valid["stale"][idx]),
            "stop_eligible_current": bool(valid["stop_ok"][idx]),
            "stop_gate_confirm_radius": float(self._stop_gate_radius),
            "pointnav_stop_radius": float(self._pointnav_stop_radius),
            "would_stop_radius_0_9": bool(
                goal_distance <= float(self._pointnav_stop_radius)
            ),
            "would_stop_radius_1_2": bool(
                goal_distance <= float(self._stop_gate_radius)
            ),
            "would_stop_numobs_ge3": bool(obs >= 3),
            "would_stop_numobs_ge5": bool(obs >= 5),
            "would_stop_gdhits_ge3": bool(gd_hits >= 3),
            "would_stop_gdhits_ge5": bool(gd_hits >= 5),
            "would_stop_gd3_r09": bool(
                goal_distance <= float(self._pointnav_stop_radius)
                and gd_hits >= 3
            ),
            "would_stop_gd5_r09": bool(
                goal_distance <= float(self._pointnav_stop_radius)
                and gd_hits >= 5
            ),
            "would_stop_obs3_r09": bool(
                goal_distance <= float(self._pointnav_stop_radius)
                and obs >= 3
            ),
            "would_stop_obs5_r09": bool(
                goal_distance <= float(self._pointnav_stop_radius)
                and obs >= 5
            ),
        }
        self._stop_shadow_logs.append(row)

    def _log_nav_goal(
        self,
        *,
        chosen_idx: int,
        goal: np.ndarray,
        robot_xy: np.ndarray,
        dists_2d: np.ndarray,
        switch_reason: str,
        tiers: List[int],
        valid: Dict[str, List[Any]],
    ) -> None:
        """Append one nav-goal selection row; no selection state is changed."""
        idx = int(chosen_idx)
        tier = int(tiers[idx])
        self._nav_goal_logs.append(
            {
                "step": int(self._num_steps),
                "memory_id": valid["memory_ids"][idx],
                "candidate_id": valid["candidate_ids"][idx],
                "centroid": np.asarray(
                    valid["centroids"][idx], dtype=np.float64
                ).tolist(),
                "goal": np.asarray(goal, dtype=np.float64).tolist(),
                "quality_score": float(valid["scores"][idx]),
                "selection_tier": tier,
                "selection_tier_name": (
                    "gd_tsp" if tier == 0
                    else "gd_only" if tier == 1
                    else "confirmed" if tier == 2
                    else "provisional" if tier == 3
                    else "other"
                ),
                "sigma_tier": int(self._sigma_tier(valid["scores"][idx])),
                "sigma_band": int(valid["bands"][idx]),
                "distance_bucket": int(
                    round(float(dists_2d[idx]) / 0.5)
                ),
                "num_obs_ge3": bool(int(valid["obs"][idx]) >= 3),
                "distance": float(dists_2d[idx]),
                "num_obs": int(valid["obs"][idx]),
                "switch_reason": str(switch_reason),
                "stale_steps": int(valid["stale"][idx]),
                "n_selectable": len(valid["centroids"]),
                "lifecycle_state": str(valid["states"][idx]),
                "origin": valid["origins"][idx],
                "gd_only": bool(valid["gd_only"][idx]),
                "gd_tsp": bool(valid["gd_tsp"][idx]),
                "gd_incremental": bool(valid["gd_incremental"][idx]),
                "gd_upgraded": bool(valid["gd_upgraded"][idx]),
                "promoted_from_gd_only": bool(
                    valid["promoted_gd_only"][idx]
                ),
                "gd_hits": int(valid["gd_hits"][idx]),
                "stop_eligible": bool(valid["stop_ok"][idx]),
                "robot_xy": np.asarray(
                    robot_xy, dtype=np.float64
                ).reshape(-1)[:2].tolist(),
                "robot_yaw": (
                    None
                    if self._last_robot_pose is None
                    else float(self._last_robot_pose[2])
                ),
            }
        )

    def _track_snapshot(self) -> List[Dict[str, Any]]:
        """Read-only snapshot of alive target tracks for GD calibration.

        The snapshot includes the last candidate box of each track (when it can
        be recovered from the memory-event stream), so offline matching can use
        the same 3D IaU / containment rule as the online merge.
        """
        candidate_by_id: Dict[str, Dict[str, Any]] = {
            str(row.get("candidate_id")): row
            for row in self._candidate_logs
            if row.get("candidate_id") is not None
        }
        last_candidate_by_memory: Dict[str, str] = {}
        for event in reversed(self._memory_event_logs):
            if event.get("event") not in ("new", "merge"):
                continue
            mid = event.get("memory_id")
            cid = event.get("candidate_id")
            if mid is None or cid is None:
                continue
            last_candidate_by_memory.setdefault(str(mid), str(cid))

        rows: List[Dict[str, Any]] = []
        for target_class, centroids in self._target_3d_memory.items():
            surfaces = self._target_surface_memory.get(target_class, [])
            for index, centroid in enumerate(centroids):
                c = np.asarray(centroid, dtype=np.float64).reshape(-1)
                state = self._memory_manager.state_of(c)
                memory_id = state.get("memory_id")
                if memory_id is None:
                    continue
                surface = (
                    np.asarray(surfaces[index], dtype=np.float64).reshape(-1)
                    if index < len(surfaces)
                    else c
                )
                cid = last_candidate_by_memory.get(str(memory_id))
                candidate = candidate_by_id.get(str(cid), {}) if cid else {}
                last_obs = state.get("last_obs_step")
                stale = None
                if last_obs is not None:
                    stale = max(0, int(self._memory_manager.frame) - 1 - int(last_obs))
                rows.append(
                    {
                        "memory_id": memory_id,
                        "target_class": str(target_class),
                        "centroid": c[:3].tolist(),
                        "surface": surface[:3].tolist(),
                        "box8": candidate.get("box"),
                        "lifecycle_state": state.get("lifecycle_state"),
                        "num_obs": state.get("num_obs"),
                        "sigma_c_max": state.get("sigma_c_max"),
                        "sigma_c_latest": state.get("sigma_c_latest"),
                        "sigma_c_ema_03": state.get("sigma_c_ema_03"),
                        "sigma_c_ema_05": state.get("sigma_c_ema_05"),
                        "sigma_c_ema_08": state.get("sigma_c_ema_08"),
                        "last_obs_step": last_obs,
                        "near_miss": state.get("near_miss"),
                        "total_miss": state.get("total_miss"),
                        "revive_count": state.get("revive_count"),
                        "stale_steps": stale,
                        "origin": state.get("origin"),
                        "gd_only": bool(state.get("gd_only", False)),
                        "gd_tsp": (
                            str(state.get("origin") or "") == "gd-tsp"
                            or bool(state.get("promoted_from_gd_only", False))
                        ),
                        "gd_hits": int(state.get("gd_hits", 0) or 0),
                        "gd_conf_max": state.get("gd_conf_max"),
                        "gd_conf_last": state.get("gd_conf_last"),
                        "gd_pos": bool(state.get("gd_pos", False)),
                        "gd_candidate_pos": bool(state.get("gd_candidate_pos", False)),
                        "gd_last_step": state.get("gd_last_step"),
                        "gd_upgraded": bool(state.get("gd_upgraded", False)),
                        "gd_upgraded_from": state.get("gd_upgraded_from"),
                        "promoted_from_gd_only": bool(
                            state.get("promoted_from_gd_only", False)
                        ),
                        "src": "gd" if str(state.get("lifecycle_state")) == "gd" else "tsp3d",
                    }
                )
        return rows

    def _log_candidates(self, records: List[Dict[str, Any]]) -> None:
        """Append per-frame candidate records to the candidate log."""
        if records:
            self._candidate_logs.extend(records)

    def _log_memory_event(self, event: Dict[str, Any]) -> None:
        """Append one memory-lifecycle event to the memory event log."""
        self._memory_event_logs.append(event)

    def _log_memory_events(self, events: List[Dict[str, Any]]) -> None:
        """Append multiple memory-lifecycle events."""
        if events:
            self._memory_event_logs.extend(events)

    def _log_accumulate_event(
        self,
        *,
        res: Dict[str, Any],
        cls: str,
        candidate_meta: Optional[Dict[str, Any]],
        centroid: np.ndarray,
        surface: Optional[np.ndarray],
        robot_xyz: np.ndarray,
        robot_yaw: float,
        sigma_c: float,
        detailed: bool,
        three_points: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one new/merge event produced by memory accumulation."""
        candidate_meta = candidate_meta or {}
        event = {
            "event": "merge" if res.get("merged") else "new",
            "step": int(self._num_steps),
            "memory_id": res.get("memory_id"),
            "candidate_id": candidate_meta.get("candidate_id"),
            "target_class": str(cls),
            "centroid": np.asarray(centroid, dtype=np.float64).tolist(),
            "surface": (
                None
                if surface is None
                else np.asarray(surface, dtype=np.float64).tolist()
            ),
            "confidence": float(sigma_c),
            "sigma_c": float(sigma_c),
            "num_obs": int(res.get("num_obs", 1)),
            "lifecycle_state": res.get("lifecycle_state"),
            "origin": res.get("origin"),
            **{
                key: res.get(key)
                for key in (
                    "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                    "sigma_c_ema_05", "sigma_c_ema_08",
                    "revive_sigma_c", "revive_num_obs_before",
                    "revive_num_obs_after", "revive_dist_to_old_centroid",
                    "revive_age_steps", "revive_last_obs_step",
                    "revive_radius", "revive_window", "revive_min_sigma",
                    "revive_detect_dist", "revive_robot_xy",
                    "revive_old_centroid", "revive_new_centroid",
                )
            },
        }
        if detailed:
            event.update(
                {
                    "new_info": res.get("new_info"),
                    "revived": bool(res.get("revived", False)),
                    "revive_count": int(res.get("revive_count", 0)),
                    "total_miss": int(res.get("total_miss", 0)),
                    "robot_xy": np.asarray(
                        robot_xyz, dtype=np.float64
                    ).reshape(-1)[:2].tolist(),
                    "robot_yaw": float(robot_yaw),
                    "detect_dist": float(
                        candidate_meta.get("distance") or 0.0
                    ),
                }
            )
        if three_points:
            event.update(three_points)
        self._log_memory_event(event)

    def _log_upgrade_event(
        self,
        *,
        res: Dict[str, Any],
        upgraded: Dict[str, Any],
        cls: str,
        candidate_meta: Optional[Dict[str, Any]],
        centroid: np.ndarray,
        surface: Optional[np.ndarray],
        robot_xyz: np.ndarray,
        robot_yaw: float,
        sigma_c: float,
    ) -> None:
        """Append one upgrade event produced by memory accumulation."""
        candidate_meta = candidate_meta or {}
        self._log_memory_event(
            {
                "event": "upgrade",
                "step": int(self._num_steps),
                "memory_id": res.get("memory_id"),
                "candidate_id": candidate_meta.get("candidate_id"),
                "target_class": str(cls),
                "centroid": np.asarray(centroid, dtype=np.float64).tolist(),
                "surface": (
                    None
                    if surface is None
                    else np.asarray(surface, dtype=np.float64).tolist()
                ),
                "confidence": float(sigma_c),
                "sigma_c": float(sigma_c),
                "num_obs": int(
                    upgraded.get("num_obs", res.get("num_obs", 1))
                ),
                "lifecycle_state": upgraded.get("to_state"),
                "origin": res.get("origin"),
                "from_state": upgraded.get("from_state"),
                "upgrade_reason": upgraded.get("reason"),
                "promoted_from_gd_only": bool(
                    res.get("promoted_from_gd_only", False)
                ),
                "detect_dist": float(upgraded.get("detect_dist", 0.0)),
                "robot_xy": np.asarray(
                    robot_xyz, dtype=np.float64
                ).reshape(-1)[:2].tolist(),
                "robot_yaw": float(robot_yaw),
                **{
                    key: res.get(key)
                    for key in (
                        "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                        "sigma_c_ema_05", "sigma_c_ema_08",
                    )
                },
            }
        )

    def _log_lifecycle_events(self, events: List[Dict[str, Any]]) -> None:
        """Append lifecycle (delete/revive/etc.) events."""
        for record in events:
            self._log_memory_event(
                {
                    "event": str(record.get("event") or "delete"),
                    "step": int(self._num_steps),
                    "memory_id": record.get("memory_id"),
                    "candidate_id": record.get("candidate_id"),
                    "target_class": record.get("target_class"),
                    "centroid": np.asarray(
                        record.get("centroid", []), dtype=np.float64
                    ).tolist(),
                    "confidence": float(record.get("conf", 0.0)),
                    "num_obs": int(record.get("num_obs", 1)),
                    "from_state": record.get("from_state"),
                    "lifecycle_state": record.get("lifecycle_state"),
                    "delete_reason": record.get("reason"),
                    "reason": record.get("reason"),
                    "near_miss": int(record.get("near_miss", 0)),
                    **{
                        key: record.get(key)
                        for key in (
                            "total_miss", "revive_count", "age_steps",
                            "last_obs_step", "dist_to_robot", "thresholds",
                            "gd_age_steps", "gd_last_step", "gd_hits",
                            "gd_conf_max", "gd_conf_last", "origin",
                        )
                    },
                    **{
                        key: record.get(key)
                        for key in (
                            "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                            "sigma_c_ema_05", "sigma_c_ema_08",
                        )
                    },
                }
            )

    def _log_try_upgrade_record(
        self,
        record: Dict[str, Any],
        *,
        robot_xyz: np.ndarray,
        robot_yaw: float,
    ) -> None:
        """Append one try_upgrade event."""
        self._log_memory_event(
            {
                "event": "upgrade",
                "step": int(self._num_steps),
                "memory_id": record.get("memory_id"),
                "candidate_id": record.get("candidate_id"),
                "target_class": record.get("target_class"),
                "centroid": np.asarray(
                    record.get("centroid", []), dtype=np.float64
                ).tolist(),
                "confidence": float(record.get("conf", 0.0)),
                "num_obs": int(record.get("num_obs", 1)),
                "from_state": record.get("from_state"),
                "upgrade_reason": record.get("reason"),
                "lifecycle_state": record.get("lifecycle_state"),
                "detect_dist": float(record.get("detect_dist", 0.0)),
                "robot_xy": np.asarray(
                    robot_xyz, dtype=np.float64
                ).reshape(-1)[:2].tolist(),
                "robot_yaw": float(robot_yaw),
                **{
                    key: record.get(key)
                    for key in (
                        "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                        "sigma_c_ema_05", "sigma_c_ema_08",
                    )
                },
            }
        )

    def _log_track_frames(
        self,
        frame_log_start: int,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        max_depth: float,
    ) -> None:
        """Per-frame visibility instrumentation for every alive target track.

        Records the geometry the offline calibration needs (distance, azimuth,
        FOV/depth eligibility, whether the track matched this frame); it never
        feeds a decision.
        """
        if not getattr(self, "_det_log_dir", None):
            return
        matched_ids = {
            row.get("memory_id")
            for row in self._memory_event_logs[frame_log_start:]
            if row.get("event") in ("new", "merge")
        }
        robot_xy = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2]
        half_fov = np.deg2rad(float(self._fov_angle)) * 0.5
        for cls in [c.strip() for c in self._target_object.split("|") if c.strip()]:
            centroids = self._target_3d_memory.get(cls) or []
            verify = self._target_verify_state.get(cls) or []
            for idx, centroid in enumerate(centroids):
                c = np.asarray(centroid, dtype=np.float64).reshape(-1)
                distance = float(np.linalg.norm(c[:2] - robot_xy))
                if distance > 8.0:
                    continue
                azimuth = float(np.arctan2(c[1] - robot_xy[1], c[0] - robot_xy[0]) - robot_yaw)
                azimuth = float((azimuth + np.pi) % (2 * np.pi) - np.pi)
                state = self._memory_manager.state_of(c)
                memory_id = state.get("memory_id")
                num_obs = int(verify[idx][0]) if idx < len(verify) and len(verify[idx]) >= 1 else 1
                _last_obs = state.get("last_obs_step")
                _age = max(0, max(0, int(self._memory_manager.frame) - 1) - (max(0, int(self._memory_manager.frame) - 1) if _last_obs is None else int(_last_obs)))
                _lifecycle = str(state.get("lifecycle_state") or "confirmed")
                _is_gd_tsp = (
                    str(state.get("origin") or "") == "gd-tsp"
                    or bool(state.get("promoted_from_gd_only", False))
                )
                self._track_frame_logs.append({
                    "step": int(self._num_steps),
                    "target_class": str(cls),
                    "memory_id": memory_id,
                    "centroid": c.tolist(),
                    "robot_xy": robot_xy.tolist(),
                    "robot_yaw": float(robot_yaw),
                    "distance": distance,
                    "azimuth_deg": float(np.rad2deg(azimuth)),
                    "in_fov": bool(abs(azimuth) <= half_fov),
                    "depth_ok": bool(distance <= float(max_depth) * 0.95),
                    "matched": bool(memory_id in matched_ids),
                    "num_obs": num_obs,
                    "lifecycle_state": _lifecycle,
                    "near_miss": int(state.get("near_miss", 0) or 0),
                    "total_miss": int(state.get("total_miss", 0) or 0),
                    "age_steps": int(_age),
                    "revive_count": int(state.get("revive_count", 0) or 0),
                    "stale_steps": int(max(0, _age)),
                    "origin": state.get("origin"),
                    "gd_only": bool(state.get("gd_only", False)),
                    "gd_tsp": _is_gd_tsp,
                    "gd_hits": int(state.get("gd_hits", 0) or 0),
                    "gd_pos": bool(state.get("gd_pos", False)),
                    "gd_candidate_pos": bool(state.get("gd_candidate_pos", False)),
                    "gd_last_step": state.get("gd_last_step"),
                    "gd_age": (
                        None if state.get("gd_last_step") is None
                        else max(0, int(self._memory_manager.frame) - 1 - int(state.get("gd_last_step")))
                    ),
                    "gd_conf_max": state.get("gd_conf_max"),
                    "gd_conf_last": state.get("gd_conf_last"),
                    "gd_upgraded": bool(state.get("gd_upgraded", False)),
                    "promoted_from_gd_only": bool(
                        state.get("promoted_from_gd_only", False)
                    ),
                    "stop_eligible": bool(
                        _lifecycle == "confirmed" and _is_gd_tsp
                    ),
                })

    def _log_stop_blacklist_event(
        self,
        *,
        state: Dict[str, Any],
        centroid: Optional[List[float]],
        turns: int,
        distance: float,
        lookaround_trace: Optional[List[Dict[str, Any]]] = None,
        sigma_c: Optional[float] = None,
    ) -> None:
        """Append one stop-gate blacklist event.

        ``lookaround_trace`` / ``sigma_*`` are log-only diagnostics used to
        judge whether extra views (or a short translation) at the gate could
        ever flip this memory to stop-eligible.
        """
        trace = list(lookaround_trace or [])
        sigmas = [
            float(row.get("sigma_c", 0.0))
            for row in trace
            if row.get("sigma_c") is not None
        ]
        sigma_now = float(
            sigma_c if sigma_c is not None else (sigmas[-1] if sigmas else 0.0)
        )
        sigma_tar = float(getattr(self, "_sigma_tar", 0.0) or 0.0)
        self._log_memory_event(
            {
                "event": "stop_blacklist",
                "step": int(self._num_steps),
                "memory_id": self._last_goal_memory_id,
                "candidate_id": state.get("last_candidate_id"),
                "target_class": str(
                    state.get("target_class")
                    or self._target_object.split("|")[0]
                ),
                "centroid": centroid,
                "reason": "stop_gate_exhausted",
                "blacklist_rule": "near_turns > stop_gate_turns",
                "near_turns": int(turns),
                "stop_gate_turns": int(self._stop_gate_turns),
                "stop_gate_confirm_radius": float(self._stop_gate_radius),
                "distance": float(distance),
                "sigma_c": sigma_now,
                "sigma_tar": sigma_tar,
                "sigma_gap": float(sigma_tar - sigma_now),
                "lookaround_sigma_max": (max(sigmas) if sigmas else sigma_now),
                "lookaround_sigma_min": (min(sigmas) if sigmas else sigma_now),
                "lookaround_sigma_delta": (
                    float(max(sigmas) - min(sigmas)) if sigmas else 0.0
                ),
                "lookaround_trace": trace[-8:],
                "lifecycle_state": state.get(
                    "lifecycle_state", self._last_goal_state
                ),
                "origin": state.get("origin", self._last_goal_origin),
                "gd_hits": int(
                    state.get("gd_hits", self._last_goal_gd_hits) or 0
                ),
                "num_obs": int(
                    state.get("num_obs", self._last_goal_num_obs) or 0
                ),
                "stop_eligible": False,
            }
        )

    # ==========================================================================
    # === Plan & Do Module ===
    # ==========================================================================

