"""Online GD mechanism: gd-only pool and gd-tsp confirmation.

This module consumes the JSON-safe record produced by
:class:`GdCalibrationLogger.observe`; it does not call GD or TSP3D itself.
That keeps the detector/association path identical to the log-only run and
makes the mechanism replayable from the same logs.

Flow (the overall navigation flow is unchanged):

    GD proposal
      -> conservative GD gate
      -> track match:
           stable                       -> blocked (no write)
           confirmed / provisional      -> gd-tsp (track centroid write)
           gd                            -> gd-only merge
      -> track_same & stable -> blocked before depth (no write, no shadow)
      -> no track match (candidate-only is diagnostics only)
           -> gd-only if class/occ/depth gates pass (no conf gate)
      -> depth gate after track association:
           confirmed / provisional depth-fail -> gd-tsp track merge
           gd-only / candidate-only depth-fail -> rejected
      -> gd-only pool / confirmed pool
      -> policy ranking

No GD value is written into ``sigma_c``.  GD-only entries are selectable but
never STOP-eligible until a later admitted TSP3D candidate promotes them; gd-tsp
entries are ordinary confirmed entries.

``process`` is called by the policy after ``_query_and_process`` has updated
the TSP3D memory.  Therefore ``record["tracks"]`` already contains candidates
that passed admission in the current step; candidate-only matches are logged
but cannot create a confirmed entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vlfm.tsp3d_models.fussion_utils.target_memory_manager import (
    CONFIRMED,
    GD,
    GD_ONLY,
    GD_TSP,
    PROVISIONAL,
    STABLE,
    TargetMemoryManager,
)


@dataclass(frozen=True)
class GdMechanismConfig:
    enabled: bool = True
    conf_min: float = 0.40
    max_dist_m: float = 5.0
    # An empty whitelist means all target classes are allowed.  Keep the
    # tuple form for an explicit class restriction when an ablation needs it.
    gd_only_classes: Tuple[str, ...] = ()
    # If False, geom_unknown is allowed when occ_passed=True; if True, only
    # HARD_CONFIRMED is accepted.
    require_hard_occ_for_gd_only: bool = False
    # Fixed S6-final behavior:
    #   * gd-only requires depth_ok (no depth exception);
    #   * depth-failed confirmed/provisional tracks may still merge via gd-tsp;
    #   * stable hits never upgrade and write no shadow event.



@dataclass(frozen=True)
class GdMechanismStats:
    proposals: int = 0
    base_rejected: int = 0
    gd_tsp: int = 0
    gd_only: int = 0
    blocked: int = 0


class GdMechanism:
    """Apply the online routing/write rules to one GD observation record."""

    def __init__(
        self,
        *,
        memory_manager: TargetMemoryManager,
        geom_gate: Any,
        obstacle_map3d: Any,
        config: Optional[GdMechanismConfig] = None,
    ) -> None:
        self.memory = memory_manager
        self.geom_gate = geom_gate
        self.obstacle_map3d = obstacle_map3d
        self.cfg = config or GdMechanismConfig()

    def process(
        self,
        record: Dict[str, Any],
        *,
        pcd: np.ndarray,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        target_classes: Sequence[str],
        step: int,
    ) -> Dict[str, Any]:
        """Route and write all proposals in ``record``.

        The returned ``record["mechanism"]`` contains one decision per proposal
        and a small counter summary.  It never raises on a bad proposal; a
        rejected proposal is logged and skipped.
        """
        if not self.cfg.enabled:
            return {"enabled": False, "decisions": []}
        proposals = record.get("proposals") or []
        candidates = {
            str(row.get("candidate_id")): row
            for row in (record.get("tsp3d", {}) or {}).get("candidates", [])
            if row.get("candidate_id") is not None
        }
        target_set = {str(c).strip().lower() for c in target_classes}
        decisions: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        counts = {
            "proposals": len(proposals),
            "base_rejected": 0,
            "gd_tsp": 0,
            "gd_only": 0,
            "blocked": 0,
        }

        for index, row in enumerate(proposals):
            decision = self._decide(
                row, candidates, target_set, pcd, robot_xyz, robot_yaw, step
            )
            decisions.append(decision)
            counts[decision["route"]] = counts.get(decision["route"], 0) + 1
            if decision.get("event") is not None:
                events.append(decision.pop("event"))
            if decision.get("upgraded") is not None:
                upgrade = decision["upgraded"]
                events.append(
                    {
                        "event": "upgrade",
                        "step": int(step),
                        "proposal_id": decision.get("proposal_id"),
                        "memory_id": decision.get("memory_id"),
                        "target_class": decision.get("target_class", decision.get("canon")),
                        "origin": GD_TSP,
                        "centroid": decision.get("centroid"),
                        "from_state": upgrade.get("from_state"),
                        "lifecycle_state": upgrade.get("to_state", CONFIRMED),
                        "reason": upgrade.get("reason", "gd_tsp"),
                        "upgrade_reason": upgrade.get("reason", "gd_tsp"),
                        "num_obs": int(upgrade.get("num_obs", 0) or 0),
                        "detect_dist": float(upgrade.get("detect_dist", 0.0) or 0.0),
                    }
                )
            if decision.get("memory_id") is not None or decision.get("written") is not None:
                pass

        # Events are attached for the policy's memory-event stream.
        out = {
            "enabled": True,
            "config": {
                "conf_min": float(self.cfg.conf_min),
                "max_dist_m": float(self.cfg.max_dist_m),
                "gd_only_classes": list(self.cfg.gd_only_classes),
                "require_hard_occ_for_gd_only": bool(self.cfg.require_hard_occ_for_gd_only),
                "gd_only_depth_required": True,
                "gd_tsp_depth_track_merge": True,
            },
            "counts": counts,
            "decisions": decisions,
            "events": events,
        }
        return out

    # ------------------------------------------------------------------
    def _decide(
        self,
        row: Dict[str, Any],
        candidates: Dict[str, Dict[str, Any]],
        target_set: set,
        pcd: np.ndarray,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        step: int,
    ) -> Dict[str, Any]:
        proposal_id = str(row.get("proposal_id") or "")
        canon = str(row.get("canon") or "").strip().lower()
        conf = float(row.get("conf", 0.0) or 0.0)
        dist = float(row.get("dist_robot", float("inf")) or float("inf"))
        depth_ok = bool(row.get("depth_ok", False))
        in_fov = bool(row.get("in_fov", False))
        decision: Dict[str, Any] = {
            "proposal_id": proposal_id,
            "canon": canon,
            "conf": conf,
            "dist_robot": dist,
            "depth_ok": depth_ok,
            "in_fov": in_fov,
            "gd_pos": bool(row.get("gd_pos", False)),
            "gd_candidate_pos": bool(row.get("gd_candidate_pos", False)),
            "gd_hits": int(row.get("gd_hits_after", 0) or 0),
            "association": "none",
            "route": "base_rejected",
            "reason": "",
        }

        if not canon:
            decision.update(reason="missing_target_class")
            return decision
        if target_set and canon not in target_set:
            decision.update(reason="not_target_class")
            return decision
        if conf < float(self.cfg.conf_min):
            decision.update(reason="conf_below_min")
            return decision
        if not in_fov:
            decision.update(reason="not_in_fov")
            return decision
        if not np.isfinite(dist) or dist > float(self.cfg.max_dist_m):
            decision.update(reason="distance_above_max")
            return decision

        match_track = row.get("match_track") or {}
        match_candidate = row.get("match_candidate") or {}
        track_state = str(match_track.get("lifecycle_state") or "")
        track_same = bool(match_track.get("same_object"))
        candidate_id = match_candidate.get("candidate_id") or row.get("matched_candidate_id")
        candidate = candidates.get(str(candidate_id)) if candidate_id is not None else None

        # S6-final stable rule: a GD hit on stable is always blocked before
        # depth evaluation; it never writes memory and never writes shadow.
        if track_same and track_state == STABLE:
            decision.update(
                association="track_stable",
                route="blocked",
                reason="stable_no_merge",
            )
            return decision

        # S6-final depth rule: only an existing confirmed/provisional track
        # may bypass depth_ok, because the merge uses the track centroid.  A
        # gd-only proposal (and candidate-only paths) must have depth_ok;
        # there is no depth exception and no shadow event.
        if not depth_ok:
            association = "none"
            if track_same:
                association = (
                    "track_confirmed"
                    if track_state == CONFIRMED
                    else "track_provisional"
                    if track_state == PROVISIONAL
                    else f"track_{track_state or 'unknown'}"
                )
            decision["association"] = association
            depth_track_relax = bool(
                track_same and track_state in (CONFIRMED, PROVISIONAL)
            )
            if not depth_track_relax:
                decision.update(reason="depth_not_ok")
                return decision
            decision.update(
                reason="depth_relaxed_track_merge",
                depth_relaxed=True,
            )

        # Routing is track-first and deliberately ignores candidate-only
        # matches: a candidate that has not passed admission is not a track.
        # This prevents `candidate match -> gd-tsp_new` from bypassing the
        # candidate gate and creating a premature confirmed entry.
        if track_same:
            if track_state in (CONFIRMED, PROVISIONAL):
                decision["association"] = (
                    "track_confirmed"
                    if track_state == CONFIRMED
                    else "track_provisional"
                )
                return self._write_gd_tsp(
                    row, decision, match_track, candidate, robot_xyz, robot_yaw, step,
                    target_class=str(match_track.get("target_class") or canon),
                )
            decision["association"] = f"track_{track_state or 'unknown'}"
        elif candidate is not None and bool(match_candidate.get("same_object")):
            # Logged for diagnostics only; never a confirmation path.  The
            # candidate must first pass admission and become a track.
            decision["association"] = "candidate_only"
            decision["candidate_admitted"] = bool(
                match_candidate.get("admitted", False)
            )
            decision["candidate_lifecycle_state"] = match_candidate.get(
                "lifecycle_state"
            )
        else:
            decision["association"] = "none"

        # A proposal matching an existing gd-only track stays gd-only; a
        # candidate-only match is also treated as no track association unless
        # the candidate later becomes a track.  Only the GD box gate below can
        # write the fallback pool.
        if self.cfg.gd_only_classes and canon not in self.cfg.gd_only_classes:
            decision.update(route="base_rejected", reason="gd_only_class_disabled")
            return decision
        is_gd_merge = decision.get("association") == "track_gd"
        # S6-final: gd-only has no confidence gate.  `conf` is still recorded
        # in the event for diagnostics, but never rejects a proposal.
        decision.update(gd_only_merge=bool(is_gd_merge))
        box8 = np.asarray(row.get("box8", ()), dtype=np.float64).reshape(-1, 3)
        if box8.shape != (8, 3):
            decision.update(route="base_rejected", reason="invalid_box8")
            return decision
        try:
            status, _unused, diag = self.geom_gate.process_detection(
                canon or "target",
                box8,
                np.asarray(pcd, dtype=np.float64),
                self.obstacle_map3d,
                conf,
                int(step),
                robot_xyz=np.asarray(robot_xyz, dtype=np.float64),
            )
        except Exception as exc:  # noqa: BLE001
            decision.update(route="base_rejected", reason=f"gd_occ_error:{type(exc).__name__}")
            return decision
        occ_diag_raw = (diag or {}).get("diagnostics", {}) or {}
        occ_metrics = occ_diag_raw.get("occ_diag", occ_diag_raw) or {}
        occ_passed = bool(occ_metrics.get("passed", False))
        occ_confirmed = str(status) == "HARD_CONFIRMED"
        decision["occ_status"] = str(status)
        decision["occ_passed"] = occ_passed
        decision["occ_confirmed"] = occ_confirmed
        decision["occ_diag"] = occ_diag_raw
        # A geom_unknown box has occ_passed=True but is not HARD_CONFIRMED.
        # Keep it as gd-only evidence unless the caller explicitly requires a
        # hard occupancy confirmation.  The gate itself uses occ_passed;
        # occ_confirmed remains available as a diagnostic/ranking feature.
        if not occ_passed:
            decision.update(route="base_rejected", reason="gd_occ_passed_false")
            return decision
        if self.cfg.require_hard_occ_for_gd_only and not occ_confirmed:
            decision.update(route="base_rejected", reason="gd_occ_geom_unknown")
            return decision

        centroid = np.asarray(row.get("centroid", ()), dtype=np.float64).reshape(-1)
        surface = np.asarray(row.get("surface", centroid), dtype=np.float64).reshape(-1)
        result = self.memory.accumulate_gd_only(
            canon,
            centroid,
            gd_conf=conf,
            near_surface=surface if surface.size else None,
            robot_xyz=np.asarray(robot_xyz, dtype=np.float64),
            robot_yaw=float(robot_yaw),
            candidate_id=proposal_id,
            # Let the manager apply its own new-view gate to gd-only hits;
            # the logger's new_info field is track-scoped and is not meaningful
            # for an unmatched proposal.
            new_info=None,
        )
        decision.update(
            route="gd_only",
            written=bool(result.get("written", False)),
            memory_id=result.get("memory_id"),
            memory_merged=bool(result.get("merged", False)),
            lifecycle_state=result.get("lifecycle_state"),
            origin=result.get("origin", GD_ONLY),
            gd_hits=int(result.get("gd_hits", 0) or 0),
            memory_reason=result.get("gd_merge_blocked") or result.get("reason"),
        )
        if decision["written"]:
            decision["event"] = {
                "event": "gd_only_new" if not decision["memory_merged"] else "gd_only_merge",
                "step": int(step),
                "proposal_id": proposal_id,
                "target_class": canon,
                "origin": GD_ONLY,
                "association": decision.get("association"),
                "lifecycle_state": decision.get("lifecycle_state"),
                "memory_id": decision.get("memory_id"),
                "gd_hits": decision.get("gd_hits"),
                "centroid": centroid[:3].tolist(),
                "conf": conf,
                "dist_robot": dist,
                "depth_ok": bool(depth_ok),
                "gd_only_merge": bool(is_gd_merge),
            }
        return decision

    # ------------------------------------------------------------------
    def _write_gd_tsp(
        self,
        row: Dict[str, Any],
        decision: Dict[str, Any],
        match_track: Optional[Dict[str, Any]],
        candidate: Optional[Dict[str, Any]],
        robot_xyz: np.ndarray,
        robot_yaw: float,
        step: int,
        *,
        target_class: str,
    ) -> Dict[str, Any]:
        # Track-first writes must use the track centroid, not the current GD
        # proposal/candidate centroid.  gd-tsp is merge-only: it enriches an
        # existing track and never creates a new one.
        if match_track is not None:
            centroid = np.asarray(
                match_track.get("centroid", row.get("centroid", ())),
                dtype=np.float64,
            ).reshape(-1)
            surface = np.asarray(
                match_track.get("surface", row.get("surface", centroid)),
                dtype=np.float64,
            ).reshape(-1)
        else:
            centroid = np.asarray(
                (candidate or {}).get("centroid", row.get("centroid", ())),
                dtype=np.float64,
            ).reshape(-1)
            surface = np.asarray(
                (candidate or {}).get("surface", row.get("surface", centroid)),
                dtype=np.float64,
            ).reshape(-1)
        if centroid.size < 3 or not np.all(np.isfinite(centroid[:3])):
            decision.update(route="base_rejected", reason="invalid_gd_tsp_centroid")
            return decision
        if not target_class:
            decision.update(route="base_rejected", reason="missing_target_class")
            return decision
        expected_memory_id = None
        if match_track is not None:
            track_state = str(match_track.get("lifecycle_state") or "")
            if track_state == "stable":
                decision.update(route="blocked", reason="stable_no_merge")
                return decision
            expected_memory_id = match_track.get("memory_id")
        if not expected_memory_id:
            decision.update(route="blocked", reason="missing_track_memory_id")
            return decision
        sigma_c = None
        if candidate is not None:
            sigma_c = candidate.get("sigma_c")
            confidence = float(candidate.get("conf", row.get("conf", 0.0)) or 0.0)
        else:
            sigma_c = match_track.get("sigma_c_latest") if match_track else None
            confidence = float(row.get("conf", 0.0) or 0.0)
        result = self.memory.accumulate(
            target_class,
            centroid,
            confidence=confidence,
            sigma_c=float(sigma_c) if sigma_c is not None else confidence,
            near_surface=surface if surface.size else None,
            robot_xyz=np.asarray(robot_xyz, dtype=np.float64),
            robot_yaw=float(robot_yaw),
            candidate_id=row.get("proposal_id"),
            new_lifecycle_state=CONFIRMED,
            origin=GD_TSP,
            force_confirmed=True,
            block_stable=True,
            gd_evidence_only=True,
            expected_memory_id=str(expected_memory_id),
        )
        if not result.get("written", False) or not result.get("merged", False):
            decision.update(
                route="blocked",
                reason=str(result.get("gd_merge_blocked") or result.get("blocked") or result.get("reason") or "not_merged"),
                memory_id=result.get("memory_id"),
                lifecycle_state=result.get("lifecycle_state"),
            )
            return decision
        gd_hits = int(row.get("gd_hits_after", 1) or 1)
        evidence = self.memory.apply_gd_evidence(
            target_class,
            centroid,
            gd_conf=float(row.get("conf", 0.0) or 0.0),
            gd_hits=gd_hits,
            gd_pos=bool(row.get("gd_pos", False)),
            gd_candidate_pos=bool(row.get("gd_candidate_pos", False)),
            candidate_id=row.get("proposal_id"),
            robot_xy=np.asarray(robot_xyz, dtype=np.float64)[:2],
            robot_yaw=float(robot_yaw),
        )
        decision.update(
            route="gd_tsp",
            target_class=target_class,
            centroid=centroid[:3].tolist(),
            written=True,
            memory_id=result.get("memory_id") or evidence.get("memory_id"),
            memory_merged=bool(result.get("merged", False)),
            lifecycle_state=result.get("lifecycle_state", CONFIRMED),
            origin=GD_TSP,
            bypass_s_penalty=True,
            gd_hits=int(evidence.get("gd_hits", gd_hits) or gd_hits),
            gd_pos=bool(row.get("gd_pos", False)),
            gd_candidate_pos=bool(row.get("gd_candidate_pos", False)),
            depth_relaxed=bool(decision.get("depth_relaxed", False)),
            upgraded=result.get("upgraded"),
        )
        decision["event"] = {
            # gd-tsp only enriches an existing track; a non-merge is blocked
            # above, so there is no gd_tsp_new event path anymore.
            "event": "gd_tsp_merge",
            "step": int(step),
            "proposal_id": row.get("proposal_id"),
            "target_class": target_class,
            "origin": GD_TSP,
            "association": decision.get("association"),
            "lifecycle_state": decision.get("lifecycle_state"),
            "memory_id": decision.get("memory_id"),
            "gd_hits": decision.get("gd_hits"),
            "gd_pos": bool(row.get("gd_pos", False)),
            "gd_candidate_pos": bool(row.get("gd_candidate_pos", False)),
            "centroid": centroid[:3].tolist(),
            "conf": float(row.get("conf", 0.0) or 0.0),
            "dist_robot": float(row.get("dist_robot", 0.0) or 0.0),
            "bypass_s_penalty": True,
            "depth_ok": bool(row.get("depth_ok", False)),
            "depth_relaxed": bool(decision.get("depth_relaxed", False)),
            "upgraded": decision.get("upgraded"),
            "from_state": (decision.get("upgraded") or {}).get("from_state"),
            "gd_incremental": bool(
                (decision.get("upgraded") or {}).get("from_state") in (PROVISIONAL, GD)
            ),
        }
        return decision


