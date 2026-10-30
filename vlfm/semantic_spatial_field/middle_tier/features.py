"""Policy-safe feature assembly for the S7 middle-tier ranker.

This module only reads the candidate snapshot and the current GD/TSP frame
record.  It never mutates policy memory, nav goals, STOP state or the chosen
action.  The offline fitter is deliberately kept in ``offline_fit.py`` and is
not imported here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_GROUPS: Tuple[str, ...] = ("cross", "repeat", "geo", "ctx")
MIDDLE_TIER_TYPES: Tuple[str, ...] = ("gd-only", "tsp-confirmed")
TARGET_CLASSES: Tuple[str, ...] = (
    "bed",
    "chair",
    "couch",
    "toilet",
    "tv",
    "potted_plant",
)

_NEAR_DIST = 0.5
_NEAR_IAU = 0.3
_NEAR_CONTAINMENT = 0.8


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _centroid(row: Dict[str, Any]) -> Optional[np.ndarray]:
    value = row.get("centroid")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return arr


def _box(row: Dict[str, Any]) -> Optional[np.ndarray]:
    value = row.get("box8")
    if value is None:
        value = row.get("box")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (8, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _bounds(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.min(box, axis=0), np.max(box, axis=0)


def _volume(lo: np.ndarray, hi: np.ndarray) -> float:
    delta = np.maximum(0.0, np.asarray(hi) - np.asarray(lo))
    return float(delta[0] * delta[1] * delta[2])


def _iau(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> Optional[float]:
    if box_a is None or box_b is None:
        return None
    lo_a, hi_a = _bounds(box_a)
    lo_b, hi_b = _bounds(box_b)
    inter = _volume(np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b))
    union = _volume(lo_a, hi_a) + _volume(lo_b, hi_b) - inter
    return float(inter / union) if union > 0 else 0.0


def _containment(
    inner: Optional[np.ndarray], outer: Optional[np.ndarray]
) -> Optional[float]:
    if inner is None or outer is None:
        return None
    lo_i, hi_i = _bounds(inner)
    lo_o, hi_o = _bounds(outer)
    vol_inner = _volume(lo_i, hi_i)
    if vol_inner <= 0:
        return 0.0
    inter = _volume(np.maximum(lo_i, lo_o), np.minimum(hi_i, hi_o))
    return float(inter / vol_inner)


def _planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def _distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    size = min(3, aa.size, bb.size)
    return float(np.linalg.norm(aa[:size] - bb[:size]))


def _class_key(name: Any) -> str:
    text = str(name or "").strip().lower().replace(" ", "_")
    return text if text else "unknown"


def _class_one_hot(name: Any) -> Dict[str, float]:
    key = _class_key(name)
    out = {f"class_{cls}": 0.0 for cls in TARGET_CLASSES}
    out["class_other"] = 0.0
    class_key = f"class_{key}"
    if class_key in out:
        out[class_key] = 1.0
    else:
        out["class_other"] = 1.0
    return out


def _switch_one_hot(reason: Any) -> Dict[str, float]:
    key = str(reason or "")
    return {
        "switch_initial": 1.0 if key == "initial" else 0.0,
        "switch_held_near": 1.0 if key == "held_near" else 0.0,
        "switch_held_far": 1.0 if key == "held_far" else 0.0,
        "switch_switched": 1.0 if key == "switched" else 0.0,
        "switch_other": 1.0 if key not in {"initial", "held_near", "held_far", "switched"} else 0.0,
    }


def _memory_key(value: Any) -> Optional[str]:
    return None if value in (None, "") else str(value)


def _proposal_memory_id(proposal: Dict[str, Any]) -> Optional[str]:
    match_track = proposal.get("match_track") or {}
    return _memory_key(
        match_track.get("memory_id") or proposal.get("matched_memory_id")
    )


def _proposal_same_object(proposal: Dict[str, Any]) -> bool:
    match_track = proposal.get("match_track") or {}
    match_candidate = proposal.get("match_candidate") or {}
    return bool(
        match_track.get("same_object") or match_candidate.get("same_object")
    )


class MiddleTierFeatureAssembler:
    """Build the four feature groups for one selection-time snapshot.

    The assembler keeps only counter state needed for features
    (``selected_before``, merge counts, TSP near-support count and GD
    consecutive hits).  It never writes back to the policy or memory manager.
    """

    def __init__(self, class_names: Sequence[str] = TARGET_CLASSES) -> None:
        self.class_names = tuple(class_names)
        self.reset()

    def reset(self) -> None:
        self._selected_before: Dict[str, int] = {}
        self._gd_only_merge_count: Dict[str, int] = {}
        self._tsp_merge_count: Dict[str, int] = {}
        self._tsp_near_support_count: Dict[str, int] = {}
        self._tsp_near_last_step: Dict[str, int] = {}
        self._gd_consecutive: Dict[str, int] = {}
        self._gd_match_last_step: Dict[str, int] = {}
        self._last_event_step: Optional[int] = None

    def _update_events(
        self, step: int, memory_events: Optional[Sequence[Dict[str, Any]]]
    ) -> None:
        if self._last_event_step == int(step):
            return
        self._last_event_step = int(step)
        for event in memory_events or []:
            memory_id = _memory_key(event.get("memory_id"))
            if memory_id is None:
                continue
            name = str(event.get("event") or "")
            if name == "gd_only_merge":
                self._gd_only_merge_count[memory_id] = (
                    self._gd_only_merge_count.get(memory_id, 0) + 1
                )
            elif name == "merge":
                self._tsp_merge_count[memory_id] = (
                    self._tsp_merge_count.get(memory_id, 0) + 1
                )

    @staticmethod
    def _frame_rows(
        gd_frame: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if not gd_frame:
            return [], [], {}, {}
        tsp_section = gd_frame.get("tsp3d") or {}
        candidates = list(tsp_section.get("candidates") or [])
        tracks = list(gd_frame.get("tracks") or [])
        proposals = list(gd_frame.get("proposals") or [])

        tsp_rows: List[Dict[str, Any]] = []
        for row in candidates:
            if str(row.get("lifecycle_state") or "") == "gd":
                continue
            tsp_rows.append(row)
        for row in tracks:
            if str(row.get("lifecycle_state") or "") == "gd":
                continue
            if str(row.get("src") or "") == "gd":
                continue
            tsp_rows.append(row)

        track_by_memory: Dict[str, Dict[str, Any]] = {}
        for row in tracks:
            memory_id = _memory_key(row.get("memory_id"))
            if memory_id is not None:
                track_by_memory[memory_id] = row
        candidate_by_memory: Dict[str, Dict[str, Any]] = {}
        for row in candidates:
            memory_id = _memory_key(row.get("memory_id"))
            if memory_id is not None:
                candidate_by_memory[memory_id] = row
        return tsp_rows, proposals, track_by_memory, candidate_by_memory

    def _gd_only_cross(
        self,
        *,
        step: int,
        memory_id: Optional[str],
        centroid: Optional[np.ndarray],
        box: Optional[np.ndarray],
        tsp_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        min_dist_xy: Optional[float] = None
        min_dist_3d: Optional[float] = None
        max_iau: Optional[float] = None
        max_cont: Optional[float] = None
        near = False
        if centroid is not None:
            for row in tsp_rows:
                other_centroid = _centroid(row)
                if other_centroid is None:
                    continue
                dist_xy = _planar_distance(centroid, other_centroid)
                dist_3d = _distance_3d(centroid, other_centroid)
                min_dist_xy = dist_xy if min_dist_xy is None else min(min_dist_xy, dist_xy)
                min_dist_3d = dist_3d if min_dist_3d is None else min(min_dist_3d, dist_3d)
                iau = _iau(box, _box(row))
                cont = _containment(box, _box(row))
                cont_rev = _containment(_box(row), box)
                if iau is not None:
                    max_iau = iau if max_iau is None else max(max_iau, iau)
                for value in (cont, cont_rev):
                    if value is not None:
                        max_cont = value if max_cont is None else max(max_cont, value)
                if (
                    dist_xy <= _NEAR_DIST
                    or (iau is not None and iau >= _NEAR_IAU)
                    or (cont is not None and cont >= _NEAR_CONTAINMENT)
                    or (cont_rev is not None and cont_rev >= _NEAR_CONTAINMENT)
                ):
                    near = True
        if memory_id is not None and near and self._tsp_near_last_step.get(memory_id) != int(step):
            self._tsp_near_last_step[memory_id] = int(step)
            self._tsp_near_support_count[memory_id] = (
                self._tsp_near_support_count.get(memory_id, 0) + 1
            )
        return {
            "tsp_min_dist_xy": min_dist_xy,
            "tsp_min_dist_3d": min_dist_3d,
            "tsp_max_iau": max_iau,
            "tsp_max_containment": max_cont,
            "tsp_near_support_now": 1.0 if near else 0.0,
            "tsp_near_support_count": float(
                0 if memory_id is None else self._tsp_near_support_count.get(memory_id, 0)
            ),
        }

    def _tsp_cross(
        self,
        *,
        step: int,
        memory_id: Optional[str],
        proposals: Sequence[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        matched = [
            p
            for p in proposals
            if memory_id is not None and _proposal_memory_id(p) == memory_id
        ]
        same = [p for p in matched if _proposal_same_object(p)]
        conf_values = [_finite(p.get("conf")) for p in same]
        conf_values = [v for v in conf_values if v is not None]
        conf_max_now = max(conf_values) if conf_values else None
        conf_last_now = conf_values[-1] if conf_values else None

        match_tracks = [p.get("match_track") or {} for p in same]
        iaus = [_finite(m.get("iau")) for m in match_tracks]
        iaus = [v for v in iaus if v is not None]
        conts = []
        for match in match_tracks:
            for key in ("containment", "containment_reverse"):
                value = _finite(match.get(key))
                if value is not None:
                    conts.append(value)
        dists = [_finite(m.get("dist_xy")) for m in match_tracks]
        dists = [v for v in dists if v is not None]

        matched_now = bool(same)
        if memory_id is not None:
            last_step = self._gd_match_last_step.get(memory_id)
            if matched_now:
                if last_step is not None and int(last_step) == int(step) - 1:
                    self._gd_consecutive[memory_id] = (
                        self._gd_consecutive.get(memory_id, 0) + 1
                    )
                else:
                    self._gd_consecutive[memory_id] = 1
                self._gd_match_last_step[memory_id] = int(step)
            else:
                self._gd_consecutive[memory_id] = 0

        return {
            "gd_match_count_now": float(len(same)),
            "gd_match_any_now": 1.0 if matched_now else 0.0,
            "gd_conf_max_now": conf_max_now,
            "gd_conf_last_now": conf_last_now,
            "gd_best_iau": max(iaus) if iaus else None,
            "gd_best_containment": max(conts) if conts else None,
            "gd_min_dist_xy": min(dists) if dists else None,
            "gd_hits": _finite(state.get("gd_hits")),
            "gd_conf_max": _finite(state.get("gd_conf_max")),
            "gd_conf_last": _finite(state.get("gd_conf_last")),
            "gd_consecutive_hits": float(
                0 if memory_id is None else self._gd_consecutive.get(memory_id, 0)
            ),
        }

    def _repeat_features(
        self,
        *,
        memory_id: Optional[str],
        candidate: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "num_obs": _finite(candidate.get("num_obs", state.get("num_obs"))),
            "stale_steps": _finite(candidate.get("stale_steps")),
            "score": _finite(candidate.get("score")),
            "gd_hits": _finite(state.get("gd_hits")),
            "gd_conf_max": _finite(state.get("gd_conf_max")),
            "gd_conf_last": _finite(state.get("gd_conf_last")),
            "gd_only_merge_count": float(
                0 if memory_id is None else self._gd_only_merge_count.get(memory_id, 0)
            ),
            "tsp_merge_count": float(
                0 if memory_id is None else self._tsp_merge_count.get(memory_id, 0)
            ),
            "revive_count": _finite(state.get("revive_count")),
        }

    def _geo_features(
        self,
        *,
        robot_xy: np.ndarray,
        centroid: Optional[np.ndarray],
        box: Optional[np.ndarray],
        candidate: Dict[str, Any],
        state: Dict[str, Any],
        depth_ok: Optional[float],
        occ_passed: Optional[float],
    ) -> Dict[str, Any]:
        if centroid is None:
            dist_xy = None
            dist_3d = None
            bucket = None
        else:
            dist_xy = _planar_distance(robot_xy, centroid)
            dist_3d = _distance_3d(
                np.asarray(robot_xy, dtype=np.float64).reshape(-1),
                centroid,
            )
            bucket = float(int(round(dist_xy / 0.5)))
        features = {
            "dist_xy": dist_xy,
            "dist_3d": dist_3d,
            "distance_bucket": bucket,
            "has_box8": 1.0 if box is not None else 0.0,
            "depth_ok": depth_ok,
            "occ_passed": occ_passed,
            "revive_count": _finite(state.get("revive_count")),
        }
        features.update(_class_one_hot(candidate.get("target_class")))
        return features

    def observe(
        self,
        *,
        step: int,
        candidates: Sequence[Dict[str, Any]],
        robot_xy: np.ndarray,
        gd_frame: Optional[Dict[str, Any]] = None,
        memory_events: Optional[Sequence[Dict[str, Any]]] = None,
        switch_reason: str = "",
    ) -> List[Dict[str, Any]]:
        """Return one shadow row per middle-tier candidate."""
        self._update_events(int(step), memory_events)
        robot = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
        tsp_rows, proposals, track_by_memory, candidate_by_memory = self._frame_rows(gd_frame)
        rows: List[Dict[str, Any]] = []
        for candidate in candidates or []:
            type_name = str(candidate.get("type") or "")
            if type_name not in MIDDLE_TIER_TYPES:
                continue
            memory_id = _memory_key(candidate.get("memory_id"))
            centroid = _centroid(candidate)
            state = dict(candidate.get("state") or {})
            box = _box(candidate)
            if box is None and memory_id is not None and memory_id in track_by_memory:
                box = _box(track_by_memory[memory_id])
            if box is None and memory_id is not None and memory_id in candidate_by_memory:
                box = _box(candidate_by_memory[memory_id])

            matched_props = [
                p for p in proposals
                if memory_id is not None and _proposal_memory_id(p) == memory_id
            ]
            depth_ok = None
            for proposal in matched_props:
                depth_ok = _finite(proposal.get("depth_ok"))
                if depth_ok is not None:
                    break
            occ_passed = None
            if memory_id is not None and memory_id in candidate_by_memory:
                occ_passed = _finite(candidate_by_memory[memory_id].get("occ_passed"))
            if occ_passed is None and memory_id is not None and memory_id in track_by_memory:
                occ_passed = _finite(track_by_memory[memory_id].get("occ_passed"))

            if type_name == "gd-only":
                cross = self._gd_only_cross(
                    step=int(step),
                    memory_id=memory_id,
                    centroid=centroid,
                    box=box,
                    tsp_rows=tsp_rows,
                )
            else:
                cross = self._tsp_cross(
                    step=int(step),
                    memory_id=memory_id,
                    proposals=proposals,
                    state=state,
                )

            ctx = {
                "selected": 1.0 if bool(candidate.get("selected")) else 0.0,
                "selected_before": float(
                    0 if memory_id is None else self._selected_before.get(memory_id, 0)
                ),
                "n_selectable": _finite(candidate.get("n_selectable")),
                "distance_bucket": _finite(candidate.get("distance_bucket")),
                "selection_tier": _finite(candidate.get("selection_tier")),
            }
            ctx.update(_switch_one_hot(switch_reason))

            rows.append(
                {
                    "step": int(step),
                    "memory_id": memory_id,
                    "candidate_id": candidate.get("candidate_id"),
                    "type": type_name,
                    "target_class": str(candidate.get("target_class") or ""),
                    "centroid": (
                        np.asarray(centroid, dtype=np.float64).reshape(-1)[:3].tolist()
                        if centroid is not None
                        else None
                    ),
                    "selected": bool(candidate.get("selected")),
                    "n_selectable": int(candidate.get("n_selectable") or 0),
                    "distance": _finite(candidate.get("distance")),
                    "distance_bucket": int(candidate.get("distance_bucket") or 0),
                    "selection_tier": int(candidate.get("selection_tier") or 0),
                    "static_rank": candidate.get("static_rank"),
                    "state": str(
                        state.get("lifecycle_state")
                        or candidate.get("lifecycle_state")
                        or ""
                    ),
                    "origin": candidate.get("origin"),
                    "features": {
                        "cross": cross,
                        "repeat": self._repeat_features(
                            memory_id=memory_id,
                            candidate=candidate,
                            state=state,
                        ),
                        "geo": self._geo_features(
                            robot_xy=robot,
                            centroid=centroid,
                            box=box,
                            candidate=candidate,
                            state=state,
                            depth_ok=depth_ok,
                            occ_passed=occ_passed,
                        ),
                        "ctx": ctx,
                    },
                }
            )

        for row in rows:
            if row["selected"] and row["memory_id"] is not None:
                key = str(row["memory_id"])
                self._selected_before[key] = self._selected_before.get(key, 0) + 1
        return rows
