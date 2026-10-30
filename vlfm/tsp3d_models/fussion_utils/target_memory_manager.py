from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vlfm.utils.geometry_utils import within_fov_cone


CONFIRMED = "confirmed"
PROVISIONAL = "provisional"
STABLE = "stable"
# S6: independent GD-only pool.  TSP3D entries stay untagged; only a GD
# detection that has no TSP3D backing gets this state and provenance.
GD = "gd"
GD_ONLY = "gd-only"
GD_TSP = "gd-tsp"
# Provenance labels for entries whose only source is TSP3D.  These are
# intentionally distinct from `None` so logs and selection can distinguish
# pure-TSP entries from legacy/unlabelled states.
TSP_CONFIRMED = "tsp-confirmed"
TSP_PROVISIONAL = "tsp-provisional"


@dataclass(frozen=True)
class ProvisionalGateConfig:
    """Thresholds for creating a new target-memory entry."""

    enabled: bool = False
    confirmed_sigma: float = 0.60
    provisional_sigma: float = 0.50
    provisional_min_dist: float = 1.5
    provisional_max_dist: float = 4.0
    # Retained for config/log compatibility.  S no longer gates new entries.
    s_min: float = 0.07


class ProvisionalGate:
    """Classify only unmatched candidates; merging is handled before this gate."""

    def __init__(self, config: Optional[ProvisionalGateConfig] = None) -> None:
        self.config = config or ProvisionalGateConfig()

    def classify_new(self, candidate: Dict[str, Any]) -> Optional[str]:
        if not self.config.enabled:
            return CONFIRMED

        sigma_c = float(candidate.get("sigma_c", candidate.get("conf_amp", 0.0)) or 0.0)
        geom_status = str(candidate.get("geom_status") or "")

        # geom_unknown is deliberately capped at provisional, even above the
        # confirmed score line.  It still has to satisfy the provisional gate.
        if sigma_c >= self.config.confirmed_sigma and geom_status != "geom_unknown":
            return CONFIRMED

        distance = candidate.get("distance")
        if distance is None:
            distance = candidate.get("dist")
        in_band = (
            distance is not None
            and self.config.provisional_min_dist
            <= float(distance)
            <= self.config.provisional_max_dist
        )
        if sigma_c >= self.config.provisional_sigma and in_band:
            return PROVISIONAL
        return None

    def describe(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        state = self.classify_new(candidate)
        return {
            "provisional_gate_enabled": bool(self.config.enabled),
            "provisional_gate_state": state,
            "provisional_gate_passed": state is not None,
            "provisional_sigma": float(self.config.provisional_sigma),
            "provisional_min_dist": float(self.config.provisional_min_dist),
            "provisional_max_dist": float(self.config.provisional_max_dist),
            "provisional_s_min": float(self.config.s_min),
        }


@dataclass
class MemoryManagerConfig:
    """
    Target-memory lifecycle hyper-parameters.
    """
    # cross-view EMA merge (memory write)
    merge_dist_thresh: float = 0.5        # merge two observations of the same class within this distance (m)
    ema_weight_old: float = 0.8           # EMA smoothing: old * 0.8 + new * 0.2

    # S5-S2 lifecycle.  Unmatched candidates may create a memory only when
    # the independent provisional gate supplies an initial state.
    confirmed_to_stable_misses: int = 3
    provisional_to_stable_misses: int = 2
    # S5 runtime upgrade: a provisional entry that is re-observed up close
    # becomes stop-eligible.  Evidence, not confidence, drives the promotion.
    # S6gd1: the only ordinary upgrade criterion is evidence count.  The
    # merge-time and frame-level paths share this threshold; distance is not
    # an upgrade gate.
    upgrade_min_obs: int = 4

    # S6: merge evidence gate + `stable` deletion pool.
    enable_stable_pool: bool = False      # master switch for the S6 lifecycle
    merge_into_stable: bool = False       # never absorb evidence into a `stable` entry
    # S6 GD pool: gd-only has its own state and is never deleted; gd-tsp is a
    # confirmed entry.  No TTL or TTL-shadow mechanism.
    enable_gd_pool: bool = False
    merge_new_info_dist: float = 0.2      # pose baseline that counts as a new observation (m)
    merge_new_info_yaw: float = 10.0      # yaw delta (deg) that counts as a new observation
    merge_new_info_steps: int = 3         # step gap that counts as a new observation
    stable_stale_purge: int = 60          # frames since the last match -> delete
    stable_miss_delete: int = 3           # consecutive near-field misses inside `stable`
    stable_total_miss: int = 6            # cumulative near-field misses (a revive does not reset it)
    stable_revive_radius: float = 0.8     # revive radius (m)
    stable_revive_window: int = 20        # revive window (frames since the last match)
    stable_revive_min_sigma: float = 0.50 # revive candidate sigma_c floor


    # Near-field miss accounting radius (m).  The old fb/anti-hallucination
    # fallback is removed; this radius now only serves the stable-pool budgets.
    near_field_radius: float = 2.5


class TargetMemoryManager:
    """
    Target lifecycle manager operating IN PLACE on the bound policy memory dicts.

    The policy binds its four memory dicts once at construction;
    because the policy clears those dicts with ``.clear()``,
    the references stay valid across episodes. 
    """

    def __init__(self, config: Optional[MemoryManagerConfig] = None) -> None:
        self.config = config or MemoryManagerConfig()
        self._mem: Dict[str, List[np.ndarray]] = {}
        self._surf: Dict[str, List[np.ndarray]] = {}
        self._ver: Dict[str, List[Tuple[Any, ...]]] = {}
        self._lifecycle: Dict[str, List[Dict[str, Any]]] = {}
        self._next_memory_id = 0
        self._frame = 0

    def reset_ids(self) -> None:
        """Reset the episode-local stable memory-id sequence."""
        self._next_memory_id = 0
        self._frame = 0

    @property
    def frame(self) -> int:
        """Current internal frame counter (incremented once per manager step)."""
        return int(self._frame)

    def _ensure_state_fields(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Backfill S6/S7 bookkeeping fields on states built by older code paths."""
        # Old pure-TSP states used ``origin=None``.  Mark them explicitly when
        # they are first touched; GD-backed origins are never overwritten.
        _life = str(state.get("lifecycle_state") or CONFIRMED)
        if (
            state.get("origin") is None
            and not bool(state.get("gd_only", False))
            and _life in (CONFIRMED, PROVISIONAL)
        ):
            state["origin"] = (
                TSP_CONFIRMED if _life == CONFIRMED else TSP_PROVISIONAL
            )
        # A gd-only entry promoted by an admitted TSP3D merge is treated as
        # gd-tsp from that point on.  Keep the promotion flag for provenance,
        # but normalize legacy states that still carry origin=gd-only.
        if bool(state.get("promoted_from_gd_only", False)):
            state["origin"] = GD_TSP
        state.setdefault("total_miss", 0)
        state.setdefault("revive_count", 0)
        state.setdefault("last_obs_step", int(self._frame))
        state.setdefault("last_obs_xy", None)
        state.setdefault("last_obs_yaw", None)
        state.setdefault("obs_xy", [])
        state.setdefault("obs_yaw", [])
        if not state["obs_xy"] and state.get("last_obs_xy") is not None:
            state["obs_xy"] = [np.asarray(state["last_obs_xy"], dtype=np.float64).reshape(-1)[:2].tolist()]
            state["obs_yaw"] = [float(state.get("last_obs_yaw") or 0.0)]
        return state

    def _append_observation(
        self,
        state: Dict[str, Any],
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> None:
        """Append one accepted (new-information) observation pose."""
        self._ensure_state_fields(state)
        state["obs_xy"].append(
            np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2].tolist()
        )
        state["obs_yaw"].append(float(robot_yaw))

    def _touch_observation(
        self,
        state: Dict[str, Any],
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> None:
        """Record the observing pose of the current matched frame."""
        state["last_obs_xy"] = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2].copy()
        state["last_obs_yaw"] = float(robot_yaw)
        state["last_obs_step"] = int(self._frame)

    def _is_new_information(self, state: Dict[str, Any], robot_xy: np.ndarray, robot_yaw: float) -> bool:
        """True when this observation comes from a genuinely different viewpoint."""
        last_xy = state.get("last_obs_xy")
        if last_xy is None:
            return True
        baseline = float(np.linalg.norm(np.asarray(last_xy, dtype=np.float64).reshape(-1)[:2] - robot_xy))
        if baseline >= float(self.config.merge_new_info_dist):
            return True
        last_yaw = state.get("last_obs_yaw")
        if last_yaw is not None:
            dyaw = abs(float((float(robot_yaw) - float(last_yaw) + np.pi) % (2 * np.pi) - np.pi))
            if np.rad2deg(dyaw) >= float(self.config.merge_new_info_yaw):
                return True
        gap = int(self._frame) - int(state.get("last_obs_step", 0))
        return gap >= int(self.config.merge_new_info_steps)

    def _is_new_gd_information(
        self,
        state: Dict[str, Any],
        robot_xy: np.ndarray,
        robot_yaw: float,
    ) -> bool:
        """New-information gate for GD evidence.

        GD evidence has its own pose history; it must not borrow the TSP3D
        observation counters, otherwise a GD hit could promote a TSP3D track.
        """
        last_xy = state.get("gd_last_xy")
        if last_xy is None:
            return True
        baseline = float(
            np.linalg.norm(
                np.asarray(last_xy, dtype=np.float64).reshape(-1)[:2] - robot_xy
            )
        )
        if baseline >= float(self.config.merge_new_info_dist):
            return True
        last_yaw = state.get("gd_last_yaw")
        if last_yaw is not None:
            dyaw = abs(
                float(
                    (float(robot_yaw) - float(last_yaw) + np.pi) % (2 * np.pi)
                    - np.pi
                )
            )
            if np.rad2deg(dyaw) >= float(self.config.merge_new_info_yaw):
                return True
        gap = int(self._frame) - int(state.get("gd_last_evidence_step", self._frame))
        return gap >= int(self.config.merge_new_info_steps)

    @staticmethod
    def _gd_fields(state: Dict[str, Any]) -> Dict[str, Any]:
        """GD evidence fields; `origin` / `gd_only` appear only for GD-only entries."""
        out: Dict[str, Any] = {
            "gd_hits": int(state.get("gd_hits", 0)),
            "gd_conf_max": float(state.get("gd_conf_max", 0.0)),
            "gd_conf_last": float(state.get("gd_conf_last", 0.0)),
            "gd_last_step": state.get("gd_last_step"),
            "gd_pos": bool(state.get("gd_pos", False)),
            "gd_candidate_pos": bool(state.get("gd_candidate_pos", False)),
            "gd_last_candidate_id": state.get("gd_last_candidate_id"),
        }
        if state.get("origin") is not None:
            out["origin"] = str(state.get("origin"))
        if bool(state.get("gd_only", False)):
            out["gd_only"] = True
        return out

    def _alloc_memory_id(self) -> str:
        memory_id = f"m{self._next_memory_id:06d}"
        self._next_memory_id += 1
        return memory_id

    @staticmethod
    def _score_fields(state: Dict[str, Any]) -> Dict[str, float]:
        return {
            "sigma_c_max": float(state.get("sigma_c_max", 0.0)),
            "sigma_c_latest": float(state.get("sigma_c_latest", 0.0)),
            "sigma_c_ema_03": float(state.get("sigma_c_ema_03", 0.0)),
            "sigma_c_ema_05": float(state.get("sigma_c_ema_05", 0.0)),
            "sigma_c_ema_08": float(state.get("sigma_c_ema_08", 0.0)),
        }

    @staticmethod
    def _update_scores(state: Dict[str, Any], sigma_c: float, *, new: bool) -> None:
        value = float(sigma_c)
        if new:
            state.update(
                sigma_c_max=value,
                sigma_c_latest=value,
                sigma_c_ema_03=value,
                sigma_c_ema_05=value,
                sigma_c_ema_08=value,
            )
            return
        previous = float(state.get("sigma_c_latest", value))
        state["sigma_c_max"] = max(float(state.get("sigma_c_max", previous)), value)
        state["sigma_c_latest"] = value
        for alpha, key in ((0.3, "sigma_c_ema_03"), (0.5, "sigma_c_ema_05"), (0.8, "sigma_c_ema_08")):
            old = float(state.get(key, previous))
            state[key] = alpha * value + (1.0 - alpha) * old

    def bind(
        self,
        target_3d_memory: Dict[str, List[np.ndarray]],
        target_surface_memory: Dict[str, List[np.ndarray]],
        target_verify_state: Dict[str, List[Tuple[Any, ...]]],
        target_lifecycle_state: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Bind the policy's memory dicts."""
        self._mem = target_3d_memory
        self._surf = target_surface_memory
        self._ver = target_verify_state
        self._lifecycle = target_lifecycle_state

    # memory write / EMA merge
    def accumulate(
        self,
        target_class: str,
        centroid: np.ndarray,
        confidence: float = 1.0,
        near_surface: Optional[np.ndarray] = None,
        robot_xyz: Optional[np.ndarray] = None,
        robot_yaw: float = 0.0,
        candidate_id: Optional[str] = None,
        new_lifecycle_state: Optional[str] = None,
        sigma_c: Optional[float] = None,
        origin: Optional[str] = None,
        force_confirmed: bool = False,
        block_stable: bool = False,
        gd_evidence_only: bool = False,
        expected_memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        In-place write / EMA-merge of one admitted detection.
        Merges within merge_dist_thresh of a same-class record 
        (EMA centroid, num_obs+1, max confidence kept, near-miss reset); 
        otherwise registers a new candidate.

        ``expected_memory_id`` makes the write merge-only: if the nearest
        same-class entry is not that memory id (or no merge is possible), the
        call returns blocked instead of creating a new entry.  GD-tsp uses
        this to guarantee that it only enriches an existing TSP track.

        Returns a small record of the write for caller-side logging / accounting.
        """
        r_xyz = np.zeros(3) if robot_xyz is None else np.asarray(robot_xyz, dtype=np.float64)
        r_xy = r_xyz[:2]
        c_np = np.asarray(centroid, dtype=np.float64)
        s_np = np.asarray(near_surface, dtype=np.float64) if near_surface is not None else c_np

        if c_np.shape[0] >= 3:
            detect_dist = float(np.linalg.norm(c_np - r_xyz))
        else:
            detect_dist = float(np.linalg.norm(c_np - r_xy))

        score = float(confidence if sigma_c is None else sigma_c)
        recs = self._mem.get(target_class)
        if not recs:
            if expected_memory_id is not None:
                return {
                    "written": False,
                    "merged": False,
                    "blocked": "expected_memory_missing",
                    "candidate_id": candidate_id,
                    "memory_id": expected_memory_id,
                }
            if new_lifecycle_state is None:
                return {
                    "written": False, "merged": False, "new_rejected": True,
                    "candidate_id": candidate_id,
                }
            self._mem[target_class] = [c_np]
            self._surf[target_class] = [s_np]
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence))]
            state = {"near_miss": 0, "memory_id": self._alloc_memory_id(),
                     "last_candidate_id": candidate_id,
                     "lifecycle_state": new_lifecycle_state or "confirmed",
                     "origin": self._tsp_origin_for_state(origin, new_lifecycle_state),
                     "origin_provisional": new_lifecycle_state == PROVISIONAL,
                     "total_miss": 0, "revive_count": 0}
            self._touch_observation(state, r_xy, robot_yaw)
            self._append_observation(state, r_xy, robot_yaw)
            self._update_scores(state, score, new=True)
            self._lifecycle[target_class] = [state]
            return {"written": True, "merged": False, "num_obs": 1,
                    "memory_id": state["memory_id"], "candidate_id": candidate_id,
                    "lifecycle_state": state["lifecycle_state"],
                    "origin": state.get("origin"),
                    **self._score_fields(state)}

        # Keep the side lists in sync
        if target_class not in self._surf:
            self._surf[target_class] = list(recs)
        if target_class not in self._ver:
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence)) for _ in recs]
        if target_class not in self._lifecycle:
            self._lifecycle[target_class] = [
                {"near_miss": 0, "memory_id": self._alloc_memory_id(),
                 "last_candidate_id": None, "lifecycle_state": "confirmed"}
                for _ in recs
            ]

        dists = np.linalg.norm(np.asarray(recs, dtype=np.float64) - c_np, axis=1)
        closest_idx = int(np.argmin(dists))
        prev_state = self._lifecycle[target_class][closest_idx]
        self._ensure_state_fields(prev_state)
        closest_lifecycle = str(prev_state.get("lifecycle_state", CONFIRMED))
        within_merge = bool(dists[closest_idx] < self.config.merge_dist_thresh)
        if expected_memory_id is not None:
            actual_memory_id = str(prev_state.get("memory_id") or "")
            if not within_merge or actual_memory_id != str(expected_memory_id):
                return {
                    "written": False,
                    "merged": False,
                    "blocked": "expected_memory_mismatch",
                    "candidate_id": candidate_id,
                    "memory_id": actual_memory_id or None,
                    "lifecycle_state": closest_lifecycle,
                }
        if block_stable and closest_lifecycle == STABLE:
            return {
                "written": False, "merged": False, "blocked": "stable",
                "candidate_id": candidate_id,
                "lifecycle_state": STABLE,
            }
        # S6: `stable` is the deletion pool -- it never absorbs evidence.
        # The revive radius is intentionally independent of merge_dist_thresh:
        # a stable entry can be revived at 0.8 m even though normal merges stay
        # at 0.5 m.  If revive fails (sigma/age), this observation is treated as
        # a new entry rather than silently merging into stable.
        if (
            self.config.enable_stable_pool
            and not self.config.merge_into_stable
            and closest_lifecycle == STABLE
        ):
            if float(dists[closest_idx]) <= float(self.config.stable_revive_radius):
                revived = self._revive_stable_entry(
                    target_class, closest_idx, prev_state, c_np, score,
                    r_xy, robot_yaw, confidence, candidate_id, detect_dist,
                )
                if revived is not None:
                    return revived
            within_merge = False
        # A gd-only entry that is later matched by an admitted TSP3D candidate
        # is not a GD-only fallback anymore: the detector has supplied the
        # geometry/identity match.  Promote it to confirmed instead of leaving
        # it permanently in the `gd` pool.
        promote_gd_only = bool(
            within_merge
            and closest_lifecycle == GD
            and not gd_evidence_only
            and new_lifecycle_state in (CONFIRMED, PROVISIONAL)
        )
        if within_merge and gd_evidence_only:
            # gd-tsp is an external confirmation/evidence path: promote the
            # lifecycle but do NOT count it as a TSP3D observation, move the
            # centroid, or alter sigma_c/num_obs. Those fields remain owned by
            # the TSP3D detector.
            v_old = self._ver[target_class][closest_idx]
            num_obs_now = int(v_old[0]) if len(v_old) >= 1 else 1
            prev_state["near_miss"] = 0
            prev_state["last_candidate_id"] = candidate_id
            self._touch_observation(prev_state, r_xy, robot_yaw)
            upgraded = None
            if force_confirmed:
                from_state = str(prev_state.get("lifecycle_state", CONFIRMED))
                prev_state["lifecycle_state"] = CONFIRMED
                if from_state != CONFIRMED:
                    prev_state["gd_upgraded"] = True
                    prev_state["gd_upgraded_from"] = from_state
                    prev_state["gd_upgraded_step"] = int(self._frame)
                    upgraded = {
                        "from_state": from_state,
                        "to_state": CONFIRMED,
                        "reason": "gd_tsp",
                        "num_obs": num_obs_now,
                        "detect_dist": float(detect_dist),
                    }
            if origin is not None:
                prev_state["origin"] = origin
            return {
                "written": True,
                "merged": True,
                "new_info": False,
                "gd_evidence_only": True,
                "revived": False,
                "num_obs": num_obs_now,
                "memory_id": prev_state.get("memory_id"),
                "candidate_id": candidate_id,
                "lifecycle_state": prev_state.get("lifecycle_state", CONFIRMED),
                "origin": prev_state.get("origin"),
                "upgraded": upgraded,
                "total_miss": int(prev_state.get("total_miss", 0)),
                "revive_count": int(prev_state.get("revive_count", 0)),
                "detect_dist": float(detect_dist),
                **self._score_fields(prev_state),
            }
        if within_merge:
            # Cross-frame consensus = strong evidence; hallucinated boxes drift and cannot merge.
            # EMA centroid, keep the max c', reset near-miss.
            old = np.asarray(recs[closest_idx], dtype=np.float64)
            recs[closest_idx] = self.config.ema_weight_old * old + (1.0 - self.config.ema_weight_old) * c_np
            v_old = self._ver[target_class][closest_idx]
            old_conf = float(v_old[3]) if len(v_old) > 3 else float(confidence)
            # S6: only a genuinely different viewpoint counts as a new observation;
            # same-pose consecutive frames keep the centroid fresh but add no evidence.
            new_info = True
            if self.config.enable_stable_pool:
                new_info = self._is_new_information(prev_state, r_xy, robot_yaw)
            if new_info:
                num_obs_now = int(v_old[0]) + 1
                self._update_scores(prev_state, score, new=False)
                self._append_observation(prev_state, r_xy, robot_yaw)
            else:
                num_obs_now = int(v_old[0])
            self._ver[target_class][closest_idx] = (
                num_obs_now, r_xy.copy(), float(robot_yaw),
                max(old_conf, float(confidence)),
            )
            prev_state["near_miss"] = 0
            prev_state["last_candidate_id"] = candidate_id
            self._touch_observation(prev_state, r_xy, robot_yaw)
            upgraded = None
            if (
                str(prev_state.get("lifecycle_state")) == PROVISIONAL
                and num_obs_now >= int(self.config.upgrade_min_obs)
            ):
                prev_state["lifecycle_state"] = CONFIRMED
                if prev_state.get("origin") not in (GD_ONLY, GD_TSP):
                    prev_state["origin"] = TSP_CONFIRMED
                upgraded = {
                    "from_state": PROVISIONAL,
                    "to_state": CONFIRMED,
                    # One ordinary upgrade reason; trigger distinguishes the
                    # merge-time path from the per-frame path.
                    "reason": "base_upgrade",
                    "trigger": "merge",
                    "num_obs": num_obs_now,
                    "detect_dist": float(detect_dist),
                }
            if force_confirmed:
                from_state = str(prev_state.get("lifecycle_state", CONFIRMED))
                prev_state["lifecycle_state"] = CONFIRMED
                if from_state != CONFIRMED and upgraded is None:
                    prev_state["gd_upgraded"] = True
                    prev_state["gd_upgraded_from"] = from_state
                    prev_state["gd_upgraded_step"] = int(self._frame)
                    upgraded = {
                        "from_state": from_state,
                        "to_state": CONFIRMED,
                        "reason": "gd_tsp",
                        "num_obs": num_obs_now,
                        "detect_dist": float(detect_dist),
                    }
            if origin is not None:
                prev_state["origin"] = origin
            if promote_gd_only:
                prev_state["lifecycle_state"] = CONFIRMED
                prev_state["gd_only"] = False
                prev_state["promoted_from_gd_only"] = True
                # Promotion is semantically gd-tsp.  Do not keep a separate
                # gd-only-tsp category downstream; the promotion flag remains
                # only as provenance in the logs.
                prev_state["origin"] = GD_TSP
                prev_state["gd_promoted_step"] = int(self._frame)
                # The gd-only entry had no sigma_c; seed it from the TSP3D
                # observation that matched it, otherwise selection would see
                # a confirmed entry with score 0.
                self._update_scores(prev_state, score, new=False)
                upgraded = {
                    "from_state": GD,
                    "to_state": CONFIRMED,
                    "reason": "gd_only_tsp_merge",
                    "num_obs": num_obs_now,
                    "detect_dist": float(detect_dist),
                }
            return {"written": True, "merged": True, "new_info": bool(new_info), "revived": False,
                    "num_obs": num_obs_now,
                    "memory_id": prev_state.get("memory_id"),
                    "candidate_id": candidate_id,
                    "lifecycle_state": prev_state.get("lifecycle_state", "confirmed"),
                    "origin": prev_state.get("origin"),
                    "upgraded": upgraded,
                    "promoted_from_gd_only": bool(promote_gd_only),
                    "total_miss": int(prev_state.get("total_miss", 0)),
                    "revive_count": int(prev_state.get("revive_count", 0)),
                    "detect_dist": float(detect_dist),
                    **self._score_fields(prev_state)}
        else:
            if new_lifecycle_state is None:
                return {
                    "written": False, "merged": False, "new_rejected": True,
                    "candidate_id": candidate_id,
                }
            recs.append(c_np)
            self._surf[target_class].append(s_np)
            self._ver[target_class].append((1, r_xy.copy(), float(robot_yaw), float(confidence)))
            state = {"near_miss": 0, "memory_id": self._alloc_memory_id(),
                     "last_candidate_id": candidate_id,
                     "lifecycle_state": new_lifecycle_state or "confirmed",
                     "origin": self._tsp_origin_for_state(origin, new_lifecycle_state),
                     "origin_provisional": new_lifecycle_state == PROVISIONAL,
                     "total_miss": 0, "revive_count": 0}
            self._touch_observation(state, r_xy, robot_yaw)
            self._append_observation(state, r_xy, robot_yaw)
            self._update_scores(state, score, new=True)
            self._lifecycle[target_class].append(state)
            return {"written": True, "merged": False, "num_obs": 1,
                    "memory_id": state["memory_id"], "candidate_id": candidate_id,
                    "lifecycle_state": state["lifecycle_state"],
                    "origin": state.get("origin"),
                    **self._score_fields(state)}

    def accumulate_gd_only(
        self,
        target_class: str,
        centroid: np.ndarray,
        gd_conf: float,
        near_surface: Optional[np.ndarray] = None,
        robot_xyz: Optional[np.ndarray] = None,
        robot_yaw: float = 0.0,
        candidate_id: Optional[str] = None,
        new_info: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Write one GD-only observation into the independent `gd` pool.

        This is the `gd-only` branch of the S6 GD mechanism: it creates/merges
        the independent `gd` state and records provenance/evidence.  It is
        called by ``GdMechanism`` when a GD proposal has no TSP3D association.

        Invariants:
          * unmatched GD observation -> new state `gd`, `origin="gd-only"`;
          * GD matching another `gd` entry -> merge GD evidence;
          * GD matching a TSP3D `confirmed` / `provisional` entry -> only write
            `gd_*` evidence fields; never change state, centroid, num_obs or
            sigma_c, so TSP3D lifecycle is untouched;
          * `stable` entries are not merged into (same rule as TSP3D);
          * no sigma_c is fabricated; GD confidence is a separate field.
        """
        if not self.config.enable_gd_pool:
            return {
                "written": False,
                "merged": False,
                "gd_pool_disabled": True,
                "candidate_id": candidate_id,
            }

        r_xyz = (
            np.zeros(3) if robot_xyz is None
            else np.asarray(robot_xyz, dtype=np.float64).reshape(-1)
        )
        r_xy = r_xyz[:2]
        c_np = np.asarray(centroid, dtype=np.float64).reshape(-1)
        s_np = (
            c_np if near_surface is None
            else np.asarray(near_surface, dtype=np.float64).reshape(-1)
        )
        if c_np.shape[0] >= 3:
            detect_dist = float(np.linalg.norm(c_np - r_xyz))
        else:
            detect_dist = float(np.linalg.norm(c_np - r_xy))

        recs = self._mem.get(target_class)
        if not recs:
            return self._new_gd_entry(
                target_class, c_np, s_np, r_xy, robot_yaw, gd_conf,
                candidate_id, detect_dist,
            )

        # Keep side lists in sync, mirroring `accumulate` without changing it.
        if target_class not in self._surf:
            self._surf[target_class] = list(recs)
        if target_class not in self._ver:
            self._ver[target_class] = [
                (1, r_xy.copy(), float(robot_yaw), float(gd_conf)) for _ in recs
            ]
        if target_class not in self._lifecycle:
            self._lifecycle[target_class] = [
                {"near_miss": 0, "memory_id": self._alloc_memory_id(),
                 "last_candidate_id": None, "lifecycle_state": CONFIRMED}
                for _ in recs
            ]

        dists = np.linalg.norm(np.asarray(recs, dtype=np.float64) - c_np, axis=1)
        idx = int(np.argmin(dists))
        prev_state = self._ensure_state_fields(self._lifecycle[target_class][idx])
        within_merge = bool(dists[idx] < self.config.merge_dist_thresh)
        life = str(prev_state.get("lifecycle_state", CONFIRMED))

        if not within_merge:
            return self._new_gd_entry(
                target_class, c_np, s_np, r_xy, robot_yaw, gd_conf,
                candidate_id, detect_dist,
            )
        if life == STABLE:
            return {
                "written": False, "merged": False,
                "gd_merge_blocked": "stable",
                "candidate_id": candidate_id, "detect_dist": float(detect_dist),
                "lifecycle_state": STABLE,
            }

        if life == GD:
            if new_info is None:
                new_info = self._is_new_gd_information(prev_state, r_xy, robot_yaw)
            old = np.asarray(recs[idx], dtype=np.float64)
            recs[idx] = (
                self.config.ema_weight_old * old
                + (1.0 - self.config.ema_weight_old) * c_np
            )
            v_old = self._ver[target_class][idx]
            old_conf = float(v_old[3]) if len(v_old) > 3 else float(gd_conf)
            num_obs_now = int(v_old[0])
            if bool(new_info):
                num_obs_now += 1
                prev_state["gd_hits"] = int(prev_state.get("gd_hits", 0)) + 1
                prev_state["gd_last_evidence_step"] = int(self._frame)
                prev_state["gd_last_xy"] = np.asarray(r_xy, dtype=np.float64).tolist()
                prev_state["gd_last_yaw"] = float(robot_yaw)
                self._append_observation(prev_state, r_xy, robot_yaw)
            self._ver[target_class][idx] = (
                num_obs_now, r_xy.copy(), float(robot_yaw),
                max(old_conf, float(gd_conf)),
            )
            prev_state["gd_conf_last"] = float(gd_conf)
            prev_state["gd_conf_max"] = max(
                float(prev_state.get("gd_conf_max", 0.0)), float(gd_conf)
            )
            prev_state["gd_last_step"] = int(self._frame)
            prev_state["last_candidate_id"] = candidate_id
            self._touch_observation(prev_state, r_xy, robot_yaw)
            return {
                "written": True, "merged": True, "revived": False,
                "new_info": bool(new_info), "num_obs": num_obs_now,
                "memory_id": prev_state.get("memory_id"),
                "candidate_id": candidate_id,
                "lifecycle_state": GD,
                "detect_dist": float(detect_dist),
                **self._gd_fields(prev_state),
            }

        if life in (CONFIRMED, PROVISIONAL):
            # Evidence-only write: TSP3D owns the geometry, state and counters.
            if new_info is None:
                new_info = self._is_new_gd_information(prev_state, r_xy, robot_yaw)
            if bool(new_info):
                prev_state["gd_hits"] = int(prev_state.get("gd_hits", 0)) + 1
                prev_state["gd_last_evidence_step"] = int(self._frame)
                prev_state["gd_last_xy"] = np.asarray(r_xy, dtype=np.float64).tolist()
                prev_state["gd_last_yaw"] = float(robot_yaw)
            prev_state["gd_conf_last"] = float(gd_conf)
            prev_state["gd_conf_max"] = max(
                float(prev_state.get("gd_conf_max", 0.0)), float(gd_conf)
            )
            prev_state["gd_last_step"] = int(self._frame)
            prev_state["gd_last_candidate_id"] = candidate_id
            return {
                "written": True, "merged": True, "revived": False,
                "new_info": bool(new_info),
                "num_obs": int(self._ver[target_class][idx][0]),
                "memory_id": prev_state.get("memory_id"),
                "candidate_id": candidate_id,
                "lifecycle_state": life,
                "detect_dist": float(detect_dist),
                **self._gd_fields(prev_state),
            }

        return {
            "written": False, "merged": False,
            "gd_merge_blocked": f"state:{life}",
            "candidate_id": candidate_id, "lifecycle_state": life,
        }

    def _new_gd_entry(
        self,
        target_class: str,
        centroid: np.ndarray,
        surface: np.ndarray,
        robot_xy: np.ndarray,
        robot_yaw: float,
        gd_conf: float,
        candidate_id: Optional[str],
        detect_dist: float,
    ) -> Dict[str, Any]:
        """Create a new GD-only entry (state `gd`, provenance `gd-only`)."""
        self._mem.setdefault(target_class, []).append(centroid)
        self._surf.setdefault(target_class, []).append(surface)
        self._ver.setdefault(target_class, []).append(
            (1, np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2].copy(),
             float(robot_yaw), float(gd_conf))
        )
        state: Dict[str, Any] = {
            "near_miss": 0,
            "memory_id": self._alloc_memory_id(),
            "last_candidate_id": candidate_id,
            "lifecycle_state": GD,
            "origin": GD_ONLY,
            "gd_only": True,
            "total_miss": 0,
            "revive_count": 0,
            "gd_hits": 1,
            "gd_conf_max": float(gd_conf),
            "gd_conf_last": float(gd_conf),
            "gd_last_step": int(self._frame),
            "gd_last_evidence_step": int(self._frame),
            "gd_last_xy": np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2].tolist(),
            "gd_last_yaw": float(robot_yaw),
            # No sigma_c is fabricated for GD-only entries.
            "sigma_c_max": 0.0, "sigma_c_latest": 0.0,
            "sigma_c_ema_03": 0.0, "sigma_c_ema_05": 0.0,
            "sigma_c_ema_08": 0.0,
        }
        self._touch_observation(state, robot_xy, robot_yaw)
        self._append_observation(state, robot_xy, robot_yaw)
        self._lifecycle.setdefault(target_class, []).append(state)
        return {
            "written": True, "merged": False, "revived": False,
            "num_obs": 1,
            "memory_id": state["memory_id"],
            "candidate_id": candidate_id,
            "lifecycle_state": GD,
            "detect_dist": float(detect_dist),
            "gd_conf": float(gd_conf),
            **self._gd_fields(state),
        }

    def apply_gd_evidence(
        self,
        target_class: str,
        centroid: np.ndarray,
        *,
        gd_conf: float,
        gd_hits: int = 1,
        gd_pos: bool = False,
        gd_candidate_pos: bool = False,
        candidate_id: Optional[str] = None,
        robot_xy: Optional[np.ndarray] = None,
        robot_yaw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Attach GD evidence without changing TSP3D state/geometry/counters."""
        recs = self._mem.get(target_class, [])
        if not recs:
            return {"written": False, "reason": "no_memory"}
        c_np = np.asarray(centroid, dtype=np.float64).reshape(-1)
        dists = np.linalg.norm(np.asarray(recs, dtype=np.float64) - c_np, axis=1)
        idx = int(np.argmin(dists))
        states = self._lifecycle.get(target_class, [])
        if idx >= len(states):
            return {"written": False, "reason": "state_missing"}
        state = self._ensure_state_fields(states[idx])
        state["gd_hits"] = max(int(state.get("gd_hits", 0)), int(gd_hits))
        state["gd_conf_last"] = float(gd_conf)
        state["gd_conf_max"] = max(float(state.get("gd_conf_max", 0.0)), float(gd_conf))
        state["gd_last_step"] = int(self._frame)
        state["gd_last_evidence_step"] = int(self._frame)
        state["gd_pos"] = bool(state.get("gd_pos", False) or gd_pos)
        state["gd_candidate_pos"] = bool(
            state.get("gd_candidate_pos", False) or gd_candidate_pos
        )
        if candidate_id is not None:
            state["gd_last_candidate_id"] = candidate_id
        if robot_xy is not None:
            state["gd_last_xy"] = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2].tolist()
        if robot_yaw is not None:
            state["gd_last_yaw"] = float(robot_yaw)
        return {
            "written": True,
            "memory_id": state.get("memory_id"),
            "lifecycle_state": state.get("lifecycle_state"),
            "origin": state.get("origin"),
            **self._gd_fields(state),
        }

    @staticmethod
    def _tsp_origin_for_state(
        origin: Optional[str], lifecycle_state: Optional[str]
    ) -> Optional[str]:
        """Set pure-TSP provenance without touching GD/gd-tsp origins."""
        if origin in (GD_ONLY, GD_TSP):
            return origin
        if lifecycle_state == PROVISIONAL:
            return TSP_PROVISIONAL
        if lifecycle_state == CONFIRMED:
            return TSP_CONFIRMED
        return origin

    def _revive_stable_entry(
        self,
        target_class: str,
        idx: int,
        state: Dict[str, Any],
        c_np: np.ndarray,
        score: float,
        robot_xy: np.ndarray,
        robot_yaw: float,
        confidence: float,
        candidate_id: Optional[str],
        detect_dist: float,
    ) -> Optional[Dict[str, Any]]:
        """S6: bring a `stable` entry back to `provisional` on a fresh sighting.

        The revive keeps `num_obs` and the score history (evidence accumulates
        instead of resetting), resets the consecutive miss counter and bumps
        `revive_count`.  Cumulative misses are never reset.
        """
        if not self.config.enable_stable_pool:
            return None
        if float(score) < float(self.config.stable_revive_min_sigma):
            return None
        self._ensure_state_fields(state)
        last_obs_before = int(state.get("last_obs_step", 0))
        age = int(self._frame) - last_obs_before
        if age > int(self.config.stable_revive_window):
            return None
        recs = self._mem[target_class]
        old = np.asarray(recs[idx], dtype=np.float64)
        dist_to_old = float(np.linalg.norm(c_np[:2] - old[:2]))
        if dist_to_old > float(self.config.stable_revive_radius):
            return None
        v_old = self._ver[target_class][idx]
        num_obs_before = int(v_old[0]) if len(v_old) >= 1 else 1
        recs[idx] = self.config.ema_weight_old * old + (1.0 - self.config.ema_weight_old) * c_np
        state["lifecycle_state"] = PROVISIONAL
        if state.get("origin") not in (GD_ONLY, GD_TSP):
            state["origin"] = TSP_PROVISIONAL
        state["near_miss"] = 0
        state["revive_count"] = int(state.get("revive_count", 0)) + 1
        state["last_candidate_id"] = candidate_id
        self._touch_observation(state, robot_xy, robot_yaw)
        self._append_observation(state, robot_xy, robot_yaw)
        self._update_scores(state, float(score), new=False)
        vers = self._ver[target_class]
        num_obs_now = num_obs_before + 1
        vers[idx] = (
            num_obs_now, np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2].copy(),
            float(robot_yaw),
            max(float(v_old[3]) if len(v_old) > 3 else float(confidence), float(confidence)),
        )
        return {
            "written": True, "merged": True, "revived": True, "new_info": True,
            "num_obs": num_obs_now,
            "memory_id": state.get("memory_id"),
            "candidate_id": candidate_id,
            "lifecycle_state": PROVISIONAL,
            "from_state": STABLE,
            "origin": state.get("origin"),
            "upgraded": None,
            "total_miss": int(state.get("total_miss", 0)),
            "revive_count": int(state.get("revive_count", 0)),
            "detect_dist": float(detect_dist),
            # Explicit revive diagnostics.  These are copied into the merge
            # event by the policy so the next run can fit sigma/obs/dist/age
            # thresholds instead of only knowing that a revive happened.
            "revive_sigma_c": float(score),
            "revive_num_obs_before": int(num_obs_before),
            "revive_num_obs_after": int(num_obs_now),
            "revive_dist_to_old_centroid": float(dist_to_old),
            "revive_age_steps": int(age),
            "revive_last_obs_step": int(last_obs_before),
            "revive_radius": float(self.config.stable_revive_radius),
            "revive_window": int(self.config.stable_revive_window),
            "revive_min_sigma": float(self.config.stable_revive_min_sigma),
            "revive_detect_dist": float(detect_dist),
            "revive_robot_xy": np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2].tolist(),
            "revive_old_centroid": old[:3].tolist(),
            "revive_new_centroid": np.asarray(recs[idx], dtype=np.float64).reshape(-1)[:3].tolist(),
            **self._score_fields(state),
        }

    def state_of(self, centroid: np.ndarray, tol: float = 0.3) -> Dict[str, Any]:
        """
        Lifecycle state dict of the entry nearest to ``centroid``.

        Returns an empty dict when no entry is within ``tol``.  The result is a
        shallow copy with `num_obs` merged in from the verify state, so callers
        can read the live observation count without mutating the stored state.
        """
        c = np.asarray(centroid, dtype=np.float64).reshape(-1)
        if c.size < 2:
            return {}
        best_state: Dict[str, Any] = {}
        best_d = float(tol)
        for target_class in list(self._mem.keys()):
            states = self._lifecycle.get(target_class, [])
            for idx, mem_c in enumerate(self._mem[target_class]):
                if idx >= len(states):
                    continue
                m = np.asarray(mem_c, dtype=np.float64).reshape(-1)
                if m.size < 2:
                    continue
                d = float(np.linalg.norm(m[:2] - c[:2]))
                if d <= best_d:
                    best_d = d
                    best_state = dict(states[idx])
                    best_state["target_class"] = str(target_class)
                    ver = self._ver.get(target_class, [])
                    if idx < len(ver) and len(ver[idx]) >= 1:
                        best_state["num_obs"] = int(ver[idx][0])
                    else:
                        best_state["num_obs"] = 1
        return best_state

    def _deletion_record(
        self,
        target_class: str,
        centroid: np.ndarray,
        v_state: Any,
        state: Dict[str, Any],
        reason: str,
        near_miss: int,
        age_steps: Optional[int] = None,
        dist_to_robot: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One deletion record.  S6: carries the counters the decision used.

        `near_miss` / `total_miss` / `age_steps` are the three budgets, and
        `thresholds` snapshots their effective values at decision time, so an
        offline pass can tell a justified deletion from a premature one.
        """
        try:
            conf = float(v_state[3]) if len(v_state) > 3 else 0.0
        except Exception:
            conf = 0.0
        self._ensure_state_fields(state)
        last_step = int(state.get("last_obs_step", 0))
        if age_steps is None:
            age_steps = int(self._frame) - last_step
        thresholds = {
            "confirmed_to_stable_misses": int(self.config.confirmed_to_stable_misses),
            "provisional_to_stable_misses": int(self.config.provisional_to_stable_misses),
            "stable_miss_delete": int(self.config.stable_miss_delete),
            "stable_total_miss": int(self.config.stable_total_miss),
            "stable_stale_purge": int(self.config.stable_stale_purge),
        }
        return {
            "event": "delete",
            "target_class": str(target_class),
            "centroid": np.asarray(centroid, dtype=np.float64).copy(),
            "reason": str(reason),
            "num_obs": int(v_state[0]) if len(v_state) >= 1 else 1,
            "conf": float(conf),
            "near_miss": int(near_miss),
            "memory_id": state.get("memory_id"),
            "candidate_id": state.get("last_candidate_id"),
            "lifecycle_state": state.get("lifecycle_state", "confirmed"),
            "from_state": state.get("lifecycle_state", "confirmed"),
            "total_miss": int(state.get("total_miss", 0)),
            "revive_count": int(state.get("revive_count", 0)),
            "age_steps": int(age_steps),
            "last_obs_step": last_step,
            "dist_to_robot": None if dist_to_robot is None else float(dist_to_robot),
            "thresholds": thresholds,
            **TargetMemoryManager._score_fields(state),
        }

    def _transition_record(
        self,
        target_class: str,
        centroid: np.ndarray,
        v_state: Any,
        state: Dict[str, Any],
        from_state: str,
        to_state: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Downgrade `confirmed` / `provisional` -> `stable` and log it."""
        self._ensure_state_fields(state)
        misses = int(state.get("near_miss", 0))
        state["lifecycle_state"] = to_state
        state["near_miss"] = 0
        return {
            "event": "transition",
            "target_class": str(target_class),
            "centroid": np.asarray(centroid, dtype=np.float64).copy(),
            "reason": str(reason),
            "from_state": from_state,
            "lifecycle_state": to_state,
            "near_miss": misses,
            "total_miss": int(state.get("total_miss", 0)),
            "revive_count": int(state.get("revive_count", 0)),
            "age_steps": int(self._frame) - int(state.get("last_obs_step", 0)),
            "memory_id": state.get("memory_id"),
            "candidate_id": state.get("last_candidate_id"),
            "num_obs": int(v_state[0]) if len(v_state) >= 1 else 1,
            "conf": float(v_state[3]) if len(v_state) > 3 else 0.0,
            **self._score_fields(state),
        }

    def try_upgrade(self, robot_xyz: np.ndarray) -> List[Dict[str, Any]]:
        """S5: promote provisional entries re-observed near the robot, every frame.

        The merge-time upgrade missed entries whose second observation arrived far
        away; this check runs every frame.  Both the merge-time and frame-level
        paths require only ``num_obs >= upgrade_min_obs``; distance is not an
        upgrade gate.  sigma_c_latest remains a diagnostic/selection field.
        """
        origin = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)
        events: List[Dict[str, Any]] = []
        for cls in list(self._lifecycle.keys()):
            states = self._lifecycle.get(cls, [])
            recs = self._mem.get(cls, [])
            vers = self._ver.get(cls, [])
            for idx, state in enumerate(states):
                if str(state.get("lifecycle_state")) != PROVISIONAL or idx >= len(recs):
                    continue
                num_obs = int(vers[idx][0]) if idx < len(vers) and len(vers[idx]) >= 1 else 1
                centroid = np.asarray(recs[idx], dtype=np.float64).reshape(-1)
                dist = float(np.linalg.norm(centroid[:3] - origin[:3]))
                if num_obs < int(self.config.upgrade_min_obs):
                    continue
                state["lifecycle_state"] = CONFIRMED
                if state.get("origin") not in (GD_ONLY, GD_TSP):
                    state["origin"] = TSP_CONFIRMED
                events.append({
                    "event": "upgrade",
                    "target_class": str(cls),
                    "centroid": centroid.copy(),
                    "reason": "base_upgrade",
                    "trigger": "frame",
                    "from_state": PROVISIONAL,
                    "lifecycle_state": CONFIRMED,
                    "num_obs": num_obs,
                    "detect_dist": dist,
                    "memory_id": state.get("memory_id"),
                    "candidate_id": state.get("last_candidate_id"),
                    "conf": float(vers[idx][3]) if idx < len(vers) and len(vers[idx]) > 3 else 0.0,
                    **self._score_fields(state),
                })
        return events

    def step(
        self,
        camera_pos: np.ndarray,
        camera_yaw: float,
        cone_fov_rad: float,
        robot_xyz: Optional[np.ndarray] = None,
        nav_goal_xy: Optional[np.ndarray] = None,
        exempt_nav_goal: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        One-step lifecycle over all memory records.

        Near-field miss accounting for the stable pool, with the nav-goal
        exemption: with `exempt_nav_goal=True` and a `nav_goal_xy`, the entry
        within 0.5 m of the current navigation goal is exempt from the
        near-field `near_miss` budget — otherwise a blocked stop (no new writes)
        would delete the entry.

        Returns per-deletion / transition logs for the stable-pool budgets.
        """
        events: List[Dict[str, Any]] = []
        if robot_xyz is None:
            robot_xyz = np.asarray(camera_pos, dtype=np.float64)
        else:
            robot_xyz = np.asarray(robot_xyz, dtype=np.float64)
        ng_xy = (
            None if nav_goal_xy is None
            else np.asarray(nav_goal_xy, dtype=np.float64).reshape(-1)[:2]
        )

        for target_class in list(self._mem.keys()):
            centroids = self._mem[target_class]
            surfaces = self._surf.get(target_class, [])
            v_states = self._ver.get(target_class, [])
            lifecycle_states = self._lifecycle.get(target_class, [])
            if len(lifecycle_states) != len(centroids):
                # Defensive: keep lifecycle state in sync with centroids.
                lifecycle_states = [
                    {"near_miss": 0, "memory_id": self._alloc_memory_id(),
                     "last_candidate_id": None, "lifecycle_state": "confirmed"}
                    for _ in centroids
                ]
                self._lifecycle[target_class] = lifecycle_states

            keep_c: List[np.ndarray] = []
            keep_s: List[np.ndarray] = []
            keep_v: List[Any] = []
            keep_lifecycle: List[Dict[str, Any]] = []

            for idx, centroid in enumerate(centroids):
                c_np = np.asarray(centroid, dtype=np.float64)
                v_state = v_states[idx] if idx < len(v_states) else (1, np.zeros(2), 0.0, 1.0)
                # S6-final: gd-only entries are never deleted by TTL or by
                # the TSP3D stable-pool near-miss/stale budgets.
                if str(lifecycle_states[idx].get("lifecycle_state", CONFIRMED)) == GD:
                    keep_c.append(c_np)
                    keep_s.append(surfaces[idx] if idx < len(surfaces) else c_np)
                    keep_v.append(v_state)
                    keep_lifecycle.append(lifecycle_states[idx])
                    continue
                # S6: `stable` is the deletion pool.  Staleness is checked even
                # when the entry has left the near field, which the miss budget
                # alone can never catch.
                if self.config.enable_stable_pool:
                    state_pre = self._ensure_state_fields(lifecycle_states[idx])
                    age = int(self._frame) - int(state_pre.get("last_obs_step", 0))
                    if age >= int(self.config.stable_stale_purge):
                        _life = str(state_pre.get("lifecycle_state"))
                        if _life == STABLE:
                            events.append(self._deletion_record(
                                target_class, c_np, v_state, state_pre,
                                "stable_stale_purge", int(state_pre.get("near_miss", 0)),
                                age_steps=age,
                                dist_to_robot=float(np.linalg.norm(c_np[:2] - robot_xyz[:2])),
                            ))
                            continue
                        if _life in {CONFIRMED, PROVISIONAL}:
                            # Staleness is state-agnostic: an entry last seen long ago
                            # is demoted first so that every delete happens in `stable`.
                            events.append(self._transition_record(
                                target_class, c_np, v_state, state_pre,
                                _life, STABLE, "stale_demote",
                            ))

                # Near-field miss accounting for the stable pool.
                query_pt = (
                    c_np.reshape(1, 3)
                    if c_np.shape[0] >= 3
                    else np.append(c_np, 0.5).reshape(1, 3)
                )
                in_cone = within_fov_cone(
                    cone_origin=camera_pos, cone_angle=camera_yaw,
                    cone_fov=cone_fov_rad, cone_range=self.config.near_field_radius,
                    points=query_pt,
                )
                if len(in_cone) > 0:
                    # navigate 模式下当前 nav goal 条目豁免 near_miss 预算。
                    # S5：provisional 与由 provisional 升级而来的条目不豁免。
                    _state_now = lifecycle_states[idx]
                    _no_exempt = (
                        str(_state_now.get("lifecycle_state")) == PROVISIONAL
                        or bool(_state_now.get("origin_provisional", False))
                    )
                    _exempt = False
                    if exempt_nav_goal and ng_xy is not None and not _no_exempt:
                        _exempt = float(np.linalg.norm(c_np[:2] - ng_xy)) <= 0.5
                    if _exempt:
                        keep_c.append(c_np)
                        keep_s.append(surfaces[idx] if idx < len(surfaces) else c_np)
                        keep_v.append(v_state)
                        keep_lifecycle.append(lifecycle_states[idx])
                        continue

                    lifecycle_states[idx]["near_miss"] += 1
                    state = self._ensure_state_fields(lifecycle_states[idx])
                    state["total_miss"] = int(state.get("total_miss", 0)) + 1
                    lifecycle = str(state.get("lifecycle_state", "confirmed"))
                    if self.config.enable_stable_pool:
                        if lifecycle in {CONFIRMED, PROVISIONAL}:
                            thr = (
                                int(self.config.confirmed_to_stable_misses)
                                if lifecycle == CONFIRMED
                                else int(self.config.provisional_to_stable_misses)
                            )
                            if int(state["near_miss"]) >= thr:
                                events.append(self._transition_record(
                                    target_class, c_np, v_state, state,
                                    lifecycle, STABLE, "near_miss_downgrade",
                                ))
                        elif int(state["near_miss"]) >= int(self.config.stable_miss_delete):
                            events.append(self._deletion_record(
                                target_class, c_np, v_state, state,
                                "stable_miss_delete", int(state["near_miss"]),
                                dist_to_robot=float(np.linalg.norm(c_np[:2] - robot_xyz[:2])),
                            ))
                            continue
                        elif int(state.get("total_miss", 0)) >= int(self.config.stable_total_miss):
                            events.append(self._deletion_record(
                                target_class, c_np, v_state, state,
                                "stable_total_miss", int(state["near_miss"]),
                                dist_to_robot=float(np.linalg.norm(c_np[:2] - robot_xyz[:2])),
                            ))
                            continue
                        keep_c.append(c_np)
                        keep_s.append(surfaces[idx] if idx < len(surfaces) else c_np)
                        keep_v.append(v_state)
                        keep_lifecycle.append(state)
                        continue

                keep_c.append(c_np)
                keep_s.append(surfaces[idx] if idx < len(surfaces) else c_np)
                keep_v.append(v_state)
                keep_lifecycle.append(lifecycle_states[idx])

            self._mem[target_class] = keep_c
            self._surf[target_class] = keep_s
            self._ver[target_class] = keep_v
            self._lifecycle[target_class] = keep_lifecycle
            if len(keep_c) == 0:
                del self._mem[target_class]
                self._surf.pop(target_class, None)
                self._ver.pop(target_class, None)
                self._lifecycle.pop(target_class, None)

        self._frame += 1
        return events
