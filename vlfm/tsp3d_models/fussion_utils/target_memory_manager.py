from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vlfm.utils.geometry_utils import within_fov_cone

# Proposer tags of a memory entry (cross-frame vote merge).
TSP3D_SRC = "tsp3d"
GD_SRC = "gd"
BOTH_SRC = "both"

def _src_tag(src: Optional[str]) -> str:
    """Missing tag means a plain TSP3D write."""
    return str(src) if src else TSP3D_SRC

def merge_src_tag(old: Optional[str], new: Optional[str]) -> str:
    """Two different proposer tags on one entry -> ``both``; otherwise unchanged."""
    a, b = _src_tag(old), _src_tag(new)
    if a == b:
        return a
    return BOTH_SRC


CONFIRMED = "confirmed"
PROVISIONAL = "provisional"
STABLE = "stable"


@dataclass(frozen=True)
class ProvisionalGateConfig:
    """Thresholds for creating a new target-memory entry."""

    enabled: bool = False
    confirmed_sigma: float = 0.60
    provisional_sigma: float = 0.55
    provisional_min_dist: float = 1.5
    provisional_max_dist: float = 4.0
    # Retained for config/log compatibility.  S no longer gates new entries.
    s_min: float = 0.10


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
    Target-memory / fallback lifecycle hyper-parameters.

    ``enable_fallback`` gates only the native hysteresis fallback.
    """
    # cross-view EMA merge (memory write)
    merge_dist_thresh: float = 0.5        # merge two observations of the same class within this distance (m)
    ema_weight_old: float = 0.8           # EMA smoothing: old * 0.8 + new * 0.2

    # S5-S2 lifecycle.  When enabled, unmatched candidates may create a memory
    # only when the independent provisional gate supplies an initial state.
    enable_lifecycle_states: bool = False
    confirmed_to_stable_misses: int = 3
    provisional_to_stable_misses: int = 2
    stable_delete_misses: int = 2


    # native anti-hallucination fallback (baseline mechanism)
    enable_fallback: bool = True          # native near-field hysteresis fallback master switch
    fb_near_radius: float = 2.5           # near-field FOV-cone radius (m); == max_depth * 0.5
    fb_hysteresis: int = 5                # trusted target: consecutive near-field miss frames before deletion
    fb_suspicious_hysteresis: int = 2     # suspicious target: consecutive near-field miss frames before deletion
    fb_suspicious_conf: float = 0.75      # trusted/suspicious confidence boundary.
    # No effect when GD is enabled: new entries are always marked suspicious 
    # and merge writes do not recompute the flag (cleared only by a `both` vote).


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
        self._fb: Dict[str, List[Dict[str, Any]]] = {}
        self._next_memory_id = 0

    def reset_ids(self) -> None:
        """Reset the episode-local stable memory-id sequence."""
        self._next_memory_id = 0

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
        target_fallback_state: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Bind the policy's memory dicts."""
        self._mem = target_3d_memory
        self._surf = target_surface_memory
        self._ver = target_verify_state
        self._fb = target_fallback_state

    # memory write / EMA merge
    def accumulate(
        self,
        target_class: str,
        centroid: np.ndarray,
        confidence: float = 1.0,
        near_surface: Optional[np.ndarray] = None,
        robot_xyz: Optional[np.ndarray] = None,
        robot_yaw: float = 0.0,
        max_depth: float = 5.0,
        suspicious_override: Optional[bool] = None,
        src: Optional[str] = None,
        new_suspicious: bool = False,
        gd_conf: float = 0.0,
        candidate_id: Optional[str] = None,
        new_lifecycle_state: Optional[str] = None,
        sigma_c: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        In-place write / EMA-merge of one admitted detection.
        Merges within merge_dist_thresh of a same-class record 
        (EMA centroid, num_obs+1, max confidence kept, near-miss reset); 
        otherwise registers a new candidate.

        `src` is the proposer tag of this write: 
        "tsp3d" (default) / "gd" (GD-only single vote) / "both". 
        Two different tags on the same entry make it "both" (cross-frame vote merge); 
        it never changes the confidence scale.

        Vote policy:
          * `both` always follows the normal path — a merged `both` entry has its
            suspicious flag cleared (the 2-frame suspicion budget no longer applies);
          * single-side evidence may be marked suspicious: `new_suspicious` marks a
            newly created entry, `suspicious_override=True` forces the flag on.

        Returns a small record of the write: 
        `{merged, src, num_obs, suspicious, suspicion_cleared, gd_conf}` 
        (for caller-side logging / accounting).
        """
        r_xyz = np.zeros(3) if robot_xyz is None else np.asarray(robot_xyz, dtype=np.float64)
        r_xy = r_xyz[:2]
        c_np = np.asarray(centroid, dtype=np.float64)
        s_np = np.asarray(near_surface, dtype=np.float64) if near_surface is not None else c_np

        # Suspicious flag: low confidence or beyond the trustworthy range.
        if c_np.shape[0] >= 3:
            detect_dist = float(np.linalg.norm(c_np - r_xyz))
        else:
            detect_dist = float(np.linalg.norm(c_np - r_xy))
        suspicious = bool(
            (float(confidence) < self.config.fb_suspicious_conf)
            or (detect_dist > max_depth * 0.95)
        )
        # External suspicious override (far / frustum-edge marks).
        if suspicious_override:
            suspicious = True
        # Single-side evidence: a newly created entry starts as suspicious.
        if new_suspicious:
            suspicious = True

        def _m_a(state: Dict[str, Any]) -> Dict[str, Any]:
            # Keep the strongest raw GD score ever written on this entry.
            # Never feeds a decision; it is only read back for logging.
            if float(gd_conf) > 0.0:
                state["gd_conf"] = max(
                    float(state.get("gd_conf", 0.0)), float(gd_conf)
                )
            return {"gd_conf": float(state.get("gd_conf", 0.0))}

        score = float(confidence if sigma_c is None else sigma_c)
        recs = self._mem.get(target_class)
        if not recs:
            if self.config.enable_lifecycle_states and new_lifecycle_state is None:
                return {
                    "written": False, "merged": False, "new_rejected": True,
                    "candidate_id": candidate_id,
                }
            self._mem[target_class] = [c_np]
            self._surf[target_class] = [s_np]
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence))]
            state = {"suspicious": suspicious, "near_miss": 0, "src": _src_tag(src),
                     "gd_conf": 0.0, "memory_id": self._alloc_memory_id(),
                     "last_candidate_id": candidate_id,
                     "lifecycle_state": new_lifecycle_state or "confirmed"}
            self._update_scores(state, score, new=True)
            self._fb[target_class] = [state]
            return {"written": True, "merged": False, "src": _src_tag(src), "num_obs": 1,
                    "suspicious": bool(suspicious), "suspicion_cleared": False,
                    "memory_id": state["memory_id"], "candidate_id": candidate_id,
                    "lifecycle_state": state["lifecycle_state"],
                    **self._score_fields(state), **_m_a(state)}

        # Keep the side lists in sync
        if target_class not in self._surf:
            self._surf[target_class] = list(recs)
        if target_class not in self._ver:
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence)) for _ in recs]
        if target_class not in self._fb:
            self._fb[target_class] = [
                {"suspicious": False, "near_miss": 0, "src": TSP3D_SRC,
                 "gd_conf": 0.0, "memory_id": self._alloc_memory_id(),
                 "last_candidate_id": None, "lifecycle_state": "confirmed"}
                for _ in recs
            ]

        dists = np.linalg.norm(np.asarray(recs, dtype=np.float64) - c_np, axis=1)
        closest_idx = int(np.argmin(dists))
        prev_state = self._fb[target_class][closest_idx]
        if dists[closest_idx] < self.config.merge_dist_thresh:
            # Cross-frame consensus = strong evidence; hallucinated boxes drift and cannot merge. 
            # EMA centroid, keep the max c', reset near-miss.
            old = np.asarray(recs[closest_idx], dtype=np.float64)
            recs[closest_idx] = self.config.ema_weight_old * old + (1.0 - self.config.ema_weight_old) * c_np
            v_old = self._ver[target_class][closest_idx]
            old_conf = float(v_old[3]) if len(v_old) > 3 else float(confidence)
            self._ver[target_class][closest_idx] = (
                int(v_old[0]) + 1, r_xy.copy(), float(robot_yaw),
                max(old_conf, float(confidence)),
            )
            prev_state["near_miss"] = 0
            prev_src = prev_state.get("src")
            merged_src = merge_src_tag(prev_src, src)
            prev_state["src"] = merged_src
            prev_state["last_candidate_id"] = candidate_id
            # S5-S2: merge resets the miss budget and records scores, but never
            # reclassifies the lifecycle state.
            self._update_scores(prev_state, score, new=False)
            cleared = False
            if merged_src == BOTH_SRC:
                # Both models corroborate the same entry -> normal target, no suspicion.
                cleared = bool(prev_state.get("suspicious"))
                prev_state["suspicious"] = False
            return {"written": True, "merged": True, "src": merged_src, "num_obs": int(v_old[0]) + 1,
                    "suspicious": bool(prev_state.get("suspicious")),
                    "suspicion_cleared": cleared,
                    "memory_id": prev_state.get("memory_id"),
                    "candidate_id": candidate_id,
                    "lifecycle_state": prev_state.get("lifecycle_state", "confirmed"),
                    **self._score_fields(prev_state), **_m_a(prev_state)}
        else:
            if self.config.enable_lifecycle_states and new_lifecycle_state is None:
                return {
                    "written": False, "merged": False, "new_rejected": True,
                    "candidate_id": candidate_id,
                }
            recs.append(c_np)
            self._surf[target_class].append(s_np)
            self._ver[target_class].append((1, r_xy.copy(), float(robot_yaw), float(confidence)))
            state = {"suspicious": suspicious, "near_miss": 0, "src": _src_tag(src),
                     "gd_conf": 0.0, "memory_id": self._alloc_memory_id(),
                     "last_candidate_id": candidate_id,
                     "lifecycle_state": new_lifecycle_state or "confirmed"}
            self._update_scores(state, score, new=True)
            self._fb[target_class].append(state)
            return {"written": True, "merged": False, "src": _src_tag(src), "num_obs": 1,
                    "suspicious": bool(suspicious), "suspicion_cleared": False,
                    "memory_id": state["memory_id"], "candidate_id": candidate_id,
                    "lifecycle_state": state["lifecycle_state"],
                    **self._score_fields(state), **_m_a(state)}

    def state_of(self, centroid: np.ndarray, tol: float = 0.3) -> Dict[str, Any]:
        """
        Lifecycle / vote state dict of the entry nearest to ``centroid``.

        Holds `suspicious`, `near_miss`, `src` and the raw GD score (`gd_conf`);
        an empty dict when no entry is within ``tol``. 

        Returns a *shallow copy* with `num_obs` merged in from the verify state, 
        so callers can read the live observation count without mutating the stored state.
        """
        c = np.asarray(centroid, dtype=np.float64).reshape(-1)
        if c.size < 2:
            return {}
        best_state: Dict[str, Any] = {}
        best_d = float(tol)
        for target_class in list(self._mem.keys()):
            states = self._fb.get(target_class, [])
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
                    ver = self._ver.get(target_class, [])
                    if idx < len(ver) and len(ver[idx]) >= 1:
                        best_state["num_obs"] = int(ver[idx][0])
                    else:
                        best_state["num_obs"] = 1
        return best_state

    @staticmethod
    def _deletion_record(
        target_class: str,
        centroid: np.ndarray,
        v_state: Any,
        state: Dict[str, Any],
        reason: str,
        near_miss: int,
    ) -> Dict[str, Any]:
        """
        One deletion record.

        Carries what the episode logger needs to label the deleted entry tp / fp 
        and to split it by proposer tag: `src` / `num_obs` / `conf` / `gd_conf` / `near_miss`
        (`conf` = the entry's stored `c'`, `gd_conf` = its strongest raw GD sigmoid).
        """
        try:
            conf = float(v_state[3]) if len(v_state) > 3 else 0.0
        except Exception:
            conf = 0.0
        return {
            "event": "delete",
            "target_class": str(target_class),
            "centroid": np.asarray(centroid, dtype=np.float64).copy(),
            "reason": str(reason),
            "suspicious": bool(state.get("suspicious", False)),
            "src": _src_tag(state.get("src")),
            "num_obs": int(v_state[0]) if len(v_state) >= 1 else 1,
            "gd_conf": float(state.get("gd_conf", 0.0) or 0.0),
            "conf": float(conf),
            "near_miss": int(near_miss),
            "memory_id": state.get("memory_id"),
            "candidate_id": state.get("last_candidate_id"),
            "lifecycle_state": state.get("lifecycle_state", "confirmed"),
            **TargetMemoryManager._score_fields(state),
        }

    # per-step lifecycle
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

        Native near-field hysteresis fallback with the near-field exemption:
        with `exempt_nav_goal=True` and a `nav_goal_xy`, 
        the entry within 0.5 m of the current navigation goal is exempt 
        from the near-field `near_miss` budget
        — otherwise a blocked stop (no new writes) would delete the entry.

        Returns per-deletion logs with `reason` in {"hysteresis_exceeded"}.
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
            f_states = self._fb.get(target_class, [])
            if len(f_states) != len(centroids):
                # Defensive: keep fallback state in sync with centroids.
                f_states = [
                    {"suspicious": False, "near_miss": 0, "src": TSP3D_SRC,
                     "gd_conf": 0.0, "memory_id": self._alloc_memory_id(),
                     "last_candidate_id": None, "lifecycle_state": "confirmed"}
                    for _ in centroids
                ]
                self._fb[target_class] = f_states

            keep_c: List[np.ndarray] = []
            keep_s: List[np.ndarray] = []
            keep_v: List[Any] = []
            keep_f: List[Dict[str, Any]] = []

            for idx, centroid in enumerate(centroids):
                c_np = np.asarray(centroid, dtype=np.float64)
                v_state = v_states[idx] if idx < len(v_states) else (1, np.zeros(2), 0.0, 1.0)
                is_suspicious = bool(f_states[idx].get("suspicious", False))

                # native near-field fallback
                if self.config.enable_fallback:
                    query_pt = (
                        c_np.reshape(1, 3)
                        if c_np.shape[0] >= 3
                        else np.append(c_np, 0.5).reshape(1, 3)
                    )
                    in_cone = within_fov_cone(
                        cone_origin=camera_pos, cone_angle=camera_yaw,
                        cone_fov=cone_fov_rad, cone_range=self.config.fb_near_radius,
                        points=query_pt,
                    )
                    if len(in_cone) > 0:
                        # navigate 模式下当前 nav goal 条目豁免 near_miss 预算。
                        _exempt = False
                        if exempt_nav_goal and ng_xy is not None:
                            _exempt = float(np.linalg.norm(c_np[:2] - ng_xy)) <= 0.5
                        if _exempt:
                            keep_c.append(c_np)
                            keep_s.append(surfaces[idx] if idx < len(surfaces) else c_np)
                            keep_v.append(v_state)
                            f_states[idx]["suspicious"] = is_suspicious
                            keep_f.append(f_states[idx])
                            continue

                        f_states[idx]["near_miss"] += 1
                        if self.config.enable_lifecycle_states:
                            state = f_states[idx]
                            lifecycle = str(state.get("lifecycle_state", "confirmed"))
                            thresholds = {
                                "confirmed": int(self.config.confirmed_to_stable_misses),
                                "provisional": int(self.config.provisional_to_stable_misses),
                                "stable": int(self.config.stable_delete_misses),
                            }
                            if state["near_miss"] >= thresholds.get(lifecycle, 2):
                                if lifecycle in {"confirmed", "provisional"}:
                                    misses = int(state["near_miss"])
                                    state["lifecycle_state"] = "stable"
                                    state["near_miss"] = 0
                                    events.append({
                                        "event": "transition",
                                        "target_class": str(target_class),
                                        "centroid": c_np.copy(),
                                        "reason": "near_miss_downgrade",
                                        "from_state": lifecycle,
                                        "lifecycle_state": "stable",
                                        "near_miss": misses,
                                        "memory_id": state.get("memory_id"),
                                        "candidate_id": state.get("last_candidate_id"),
                                        "src": _src_tag(state.get("src")),
                                        "num_obs": int(v_state[0]) if len(v_state) >= 1 else 1,
                                        "conf": float(v_state[3]) if len(v_state) > 3 else 0.0,
                                        "suspicious": bool(state.get("suspicious", False)),
                                        **self._score_fields(state),
                                    })
                                else:
                                    rec = self._deletion_record(
                                        target_class, c_np, v_state, state,
                                        "stable_hysteresis_exceeded", int(state["near_miss"]),
                                    )
                                    events.append(rec)
                                    continue
                        else:
                            thr = (
                                int(self.config.fb_suspicious_hysteresis)
                                if is_suspicious
                                else int(self.config.fb_hysteresis)
                            )
                            if f_states[idx]["near_miss"] >= thr:
                                rec = self._deletion_record(
                                    target_class, c_np, v_state, f_states[idx],
                                    "hysteresis_exceeded", int(f_states[idx]["near_miss"]),
                                )
                                events.append(rec)
                                continue

                keep_c.append(c_np)
                keep_s.append(surfaces[idx] if idx < len(surfaces) else c_np)
                keep_v.append(v_state)
                f_states[idx]["suspicious"] = is_suspicious
                keep_f.append(f_states[idx])

            self._mem[target_class] = keep_c
            self._surf[target_class] = keep_s
            self._ver[target_class] = keep_v
            self._fb[target_class] = keep_f
            if len(keep_c) == 0:
                del self._mem[target_class]
                self._surf.pop(target_class, None)
                self._ver.pop(target_class, None)
                self._fb.pop(target_class, None)

        return events
