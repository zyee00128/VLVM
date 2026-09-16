from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vlfm.utils.geometry_utils import within_fov_cone

# Proposer tags of a memory entry (§3.6.4 cross-frame vote merge).
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


@dataclass
class MemoryManagerConfig:
    """
    Target-memory / fallback lifecycle hyper-parameters.

    ``enable_fallback`` gates only the native hysteresis fallback;
    the free-space erasure runs on its own switch even when the fallback is off.
    """
    # cross-view EMA merge (memory write)
    merge_dist_thresh: float = 0.5        # merge two observations of the same class within this distance (m)
    ema_weight_old: float = 0.8           # EMA smoothing: old * 0.8 + new * 0.2

    # 3-5 shadow (§优化方向 3, 零行为量测): geometric independence of successive writes
    # to one entry — displacement >= `gd_geo_move_m` (m) OR heading change >=
    # `gd_geo_yaw_deg` (deg). Counted per entry (`gd_geo_obs` / `gd_geo_ind`); the
    # 3-5 gate itself is `d2_geo_lock` in the policy. 3-4 帧级补框门由调用方按帧
    # 判定（`accumulate(..., new_gate=True)`，带 TSP3D 候选的帧不得新建条目）。
    gd_geo_move_m: float = 0.5
    gd_geo_yaw_deg: float = 30.0

    # native anti-hallucination fallback (baseline mechanism)
    enable_fallback: bool = True          # native near-field hysteresis fallback master switch
    fb_near_radius: float = 2.5           # near-field FOV-cone radius (m); == max_depth * 0.5
    fb_hysteresis: int = 5                # trusted target: consecutive near-field miss frames before deletion
    fb_suspicious_hysteresis: int = 2     # suspicious target: consecutive near-field miss frames before deletion
    fb_suspicious_conf: float = 0.75      # trusted/suspicious confidence boundary (must be > sigma_tar)


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
        new_gate: bool = False,
    ) -> Dict[str, Any]:
        """
        In-place write / EMA-merge of one admitted detection.
        Merges within merge_dist_thresh of a same-class record (EMA centroid, num_obs+1,
        max confidence kept, near-miss reset); otherwise registers a new candidate.

        `src` is the proposer tag of this write: "tsp3d" (default) / "gd" (GD-only
        single vote) / "both". Two different tags on the same entry make it "both"
        (cross-frame vote merge, §3.6.4); it never changes the confidence scale.

        Vote policy (§3.6.4):
          * `both` always follows the normal path — a merged `both` entry has its
            suspicious flag cleared (the 2-frame suspicion budget no longer applies);
          * single-side evidence may be marked suspicious: `new_suspicious` marks a
            newly created entry, `suspicious_override=True` forces the flag on.

        3-4 帧级补框门（§优化方向 3）: `new_gate=True` 时本写入**不得新建条目**
        （该帧 TSP3D 已有目标类 admitted 候选，由调用方判定）—— 能并入已有条目则
        照常合并，否则直接丢弃并返回 `{"suppressed": True}`（不写记忆、不计数）。

        M-a (§五 量测, default-neutral): `gd_conf` is the RAW GD sigmoid score of
        this write (0.0 = the write carries no GD score, i.e. a pure `tsp3d`
        write). Unlike `confidence` — which for a `gd` entry is the constant
        `sigma_tar` and for a `both` entry is the TSP3D `c'` — `gd_conf` keeps the
        GD side's own score so that later measurement can test whether the GD
        score separates tp from fp entries. It is stored as the running MAX over
        all writes, i.e. the strongest GD evidence ever seen on that entry.

        Returns a small record of the write: `{merged, src, num_obs, suspicious,
        suspicion_cleared, gd_conf, gd_geo_obs, gd_geo_ind}` (for caller-side
        logging / accounting; the `gd_geo_*` pair is the 3-5 shadow count).
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
            # M-a (§五 量测): keep the strongest raw GD score ever written on this
            # entry. Never feeds a decision; it is only read back for logging.
            if float(gd_conf) > 0.0:
                state["gd_conf"] = max(
                    float(state.get("gd_conf", 0.0)), float(gd_conf)
                )
            return {"gd_conf": float(state.get("gd_conf", 0.0))}

        def _geo(state: Dict[str, Any]) -> Dict[str, Any]:
            # 3-5 shadow (§优化方向 3, 零行为量测): count this write as a geometrically
            # independent vote when the robot moved >= gd_geo_move_m or its heading
            # changed >= gd_geo_yaw_deg since the last counted vote of this entry.
            # (`gd_geo_obs` = total writes counted, `gd_geo_ind` = independent ones.)
            obs = int(state.get("gd_geo_obs", 0)) + 1
            state["gd_geo_obs"] = obs
            pose = state.get("gd_geo_pose")
            if pose is None:
                ind = 1
                state["gd_geo_ind"] = ind
                state["gd_geo_pose"] = [float(r_xy[0]), float(r_xy[1]), float(robot_yaw)]
            else:
                moved = float(np.hypot(
                    float(r_xy[0]) - float(pose[0]), float(r_xy[1]) - float(pose[1])
                )) >= float(self.config.gd_geo_move_m)
                dyaw = abs(float(np.arctan2(
                    np.sin(float(robot_yaw) - float(pose[2])),
                    np.cos(float(robot_yaw) - float(pose[2])),
                )))
                if moved or dyaw >= float(np.deg2rad(float(self.config.gd_geo_yaw_deg))):
                    ind = int(state.get("gd_geo_ind", 0)) + 1
                    state["gd_geo_ind"] = ind
                    state["gd_geo_pose"] = [float(r_xy[0]), float(r_xy[1]), float(robot_yaw)]
                else:
                    ind = int(state.get("gd_geo_ind", 0))
            return {"gd_geo_obs": obs, "gd_geo_ind": int(ind)}

        recs = self._mem.get(target_class)
        if not recs:
            if new_gate:
                # 3-4：带 TSP3D 候选的帧不允许新建条目（无同类记录可合并）。
                return {"merged": False, "suppressed": True, "src": _src_tag(src),
                        "num_obs": 0, "suspicious": False, "suspicion_cleared": False,
                        "gd_conf": float(gd_conf) if float(gd_conf) > 0.0 else 0.0,
                        "gd_geo_obs": 0, "gd_geo_ind": 0}
            self._mem[target_class] = [c_np]
            self._surf[target_class] = [s_np]
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence))]
            state = {"suspicious": suspicious, "near_miss": 0, "src": _src_tag(src),
                     "gd_conf": 0.0}
            self._fb[target_class] = [state]
            return {"merged": False, "src": _src_tag(src), "num_obs": 1,
                    "suspicious": bool(suspicious), "suspicion_cleared": False,
                    **_m_a(state), **_geo(state)}

        # Keep the side lists in sync
        if target_class not in self._surf:
            self._surf[target_class] = list(recs)
        if target_class not in self._ver:
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence)) for _ in recs]
        if target_class not in self._fb:
            self._fb[target_class] = [{"suspicious": False, "near_miss": 0, "src": TSP3D_SRC,
                                      "gd_conf": 0.0} for _ in recs]

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
            cleared = False
            if merged_src == BOTH_SRC:
                # Both models corroborate the same entry -> normal target, no suspicion.
                cleared = bool(prev_state.get("suspicious"))
                prev_state["suspicious"] = False
            return {"merged": True, "src": merged_src, "num_obs": int(v_old[0]) + 1,
                    "suspicious": bool(prev_state.get("suspicious")),
                    "suspicion_cleared": cleared, **_m_a(prev_state), **_geo(prev_state)}
        else:
            if new_gate:
                # 3-4：未命中已有条目（会新建）且当前帧带 TSP3D 候选 ⇒ 丢弃本条写入。
                return {"merged": False, "suppressed": True, "src": _src_tag(src),
                        "num_obs": 0, "suspicious": False, "suspicion_cleared": False,
                        "gd_conf": float(gd_conf) if float(gd_conf) > 0.0 else 0.0,
                        "gd_geo_obs": 0, "gd_geo_ind": 0}
            recs.append(c_np)
            self._surf[target_class].append(s_np)
            self._ver[target_class].append((1, r_xy.copy(), float(robot_yaw), float(confidence)))
            state = {"suspicious": suspicious, "near_miss": 0, "src": _src_tag(src),
                     "gd_conf": 0.0}
            self._fb[target_class].append(state)
            return {"merged": False, "src": _src_tag(src), "num_obs": 1,
                    "suspicious": bool(suspicious), "suspicion_cleared": False,
                    **_m_a(state), **_geo(state)}

    def state_of(self, centroid: np.ndarray, tol: float = 0.3) -> Dict[str, Any]:
        """Lifecycle / vote state dict of the entry nearest to ``centroid``.

        Holds `suspicious`, `near_miss`, `src` and the M-a raw GD score (`gd_conf`);
        an empty dict when no entry is within ``tol``. Returns a *shallow copy* with
        `num_obs` merged in from the verify state, so callers (e.g. the 3-9 stop gate)
        can read the live observation count without mutating the stored state.
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
        """One deletion record (A2 measurement #3, §3.9).

        Carries what the episode logger needs to label the deleted entry tp / fp and to
        split it by proposer tag: `src` / `num_obs` / `conf` / `gd_conf` /
        `near_miss` (`conf` = the entry's stored `c'`, `gd_conf` = its strongest raw GD
        sigmoid, M-a §五 量测).
        """
        try:
            conf = float(v_state[3]) if len(v_state) > 3 else 0.0
        except Exception:
            conf = 0.0
        return {
            "target_class": str(target_class),
            "centroid": np.asarray(centroid, dtype=np.float64).copy(),
            "reason": str(reason),
            "suspicious": bool(state.get("suspicious", False)),
            "src": _src_tag(state.get("src")),
            "num_obs": int(v_state[0]) if len(v_state) >= 1 else 1,
            "gd_conf": float(state.get("gd_conf", 0.0) or 0.0),
            "conf": float(conf),
            "near_miss": int(near_miss),
        }

    @staticmethod
    def _log_deletion(rec: Dict[str, Any]) -> None:
        """Print one deletion together with the A2 fields (one line, grep-able)."""
        head, c = rec["reason"], np.round(np.asarray(rec["centroid"], dtype=np.float64), 3)
        tail = (
            f"suspicious={rec['suspicious']}, near_miss={rec['near_miss']}, "
            f"src={rec['src']}, n_obs={rec['num_obs']}, "
            f"conf={rec['conf']:.2f}, gd_conf={float(rec.get('gd_conf', 0.0) or 0.0):.2f}, "
            f"reason={head}"
        )
        print(
            f"[AntiHallucination] Deleted {rec['target_class']} centroid "
            f"{c} ({tail}) -> fallback to explore"
        )

    # per-step lifecycle
    def step(
        self,
        camera_pos: np.ndarray,
        camera_yaw: float,
        cone_fov_rad: float,
        obstacle_map_3d: Optional[Any] = None,
        robot_xyz: Optional[np.ndarray] = None,
        nav_goal_xy: Optional[np.ndarray] = None,
        exempt_nav_goal: bool = False,
    ) -> List[Dict[str, Any]]:
        """One-step lifecycle over all memory records.

        Order: (1) free-space erasure; (2) native near-field hysteresis fallback
        with the near-field exemption.

        3-9 配套（§优化方向 3）: with `exempt_nav_goal=True` and a `nav_goal_xy`,
        the entry within 0.5 m of the current navigation goal is exempt from the
        near-field `near_miss` budget — otherwise a blocked stop (robot parked at
        the goal, no new writes) would delete the entry and turn "delayed stop"
        into "lost entry" (results/VLVM-V8.md §3.15.8 chain 3).

        Returns per-deletion logs with `reason="hysteresis_exceeded"`.
        """
        deleted: List[Dict[str, Any]] = []
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
                f_states = [{"suspicious": False, "near_miss": 0, "src": TSP3D_SRC} for _ in centroids]
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
                        # 3-9 配套：navigate 模式下当前 nav goal 条目豁免 near_miss 预算。
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
                            deleted.append(rec)
                            self._log_deletion(rec)
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

        return deleted
