from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vlfm.utils.geometry_utils import within_fov_cone


@dataclass
class MemoryManagerConfig:
    """
    Target-memory / fallback lifecycle hyper-parameters.

    ``enable_fallback`` gates only the native hysteresis fallback; 
    M6 keeps running on its own switch even when the fallback is disabled.
    """
    # cross-view EMA merge (memory write)
    merge_dist_thresh: float = 0.5        # merge two observations of the same class within this distance (m)
    ema_weight_old: float = 0.8           # EMA smoothing: old * 0.8 + new * 0.2
    
    # native anti-hallucination fallback (baseline mechanism)
    enable_fallback: bool = True          # native near-field hysteresis fallback master switch
    fb_near_radius: float = 2.5           # near-field FOV-cone radius (m); == max_depth * 0.5
    fb_hysteresis: int = 5                # trusted target: consecutive near-field miss frames before deletion
    fb_suspicious_hysteresis: int = 2     # suspicious target: consecutive near-field miss frames before deletion
    fb_suspicious_conf: float = 0.75      # trusted/suspicious confidence boundary (must be > sigma_tar)

    # M7: near-field confirmation exemption (default OFF -> pure baseline)
    enable_mechanism_7: bool = False      # independent switch
    exempt_near_radius: float = 1.0       # robot-target distance below which a trusted target is exempt (m)
    exempt_min_merges: int = 1            # min num_obs of a (trusted) target to be exempt

    # M6: free-space line-of-sight erasure (default OFF -> pure baseline)
    enable_mechanism_6: bool = False      # independent switch
    free_erasure_soft_only: bool = True   # True: erase suspicious targets only (conservative)
    free_erasure_radius: float = 0.3      # half-side (m) of the box around the centroid to inspect
    free_erasure_min_explored: int = 5    # min explored voxels inside the box to trust "this space was seen"
    free_erasure_free_ratio: float = 0.85 # free-voxel ratio above which the box is judged "pure air"
    
    # M9: navigate identity freeze (default OFF -> pure baseline)
    enable_mechanism_9: bool = False      # independent switch

    

class TargetMemoryManager:
    """
    Target lifecycle manager operating IN PLACE on the bound policy memory dicts.

    The policy binds its four memory dicts once at construction;
    because the policy clears those dicts with ``.clear()``,
    the references stay valid across episodes. 
    ``reset()`` only clears manager-local bookkeeping.
    """

    def __init__(self, config: Optional[MemoryManagerConfig] = None) -> None:
        self.config = config or MemoryManagerConfig()
        self._mem: Dict[str, List[np.ndarray]] = {}
        self._surf: Dict[str, List[np.ndarray]] = {}
        self._ver: Dict[str, List[Tuple[Any, ...]]] = {}
        self._fb: Dict[str, List[Dict[str, Any]]] = {}
        # M9 lock state (class, in-class index + frozen centroid snapshot)
        self._lock_cls: Optional[str] = None
        self._lock_idx: int = -1
        self._lock_centroid: Optional[np.ndarray] = None

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

    def reset(self) -> None:
        """Reset manager-local state."""
        self.unlock_target()

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
    ) -> None:
        """
        In-place write / EMA-merge of one S-penalty-admitted detection.

        A detection within ``merge_dist_thresh`` of an existing same-class record is merged 
        (EMA centroid, num_obs+1, max confidence kept, near-miss reset = fresh evidence);
        otherwise a new candidate record is registered with a suspicious flag.
        """
        r_xyz = np.zeros(3) if robot_xyz is None else np.asarray(robot_xyz, dtype=np.float64)
        r_xy = r_xyz[:2]
        c_np = np.asarray(centroid, dtype=np.float64)
        s_np = np.asarray(near_surface, dtype=np.float64) if near_surface is not None else c_np

        # Suspicious two-tier: [sigma_tar-adjacent low conf, fb_suspicious_conf).
        if c_np.shape[0] >= 3:
            detect_dist = float(np.linalg.norm(c_np - r_xyz))
        else:
            detect_dist = float(np.linalg.norm(c_np - r_xy))
        suspicious = bool(
            (float(confidence) < self.config.fb_suspicious_conf)
            or (detect_dist > max_depth * 0.95)
        )
        # M4 geometric suspicious override (far / frustum-edge mark from module 1).
        if suspicious_override:
            suspicious = True

        recs = self._mem.get(target_class)
        if not recs:
            self._mem[target_class] = [c_np]
            self._surf[target_class] = [s_np]
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence))]
            self._fb[target_class] = [{"suspicious": suspicious, "near_miss": 0}]
            return

        # Keep the side lists in sync
        if target_class not in self._surf:
            self._surf[target_class] = list(recs)
        if target_class not in self._ver:
            self._ver[target_class] = [(1, r_xy.copy(), float(robot_yaw), float(confidence)) for _ in recs]
        if target_class not in self._fb:
            self._fb[target_class] = [{"suspicious": False, "near_miss": 0} for _ in recs]

        dists = np.linalg.norm(np.asarray(recs, dtype=np.float64) - c_np, axis=1)
        closest_idx = int(np.argmin(dists))
        if dists[closest_idx] < self.config.merge_dist_thresh:
            # Cross-frame consensus = strong evidence; hallucinated boxes drift and cannot merge. 
            # EMA centroid, keep the max c', reset near-miss.
            old = np.asarray(recs[closest_idx], dtype=np.float64)
            recs[closest_idx] = self.config.ema_weight_old * old + (1.0 - self.config.ema_weight_old) * c_np
            num_obs, _, _, old_conf = self._ver[target_class][closest_idx]
            self._ver[target_class][closest_idx] = (
                int(num_obs) + 1, r_xy.copy(), float(robot_yaw),
                max(float(old_conf), float(confidence)),
            )
            self._fb[target_class][closest_idx]["near_miss"] = 0
        else:
            recs.append(c_np)
            self._surf[target_class].append(s_np)
            self._ver[target_class].append((1, r_xy.copy(), float(robot_yaw), float(confidence)))
            self._fb[target_class].append({"suspicious": suspicious, "near_miss": 0})

    # per-step lifecycle  (M6 -> native fallback with M7 exemption)
    def step(
        self,
        camera_pos: np.ndarray,
        camera_yaw: float,
        cone_fov_rad: float,
        obstacle_map_3d: Optional[Any] = None,
        robot_xyz: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """One-step lifecycle over all memory records.

        Order: (1) M6 free-space erasure; (2) native near-field hysteresis fallback
        with the M7 near-field exemption; deleted records are dropped and the M9
        lock is released if it pointed at them.

        Returns per-deletion logs: ``{"target_class", "centroid", "reason",
        "suspicious"}`` with ``reason`` in {"m6_free_erasure", "hysteresis_exceeded"}.
        """
        deleted: List[Dict[str, Any]] = []
        if robot_xyz is None:
            robot_xyz = np.asarray(camera_pos, dtype=np.float64)
        else:
            robot_xyz = np.asarray(robot_xyz, dtype=np.float64)
        r_xy = robot_xyz[:2]

        for target_class in list(self._mem.keys()):
            centroids = self._mem[target_class]
            surfaces = self._surf.get(target_class, [])
            v_states = self._ver.get(target_class, [])
            f_states = self._fb.get(target_class, [])
            if len(f_states) != len(centroids):
                # Defensive: keep fallback state in sync with centroids.
                f_states = [{"suspicious": False, "near_miss": 0} for _ in centroids]
                self._fb[target_class] = f_states

            keep_c: List[np.ndarray] = []
            keep_s: List[np.ndarray] = []
            keep_v: List[Any] = []
            keep_f: List[Dict[str, Any]] = []

            for idx, centroid in enumerate(centroids):
                c_np = np.asarray(centroid, dtype=np.float64)
                v_state = v_states[idx] if idx < len(v_states) else (1, np.zeros(2), 0.0, 1.0)
                num_obs = int(v_state[0])
                is_suspicious = bool(f_states[idx].get("suspicious", False))

                # (1) M6: free-space line-of-sight erasure (single step)
                if self._free_erasure(c_np, obstacle_map_3d, is_suspicious):
                    deleted.append({
                        "target_class": target_class, "centroid": c_np.copy(),
                        "reason": "m6_free_erasure", "suspicious": is_suspicious,
                    })
                    print(
                        f"[TargetMemory][M6] Free-erased {target_class} centroid "
                        f"{np.round(c_np, 3)} (suspicious={is_suspicious}) -> fallback to explore"
                    )
                    continue

                # (2) native near-field fallback (with M7 exemption)
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
                        dist_to_agent = float(np.linalg.norm(c_np[:2] - r_xy))

                        # M7: near-field confirmation exemption -
                        # a target the robot is parked next to must not be deleted for lack of fresh re-detections 
                        # (fresh merges legitimately stop near-field).
                        exempt = False
                        if self.config.enable_mechanism_7:
                            exempt = (
                                (not is_suspicious)
                                and (num_obs >= int(self.config.exempt_min_merges))
                                and (dist_to_agent < float(self.config.exempt_near_radius))
                            )

                        if not exempt:
                            f_states[idx]["near_miss"] += 1
                            thr = (
                                int(self.config.fb_suspicious_hysteresis)
                                if is_suspicious
                                else int(self.config.fb_hysteresis)
                            )
                            if f_states[idx]["near_miss"] >= thr:
                                deleted.append({
                                    "target_class": target_class, "centroid": c_np.copy(),
                                    "reason": "hysteresis_exceeded", "suspicious": is_suspicious,
                                })
                                print(
                                    f"[AntiHallucination] Deleted {target_class} centroid "
                                    f"{np.round(c_np, 3)} (suspicious={is_suspicious}, "
                                    f"near_miss={f_states[idx]['near_miss']}) -> fallback to explore"
                                )
                                continue

                keep_c.append(c_np)
                keep_s.append(surfaces[idx] if idx < len(surfaces) else c_np)
                keep_v.append(v_state)
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

        self._prune_lock()
        return deleted


    # M6: free-space line-of-sight erasure
    def _free_erasure(self, centroid: np.ndarray, obstacle_map_3d: Any, suspicious: bool) -> bool:
        """
        M6: erase a ghost whose centroid neighbourhood was seen as explored & free.

        Uses the accumulated 3D occupancy grid: the box around the centroid must be
        (a) sufficiently explored (the robot actually looked through that space) and
        (b) contain zero occupied voxels with free ratio >= threshold. ``soft_only``
        keeps this to suspicious targets by default because trusted centroids carry
        box error and may float in air above a real object.
        """
        if not self.config.enable_mechanism_6 or obstacle_map_3d is None:
            return False
        if self.config.free_erasure_soft_only and not suspicious:
            return False

        c = np.asarray(centroid, dtype=np.float64)
        if c.ndim != 1 or c.shape[0] < 3:
            return False

        r = float(self.config.free_erasure_radius)
        idx_min = obstacle_map_3d._xyz_to_grid_index((c - r).reshape(1, 3))[0]
        idx_max = obstacle_map_3d._xyz_to_grid_index((c + r).reshape(1, 3))[0]

        size = obstacle_map_3d.size
        h_size = obstacle_map_3d._height_size
        # Target fully outside the grid -> nothing to verify (conservative: keep).
        if not (idx_max[0] >= 0 and idx_min[0] < size and
                idx_max[1] >= 0 and idx_min[1] < size and
                idx_max[2] >= 0 and idx_min[2] < h_size):
            return False

        px_min = max(0, min(int(idx_min[0]), int(idx_max[0])))
        px_max = min(size - 1, max(int(idx_min[0]), int(idx_max[0])))
        py_min = max(0, min(int(idx_min[1]), int(idx_max[1])))
        py_max = min(size - 1, max(int(idx_min[1]), int(idx_max[1])))
        cz_min = max(0, min(int(idx_min[2]), int(idx_max[2])))
        cz_max = min(h_size - 1, max(int(idx_min[2]), int(idx_max[2])))

        occ_sub = obstacle_map_3d._map[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]
        exp_sub = obstacle_map_3d.explored_area[py_min:py_max + 1, px_min:px_max + 1, cz_min:cz_max + 1]
        num_occ = int(np.count_nonzero(occ_sub))
        num_exp = int(np.count_nonzero(exp_sub))

        if num_exp >= int(self.config.free_erasure_min_explored) and num_occ == 0:
            free_ratio = (num_exp - num_occ) / float(num_exp)
            if free_ratio >= float(self.config.free_erasure_free_ratio):
                return True
        return False

    # M9: navigate identity freeze
    def is_locked(self) -> bool:
        return bool(self.config.enable_mechanism_9 and self._lock_cls is not None)

    def unlock_target(self) -> None:
        """Release the M9 lock (target deleted / fallback to explore / episode reset)."""
        self._lock_cls = None
        self._lock_idx = -1
        self._lock_centroid = None

    def _resolve_lock_index(self, recs: List[np.ndarray]) -> Optional[int]:
        """Re-resolve the locked record: same index first, then nearest within merge radius."""
        snap = self._lock_centroid
        if snap is None:
            return None
        if 0 <= self._lock_idx < len(recs):
            cand = np.asarray(recs[self._lock_idx], dtype=np.float64)
            if float(np.linalg.norm(cand - snap)) <= float(self.config.merge_dist_thresh):
                return self._lock_idx
        if len(recs) == 0:
            return None
        dists = [float(np.linalg.norm(np.asarray(c, dtype=np.float64) - snap)) for c in recs]
        best = int(np.argmin(dists))
        return best if dists[best] <= float(self.config.merge_dist_thresh) else None

    def locked_goal(self) -> Optional[np.ndarray]:
        """M9: 2D nav goal of the locked record (identity frozen).

        The record is re-resolved by proximity to the frozen centroid snapshot
        (indexes may shift after deletions / EMA drift). 
        ``None`` when unlocked, disabled, or the locked record was deleted 
        (in which case the lock clears).
        """
        if not self.is_locked():
            return None
        recs = self._mem.get(self._lock_cls)
        if not recs:
            self.unlock_target()
            return None
        idx = self._resolve_lock_index(recs)
        if idx is None:
            self.unlock_target()
            return None
        
        self._lock_idx = idx
        self._lock_centroid = np.asarray(recs[idx], dtype=np.float64).copy()
        surf = self._surf.get(self._lock_cls)
        goal = (
            np.asarray(surf[idx], dtype=np.float64)
            if (surf is not None and idx < len(surf))
            else np.asarray(recs[idx], dtype=np.float64)
        )
        return goal[:2].copy()

    def lock_target(self, target_class: str, index: int) -> None:
        """M9: freeze record ``(target_class, index)`` as the active navigation target.

        No-op when mechanism 9 is disabled or the record is missing.
        """
        if not self.config.enable_mechanism_9:
            return
        recs = self._mem.get(target_class)
        if not recs or not (0 <= index < len(recs)):
            return
        self._lock_cls = target_class
        self._lock_idx = int(index)
        self._lock_centroid = np.asarray(recs[index], dtype=np.float64).copy()

    def locked_identity(self) -> Optional[Tuple[str, int]]:
        """M9: (target_class, in-class index) of the currently locked record."""
        if not self.is_locked():
            return None
        recs = self._mem.get(self._lock_cls)
        if not recs:
            self.unlock_target()
            return None
        idx = self._resolve_lock_index(recs)
        if idx is None:
            self.unlock_target()
            return None
        self._lock_idx = idx
        return (self._lock_cls, idx)

    def _prune_lock(self) -> None:
        """Drop the M9 lock if its record was removed by ``step()``."""
        if self._lock_cls is None:
            return
        recs = self._mem.get(self._lock_cls)
        if not recs or self._resolve_lock_index(recs) is None:
            self.unlock_target()
