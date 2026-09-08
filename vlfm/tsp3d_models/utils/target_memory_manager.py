from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from vlfm.utils.geometry_utils import within_fov_cone


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
    
    # native anti-hallucination fallback (baseline mechanism)
    enable_fallback: bool = True          # native near-field hysteresis fallback master switch
    fb_near_radius: float = 2.5           # near-field FOV-cone radius (m); == max_depth * 0.5
    fb_hysteresis: int = 5                # trusted target: consecutive near-field miss frames before deletion
    fb_suspicious_hysteresis: int = 2     # suspicious target: consecutive near-field miss frames before deletion
    fb_suspicious_conf: float = 0.75      # trusted/suspicious confidence boundary (must be > sigma_tar)

    # near-field confirmation exemption (default OFF = pure baseline)
    enable_near_field_exempt: bool = False  # keep a trusted target parked next to the robot from being deleted
    exempt_near_radius: float = 1.0         # robot-target distance below which a trusted target is exempt (m)
    exempt_min_merges: int = 1              # min merges of a trusted target to be exempt

    # free-space line-of-sight erasure (default OFF = pure baseline)
    enable_free_space_erasure: bool = False  # single-step ghost removal via the occupancy grid
    free_erasure_soft_only: bool = True      # True: erase suspicious targets only (conservative)
    free_erasure_radius: float = 0.3      # half-side (m) of the box around the centroid to inspect
    free_erasure_min_explored: int = 5    # min explored voxels inside the box to trust "this space was seen"
    free_erasure_free_ratio: float = 0.85 # free-voxel ratio above which the box is judged "pure air"


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
    ) -> None:
        """
        In-place write / EMA-merge of one admitted detection.
        Merges within merge_dist_thresh of a same-class record (EMA centroid, num_obs+1,
        max confidence kept, near-miss reset); otherwise registers a new candidate.
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

    # per-step lifecycle
    def step(
        self,
        camera_pos: np.ndarray,
        camera_yaw: float,
        cone_fov_rad: float,
        obstacle_map_3d: Optional[Any] = None,
        robot_xyz: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """One-step lifecycle over all memory records.

        Order: (1) free-space erasure; (2) native near-field hysteresis fallback
        with the near-field exemption.

        Returns per-deletion logs with reason in {"free_space_erasure", "hysteresis_exceeded"}.
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

                # (1) free-space line-of-sight erasure (single step)
                if self._free_erasure(c_np, obstacle_map_3d, is_suspicious):
                    deleted.append({
                        "target_class": target_class, "centroid": c_np.copy(),
                        "reason": "free_space_erasure", "suspicious": is_suspicious,
                    })
                    print(
                        f"[TargetMemory][Erasure] Free-erased {target_class} centroid "
                        f"{np.round(c_np, 3)} (suspicious={is_suspicious}) -> fallback to explore"
                    )
                    continue

                # (2) native near-field fallback (with exemption)
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

                        exempt = False
                        if self.config.enable_near_field_exempt:
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

        return deleted


    # free-space line-of-sight erasure
    def _free_erasure(self, centroid: np.ndarray, obstacle_map_3d: Any, suspicious: bool) -> bool:
        """Erase a ghost whose centroid neighbourhood was explored and is free.
        Uses the accumulated occupancy grid; soft_only keeps it to suspicious
        targets (trusted centroids carry box error and may float above an object).
        """
        if not self.config.enable_free_space_erasure or obstacle_map_3d is None:
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
