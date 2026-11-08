import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from hydra.core.config_store import ConfigStore
try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.tsp3d import TSP3DClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.base_policy import BasePolicy
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import rho_theta, extract_yaw, get_fov
from vlfm.utils.vlvm_stats_logger import VlvmStatsLogger

from vlfm.tsp3d_models.fussion_utils.pipeline import TSP3DInputPreprocessor
from vlfm.vlvm.memory_life.s_penalty import (
    SPenaltyConfig,
    apply_s_penalty,
    compute_w_s,
    near_surface_point,
    query_point_distance,
    query_semantic_at,
)
from vlfm.vlvm.memory_life.target_memory_manager import (
    MemoryManagerConfig,
    ProvisionalGate,
    ProvisionalGateConfig,
    TargetMemoryManager,
)
from vlfm.vlvm.memory_life.target_geometric_gating import (
    TargetGeometricGatingEngine,
    BoxOccupancyConfig,
    apply_geom_gate,
)
from vlfm.vlvm.recheck import ExplRecheckMixin, ExploreRuntimeMixin
from vlfm.vlvm.utils.policy_init import PolicyInitMixin
from vlfm.vlvm.utils.policy_logging import PolicyLoggingMixin
from vlfm.vlvm.utils.three_points import three_point_fields_for


PROMPT_SEPARATOR = "|"

class TSP3DObjectNavPolicy(
    PolicyInitMixin,
    PolicyLoggingMixin,
    ExplRecheckMixin,
    ExploreRuntimeMixin,
    BasePolicy,
):
    """TSP3D-based 3D active semantic target navigation policy."""
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _observations_cache: Dict[str, Any] = {}

    def __init__(
            self,
            pointnav_policy_path: str = "data/pointnav_weights.pth",
            depth_image_shape: Tuple[int, int] = (224, 224),
            fov_angle: float = 79.0,
            camera_height: float = 0.88,
            text_prompt: str = "Seems like there is a target_object ahead.",
            visualize: bool = False,
            init_turn_steps: int = 12,
            min_depth: float = 0.5,
            max_depth: float = 5.0,
            om_style: str = "obstacle",
            voxel_size: float = 0.01,
            min_obstacle_height: float = 0.05,
            max_obstacle_height: float = 1.50,
            agent_radius: float = 0.18,
            nav_slice_height: float = 0.35,
            agent_height: float = 0.88,
            hole_area_thresh: int = 100000,
            obstacle_map_area_threshold: float = 3,
            log_odds_occ: float = 2.0,
            log_odds_free: float = -2.0,
            occ_threshold: float = 0.0,
            free_threshold: float = 0.0,
            use_vlfm_nlp: bool = False,
            pointnav_stop_radius: float = 0.90, 
            det_seed: int = 0,
            det_log_dir: Optional[str] = None,
            enable_stats: bool = True,
            # TSP3D query
            sigma_sce: float = 0.15,
            sigma_tar: float = 0.60,
            sigma_prop: Optional[float] = None,
            tau: float = 0.15,
            near_field_dist: float = 1.0,
            near_field_sigma_scale: float = 0.8,
            fusion_style: str = "world",  # "world" / "none"
            wm_voxel_size: float = 0.02,
            wm_radius: float = 6.0,
            wm_min_view_disp: float = 0.15,
            wm_min_view_yaw: float = 15.0,
            wm_max_voxels: int = 400000,
            wm_max_frames: Optional[int] = 8, 
            wm_near_refresh_radius: Optional[float] = None,
            wm_near_refresh_value: bool = True,
            distance_sample: bool = True,
            near_dist: float = 1.5,
            mid_dist: float = 3.0,
            near_voxel: float = 0.01,
            mid_voxel: float = 0.02,
            far_voxel: float = 0.05,
            fuse_max_points: int = 200000,
            cap_style: str = "random",
            # Mem-life
            enable_s_penalty: bool = True,
            s_penalty_thresh: float = 0.14,
            s_penalty_uncovered_w: float = 1.0,
            s_min: float = 0.07,
            s_query_radius_m: float = 0.5,
            enable_occ_consistency: bool = True,
            occ_min_voxels: int = 8,
            occ_min_ratio: float = 0.001,
            merge_dist_thresh: float = 0.5,
            ema_weight_old: float = 0.8,
            enable_provisional_gate: bool = True,
            provisional_sigma: float = 0.50,
            provisional_min_dist: float = 1.0,
            provisional_max_dist: float = 4.0,
            provisional_s_min: float = 0.07,
            upgrade_min_obs: int = 4,
            stop_gate_confirm_radius: float = 1.2,
            stop_gate_turns: int = 3,
            enable_stable_pool: bool = True,
            merge_new_info_dist: float = 0.2,
            merge_new_info_yaw: float = 10.0,
            merge_new_info_steps: int = 3,
            stable_stale_purge: int = 60,
            stable_miss_delete: int = 5,
            stable_total_miss: int = 8,
            stable_revive_radius: float = 0.8,
            stable_revive_window: int = 20,
            stable_revive_min_sigma: float = 0.50,
            confirmed_to_stable_misses: int = 3,
            provisional_to_stable_misses: int = 3,
            s_penalty_use_surface: bool = True,
            goal_use_surface: bool = True,
            lock_exempt: bool = True,
            lock_exempt_identity: bool = True,
            # GD query
            enable_gd_mechanism: bool = False,
            gd_conf_min: float = 0.40,
            gd_max_dist: float = 5.0,
            gd_only_classes: str = "",
            # Recheck & Explore mode
            explore_stuck_steps: int = 15,
            explore_stuck_move_m: float = 0.05,
            recheck_enable: bool = True,
            recheck_arrive_steps: int = 45,
            explore_auto_mode: bool = True,
            explore_sem_value_min: float = 0.14,
            *args: Any,
            **kwargs: Any,
        ) -> None:
        super().__init__()
        self._depth_image_shape = tuple(depth_image_shape)
        self._fov_angle = fov_angle
        self._camera_height = camera_height
        self._visualize = visualize
        self._init_turn_steps = init_turn_steps
        self._cached_grid_size = None
        self._cached_u_flat = None
        self._cached_v_flat = None
        # Focal length from FOV and camera resolution
        fov_rad = np.deg2rad(fov_angle)
        self._fx = self._fy = depth_image_shape[1] / (2 * np.tan(fov_rad / 2))
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._om_style = om_style
        self._voxel_size = voxel_size
        self._agent_height = agent_height
        self._log_odds_occ = log_odds_occ
        self._log_odds_free = log_odds_free
        self._occ_threshold = occ_threshold
        self._free_threshold = free_threshold
        # Episode / action state.
        self._num_steps = 0
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._did_reset = False
        self._stop_action = torch.tensor([[0]], dtype=torch.long)
        self._turn_left_action = torch.tensor([[2]], dtype=torch.long)
        self._last_tsp3d_local_pcd: Optional[np.ndarray] = None
        self._last_robot_pose: Optional[Tuple[float, float, float]] = None

        # TSP3D query (must run BEFORE `_init_memory_manager`, which reads `_sigma_tar`)
        self._init_tsp3d_query(
            sigma_tar=sigma_tar,
            sigma_prop=sigma_prop,
            sigma_sce=sigma_sce,
            tau=tau,
            s_query_radius_m=s_query_radius_m,
            near_field_dist=near_field_dist,
            near_field_sigma_scale=near_field_sigma_scale,
            use_vlfm_nlp=use_vlfm_nlp,
        )
        # Per-module state. Each init owns the fields of its own mechanism.
        self._init_policy_logging()
        self._init_explore_mode(
            enable=explore_auto_mode,
            sem_value_min=explore_sem_value_min,
            stuck_steps=explore_stuck_steps,
            stuck_move_m=explore_stuck_move_m,
        )
        self._init_recheck(
            enable=recheck_enable,
            arrive_steps=recheck_arrive_steps,
        )
        # Mechanism initialisation
        self._init_spatial_representations(
            om_style=om_style,
            voxel_size=voxel_size,
            min_obstacle_height=min_obstacle_height,
            max_obstacle_height=max_obstacle_height,
            agent_radius=agent_radius,
            nav_slice_height=nav_slice_height,
            agent_height=agent_height,
            hole_area_thresh=hole_area_thresh,
            obstacle_map_area_threshold=obstacle_map_area_threshold,
            log_odds_occ=log_odds_occ,
            log_odds_free=log_odds_free,
            occ_threshold=occ_threshold,
            free_threshold=free_threshold,
        )
        self._init_memory_manager(
            merge_dist_thresh=merge_dist_thresh,
            ema_weight_old=ema_weight_old,
            upgrade_min_obs=upgrade_min_obs,
            enable_stable_pool=enable_stable_pool,
            enable_gd_pool=bool(enable_gd_mechanism),
            enable_provisional_gate=enable_provisional_gate,
            provisional_sigma=provisional_sigma,
            provisional_min_dist=provisional_min_dist,
            provisional_max_dist=provisional_max_dist,
            provisional_s_min=provisional_s_min,
            merge_new_info_dist=merge_new_info_dist,
            merge_new_info_yaw=merge_new_info_yaw,
            merge_new_info_steps=merge_new_info_steps,
            stable_stale_purge=stable_stale_purge,
            stable_miss_delete=stable_miss_delete,
            stable_total_miss=stable_total_miss,
            stable_revive_radius=stable_revive_radius,
            stable_revive_window=stable_revive_window,
            stable_revive_min_sigma=stable_revive_min_sigma,
            confirmed_to_stable_misses=confirmed_to_stable_misses,
            provisional_to_stable_misses=provisional_to_stable_misses,
            lock_exempt=lock_exempt,
            lock_exempt_identity=lock_exempt_identity,
            stop_gate_confirm_radius=stop_gate_confirm_radius,
            stop_gate_turns=stop_gate_turns,
            pointnav_stop_radius=pointnav_stop_radius,
        )
        self._init_fusion(
            fusion_style=fusion_style,
            fuse_max_points=fuse_max_points,
            wm_voxel_size=wm_voxel_size,
            wm_radius=wm_radius,
            wm_min_view_disp=wm_min_view_disp,
            wm_min_view_yaw=wm_min_view_yaw,
            wm_max_voxels=wm_max_voxels,
            wm_max_frames=wm_max_frames,
            wm_near_refresh_radius=wm_near_refresh_radius,
            wm_near_refresh_value=wm_near_refresh_value,
            cap_style=cap_style,
            distance_sample=distance_sample,
            near_dist=near_dist,
            mid_dist=mid_dist,
            near_voxel=near_voxel,
            mid_voxel=mid_voxel,
            far_voxel=far_voxel,
        )
        self._init_s_penalty(
            enable_s_penalty=enable_s_penalty,
            s_penalty_thresh=s_penalty_thresh,
            s_penalty_uncovered_w=s_penalty_uncovered_w,
            s_penalty_use_surface=s_penalty_use_surface,
            goal_use_surface=goal_use_surface,
            s_min=float(s_min),
            s_query_radius_m=float(s_query_radius_m),
            det_log_dir=det_log_dir,
        )
        self._init_geom_gate(
            enable_occ_consistency=enable_occ_consistency,
            occ_min_voxels=occ_min_voxels,
            occ_min_ratio=occ_min_ratio,
        )
        
        # GD mechanism
        self._init_gd_mechanism(
            enabled=enable_gd_mechanism,
            conf_min=gd_conf_min,
            max_dist=gd_max_dist,
            gd_only_classes=gd_only_classes,
        )
        # TSP3D client
        self._tsp3d_client = TSP3DClient(port=int(os.environ.get("TSP3D_PORT", "12186")))
        self._itm_client = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._text_prompt = text_prompt

        self._det_seed = int(det_seed)
        self._rng_step = 0            # monotonic numpy-RNG seed counter (never reset)
        # 通用前置量测器
        self._stats = VlvmStatsLogger(self, enabled=enable_stats)
        
    def _reset(self) -> None:
        """Reset memories, step counters, pointnav model, and 3D occupancy map."""
        self._target_object = ""
        self._num_steps = 0
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._stats.episode_summary()
        self._geom_gate_engine.reset()
        self._pointnav_policy.reset()
        self._obstacle_map3d.reset()
        self._preprocessor.reset()
        self._last_tsp3d_local_pcd = None
        if self._gd_calib is not None:
            self._gd_calib.reset()
        self._gd_calib_logs = []
        self._gd_mech_last_step = -1
        self._gd_mech_error = None
        # Per-module resets
        self._reset_memory_state()
        self._reset_policy_logging()
        self._reset_explore_mode()
        self._recheck_reset()
        self._stats.reset()
        self._did_reset = True

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        """Reset on episode end, cache observations, clear logging dict."""
        assert masks.shape[1] == 1, "Currently only supporting single-environment instances."
        if self._did_reset or masks[0] == 0:
            if not self._did_reset:
                self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(f"Index error while caching observations: {e}")
            raise StopIteration
        self._policy_info = {}

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extract/normalize rgb, depth, extrinsics; implemented by subclasses."""
        raise NotImplementedError

    def _query_semantic_at(self, x: float, y: float, radius_m: Optional[float] = None) -> Union[float, None]:
        """Hook: query the 2.5D semantic-field value S at world (x, y).

        Base implementation delegates to the module default (None = no coverage);
        subclasses override to query the value map."""
        return query_semantic_at(x, y, radius_m)

    def _query_tsp3d_client(
        self,
        aligned_pcd: np.ndarray,
        target_query: str,
        *,
        camera_height: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query TSP3D, scaling the pruning threshold when extremely close to surfaces."""
        if len(aligned_pcd) == 0:
            return [], {}

        camera_height = (
            getattr(self, "_camera_height", 0.88)
            if camera_height is None
            else float(camera_height)
        )
        camera_pos_local = np.array([0.0, 0.0, camera_height])
        pcd_pts = aligned_pcd[:, :3]
        dists = np.linalg.norm(pcd_pts - camera_pos_local, axis=1)
        min_dist = np.min(dists) if len(dists) > 0 else 10.0

        # Adaptive soft-pruning adjustment near obstacle surfaces
        dynamic_sigma_sce = self._sigma_sce
        if min_dist < self._near_field_dist:
            scale = max(self._near_field_sigma_scale, min_dist / self._near_field_dist)
            dynamic_sigma_sce = self._sigma_sce * scale

        raw_preds, diagnostics = self._tsp3d_client.predict(
            pcd=aligned_pcd,
            text=target_query,
            # The wire API retains the historical name ``sigma_tar``; at the
            # policy boundary this value has proposal-only semantics.
            sigma_tar=self._sigma_prop,
            sigma_sce=dynamic_sigma_sce,
            tau=self._tau,
            use_vlfm_nlp=self._nlp_mode,
        )

        return raw_preds, diagnostics

    def _project_rgbd_to_3d_point_cloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        fx: float,
        fy: float,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> np.ndarray:
        """Project RGB-D into a colored world-frame point cloud (N, 6)."""
        H, W = depth.shape[:2]

        if self._cached_grid_size != (H, W) or self._cached_grid_size is None:
            self._cached_grid_size = (H, W)
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            self._cached_u_flat = u.flatten()
            self._cached_v_flat = v.flatten()
        depth_flat = depth.flatten()
        valid = (depth_flat > 0.0) & (depth_flat < 1.0)

        if not np.any(valid):
            return np.empty((0, 6))

        u_v = self._cached_u_flat[valid]
        v_v = self._cached_v_flat[valid]
        d_v = depth_flat[valid] * (max_depth - min_depth) + min_depth

        # Back-project pixels to 3D camera coordinates
        xc = (u_v - W / 2.0) * d_v / fx
        yc = (v_v - H / 2.0) * d_v / fy
        zc = d_v
        # Camera frame: [forward, right, down] → episodic base frame: [forward, left, up]
        pts_base = np.stack([zc, -xc, -yc], axis=1)
        # Transform 3D coordinates from camera frame to episodic world frame
        R_T = tf_camera_to_episodic[:3, :3].T
        t = tf_camera_to_episodic[:3, 3]
        pts_w = pts_base @ R_T + t
        colors = rgb.reshape(-1, 3)[valid] / 255.0

        return np.hstack([pts_w, colors])

    def _get_shared_pcd(self, i: int) -> np.ndarray:
        """Return the i-th shared back-projected world-frame cloud (N, 6)."""
        shared_pcds = self._observations_cache.get("shared_pcds")
        if shared_pcds is not None and i < len(shared_pcds):
            return shared_pcds[i]
        rgb, depth, tf, min_depth, max_depth, fx, fy = self._observations_cache["object_map_rgbd"][i]
        return self._project_rgbd_to_3d_point_cloud(
            rgb, depth, fx, fy, tf, min_depth, max_depth
        )

    @staticmethod
    def _sigma_band(score: float) -> int:
        """band-pass order: 0.65-0.70 > 0.60-0.65 > 0.55-0.60 > >=0.70."""
        if 0.65 <= score < 0.70:
            return 0
        if 0.60 <= score < 0.65:
            return 1
        if 0.55 <= score < 0.60:
            return 2
        if score >= 0.70:
            return 3
        return 4

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """
        Navigation goal among the memorized target candidates.

        Current four-tier selection:
          gd-tsp (including promoted gd-only) > gd-only > confirmed > provisional.
        Internal keys: gd-tsp by sigma band -> gd_hits -> distance ->
        num_obs/freshness; gd-only by gd_hits -> distance -> num_obs/freshness;
        confirmed/provisional by sigma band -> distance -> num_obs/freshness.

        Field / region_semantic shadow is intentionally NOT read here: those
        diagnostics are log-only and must not change the nav goal or STOP.
        """
        valid_centroids: List[np.ndarray] = []
        valid_goals: List[np.ndarray] = []
        valid_obs: List[int] = []
        valid_memory_ids: List[Optional[str]] = []
        valid_candidate_ids: List[Optional[str]] = []
        valid_scores: List[float] = []
        valid_stop_ok: List[bool] = []
        valid_stale: List[int] = []
        valid_bands: List[int] = []
        valid_states: List[str] = []
        valid_origins: List[Optional[str]] = []
        valid_classes: List[str] = []
        valid_gd_hits: List[int] = []
        valid_gd_only: List[bool] = []
        valid_gd_tsp: List[bool] = []
        valid_gd_incremental: List[bool] = []
        valid_gd_upgraded: List[bool] = []
        valid_promoted_gd_only: List[bool] = []
        target_classes = self._target_object.split("|")
        for cls in target_classes:
            if cls in self._target_3d_memory and len(self._target_3d_memory[cls]) > 0:
                surf_list = self._target_surface_memory.get(cls, [])
                verify_list = self._target_verify_state.get(cls, [])
                for i, centroid in enumerate(self._target_3d_memory[cls]):
                    obs = 1
                    if i < len(verify_list):
                        v = verify_list[i]
                        obs = int(v[0]) if len(v) >= 1 else 1

                    st = self._memory_manager.state_of(centroid)
                    state_name = str(st.get("lifecycle_state") or "confirmed")
                    if state_name == "stable":
                        continue
                    if st.get("memory_id") in self._stop_blacklist:
                        continue

                    score = (
                        float(v[3]) if i < len(verify_list) and len(v) > 3 else 0.0
                    )
                    last_obs = int(st.get("last_obs_step", self._num_steps))
                    current_frame = max(0, int(self._memory_manager.frame) - 1)
                    stale = max(0, current_frame - last_obs)

                    valid_centroids.append(np.array(centroid))
                    if self._goal_use_surface and i < len(surf_list):
                        valid_goals.append(np.array(surf_list[i]))
                    else:
                        valid_goals.append(np.array(centroid))
                    valid_obs.append(obs)
                    valid_memory_ids.append(st.get("memory_id"))
                    valid_candidate_ids.append(st.get("last_candidate_id"))
                    valid_scores.append(score)
                    valid_stale.append(int(stale))
                    valid_bands.append(self._sigma_band(score))
                    valid_states.append(state_name)
                    origin = st.get("origin")
                    valid_origins.append(str(origin) if origin is not None else None)
                    valid_classes.append(str(cls))
                    valid_gd_hits.append(int(st.get("gd_hits", 0) or 0))
                    valid_gd_only.append(bool(st.get("gd_only", False)))
                    promoted_gd_only = bool(
                        st.get("promoted_from_gd_only", False)
                    )
                    is_gd_tsp = (
                        str(origin or "") == "gd-tsp" or promoted_gd_only
                    )
                    valid_gd_tsp.append(is_gd_tsp)
                    gd_upgraded = bool(st.get("gd_upgraded", False))
                    valid_gd_upgraded.append(gd_upgraded)
                    valid_promoted_gd_only.append(promoted_gd_only)
                    valid_gd_incremental.append(
                        gd_upgraded or promoted_gd_only
                    )
                    valid_stop_ok.append(
                        state_name == "confirmed" and is_gd_tsp
                    )
        if len(valid_centroids) == 0:
            self._last_stop_eligible = False
            self._last_goal_memory_id = None
            return None

        robot_xy = np.asarray(position)[:2]
        centroids = np.array(valid_centroids)
        dists_2d = np.linalg.norm(centroids[:, :2] - robot_xy, axis=1)

        def selection_tier(i: int) -> int:
            if bool(valid_gd_tsp[i]):
                return 0
            if bool(valid_gd_only[i]) or valid_states[i] == "gd":
                return 1
            if valid_states[i] == "confirmed":
                return 2
            if valid_states[i] == "provisional":
                return 3
            return 4

        def sigma_band_key(i: int) -> int:
            t = selection_tier(i)
            return int(valid_bands[i]) if t in (0, 2, 3) else 0

        def gd_hits_key(i: int) -> int:
            t = selection_tier(i)
            return -int(valid_gd_hits[i]) if t in (0, 1) else 0

        def distance_bucket(i: int) -> int:
            return int(round(float(dists_2d[i]) / 0.5))

        def num_obs_key(i: int) -> int:
            return 0 if int(valid_obs[i]) >= 3 else 1

        tiers = [selection_tier(i) for i in range(len(valid_centroids))]
        chosen_idx = min(
            range(len(valid_centroids)),
            key=lambda i: (
                int(tiers[i]),
                sigma_band_key(i),
                gd_hits_key(i),
                distance_bucket(i),
                num_obs_key(i),
                -int(valid_obs[i]),
                int(valid_stale[i]),
            ),
        )

        self._last_stop_eligible = bool(valid_stop_ok[chosen_idx])
        self._last_goal_memory_id = valid_memory_ids[chosen_idx]
        self._last_goal_num_obs = int(valid_obs[chosen_idx])
        self._last_goal_state = valid_states[chosen_idx]
        self._last_goal_stale = int(valid_stale[chosen_idx])
        self._last_goal_origin = valid_origins[chosen_idx]
        self._last_goal_gd_hits = int(valid_gd_hits[chosen_idx])
        self._last_goal_score = float(valid_scores[chosen_idx])
        chosen_2d = valid_goals[chosen_idx][:2].copy()
        self._last_target_centroid = np.asarray(
            valid_centroids[chosen_idx], dtype=np.float64
        ).reshape(-1)[:2].copy()

        valid_payload = {
            "centroids": valid_centroids,
            "obs": valid_obs,
            "memory_ids": valid_memory_ids,
            "candidate_ids": valid_candidate_ids,
            "scores": valid_scores,
            "bands": valid_bands,
            "states": valid_states,
            "origins": valid_origins,
            "gd_only": valid_gd_only,
            "gd_tsp": valid_gd_tsp,
            "gd_incremental": valid_gd_incremental,
            "gd_upgraded": valid_gd_upgraded,
            "promoted_gd_only": valid_promoted_gd_only,
            "gd_hits": valid_gd_hits,
            "stale": valid_stale,
            "stop_ok": valid_stop_ok,
        }
        
        switch_reason = "initial"
        if self._last_target_coord is None:
            self._last_target_coord = chosen_2d
        else:
            # distance-latch: keep the current goal unless the switch is meaningful.
            delta_dist = np.linalg.norm(chosen_2d - self._last_target_coord)
            dist_to_new = float(np.linalg.norm(valid_centroids[chosen_idx][:2] - robot_xy))
            if delta_dist < 0.1:
                switch_reason = "held_near"
            elif delta_dist < 0.5 and dist_to_new > 2.0:
                switch_reason = "held_far"
            else:
                self._last_target_coord = chosen_2d
                switch_reason = "switched"

        self._log_nav_goal(
            chosen_idx=chosen_idx,
            goal=self._last_target_coord,
            robot_xy=robot_xy,
            dists_2d=dists_2d,
            switch_reason=switch_reason,
            tiers=tiers,
            valid=valid_payload,
        )
        self._stats.note_pick(valid_centroids, valid_obs, chosen_idx, position)

        return self._last_target_coord

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        pcd: Union[np.ndarray, None] = None,
    ) -> ObjectDetections:
        """Update occupancy map, build fused PCD, query TSP3D, filter & memorize detections."""
        if pcd is None:
            pcd = self._project_rgbd_to_3d_point_cloud(
                rgb, depth, fx, fy, tf_camera_to_episodic, min_depth, max_depth
            )
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        robot_yaw = self._observations_cache.get("robot_heading", 0.0)

        # Update 3D geometric obstacle occupancy grid for navigation.
        self._obstacle_map3d.update_map(
            pcd=pcd,
            tf_camera_to_episodic=tf_camera_to_episodic,
            depth=depth,
            min_depth=min_depth,
            max_depth=max_depth,
            fx=fx,
            fy=fy,
        )

        # Unified input pipeline: world-frame incremental map -> update -> prepare.
        self._preprocessor.update(pcd, robot_xyz, robot_yaw)
        fused_pcd_local = self._preprocessor.prepare(
            robot_xyz,
            robot_yaw,
            camera_height=self._camera_height,
        )

        detections = self._query_and_process(
            fused_pcd_local,
            robot_xyz,
            robot_yaw,
            pcd,
            rgb,
            tf_camera_to_episodic,
            max_depth,
            fx,
            fy,
        )

        if self._gd_calib is not None and self._done_initializing:
            try:
                record = self._gd_calib.observe(
                    step=int(self._num_steps),
                    rgb=rgb,
                    depth_norm=depth,
                    tf_camera_to_episodic=tf_camera_to_episodic,
                    min_depth=float(min_depth),
                    max_depth=float(max_depth),
                    fx=float(fx),
                    fy=float(fy),
                    target_classes=[
                        c.strip()
                        for c in str(self._target_object).split("|")
                        if c.strip()
                    ],
                    robot_xyz=robot_xyz,
                    robot_yaw=float(robot_yaw),
                    fov_deg=float(self._fov_angle),
                    candidates=[
                        row for row in self._candidate_logs
                        if int(row.get("step", -1)) == int(self._num_steps)
                    ],
                    memory_events=[
                        row for row in self._memory_event_logs
                        if int(row.get("step", -1)) == int(self._num_steps)
                    ],
                    tracks=self._track_snapshot(),
                    done_initializing=bool(self._done_initializing),
                )
                if record is not None:
                    if self._gd_mech is not None and int(self._gd_mech_last_step) != int(self._num_steps):
                        try:
                            mechanism_record = self._gd_mech.process(
                                record,
                                pcd=pcd,
                                robot_xyz=robot_xyz,
                                robot_yaw=float(robot_yaw),
                                target_classes=[
                                    c.strip()
                                    for c in str(self._target_object).split("|")
                                    if c.strip()
                                ],
                                step=int(self._num_steps),
                            )
                            record["mechanism"] = mechanism_record
                            events = mechanism_record.get("events", []) or []
                            self._log_memory_events(events)
                            self._gd_mech_last_step = int(self._num_steps)
                        except BaseException as exc:  # noqa: BLE001 - GD must not kill nav
                            if self._gd_mech_error is None:
                                self._gd_mech_error = f"{type(exc).__name__}: {exc}"
                                print(
                                    f"[GD-MECH] disabled after error: {self._gd_mech_error}",
                                    flush=True,
                                )
                    self._gd_calib_logs.append(record)
            except BaseException as exc:  # noqa: BLE001 - calibration must not kill nav
                if self._gd_calib_error is None:
                    self._gd_calib_error = f"{type(exc).__name__}: {exc}"
                    print(f"[GD-CALIB] disabled after error: {self._gd_calib_error}", flush=True)

        return detections

    def _query_and_process(
        self,
        pcd_local: np.ndarray,
        robot_xyz: np.ndarray,
        robot_yaw: float,
        pcd_world: np.ndarray,
        image_rgb: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Query TSP3D on a local cloud, restore boxes to world frame, 
        then run the full detection pipeline (filter -> S-penalty -> memory lifecycle).
        """
        self._last_tsp3d_local_pcd = np.asarray(
            pcd_local, dtype=np.float64
        ).copy()
        target_classes = [c.strip() for c in self._target_object.split("|") if c.strip()]
        self._last_robot_pose = (float(robot_xyz[0]), float(robot_xyz[1]), float(robot_yaw))
        frame_log_start = len(self._memory_event_logs)
        raw_detections, _ = self._query_tsp3d_client(pcd_local, self._target_object)
        # Restore predicted boxes to global coordinates (inverse rotation + translation)
        valid_raw = [det for det in raw_detections if det.get("box_3d") is not None]
        cos_yaw_r, sin_yaw_r = np.cos(robot_yaw), np.sin(robot_yaw)
        boxes_3d_global = []
        for det in valid_raw:
            box_np = np.array(det["box_3d"])  # shape (8, 3) corners
            # Step 1: rotate back by +yaw around Z
            bx, by = box_np[:, 0].copy(), box_np[:, 1].copy()
            box_np[:, 0] = cos_yaw_r * bx - sin_yaw_r * by
            box_np[:, 1] = sin_yaw_r * bx + cos_yaw_r * by
            # Step 2: add world XY translation
            box_np[:, :2] += robot_xyz[:2]
            boxes_3d_global.append(box_np.tolist())
        logits = [det.get("confidence", 0.0) for det in valid_raw]
        phrases = [self._target_object for _ in valid_raw]

        detections = ObjectDetections(
            boxes=boxes_3d_global,
            logits=logits,
            phrases=phrases,
            pcd_source=pcd_world,
            image_source=image_rgb,
            fx=fx,
            fy=fy,
            tf_camera_to_episodic=tf_camera_to_episodic
        )
        detections.filter_by_conf(self._sigma_prop)
        detections.filter_by_class(target_classes, use_vlfm_nlp=self._nlp_mode)
        if not self._done_initializing:
            return detections

        candidate_records: List[Dict[str, Any]] = []
        for centroid, conf, box in zip(
            detections.centroids, detections.logits, detections.boxes
        ):
            centroid_np = centroid.detach().cpu().numpy()
            box_np = box.detach().cpu().numpy()
            conf_f = float(conf.detach().cpu().item())
            candidate_id = f"c{self._candidate_seq:06d}"
            self._candidate_seq += 1
            s_pt = centroid_np
            s_pt_source = "centroid"
            if self._s_penalty_use_surface:
                near_pt = near_surface_point(box_np, robot_xyz)
                if near_pt is not None:
                    s_pt = near_pt
                    s_pt_source = "surface"
            s_value = self._query_semantic_at(
                float(s_pt[0]), float(s_pt[1]), self._s_query_radius_m
            )
            w_s, _ = compute_w_s(s_value, self._s_penalty_cfg)
            if s_value is None or float(s_value) <= 0.0:
                branch = "uncovered"
            elif float(s_value) >= float(self._s_penalty_cfg.thresh):
                branch = "covered-high"
            else:
                branch = "covered-low"
            candidate_records.append(
                {
                    "candidate_id": candidate_id,
                    "step": int(self._num_steps),
                    "target_class": target_classes[0] if target_classes else "unknown",
                    "conf": conf_f,
                    "pass_prop_045": bool(conf_f >= 0.45),
                    "pass_prop_050": bool(conf_f >= 0.50),
                    "pass_prop_055": bool(conf_f >= 0.55),
                    "pass_prop_060": bool(conf_f >= 0.60),
                    "sigma_prop_floor": float(self._sigma_prop),
                    "sigma_tar_admit": float(self._sigma_tar),
                    "box": box_np.tolist(),
                    "centroid": centroid_np.tolist(),
                    "surface": np.asarray(s_pt, dtype=np.float64).reshape(-1)[:3].tolist(),
                    "distance": query_point_distance(s_pt, robot_xyz),
                    "s": s_value,
                    "s_pt": np.asarray(s_pt, dtype=np.float64).reshape(-1)[:3].tolist(),
                    "s_pt_source": s_pt_source,
                    "w_s": float(w_s),
                    "conf_amp": conf_f * float(w_s),
                    "sigma_c": conf_f * float(w_s),
                    "branch": branch,
                    "s_stage_seen": False,
                    "admitted": False,
                    "reject": None,
                }
            )
        self._log_candidates(candidate_records)

        # REJECTED (empty-air) -> dropped before S-penalty
        post_occ_records: List[Dict[str, Any]] = []
        if self._geom_gate_engine.enabled:
            detections = apply_geom_gate(
                detections, pcd_world,
                engine=self._geom_gate_engine,
                obstacle_map_3d=self._obstacle_map3d,
                step=self._num_steps,
                robot_xyz=robot_xyz,
                per_det_meta=candidate_records,
                meta_out=post_occ_records,
            )
            for record in candidate_records:
                if not record.get("occ_passed", False):
                    record["reject"] = "occ"
        else:
            post_occ_records = list(candidate_records)
            for record in post_occ_records:
                record.update(
                    {
                        "occ_passed": True,
                        "occ_status": "gate_off",
                        "occ_reason": "gate_off",
                        "geom_status": "gate_off",
                    }
                )
        # S-penalty gate cross-validation -> write passed detections to memory.
        gate_enabled = self._provisional_gate.config.enabled
        admitted_meta: Optional[List[Any]] = None if gate_enabled else []
        s_penalty_pending = apply_s_penalty(
            detections, target_classes, robot_xyz,
            query_semantic=self._query_semantic_at,
            cfg=self._s_penalty_cfg,
            sigma_tar=self._sigma_tar,
            sigma_prop=self._sigma_prop,
            use_surface=self._s_penalty_use_surface,
            goal_use_surface=self._goal_use_surface,
            nlp_mode=self._nlp_mode,
            out_log=self._detect_logs,
            per_det_meta=post_occ_records,
            meta_out=admitted_meta,
            step=self._num_steps,
            candidate_records=post_occ_records,
        )
        pending: List[Tuple[float, np.ndarray, Optional[np.ndarray], List[str]]] = []
        if gate_enabled:
            for centroid, box, candidate_meta in zip(
                detections.centroids, detections.boxes, post_occ_records
            ):
                centroid_np = centroid.detach().cpu().numpy()
                box_np = box.detach().cpu().numpy() if torch.is_tensor(box) else np.asarray(box)
                sigma_c = float(candidate_meta.get("sigma_c", candidate_meta.get("conf_amp", 0.0)))
                active_classes = target_classes if self._nlp_mode else ([target_classes[0]] if target_classes else [])
                near_surface = None
                if self._goal_use_surface:
                    near_surface = near_surface_point(box_np, robot_xyz)
                gate_fields = self._provisional_gate.describe(candidate_meta)
                gate_state = gate_fields["provisional_gate_state"]
                candidate_meta.update(gate_fields)
                wrote = False
                merged = False
                lifecycle_states: List[str] = []
                for cls in active_classes:
                    candidate_distance = float(
                        candidate_meta.get("distance") or 0.0
                    )
                    candidate_meta["depth_ok"] = bool(
                        candidate_distance <= float(self._max_depth) * 0.95
                    )
                    res = self._memory_manager.accumulate(
                        cls, centroid_np, confidence=sigma_c, sigma_c=sigma_c,
                        near_surface=near_surface, robot_xyz=robot_xyz,
                        robot_yaw=robot_yaw,
                        candidate_id=candidate_meta.get("candidate_id"),
                        new_lifecycle_state=gate_state,
                        blocked_memory_ids=self._stop_blacklist,
                    )
                    candidate_meta["gd_only_promotion_gate"] = res.get(
                        "promotion_gate_reason"
                    )
                    if not res.get("written", False):
                        continue
                    wrote = True
                    merged = merged or bool(res.get("merged", False))
                    lifecycle_states.append(str(res.get("lifecycle_state") or "confirmed"))
                    self._log_accumulate_event(
                        res=res,
                        cls=cls,
                        candidate_meta=candidate_meta,
                        centroid=centroid_np,
                        surface=near_surface,
                        robot_xyz=robot_xyz,
                        robot_yaw=robot_yaw,
                        sigma_c=sigma_c,
                        detailed=True,
                        three_points=three_point_fields_for(
                            self._memory_manager, centroid_np
                        ),
                    )
                    upgraded = res.get("upgraded")
                    if upgraded:
                        self._log_upgrade_event(
                            res=res,
                            upgraded=upgraded,
                            cls=cls,
                            candidate_meta=candidate_meta,
                            centroid=centroid_np,
                            surface=near_surface,
                            robot_xyz=robot_xyz,
                            robot_yaw=robot_yaw,
                            sigma_c=sigma_c,
                        )
                candidate_meta["memory_written"] = wrote
                candidate_meta["memory_merged"] = merged
                candidate_meta["memory_lifecycle_states"] = lifecycle_states
                candidate_meta["admitted"] = wrote
                candidate_meta["reject"] = (
                    None if wrote else "provisional_gate"
                )
                candidate_meta["provisional_gate_outcome"] = (
                    "merge_existing" if merged else (gate_state or "rejected")
                )
                if wrote:
                    pending.append((sigma_c, centroid_np, near_surface, active_classes))

            by_candidate = {
                row.get("candidate_id"): row for row in post_occ_records
                if row.get("candidate_id") is not None
            }
            for row in self._detect_logs:
                source = by_candidate.get(row.get("candidate_id"))
                if source is not None:
                    row.update({key: source.get(key) for key in (
                        "admitted", "reject", "memory_written", "memory_merged",
                        "memory_lifecycle_states", "provisional_gate_enabled",
                        "provisional_gate_state", "provisional_gate_passed",
                        "provisional_gate_outcome",
                        "provisional_sigma", "provisional_min_dist", "provisional_max_dist", "provisional_s_min",
                    )})
        else:
            pending = list(s_penalty_pending)
            for (conf_amp, centroid_np, near_surface, active_classes), candidate_meta in zip(
                pending, admitted_meta or []
            ):
                for cls in active_classes:
                    res = self._memory_manager.accumulate(
                        cls, centroid_np, confidence=conf_amp, near_surface=near_surface,
                        robot_xyz=robot_xyz, robot_yaw=robot_yaw,
                        candidate_id=(candidate_meta or {}).get("candidate_id"),
                        new_lifecycle_state="confirmed",
                        sigma_c=conf_amp,
                        blocked_memory_ids=self._stop_blacklist,
                    )
                    self._log_accumulate_event(
                        res=res,
                        cls=cls,
                        candidate_meta=candidate_meta,
                        centroid=centroid_np,
                        surface=near_surface,
                        robot_xyz=robot_xyz,
                        robot_yaw=robot_yaw,
                        sigma_c=float(conf_amp),
                        detailed=False,
                        three_points=three_point_fields_for(
                            self._memory_manager, centroid_np
                        ),
                    )

        self._stats.note_frame_admitted(pending)

        # Target Memory Manager
        camera_pos = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)
        cone_fov = get_fov(self._fx, self._depth_image_shape[1])
        # `lock_exempt`: in navigate mode the nav-goal entry is exempt from the near_miss budget.
        _nav_goal = self._last_target_coord if int(self._lock_step) >= 0 else None
        # `lock_exempt_identity`: match on the pursued entry's centroid instead of the nav goal (0.5 m).
        _exempt_ref = _nav_goal
        if self._lock_exempt_identity:
            _pxy_ref = self._last_target_centroid
            if _pxy_ref is not None:
                _exempt_ref = _pxy_ref
        lifecycle_events = self._memory_manager.step(
            camera_pos, camera_yaw, cone_fov,
            robot_xyz=robot_xyz,
            nav_goal_xy=_exempt_ref,
            exempt_nav_goal=bool(self._lock_exempt),
        )
        self._log_lifecycle_events(lifecycle_events)

        for record in self._memory_manager.try_upgrade(robot_xyz):
            self._log_try_upgrade_record(
                record, robot_xyz=robot_xyz, robot_yaw=robot_yaw
            )

        self._log_track_frames(frame_log_start, robot_xyz, robot_yaw, max_depth)

        return detections

    def _initialize(self) -> Tensor:
        raise NotImplementedError
    
    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """PointNav to goal via rho/theta; supports 2D (x,y) and 3D (x,y,z) goals."""
        device = next(self._pointnav_policy.policy.parameters()).device
        masks = torch.tensor([[self._num_steps != 0]], dtype=torch.bool, device=device)

        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.5:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)

            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho = max(1e-4, rho)
        rho_theta_tensor = torch.tensor([[rho, theta]], device=device, dtype=torch.float32)

        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }

        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._stats.note_stop(allowed=True)
            self._called_stop = True
            return self._stop_action

        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        if not stop:
            action = action.clamp(min=1)
        return action

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        np.random.seed((self._det_seed + self._rng_step) & 0x7FFFFFFF)
        self._rng_step += 1

        self._pre_step(observations, masks)
        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = []
        for i, (rgb, depth, tf, min_depth, max_depth, fx, fy) in enumerate(object_map_rgbd):
            pcd = self._get_shared_pcd(i)
            detections.append(
                self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy, pcd=pcd)
            )
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        goal_3d = self._get_target_object_location(robot_xyz)
        # Exploration via habitat frontier_sensor.
        if not self._done_initializing:
            mode = "initialize"
            action = self._initialize()
        elif goal_3d is None:
            # No lockable entry: drop the lock bookkeeping and keep exploring.
            self._lock_step = -1
            mode = "explore"
            action = self._explore_safe(observations)
        else:
            if self._lock_step < 0:
                self._lock_step = self._num_steps
            mode = "navigate"
            if self._recheck_active:
                self._recheck_active = False
            print(f"[TSP3D Mode] Target '{self._target_object}' located at {goal_3d}. Navigating.")
            self._stats.note_approach(goal_3d, robot_xyz)
            
            stop_ok = bool(self._last_stop_eligible)
            robot_xy2d = np.asarray(
                self._observations_cache.get("robot_xy", robot_xyz[:2]),
                dtype=np.float64,
            ).reshape(-1)[:2]
            heading = float(self._observations_cache.get("robot_heading", 0.0))
            rho, _ = rho_theta(robot_xy2d, heading, goal_3d[:2])
            if stop_ok:
                self._stop_confirm_counts.pop(str(self._last_goal_memory_id), None)
                self._lookaround_trace = []
                action = self._pointnav(goal_3d[:2], stop=True)
            else:
                goal_id = str(self._last_goal_memory_id)
                if rho <= self._stop_gate_radius:
                    turns = self._stop_confirm_counts.get(goal_id, 0) + 1
                    self._stop_confirm_counts[goal_id] = turns
                    # Log-only: one row per look-around frame, so offline analysis
                    # can tell whether extra views ever moved sigma_c at all.
                    self._lookaround_trace.append(
                        {
                            "turn": int(turns),
                            "dist": round(float(rho), 3),
                            "sigma_c": round(
                                float(getattr(self, "_last_goal_score", 0.0) or 0.0), 4
                            ),
                            "gd_hits": int(
                                getattr(self, "_last_goal_gd_hits", 0) or 0
                            ),
                            "state": str(self._last_goal_state),
                            "origin": (
                                str(self._last_goal_origin)
                                if self._last_goal_origin
                                else None
                            ),
                        }
                    )
                    if turns > self._stop_gate_turns:
                        self._stop_blacklist.add(
                            str(self._last_goal_memory_id)
                        )
                        print(
                            "[BLACKLIST] add stop blacklist: "
                            f"memory_id={self._last_goal_memory_id} "
                            f"step={int(self._num_steps)} "
                            f"near_turns={int(turns)} "
                            f"reason=stop_gate_exhausted",
                            flush=True,
                        )
                        _blacklist_state = {}
                        if self._last_target_centroid is not None:
                            _blacklist_state = self._memory_manager.state_of(
                                np.asarray(self._last_target_centroid, dtype=np.float64)
                            )
                        _blacklist_centroid = (
                            np.asarray(self._last_target_centroid, dtype=np.float64).reshape(-1)[:3].tolist()
                            if self._last_target_centroid is not None
                            else None
                        )
                        self._log_stop_blacklist_event(
                            state=_blacklist_state,
                            centroid=_blacklist_centroid,
                            turns=int(turns),
                            distance=float(rho),
                            lookaround_trace=list(self._lookaround_trace),
                            sigma_c=float(
                                getattr(self, "_last_goal_score", 0.0) or 0.0
                            ),
                        )
                        self._lookaround_trace = []
                        self._stop_confirm_counts.pop(goal_id, None)
                        self._lock_step = -1
                        self._last_target_coord = None
                        self._last_target_centroid = None
                        mode = "explore"
                        action = self._explore_safe(observations)
                    else:
                        self._stats.note_stop(allowed=False)
                        action = self._turn_left_action
                else:
                    action = self._pointnav(goal_3d[:2], stop=False)

        action_np = action.detach().cpu().numpy()[0]
        if len(action_np) == 1:
            action_np = action_np[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_np}")

        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1
        self._observations_cache = {}
        self._did_reset = False

        return action, rnn_hidden_states


    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        has_target = any(cls in self._target_3d_memory 
                        for cls in self._target_object.split("|"))

        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache.get("robot_xy_z", np.zeros(3))),
            "yaw": np.rad2deg(self._observations_cache.get("robot_heading", 0.0)),
            "target_detected": has_target,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            "explore_exhausted": bool(getattr(self, "_explore_exhausted", False)),
            "explore_stop_suppressed": int(
                getattr(self, "_explore_stop_suppressed", 0)
            ),
            "recheck_active": bool(getattr(self, "_recheck_active", False)),
            "recheck_index": int(getattr(self, "_recheck_idx", 0)),
            "recheck_total_steps": int(
                getattr(self, "_recheck_total_steps", 0)
            ),
            "render_below_images": ["target_object"],
        }

        if not self._visualize:
            return policy_info

        annotated_rgb = (
        detections.annotated_frame
        if detections.annotated_frame is not None
        else self._observations_cache["object_map_rgbd"][0][0])
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["obstacle_map"] = self._obstacle_map3d.visualize_slice()

        return policy_info


@dataclass
class VLVMConfig:
    name: str = "VLVMPolicy"

    # Phase 1: System & Base Camera Configuration
    pointnav_policy_path: str = "data/pointnav_weights.pth"  # Path to the pretrained PointNav weights.
    depth_image_shape: Tuple[int, int] = (224, 224)          # (H, W) resolution fed to the PointNav policy; larger keeps more detail but is slower.
    fov_angle: float = 79.0        # Camera horizontal FOV (deg); larger widens the view, smaller zooms in (used to derive the focal length).
    camera_height: float = 0.88    # Camera height above ground (m); must match the sensor pose for correct 3D projection.
    text_prompt: str = "Seems like there is a target_object ahead."  # Language prompt used for visual grounding.
    visualize: bool = False        # Enable visualization/debug rendering; True slows down inference.
    init_turn_steps: int = 12      # Number of in-place turns at episode start to scan the scene; higher = wider initial view but slower reset.
    # Phase 2: Depth Sensor Filtering
    min_depth: float = 0.5   # Minimum valid depth (m); points closer than this are discarded.
    max_depth: float = 5.0   # Maximum valid depth (m); larger sees farther but adds far-field noise/clutter.
    # Phase 6a: Navigation Execution & Termination
    pointnav_stop_radius: float = 0.90  # Final S5 stop distance (m).

    # Phase 3a: 3D Mapping & Occupancy Grid
    om_style: str = "obstacle"
    voxel_size: float = 0.01                 # Voxel grid resolution (m); smaller = finer map but more memory/compute.
    min_obstacle_height: float = 0.15        # Lower height bound (m) for obstacle voxels; too low includes floor noise, too high misses low obstacles.
    max_obstacle_height: float = 1.50        # Upper height bound (m) for obstacle voxels; too low ignores tall obstacles.
    hole_area_thresh: int = 100000           # Hole area threshold (px) for depth gap filling; -1 disables filling (assume max depth).
    # Phase 3b: ProbabilisticGrid log-odds inverse-sensor model (om_style=probabilistic only)
    log_odds_occ: float = 2.0                # Occupied endpoint evidence weight (default symmetric: |free|==occ)
    log_odds_free: float = -2.0              # Free ray-body evidence weight; |free| < occ -> conservative bias (offset not cancelled to 0)
    occ_threshold: float = 0.0               # Occupied decision threshold (l > occ_thr); >0 -> hysteresis buffer (Unknown band)
    free_threshold: float = 0.0              # Free decision threshold (l < free_thr); <0 -> hysteresis buffer (Unknown band)
    obstacle_map_area_threshold: float = 1.5 # Frontier area threshold (m^2) to filter small isolated regions; higher = fewer, larger frontiers.
    ## Dilation / structuring element 
    agent_radius: float = 0.18               # Robot physical radius (m) used for collision inflation; larger = more conservative navigation.
    agent_height: float = 0.88               # Robot height (m) for the 3D cylinder structuring element (dilation / collision).
    nav_slice_height: float = 0.35           # Reference navigation height (m) for frontier slicing and visualization.

    # Phase 4: 2.5D Semantic Value Plane (BEV)
    vm_style: str = "region"          # Semantic value mapping mode: "region" (V1) / "surface" (V2)
    h_lam: float = 0.3                # Bonus weight lambda (lambda=0 reduces to VLFM baseline)
    h_norm_max: float = 1.0           # Upper bound for normalizing H (locks the value score range)
    h_z_min: float = 0.15             # Lower bound of the H1 passable band (m)
    h_z_max: float = 0.88             # Upper bound of the H1 passable band (m, robot height)
    query_radius_m: float = 0.5       # Horizontal query radius r_h for route 2 (includes dilation semantics)
    query_z_min: float = 0.15         # Lower bound of the fixed query height band (m, surface landing)
    query_z_max: float = 1.50         # Upper bound of the fixed query height band (m)

    # Phase 5a: TSP3D Perception & Visual Grounding
    sigma_tar: float = 0.60             # P0/P1 fixed raw/admission threshold (c_base); no longer controls service proposal supply when sigma_prop is set.
    sigma_prop: Optional[float] = None   # TSP3D service return threshold; None inherits sigma_tar for exact legacy behaviour.
    sigma_sce: float = 0.15             # TGP scene voxel retention threshold; higher = more aggressive pruning (fewer hallucinations, may miss targets).
    tau: float = 0.15                   # Soft-pruning temperature; higher = smoother/looser pruning, lower = harder thresholding.
    near_field_dist: float = 1.0        # Distance (m) below which near-field adaptive sigma scaling activates; larger = scaling kicks in earlier.
    near_field_sigma_scale: float = 0.8 # Minimum voxel retention ratio near surfaces; higher = keep more voxels near obstacles.
    use_vlfm_nlp: bool = False          # Use raw NLP prompt formatting; True = multi-class synonym merging, False = use only the primary class.
    cap_style: str = "random"           # send-side point cap (shared): "random" (baseline) / "near_first" (near-field priority)
    det_seed: int = 0                   # base of the per-step numpy reseed
    distance_sample: bool = True        # distance-adaptive sampling (dense near / sparse far, shared post-processing)
    near_dist: float = 1.5              # distance sampling: near/mid band boundary (m)
    mid_dist: float = 3.0               # distance sampling: mid/far band boundary (m)
    near_voxel: float = 0.01            # distance sampling: near-band voxel (m)
    mid_voxel: float = 0.02             # distance sampling: mid-band voxel (m)
    far_voxel: float = 0.05             # distance sampling: far-band voxel (m)
    fusion_style: str = "world"         # Fusion route: "world" (incremental map) / "none" (raw current frame).
    fuse_max_points: int = 200000           # Cap on fused point count; higher = more detail but heavier sparse-conv inference.
    # Phase 5b: World-frame Local Map (TSP3DInputPreprocessor)
    wm_max_frames: Optional[int] = 8                # world: frame-window cap (keep voxels of the most recent N frames); None = all history
    wm_voxel_size: float = 0.02                     # world: local map voxel size (m) for fixed-grid dedup
    wm_max_voxels: int = 400000                     # world: hard cap on map voxels
    wm_radius: Optional[float] = 6.0                # world: local map radius (m), None = no crop (only max_voxels hard cap); cam_radius=None is the symmetric case
    wm_min_view_disp: float = 0.15                  # world: min displacement (m) to accept a frame into fusion
    wm_min_view_yaw: float = 15.0                   # world: min yaw change (deg) to accept a frame into fusion
    wm_near_refresh_radius: Optional[float] = 3.0   # world: near-field refresh radius (m). None=off (first-observation accounting); >0: re-observed near-field voxels are re-owned by the current frame (cam-style accounting, survives frame-window slide-out).
    wm_near_refresh_value: bool = False             # world: also overwrite the stored point of refreshed near-field voxels with the current observation (cam-style sliding refresh of the value layer). False = value layer stays first.

    # Phase 7: Mem-Life
    enable_occ_consistency: bool = True     # box must have occupied-voxel support in the 3D grid
    occ_min_voxels: int = 8                 # occ gate: min occupied voxels supporting the box
    occ_min_ratio: float = 0.001            # occ gate: min occupied/(box volume) ratio

    enable_s_penalty: bool = True           # Master switch: c' = c * w_S(S) — multiply TSP3D confidence by a semantic-field weight (BLIP2 ITM, zero extra queries) to amplify TP/FP discrimination.
    s_penalty_thresh: float = 0.14          # Covered w_S=min(1,max(S,s_min)/thresh).
    s_penalty_uncovered_w: float = 1.0      # weight when S=None/<=0 (no coverage): 1.0 = free pass (A1), <1.0 = mild penalty on unverifiable detections.
    s_min: float = 0.07                     # S lower clamp used by w_S (was a module constant).
    s_query_radius_m: float = 0.5           # Radius for the S query on the value map (was a module constant).

    merge_dist_thresh: float = 0.5          # cross-view EMA merge distance threshold (m)
    ema_weight_old: float = 0.8             # cross-view EMA smoothing (old * w + new * (1 - w))
    enable_provisional_gate: bool = True    # Final S5: unmatched candidates use the provisional gate.
    provisional_sigma: float = 0.50         # Provisional admission line.
    provisional_min_dist: float = 1.0       # Provisional new-memory distance floor (m).
    provisional_max_dist: float = 4.0       # Provisional new-memory distance ceiling (m).
    provisional_s_min: float = 0.07         # Logging only; S does not gate provisional entries.
    upgrade_min_obs: int = 4                 # provisional -> confirmed: new-info observations.
    stop_gate_confirm_radius: float = 1.2    # Arrival radius that starts the look-around.
    stop_gate_turns: int = 3                 # Look-around frames before dropping the track.
    enable_stable_pool: bool = True          # `stable` is the single deletion pool; stable is always hidden from selection.
    merge_new_info_dist: float = 0.2         # Pose baseline that counts as a new observation (m).
    merge_new_info_yaw: float = 10.0         # Yaw delta (deg) that counts as a new observation.
    merge_new_info_steps: int = 3            # Step gap that counts as a new observation.
    stable_stale_purge: int = 60             # Frames since the last match -> delete from `stable`.
    stable_miss_delete: int = 5              # Consecutive near-field misses inside `stable`.
    stable_total_miss: int = 8               # Cumulative near-field misses (a revive does not reset it).
    stable_revive_radius: float = 0.8        # Revive radius (m).
    stable_revive_window: int = 20           # Revive window (frames since the last match).
    stable_revive_min_sigma: float = 0.50    # Revive candidate sigma_c floor.
    confirmed_to_stable_misses: int = 3      # confirmed -> stable near-field miss budget.
    provisional_to_stable_misses: int = 3    # provisional -> stable near-field miss budget.
    explore_stuck_steps: int = 15  # Consecutive near-zero movements before treating explore as stuck.
    explore_stuck_move_m: float = 0.05  # Movement below this is considered no progress.
    explore_action_repeat_steps: int = 8  # Same action + unchanged BLIP cosine before replacing the frontier.
    explore_action_cosine_eps: float = 1e-4  # BLIP cosine equality tolerance for repeat detection.
    explore_frontier_repeat_steps: int = 8  # Same selected frontier before replacing it.
    lock_exempt: bool = True             # Lock hold: while locked in navigate, the nav-goal entry is exempt from the near-field near_miss budget.
    lock_exempt_identity: bool = True    # Exemption reference: True = pursued entry centroid (identity); False = nav-goal near-surface point.
    s_penalty_use_surface: bool = True      # query S at the bbox near-surface point (facing camera) instead of the centroid.
    goal_use_surface: bool = True           # navigate to the stored near-surface point (first-write fixed) instead of the centroid.
    det_log_dir: Optional[str] = None       # SP1 per-detection jsonl directory; None = off (logging only, no behaviour change).

    enable_gd_mechanism: bool = False        # Enable GD-only/gd-tsp routing and writes.
    gd_conf_min: float = 0.40                # GD base-gate confidence (initial filter only).
    gd_max_dist: float = 5.0                 # GD base-gate robot distance (m).
    gd_only_classes: str = ""                # empty = all target classes; comma-separated otherwise.

    # Recheck & Explore mode.
    recheck_enable: bool = True             # Master switch for the recheck queue.
    recheck_arrive_steps: int = 45          # Per-waypoint arrival budget (fixed); measured need p100 = 39, tail 42-48.
    explore_auto_mode: bool = True          # Master switch; False keeps the pure value-map ordering.
    explore_sem_value_min: float = 0.14     # Pure ValueMap S of the best frontier (excludes h_lam*H1).

    enable_stats: bool = True     # 通用前置量测开关（[NAV]/[STOP] 打印与逐集摘要；零行为）

cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
