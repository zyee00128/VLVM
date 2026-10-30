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
from vlfm.mapping.obstacle_map import ObstacleMap3D, ProbabilisticGrid

from vlfm.tsp3d_models.fussion_utils.pipeline import TSP3DInputPreprocessor
from vlfm.tsp3d_models.fussion_utils.s_penalty import (
    S_QUERY_RADIUS_M,
    SPenaltyConfig, 
    apply_s_penalty, 
    compute_w_s,
    near_surface_point,
    query_point_distance,
    query_semantic_at,
)
from vlfm.tsp3d_models.fussion_utils.target_memory_manager import (
    MemoryManagerConfig,
    ProvisionalGate,
    ProvisionalGateConfig,
    TargetMemoryManager,
)
from vlfm.utils.vlvm_stats_logger import VlvmStatsLogger
from vlfm.semantic_spatial_field.middle_tier import (
    MiddleTierFeatureAssembler,
    MiddleTierRanker,
    MiddleTierShadowLogger,
)
from vlfm.tsp3d_models.fussion_utils.target_geometric_gating import (
    TargetGeometricGatingEngine,
    BoxOccupancyConfig,
    apply_geom_gate,
)


PROMPT_SEPARATOR = "|"


class TSP3DObjectNavPolicy(BasePolicy):
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
            sigma_sce: float = 0.15,
            sigma_tar: float = 0.60,
            sigma_prop: Optional[float] = None,
            tau: float = 0.15,
            near_field_dist: float = 1.0,
            near_field_sigma_scale: float = 0.8,
            use_vlfm_nlp: bool = False,
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
            det_seed: int = 0,
            pointnav_stop_radius: float = 0.90,
            enable_s_penalty: bool = True,
            s_penalty_thresh: float = 0.14,
            s_penalty_uncovered_w: float = 1.0,
            det_log_dir: Optional[str] = None,
            # middle-tier promotion-probability shadow / ranker
            enable_middle_tier_shadow: bool = True,
            middle_tier_shadow_every_n: int = 1,
            middle_tier_model_path: str = "",
            enable_middle_tier_ranking: bool = False,
            # geometric admission module
            enable_occ_consistency: bool = True,
            occ_min_voxels: int = 8,
            occ_min_ratio: float = 0.001,
            # target-memory manager
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
            enable_gd_mechanism: bool = False,
            gd_conf_min: float = 0.40,
            gd_max_dist: float = 5.0,
            gd_only_classes: str = "",
            enable_stats: bool = True,
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
        # Keep proposal supply separate from policy admission.  ``sigma_tar``
        # remains the locked P0/P1 raw/admission line; ``sigma_prop`` only
        # controls what the TSP3D service returns.  None preserves the legacy
        # one-threshold behaviour exactly.
        self._sigma_tar = float(sigma_tar)
        self._sigma_prop = (
            self._sigma_tar if sigma_prop is None else float(sigma_prop)
        )
        if not 0.0 <= self._sigma_prop <= 1.0:
            raise ValueError(f"sigma_prop must be in [0, 1], got {self._sigma_prop}")
        self._sigma_sce = sigma_sce
        self._tau = tau
        self._enable_stable_pool = bool(enable_stable_pool)
        self._stop_gate_radius = float(stop_gate_confirm_radius)
        self._stop_gate_turns = int(stop_gate_turns)
        self._stop_blacklist: set = set()
        self._stop_confirm_counts: Dict[str, int] = {}
        self._last_stop_eligible: bool = True
        self._last_goal_memory_id: Optional[str] = None
        self._last_goal_num_obs: int = 0
        self._last_goal_state: str = "confirmed"
        self._last_goal_stale: int = 0
        self._last_goal_origin: Optional[str] = None
        self._last_goal_gd_hits: int = 0
        self._near_field_dist = near_field_dist
        self._near_field_sigma_scale = near_field_sigma_scale
        self._nlp_mode = use_vlfm_nlp
        self._pointnav_stop_radius = pointnav_stop_radius
        self._init_step_count = 0
        self._num_steps = 0
        self._last_goal = np.zeros(2)  # Current 3D goal coordinate [x, y, z]
        self._done_initializing = False
        self._called_stop = False
        self._did_reset = False
        self._stop_action = torch.tensor([[0]], dtype=torch.long)
        self._turn_left_action = torch.tensor([[2]], dtype=torch.long)
        self._look_up_action = torch.tensor([[4]], dtype=torch.long)
        self._look_down_action = torch.tensor([[5]], dtype=torch.long)
        self._target_3d_memory: Dict[str, List[np.ndarray]] = {}
        self._target_surface_memory: Dict[str, List[np.ndarray]] = {}  # per-centroid near-surface point
        # (num_obs, last robot xy, last robot yaw, c' = S-penalty score)
        self._target_verify_state: Dict[str, List[Tuple[int, np.ndarray, float, float]]] = {}
        self._target_lifecycle_state: Dict[str, List[Dict[str, Any]]] = {}
        self._last_target_coord: Union[None, np.ndarray] = None
        self._last_target_centroid: Union[None, np.ndarray] = None
        self._candidate_seq = 0
        self._candidate_logs: List[Dict[str, Any]] = []
        self._memory_event_logs: List[Dict[str, Any]] = []
        self._nav_goal_logs: List[Dict[str, Any]] = []
        self._track_frame_logs: List[Dict[str, Any]] = []
        self._stop_shadow_logs: List[Dict[str, Any]] = []
        self._middle_rank_shadow_logs: List[Dict[str, Any]] = []
        self._middle_tier_shadow = None
        self._middle_tier_shadow_error: Optional[str] = None
        # Optional raw observation side-channel. It is never read by act's
        # decision path and is enabled only with det_log_dir.
        self._field_observation_logger = None
        self._field_log_error: Optional[str] = None
        # Shadow-only field exploration coverage.  It has its own region grid
        # and is called after the action has been selected; no field value is
        # read by target selection or STOP.
        self._field_shadow = None
        self._field_shadow_logs: List[Dict[str, Any]] = []
        self._field_shadow_error: Optional[str] = None
        self._last_robot_pose: Optional[Tuple[float, float, float]] = None

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
        self._enable_gd_mechanism = bool(enable_gd_mechanism)
        self._gd_conf_min = float(gd_conf_min)
        self._gd_max_dist = float(gd_max_dist)
        self._gd_only_classes = tuple(
            part.strip().lower() for part in str(gd_only_classes).split(",") if part.strip()
        )
        self._init_memory_manager(
            merge_dist_thresh=merge_dist_thresh,
            ema_weight_old=ema_weight_old,
            upgrade_min_obs=upgrade_min_obs,
            enable_stable_pool=enable_stable_pool,
            enable_gd_pool=self._enable_gd_mechanism,
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
            det_log_dir=det_log_dir,
        )
        self._enable_middle_tier_ranking = bool(enable_middle_tier_ranking)
        self._middle_tier_model_path = str(middle_tier_model_path or "")
        self._middle_tier_shadow_every_n = int(middle_tier_shadow_every_n)
        try:
            self._middle_tier_shadow = MiddleTierShadowLogger(
                enabled=bool(enable_middle_tier_shadow),
                assembler=MiddleTierFeatureAssembler(),
                ranker=MiddleTierRanker(
                    model_path=str(middle_tier_model_path or ""),
                    enabled=True,
                ),
                every_n=int(middle_tier_shadow_every_n),
            )
        except Exception as exc:  # noqa: BLE001 - shadow must not kill nav
            self._middle_tier_shadow = None
            self._middle_tier_shadow_error = f"{type(exc).__name__}: {exc}"
        self._provisional_gate = ProvisionalGate(ProvisionalGateConfig(
            enabled=enable_provisional_gate,
            confirmed_sigma=self._sigma_tar,
            provisional_sigma=provisional_sigma,
            provisional_min_dist=provisional_min_dist,
            provisional_max_dist=provisional_max_dist,
            s_min=provisional_s_min,
        ))
        self._init_geom_gate(
            enable_occ_consistency=enable_occ_consistency,
            occ_min_voxels=occ_min_voxels,
            occ_min_ratio=occ_min_ratio,
        )
        # 通用前置量测器
        self._stats = VlvmStatsLogger(self, enabled=enable_stats)

        # GD calibration logger (log-only; never feeds a navigation decision).
        # Enabled by environment variables so the Hydra/YAML surface stays fixed.
        self._gd_calib = None
        self._gd_calib_logs: List[Dict[str, Any]] = []
        self._gd_calib_error: Optional[str] = None
        self._gd_mech = None
        self._gd_mech_error: Optional[str] = None
        self._gd_mech_last_step: int = -1
        _gd_requested = self._enable_gd_mechanism or (
            os.environ.get("GD_CALIB_ENABLE", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if _gd_requested:
            try:
                from vlfm.tsp3d_models.grounding_dino.calibration import (
                    GdCalibConfig,
                    GdCalibrationLogger,
                )

                _gd_cfg = GdCalibConfig.from_env()
                if self._enable_gd_mechanism:
                    _gd_cfg.enabled = True
                if _gd_cfg.enabled:
                    self._gd_calib = GdCalibrationLogger(_gd_cfg)
                    print(
                        "[GD-CALIB] enabled "
                        f"every_n={_gd_cfg.every_n} record_every_n={_gd_cfg.record_every_n} "
                        f"box_thr={_gd_cfg.box_thr} max_boxes={_gd_cfg.max_boxes} "
                        f"mechanism={self._enable_gd_mechanism}",
                        flush=True,
                    )
                if self._enable_gd_mechanism:
                    from vlfm.tsp3d_models.grounding_dino.mechanism import (
                        GdMechanism,
                        GdMechanismConfig,
                    )

                    self._gd_mech = GdMechanism(
                        memory_manager=self._memory_manager,
                        geom_gate=self._geom_gate_engine,
                        obstacle_map3d=self._obstacle_map3d,
                        config=GdMechanismConfig(
                            enabled=True,
                            conf_min=float(self._gd_conf_min),
                            max_dist_m=float(self._gd_max_dist),
                            gd_only_classes=tuple(self._gd_only_classes),
                        ),
                    )
                    print(
                        "[GD-MECH] enabled "
                        f"conf_min={self._gd_conf_min} "
                        f"max_dist={self._gd_max_dist} classes={self._gd_only_classes or 'ALL'} "
                        "depth=gd-only-required/gd-tsp-track-merge no-conf-gate no-ttl",
                        flush=True,
                    )
            except BaseException as exc:  # noqa: BLE001 - GD must not kill nav
                self._gd_calib_error = f"{type(exc).__name__}: {exc}"
                self._gd_mech_error = self._gd_calib_error
                print(f"[GD-MECH] init failed: {self._gd_calib_error}", flush=True)

        # 3D visual grounding and vision-language evaluation clients
        self._tsp3d_client = TSP3DClient(port=int(os.environ.get("TSP3D_PORT", "12186")))
        self._itm_client = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._text_prompt = text_prompt

        self._det_seed = int(det_seed)
        self._rng_step = 0            # monotonic numpy-RNG seed counter (never reset)

    def _reset(self) -> None:
        """Reset memories, step counters, pointnav model, and 3D occupancy map."""
        self._target_object = ""
        self._init_step_count = 0
        self._num_steps = 0
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._target_3d_memory.clear()
        self._target_surface_memory.clear()
        self._target_verify_state.clear()
        self._target_lifecycle_state.clear()
        self._stats.episode_summary()
        self._memory_manager.reset_ids()
        self._last_target_coord = None
        self._last_target_centroid = None
        self._geom_gate_engine.reset()
        self._last_lock_step = 0
        self._pointnav_policy.reset()
        self._obstacle_map3d.reset()
        self._preprocessor.reset()
        self._detect_logs = []
        self._candidate_seq = 0
        self._candidate_logs = []
        self._memory_event_logs = []
        self._nav_goal_logs = []
        self._track_frame_logs = []
        self._stop_shadow_logs = []
        self._middle_rank_shadow_logs = []
        self._field_shadow_logs = []
        self._field_shadow_error = None
        middle_shadow = getattr(self, "_middle_tier_shadow", None)
        if middle_shadow is not None:
            try:
                middle_shadow.reset()
            except Exception:
                pass
        shadow = getattr(self, "_field_shadow", None)
        if shadow is not None:
            try:
                shadow.reset()
            except Exception:
                pass
        self._gd_calib_logs = []
        self._gd_mech_last_step = -1
        self._gd_mech_error = None
        logger = getattr(self, "_field_observation_logger", None)
        if logger is not None:
            try:
                logger.abort()
            except Exception:
                pass
        self._field_observation_logger = None
        self._field_log_error = None
        if self._gd_calib is not None:
            self._gd_calib.reset()
        self._stop_blacklist = set()
        self._stop_confirm_counts = {}
        self._last_stop_eligible = True
        self._last_goal_memory_id = None
        self._last_goal_num_obs = 0
        self._last_goal_state = "confirmed"
        self._last_goal_stale = 0
        self._last_goal_origin = None
        self._last_goal_gd_hits = 0
        self._lock_step = -1
        self._stats.reset()
        self._did_reset = True

    def _log_field_shadow_frame(self, *, mode: Optional[str] = None) -> None:
        """Append one read-only field exploration row after action selection.

        The shadow owns its own incremental region grid.  It never mutates
        policy memory, nav goals, STOP state or the chosen action; exceptions
        are captured and do not interrupt navigation.
        """
        if not getattr(self, "_det_log_dir", None):
            return
        enabled = os.environ.get("FIELD_SHADOW_ENABLE", "0").strip().lower()
        if enabled not in ("1", "true", "yes", "on"):
            return
        try:
            if self._field_shadow is None:
                from vlfm.semantic_spatial_field.shadow import (
                    FieldExplorationShadow,
                    FieldShadowConfig,
                )

                self._field_shadow = FieldExplorationShadow(
                    FieldShadowConfig(
                        enabled=True,
                        config_path=os.environ.get("FIELD_SHADOW_CONFIG") or None,
                        cell_size_m=float(os.environ.get("FIELD_SHADOW_CELL_SIZE_M", "1.0")),
                        top_k=int(os.environ.get("FIELD_SHADOW_TOP_K", "5")),
                    )
                )
            cache = self._observations_cache
            robot_xyz = np.asarray(cache.get("robot_xy_z", np.zeros(3)))
            robot_xy = np.asarray(cache.get("robot_xy", robot_xyz[:2]))
            candidate_points: List[np.ndarray] = []
            for cls in [c.strip() for c in str(self._target_object).split("|") if c.strip()]:
                for centroid in self._target_3d_memory.get(cls, []) or []:
                    point = np.asarray(centroid, dtype=np.float64).reshape(-1)
                    if point.size >= 2 and np.all(np.isfinite(point[:2])):
                        candidate_points.append(point[:2].copy())
            nav_goal = self._last_target_coord
            row = self._field_shadow.observe(
                step=int(self._num_steps),
                robot_xy=robot_xy,
                frontiers_3d=cache.get("frontier_sensor_3d", np.empty((0, 3))),
                frontiers_2d=cache.get("frontier_sensor", np.empty((0, 2))),
                candidate_points=candidate_points,
                nav_goal_xy=None if nav_goal is None else np.asarray(nav_goal),
                mode=mode,
            )
            if row is not None:
                self._field_shadow_logs.append(row)
        except Exception as exc:  # noqa: BLE001 - shadow must never kill nav
            self._field_shadow_error = f"{type(exc).__name__}: {exc}"

    def _log_field_observation_frame(self) -> None:
        """Record offline inputs without invoking the field or changing actions."""
        if not getattr(self, "_det_log_dir", None):
            return
        try:
            from vlfm.semantic_spatial_field.observation_logger import FieldObservationLogger

            cache = self._observations_cache
            rgbd_views = cache.get("object_map_rgbd", ())
            if not rgbd_views:
                return
            if self._field_observation_logger is None:
                om = self._obstacle_map3d
                map_spec = {
                    "map_kind": self._om_style,
                    "size_pixels": int(om.size),
                    "pixels_per_meter": float(om.pixels_per_meter),
                    "episode_pixel_origin_initial": np.asarray(om._episode_pixel_origin),
                    "voxel_size_m": float(om._voxel_size),
                    "height_size_voxels": int(om._height_size),
                    "min_height_m": float(om._min_height),
                    "max_height_m": float(om._max_height),
                    **getattr(self, "_field_map_parameters", {}),
                    "agent_radius_m": float(om._agent_radius),
                    "agent_height_m": float(om._agent_height),
                    "nav_slice_height_m": float(om._nav_slice_height),
                    "hole_area_thresh": int(om._hole_area_thresh),
                    "depth_image_shape": list(self._depth_image_shape),
                    "camera_height_m": float(self._camera_height),
                    "fov_horizontal_deg": float(self._fov_angle),
                }
                self._field_observation_logger = FieldObservationLogger(
                    self._det_log_dir, map_spec
                )
            frontiers_3d = np.asarray(cache.get("frontier_sensor_3d", np.empty((0, 3))))
            frontiers_2d = np.asarray(cache.get("frontier_sensor", np.empty((0, 2))))
            robot_xyz = np.asarray(cache.get("robot_xy_z", np.zeros(3)))
            robot_xy = np.asarray(cache.get("robot_xy", robot_xyz[:2]))
            robot_yaw = float(cache.get("robot_heading", 0.0))
            for view_idx, (rgb, depth, tf, min_depth, max_depth, fx, fy) in enumerate(rgbd_views):
                self._field_observation_logger.append_frame(
                    step=int(self._num_steps),
                    view_index=int(view_idx),
                    target_object=str(self._target_object),
                    rgb=rgb,
                    depth=depth,
                    camera_to_episodic=tf,
                    depth_range_intrinsics=(min_depth, max_depth, fx, fy),
                    robot_xyz=robot_xyz,
                    robot_xy=robot_xy,
                    robot_yaw=robot_yaw,
                    frontiers_3d=frontiers_3d,
                    frontiers_2d=frontiers_2d,
                    pixel_origin=self._obstacle_map3d._episode_pixel_origin,
                )
        except Exception as exc:  # logging must not interrupt navigation
            self._field_log_error = f"{type(exc).__name__}: {exc}"

    def _finalize_field_observations(
        self, *, scene_id: str, episode_id: int, target_object: str, infos: Dict[str, Any]
    ) -> Optional[str]:
        logger = getattr(self, "_field_observation_logger", None)
        if logger is None:
            return None
        try:
            path = logger.finalize(
                scene_id=scene_id,
                episode_id=episode_id,
                target_object=target_object,
                infos=infos,
            )
            self._field_observation_logger = None
            return path
        except Exception as exc:  # diagnostics must not interrupt episode accounting
            self._field_log_error = f"{type(exc).__name__}: {exc}"
            try:
                logger.abort()
            except Exception:
                pass
            self._field_observation_logger = None
            return None
    
    # ==========================================================================
    # Mechanism initialisation (one `_init_*` per module)
    # 3D spatial representations / target-memory lifecycle /
    # fusion / S-penalty / geometric admission
    # ==========================================================================
    def _init_spatial_representations(
        self,
        *,
        om_style: str,
        voxel_size: float,
        min_obstacle_height: float,
        max_obstacle_height: float,
        agent_radius: float,
        nav_slice_height: float,
        agent_height: float,
        hole_area_thresh: int,
        obstacle_map_area_threshold: float,
        log_odds_occ: float,
        log_odds_free: float,
        occ_threshold: float,
        free_threshold: float,
    ) -> None:
        """Core 3D spatial representations: the occupancy / obstacle map backend."""
        self._field_map_parameters = {
            "min_obstacle_height_m": float(min_obstacle_height),
            "max_obstacle_height_m": float(max_obstacle_height),
            "obstacle_area_threshold_m2": float(obstacle_map_area_threshold),
            "log_odds_occ": float(log_odds_occ),
            "log_odds_free": float(log_odds_free),
            "occ_threshold": float(occ_threshold),
            "free_threshold": float(free_threshold),
        }
        height_range = max_obstacle_height - min_obstacle_height
        height_size = int(height_range / voxel_size) + 1
        pixels_per_meter = int(1.0 / voxel_size)
        size = 400
        if self._om_style == "obstacle":
            self._obstacle_map3d = ObstacleMap3D(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                agent_radius=agent_radius,
                area_thresh=obstacle_map_area_threshold,
                hole_area_thresh=hole_area_thresh,
                size=size,
                pixels_per_meter=pixels_per_meter,
                voxel_size=voxel_size,
                height_size=height_size,
                nav_slice_height=nav_slice_height,
                agent_height=agent_height,
                compute_navigable=self._visualize,
            )
        elif self._om_style == "probabilistic":
            self._obstacle_map3d = ProbabilisticGrid(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                agent_radius=agent_radius,
                area_thresh=obstacle_map_area_threshold,
                hole_area_thresh=hole_area_thresh,
                size=size,
                pixels_per_meter=pixels_per_meter,
                voxel_size=voxel_size,
                height_size=height_size,
                nav_slice_height=nav_slice_height,
                agent_height=agent_height,
                compute_navigable=self._visualize,
                log_odds_occ=self._log_odds_occ,
                log_odds_free=self._log_odds_free,
                occ_threshold=self._occ_threshold,
                free_threshold=self._free_threshold,
            )

    def _init_memory_manager(
        self,
        *,
        merge_dist_thresh: float,
        ema_weight_old: float,
        upgrade_min_obs: int = 4,
        enable_stable_pool: bool = True,
        enable_gd_pool: bool = False,
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
        lock_exempt: bool = True,
        lock_exempt_identity: bool = True,
    ) -> None:
        """
        Target-memory lifecycle manager (EMA merge + stable pool + erasure)
        bound to the policy's memory dicts.
        """
        self._memory_manager = TargetMemoryManager(MemoryManagerConfig(
            near_field_radius=float(self._max_depth) * 0.5,
            merge_dist_thresh=merge_dist_thresh,
            ema_weight_old=ema_weight_old,
            upgrade_min_obs=upgrade_min_obs,
            enable_stable_pool=enable_stable_pool,
            enable_gd_pool=enable_gd_pool,
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
        ))
        self._memory_manager.bind(
            self._target_3d_memory, self._target_surface_memory,
            self._target_verify_state, self._target_lifecycle_state,
        )
        self._lock_exempt = bool(lock_exempt)
        self._lock_exempt_identity = bool(lock_exempt_identity)
        self._lock_step: int = -1

    def _init_fusion(
        self,
        *,
        fusion_style: str,
        fuse_max_points: int,
        wm_voxel_size: float,
        wm_radius: Optional[float],
        wm_min_view_disp: float,
        wm_min_view_yaw: float,
        wm_max_voxels: int,
        wm_max_frames: Optional[int],
        wm_near_refresh_radius: Optional[float],
        wm_near_refresh_value: bool,
        cap_style: str,
        distance_sample: bool,
        near_dist: float,
        mid_dist: float,
        near_voxel: float,
        mid_voxel: float,
        far_voxel: float,
    ) -> None:
        """Fusion route: world-frame local map or no fusion (raw current frame)."""
        if fusion_style not in ("world", "none"):
            raise ValueError("fusion_style MUST BE 'world' or 'none'")
        self._fusion_style = fusion_style
        self._last_lock_step: int = 0
        self._preprocessor = TSP3DInputPreprocessor(
            fusion_style=fusion_style,
            map_voxel_size=wm_voxel_size,
            map_radius=wm_radius,
            min_view_disp=wm_min_view_disp,
            min_view_yaw=np.deg2rad(wm_min_view_yaw),
            max_map_voxels=wm_max_voxels,
            map_max_frames=wm_max_frames,
            map_near_refresh_radius=wm_near_refresh_radius,
            map_near_refresh_value=wm_near_refresh_value,
            max_points=fuse_max_points,
            cap_style=cap_style,
            use_distance_sampling=distance_sample,
            near_dist=near_dist,
            mid_dist=mid_dist,
            near_voxel=near_voxel,
            mid_voxel=mid_voxel,
            far_voxel=far_voxel,
        )

    def _init_s_penalty(
        self,
        *,
        enable_s_penalty: bool,
        s_penalty_thresh: float,
        s_penalty_uncovered_w: float,
        s_penalty_use_surface: bool,
        goal_use_surface: bool,
        det_log_dir: Optional[str] = None,
    ) -> None:
        """
        S-penalty: c' = c * w_S (semantic field), plus the surface-point goal option.

        ``det_log_dir`` (when set) only turns on the SP1 per-detection jsonl dump; it
        changes no admission behaviour.
        """
        self._s_penalty_cfg = SPenaltyConfig(
            enable=enable_s_penalty,
            thresh=s_penalty_thresh,
            uncovered_w=s_penalty_uncovered_w,
        )
        self._s_penalty_use_surface = s_penalty_use_surface
        self._goal_use_surface = goal_use_surface
        self._det_log_dir = None if det_log_dir in (None, "", "null") else str(det_log_dir)

    def _init_geom_gate(
        self,
        *,
        enable_occ_consistency: bool,
        occ_min_voxels: int,
        occ_min_ratio: float,
    ) -> None:
        """
        Geometric admission gate (occupancy-consistency). Runs before S-penalty.
        """
        self._geom_gate_engine = TargetGeometricGatingEngine(
            occ_cfg=BoxOccupancyConfig(
                enable=enable_occ_consistency,
                min_occupied_voxels=occ_min_voxels,
                min_occupancy_ratio=occ_min_ratio,
            ),
        )

    # ==========================================================================
    # === Input & Mapping Module ===
    # ==========================================================================
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
    ) -> List[Dict[str, Any]]:
        """Query TSP3D, scaling the pruning threshold when extremely close to surfaces."""
        if len(aligned_pcd) == 0:
            return [], {}

        camera_height = getattr(self, "_camera_height", 0.88)
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
    def _sigma_tier(score: float) -> int:
        """Legacy S4 tier, kept only for log compatibility."""
        if score >= 0.70:
            return 3
        if score >= 0.65:
            return 2
        if score >= 0.60:
            return 1
        return 0

    @staticmethod
    def _sigma_band(score: float) -> int:
        """Final S5 band-pass order: 0.65-0.70 > 0.60-0.65 > 0.55-0.60 > >=0.70."""
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
          gd-tsp (including promoted gd-only) > gd-only > confirmed >
          provisional.
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
        # Pure confirmed and gd-only are stop-shadow candidates for the next
        # experiment.  They are logged below but are not stop-eligible now.
        self._stop_shadow_logs = getattr(self, "_stop_shadow_logs", [])
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
                    # Stop eligibility is assigned after origin/flags are read:
                    # pure TSP confirmed is temporarily NOT stop-eligible.
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
                    # A gd-only entry promoted by a TSP3D merge is gd-tsp,
                    # not a separate category.  Keep the flag for provenance.
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
                    # Only GD-backed confirmations may STOP; promoted gd-only
                    # entries are included via is_gd_tsp.
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

        # Current middle-tier order: gd-only > confirmed.  A gd-only entry
        # promoted by a TSP3D merge remains gd-tsp and uses tier 0; there is
        # no separate promoted tier.  `stable` and stop-blacklisted entries
        # were already filtered above.
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

        def distance_bucket(i: int) -> int:
            # 0.5 m buckets: close goals first, but only after the tier's
            # primary evidence key.  Keep the raw 0.5 m convention used by the
            # previous selection for continuity.
            return int(round(float(dists_2d[i]) / 0.5))

        def sigma_band_key(i: int) -> int:
            t = selection_tier(i)
            # gd-tsp / confirmed / provisional use sigma band; gd-only uses
            # gd_hits as its primary evidence.
            return int(valid_bands[i]) if t in (0, 2, 3) else 0

        def gd_hits_key(i: int) -> int:
            t = selection_tier(i)
            # gd-tsp and gd-only use GD hit count; confirmed/provisional do not.
            return -int(valid_gd_hits[i]) if t in (0, 1) else 0

        def num_obs_key(i: int) -> int:
            # S6gd2: confirmed/provisional rows with >=3 observations were more
            # precise than 1-2 observation rows; use 3 as the tie threshold.
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
        chosen_2d = valid_goals[chosen_idx][:2].copy()
        self._last_target_centroid = np.asarray(
            valid_centroids[chosen_idx], dtype=np.float64
        ).reshape(-1)[:2].copy()

        # STOP shadow: log every selected confirmed (including gd-tsp and
        # gd-only-promoted-as-gd-tsp) and gd-only goal, with candidate STOP
        # features and hypothetical thresholds.  This never changes an action.
        _chosen_state = str(valid_states[chosen_idx])
        _chosen_is_gd_tsp = bool(valid_gd_tsp[chosen_idx])
        _chosen_is_promoted = bool(valid_promoted_gd_only[chosen_idx])
        _chosen_is_gd_only = bool(valid_gd_only[chosen_idx]) or _chosen_state == "gd"
        _should_log_stop_shadow = (
            _chosen_is_gd_only or _chosen_state == "confirmed"
        )
        if _should_log_stop_shadow:
            _goal_distance = float(np.linalg.norm(chosen_2d - robot_xy))
            _obs = int(valid_obs[chosen_idx])
            _gd_hits = int(valid_gd_hits[chosen_idx])
            _stop_shadow = {
                "event": "stop_shadow",
                "shadow_only": True,
                "step": int(self._num_steps),
                "memory_id": valid_memory_ids[chosen_idx],
                "candidate_id": valid_candidate_ids[chosen_idx],
                "target_class": str(self._target_object.split("|")[0]),
                "centroid": np.asarray(valid_centroids[chosen_idx], dtype=np.float64).reshape(-1)[:3].tolist(),
                "goal": np.asarray(chosen_2d, dtype=np.float64).reshape(-1)[:2].tolist(),
                "distance": float(dists_2d[chosen_idx]),
                "goal_distance": _goal_distance,
                "lifecycle_state": _chosen_state,
                "origin": valid_origins[chosen_idx],
                "gd_only": bool(_chosen_is_gd_only),
                "gd_tsp": bool(_chosen_is_gd_tsp),
                "promoted_from_gd_only": bool(_chosen_is_promoted),
                "sigma_c": float(valid_scores[chosen_idx]),
                "sigma_band": int(valid_bands[chosen_idx]),
                "gd_hits": _gd_hits,
                "num_obs": _obs,
                "stale_steps": int(valid_stale[chosen_idx]),
                "stop_eligible_current": bool(valid_stop_ok[chosen_idx]),
                "stop_gate_confirm_radius": float(self._stop_gate_radius),
                "pointnav_stop_radius": float(self._pointnav_stop_radius),
                "would_stop_radius_0_9": bool(_goal_distance <= float(self._pointnav_stop_radius)),
                "would_stop_radius_1_2": bool(_goal_distance <= float(self._stop_gate_radius)),
                "would_stop_numobs_ge3": bool(_obs >= 3),
                "would_stop_numobs_ge5": bool(_obs >= 5),
                "would_stop_gdhits_ge3": bool(_gd_hits >= 3),
                "would_stop_gdhits_ge5": bool(_gd_hits >= 5),
                "would_stop_gd3_r09": bool(_goal_distance <= float(self._pointnav_stop_radius) and _gd_hits >= 3),
                "would_stop_gd5_r09": bool(_goal_distance <= float(self._pointnav_stop_radius) and _gd_hits >= 5),
                "would_stop_obs3_r09": bool(_goal_distance <= float(self._pointnav_stop_radius) and _obs >= 3),
                "would_stop_obs5_r09": bool(_goal_distance <= float(self._pointnav_stop_radius) and _obs >= 5),
            }
            self._stop_shadow_logs.append(_stop_shadow)
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

        # Middle-tier shadow: build the candidate snapshot after selection.
        # This block is read-only; exceptions are captured by the helper.
        middle_candidates: List[Dict[str, Any]] = []
        for i in range(len(valid_centroids)):
            tier = int(tiers[i])
            if tier not in (1, 2):
                continue
            st_i = self._memory_manager.state_of(
                np.asarray(valid_centroids[i], dtype=np.float64)
            )
            middle_candidates.append(
                {
                    "index": int(i),
                    "memory_id": valid_memory_ids[i],
                    "candidate_id": valid_candidate_ids[i],
                    "type": "gd-only" if tier == 1 else "tsp-confirmed",
                    "target_class": valid_classes[i],
                    "centroid": np.asarray(valid_centroids[i], dtype=np.float64),
                    "score": float(valid_scores[i]),
                    "state": st_i,
                    "origin": valid_origins[i],
                    "num_obs": int(valid_obs[i]),
                    "stale_steps": int(valid_stale[i]),
                    "gd_hits": int(valid_gd_hits[i]),
                    "gd_only": bool(valid_gd_only[i]),
                    "gd_tsp": bool(valid_gd_tsp[i]),
                    "gd_conf_max": st_i.get("gd_conf_max"),
                    "gd_conf_last": st_i.get("gd_conf_last"),
                    "revive_count": st_i.get("revive_count"),
                    "origin_provisional": bool(st_i.get("origin_provisional", False)),
                    "selected": bool(i == chosen_idx),
                    "n_selectable": len(valid_centroids),
                    "distance": float(dists_2d[i]),
                    "distance_bucket": int(distance_bucket(i)),
                    "selection_tier": int(tier),
                }
            )

        def _middle_static_key(row: Dict[str, Any]) -> tuple:
            type_name = str(row.get("type") or "")
            return (
                int(row.get("selection_tier") or 0),
                int(self._sigma_band(float(row.get("score") or 0.0)))
                if type_name == "tsp-confirmed"
                else 0,
                -int(row.get("gd_hits") or 0)
                if type_name == "gd-only"
                else 0,
                int(row.get("distance_bucket") or 0),
                0 if int(row.get("num_obs") or 0) >= 3 else 1,
                -int(row.get("num_obs") or 0),
                int(row.get("stale_steps") or 0),
            )

        middle_candidates.sort(key=_middle_static_key)
        for rank, row in enumerate(middle_candidates):
            row["static_rank"] = int(rank)
            row["switch_reason"] = switch_reason

        gd_frame = None
        for record in reversed(getattr(self, "_gd_calib_logs", []) or []):
            if int(record.get("step", -1)) == int(self._num_steps):
                gd_frame = record
                break
        current_events = [
            event
            for event in self._memory_event_logs
            if int(event.get("step", -1)) == int(self._num_steps)
        ]
        self._log_middle_tier_shadow(
            step=int(self._num_steps),
            candidates=middle_candidates,
            robot_xy=robot_xy,
            gd_frame=gd_frame,
            memory_events=current_events,
            switch_reason=switch_reason,
        )

        self._nav_goal_logs.append(
            {
                "step": int(self._num_steps),
                "memory_id": valid_memory_ids[chosen_idx],
                "candidate_id": valid_candidate_ids[chosen_idx],
                "centroid": np.asarray(valid_centroids[chosen_idx], dtype=np.float64).tolist(),
                "goal": np.asarray(self._last_target_coord, dtype=np.float64).tolist(),
                "quality_score": float(valid_scores[chosen_idx]),
                "selection_tier": int(tiers[chosen_idx]),
                "selection_tier_name": (
                    "gd_tsp" if int(tiers[chosen_idx]) == 0
                    else "gd_only" if int(tiers[chosen_idx]) == 1
                    else "confirmed" if int(tiers[chosen_idx]) == 2
                    else "provisional" if int(tiers[chosen_idx]) == 3
                    else "other"
                ),
                "sigma_tier": int(self._sigma_tier(valid_scores[chosen_idx])),
                "sigma_band": int(valid_bands[chosen_idx]),
                "distance_bucket": int(distance_bucket(chosen_idx)),
                "num_obs_ge3": bool(int(valid_obs[chosen_idx]) >= 3),
                "distance": float(dists_2d[chosen_idx]),
                "num_obs": int(valid_obs[chosen_idx]),
                "switch_reason": switch_reason,
                "stale_steps": int(valid_stale[chosen_idx]),
                "n_selectable": len(valid_centroids),
                "lifecycle_state": str(valid_states[chosen_idx]),
                "origin": valid_origins[chosen_idx],
                "gd_only": bool(valid_gd_only[chosen_idx]),
                "gd_tsp": bool(valid_gd_tsp[chosen_idx]),
                "gd_incremental": bool(valid_gd_incremental[chosen_idx]),
                "gd_upgraded": bool(valid_gd_upgraded[chosen_idx]),
                "promoted_from_gd_only": bool(
                    valid_promoted_gd_only[chosen_idx]
                ),
                "gd_hits": int(valid_gd_hits[chosen_idx]),
                "stop_eligible": bool(valid_stop_ok[chosen_idx]),
                "robot_xy": np.asarray(position, dtype=np.float64).reshape(-1)[:2].tolist(),
                "robot_yaw": None if self._last_robot_pose is None else float(self._last_robot_pose[2]),
            }
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

        # Update 3D geometric obstacle occupancy grid
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
            fused_pcd_local, robot_xyz, robot_yaw, pcd, rgb,
            tf_camera_to_episodic, max_depth, fx, fy,
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
                    tracks=self._gd_track_snapshot(),
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
                            self._memory_event_logs.extend(
                                mechanism_record.get("events", []) or []
                            )
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

    def _log_middle_tier_shadow(
        self,
        *,
        step: int,
        candidates: List[Dict[str, Any]],
        robot_xy: np.ndarray,
        gd_frame: Optional[Dict[str, Any]],
        memory_events: List[Dict[str, Any]],
        switch_reason: str,
    ) -> None:
        """Append read-only middle-tier rows; never affect selection or STOP."""
        if not getattr(self, "_det_log_dir", None):
            return
        shadow = getattr(self, "_middle_tier_shadow", None)
        if shadow is None or not candidates:
            return
        try:
            rows = shadow.observe(
                step=int(step),
                candidates=candidates,
                robot_xy=robot_xy,
                gd_frame=gd_frame,
                memory_events=memory_events,
                switch_reason=str(switch_reason),
            )
            if rows:
                self._middle_rank_shadow_logs.extend(rows)
        except Exception as exc:  # noqa: BLE001 - shadow must not kill nav
            if self._middle_tier_shadow_error is None:
                self._middle_tier_shadow_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[MIDDLE-RANK] disabled after error: {self._middle_tier_shadow_error}",
                    flush=True,
                )

    def _gd_track_snapshot(self) -> List[Dict[str, Any]]:
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
        # Service-side filtering already applies sigma_prop.  Repeating it here
        # protects compatibility with older/mocked servers without imposing the
        # independent P0/P1 admission line before S can be logged.
        detections.filter_by_conf(self._sigma_prop)
        detections.filter_by_class(target_classes, use_vlfm_nlp=self._nlp_mode)
        # Skip memory accumulation during initialization turning (repeated surfaces pollute memory).
        if not self._done_initializing:
            return detections

        # Phase-G candidate contract: create one stable record before occ so
        # rejected candidates remain available for causal gate analysis.
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
                float(s_pt[0]), float(s_pt[1]), S_QUERY_RADIUS_M
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
        self._candidate_logs.extend(candidate_records)

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
            # S5-S2 ordering is intentional: every post-occ candidate first
            # gets a chance to merge; only an unmatched candidate is governed
            # by the independent NEW-memory provisional gate.
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
                    res = self._memory_manager.accumulate(
                        cls, centroid_np, confidence=sigma_c, sigma_c=sigma_c,
                        near_surface=near_surface, robot_xyz=robot_xyz,
                        robot_yaw=robot_yaw,
                        candidate_id=candidate_meta.get("candidate_id"),
                        new_lifecycle_state=gate_state,
                    )
                    if not res.get("written", False):
                        continue
                    wrote = True
                    merged = merged or bool(res.get("merged", False))
                    lifecycle_states.append(str(res.get("lifecycle_state") or "confirmed"))
                    self._memory_event_logs.append({
                        "event": "merge" if res.get("merged") else "new",
                        "step": int(self._num_steps),
                        "memory_id": res.get("memory_id"),
                        "candidate_id": candidate_meta.get("candidate_id"),
                        "target_class": str(cls),
                        "centroid": np.asarray(centroid_np, dtype=np.float64).tolist(),
                        "surface": None if near_surface is None else np.asarray(near_surface, dtype=np.float64).tolist(),
                        "confidence": sigma_c,
                        "sigma_c": sigma_c,
                        "num_obs": int(res.get("num_obs", 1)),
                        "lifecycle_state": res.get("lifecycle_state"),
                        "origin": res.get("origin"),
                        "new_info": res.get("new_info"),
                        "revived": bool(res.get("revived", False)),
                        "revive_count": int(res.get("revive_count", 0)),
                        "total_miss": int(res.get("total_miss", 0)),
                        "robot_xy": np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2].tolist(),
                        "robot_yaw": float(robot_yaw),
                        "detect_dist": float(candidate_meta.get("distance") or 0.0),
                        **{key: res.get(key) for key in (
                            "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                            "sigma_c_ema_05", "sigma_c_ema_08",
                            "revive_sigma_c", "revive_num_obs_before",
                            "revive_num_obs_after", "revive_dist_to_old_centroid",
                            "revive_age_steps", "revive_last_obs_step",
                            "revive_radius", "revive_window", "revive_min_sigma",
                            "revive_detect_dist", "revive_robot_xy",
                            "revive_old_centroid", "revive_new_centroid",
                        )},
                    })
                    upgraded = res.get("upgraded")
                    if upgraded:
                        self._memory_event_logs.append({
                            "event": "upgrade",
                            "step": int(self._num_steps),
                            "memory_id": res.get("memory_id"),
                            "candidate_id": candidate_meta.get("candidate_id"),
                            "target_class": str(cls),
                            "centroid": np.asarray(centroid_np, dtype=np.float64).tolist(),
                            "surface": None if near_surface is None else np.asarray(near_surface, dtype=np.float64).tolist(),
                            "confidence": sigma_c,
                            "sigma_c": sigma_c,
                            "num_obs": int(upgraded.get("num_obs", res.get("num_obs", 1))),
                            "lifecycle_state": upgraded.get("to_state"),
                            "origin": res.get("origin"),
                            "from_state": upgraded.get("from_state"),
                            "upgrade_reason": upgraded.get("reason"),
                            "promoted_from_gd_only": bool(
                                res.get("promoted_from_gd_only", False)
                            ),
                            "detect_dist": float(upgraded.get("detect_dist", 0.0)),
                            "robot_xy": np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2].tolist(),
                            "robot_yaw": float(robot_yaw),
                            **{key: res.get(key) for key in (
                                "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                                "sigma_c_ema_05", "sigma_c_ema_08",
                            )},
                        })
                candidate_meta["memory_written"] = wrote
                candidate_meta["memory_merged"] = merged
                candidate_meta["memory_lifecycle_states"] = lifecycle_states
                candidate_meta["admitted"] = wrote
                candidate_meta["reject"] = None if wrote else "provisional_gate"
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
                    )
                    self._memory_event_logs.append({
                        "event": "merge" if res.get("merged") else "new",
                        "step": int(self._num_steps),
                        "memory_id": res.get("memory_id"),
                        "candidate_id": (candidate_meta or {}).get("candidate_id"),
                        "target_class": str(cls),
                        "centroid": np.asarray(centroid_np, dtype=np.float64).tolist(),
                        "surface": None if near_surface is None else np.asarray(near_surface, dtype=np.float64).tolist(),
                        "confidence": float(conf_amp),
                        "sigma_c": float(conf_amp),
                        "num_obs": int(res.get("num_obs", 1)),
                        "lifecycle_state": res.get("lifecycle_state"),
                        "origin": res.get("origin"),
                        **{key: res.get(key) for key in (
                            "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                            "sigma_c_ema_05", "sigma_c_ema_08",
                            "revive_sigma_c", "revive_num_obs_before",
                            "revive_num_obs_after", "revive_dist_to_old_centroid",
                            "revive_age_steps", "revive_last_obs_step",
                            "revive_radius", "revive_window", "revive_min_sigma",
                            "revive_detect_dist", "revive_robot_xy",
                            "revive_old_centroid", "revive_new_centroid",
                        )},
                    })

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
        for record in lifecycle_events:
            self._memory_event_logs.append(
                {
                    "event": str(record.get("event") or "delete"),
                    "step": int(self._num_steps),
                    "memory_id": record.get("memory_id"),
                    "candidate_id": record.get("candidate_id"),
                    "target_class": record.get("target_class"),
                    "centroid": np.asarray(record.get("centroid", []), dtype=np.float64).tolist(),
                    "confidence": float(record.get("conf", 0.0)),
                    "num_obs": int(record.get("num_obs", 1)),
                    "from_state": record.get("from_state"),
                    "lifecycle_state": record.get("lifecycle_state"),
                    "delete_reason": record.get("reason"),
                    "reason": record.get("reason"),
                    "near_miss": int(record.get("near_miss", 0)),
                    **{key: record.get(key) for key in (
                        "total_miss", "revive_count", "age_steps", "last_obs_step",
                        "dist_to_robot", "thresholds", "gd_age_steps", "gd_last_step",
                        "gd_hits", "gd_conf_max", "gd_conf_last", "origin",
                    )},
                    **{key: record.get(key) for key in (
                        "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                        "sigma_c_ema_05", "sigma_c_ema_08",
                    )},
                }
            )

        for record in self._memory_manager.try_upgrade(robot_xyz):
            self._memory_event_logs.append(
                {
                    "event": "upgrade",
                    "step": int(self._num_steps),
                    "memory_id": record.get("memory_id"),
                    "candidate_id": record.get("candidate_id"),
                    "target_class": record.get("target_class"),
                    "centroid": np.asarray(record.get("centroid", []), dtype=np.float64).tolist(),
                    "confidence": float(record.get("conf", 0.0)),
                    "num_obs": int(record.get("num_obs", 1)),
                    "from_state": record.get("from_state"),
                    "upgrade_reason": record.get("reason"),
                    "lifecycle_state": record.get("lifecycle_state"),
                    "detect_dist": float(record.get("detect_dist", 0.0)),
                    "robot_xy": np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2].tolist(),
                    "robot_yaw": float(robot_yaw),
                    **{key: record.get(key) for key in (
                        "sigma_c_max", "sigma_c_latest", "sigma_c_ema_03",
                        "sigma_c_ema_05", "sigma_c_ema_08",
                    )},
                }
            )

        self._log_track_frames(frame_log_start, robot_xyz, robot_yaw, max_depth)
        return detections

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

    # ==========================================================================
    # === Plan & Do Module ===
    # ==========================================================================
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
        # Hard guard: only a confirmed gd-tsp goal reaches this method with
        # stop=True.  For pure TSP confirmed / gd-only / provisional we still
        # use PointNav for movement, but its own STOP action must not bypass
        # the provenance gate.
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
        self._log_field_observation_frame()
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
            action = self._explore(observations)
        else:
            if self._lock_step < 0:
                self._lock_step = self._num_steps
            mode = "navigate"
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
                action = self._pointnav(goal_3d[:2], stop=True)
            else:
                # S5: an un-upgraded entry may be pursued but never stopped on.
                goal_id = str(self._last_goal_memory_id)
                if rho <= self._stop_gate_radius:
                    turns = self._stop_confirm_counts.get(goal_id, 0) + 1
                    self._stop_confirm_counts[goal_id] = turns
                    if turns > self._stop_gate_turns:
                        self._stop_blacklist.add(self._last_goal_memory_id)
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
                        self._memory_event_logs.append({
                            "event": "stop_blacklist",
                            "step": int(self._num_steps),
                            "memory_id": self._last_goal_memory_id,
                            "candidate_id": _blacklist_state.get("last_candidate_id"),
                            "target_class": str(
                                _blacklist_state.get("target_class")
                                or self._target_object.split("|")[0]
                            ),
                            "centroid": _blacklist_centroid,
                            "reason": "stop_gate_exhausted",
                            "blacklist_rule": "near_turns > stop_gate_turns",
                            "near_turns": int(turns),
                            "stop_gate_turns": int(self._stop_gate_turns),
                            "stop_gate_confirm_radius": float(self._stop_gate_radius),
                            "distance": float(rho),
                            "lifecycle_state": _blacklist_state.get("lifecycle_state", self._last_goal_state),
                            "origin": _blacklist_state.get("origin", self._last_goal_origin),
                            "gd_hits": int(_blacklist_state.get("gd_hits", self._last_goal_gd_hits) or 0),
                            "num_obs": int(_blacklist_state.get("num_obs", self._last_goal_num_obs) or 0),
                            "stop_eligible": False,
                        })
                        self._stop_confirm_counts.pop(goal_id, None)
                        self._lock_step = -1
                        self._last_target_coord = None
                        self._last_target_centroid = None
                        mode = "explore"
                        action = self._explore(observations)
                    else:
                        self._stats.note_stop(allowed=False)
                        action = self._turn_left_action
                else:
                    action = self._pointnav(goal_3d[:2], stop=False)

        # Read-only field exploration shadow: run after the action has been
        # selected, so it cannot affect target choice, navigation or STOP.
        self._log_field_shadow_frame(mode=mode)

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
    # Phase 5c: World-frame Local Map (TSP3DInputPreprocessor)
    wm_max_frames: Optional[int] = 8                # world: frame-window cap (keep voxels of the most recent N frames); None = all history
    wm_voxel_size: float = 0.02                     # world: local map voxel size (m) for fixed-grid dedup
    wm_max_voxels: int = 400000                     # world: hard cap on map voxels
    wm_radius: Optional[float] = 6.0                # world: local map radius (m), None = no crop (only max_voxels hard cap); cam_radius=None is the symmetric case
    wm_min_view_disp: float = 0.15                  # world: min displacement (m) to accept a frame into fusion
    wm_min_view_yaw: float = 15.0                   # world: min yaw change (deg) to accept a frame into fusion
    wm_near_refresh_radius: Optional[float] = 3.0   # world: near-field refresh radius (m). None=off (first-observation accounting); >0: re-observed near-field voxels are re-owned by the current frame (cam-style accounting, survives frame-window slide-out).
    wm_near_refresh_value: bool = False             # world: also overwrite the stored point of refreshed near-field voxels with the current observation (cam-style sliding refresh of the value layer). False = value layer stays first.
    # Phase 5d: Panoramic Fusion


    # Phase 7: S-penalty (semantic-field cross-validation)
    enable_s_penalty: bool = True           # Master switch: c' = c * w_S(S) — multiply TSP3D confidence by a semantic-field weight (BLIP2 ITM, zero extra queries) to amplify TP/FP discrimination.
    s_penalty_thresh: float = 0.14          # Covered w_S=min(1,max(S,s_min)/thresh).
    s_penalty_uncovered_w: float = 1.0      # weight when S=None/<=0 (no coverage): 1.0 = free pass (A1), <1.0 = mild penalty on unverifiable detections.
    # Phase 7b: geometric admission module (occupancy-consistency gate)
    enable_occ_consistency: bool = True     # box must have occupied-voxel support in the 3D grid
    occ_min_voxels: int = 8                 # occ gate: min occupied voxels supporting the box
    occ_min_ratio: float = 0.001            # occ gate: min occupied/(box volume) ratio
    # Phase 7c: target-memory lifecycle
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
    enable_gd_mechanism: bool = False        # Enable GD-only/gd-tsp routing and writes.
    gd_conf_min: float = 0.40                # GD base-gate confidence (initial filter only).
    gd_max_dist: float = 5.0                 # GD base-gate robot distance (m).
    gd_only_classes: str = ""                # empty = all target classes; comma-separated otherwise.
    # Fixed S6-final: gd-only has no conf gate, no TTL, requires depth_ok;
    # depth-failed confirmed/provisional tracks may merge via gd-tsp; stable is
    # never selectable and never upgraded.
    s_penalty_use_surface: bool = True      # query S at the bbox near-surface point (facing camera) instead of the centroid.
    goal_use_surface: bool = True           # navigate to the stored near-surface point (first-write fixed) instead of the centroid.
    det_log_dir: Optional[str] = None       # SP1 per-detection jsonl directory; None = off (logging only, no behaviour change).
    enable_middle_tier_shadow: bool = True  # Read-only middle-tier feature/probability shadow; never changes selection.
    middle_tier_shadow_every_n: int = 1     # Shadow sampling stride (steps).
    middle_tier_model_path: str = ""        # Offline-fitted middle-tier model JSON; empty = features-only shadow.
    enable_middle_tier_ranking: bool = False  # Future gate: use p for middle-tier ordering; false = static order.
    lock_exempt: bool = True             # Lock hold: while locked in navigate, the nav-goal entry is exempt from the near-field near_miss budget.
    lock_exempt_identity: bool = True    # Exemption reference: True = pursued entry centroid (identity); False = nav-goal near-surface point.

    enable_stats: bool = True     # 通用前置量测开关（[NAV]/[STOP] 打印与逐集摘要；零行为）

cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
