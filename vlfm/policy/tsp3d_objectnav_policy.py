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

from vlfm.tsp3d_models.utils.pipeline import TSP3DInputPreprocessor
from vlfm.tsp3d_models.utils.scan_behavior import (
    _act_panoramic,
    _act_world_scan,
)
from vlfm.tsp3d_models.utils.s_penalty import (
    SPenaltyConfig, 
    apply_s_penalty, 
    query_semantic_at,
)
from vlfm.tsp3d_models.utils.target_memory_manager import MemoryManagerConfig, TargetMemoryManager
from vlfm.utils.vlvm_stats_logger import VlvmStatsLogger
from vlfm.tsp3d_models.utils.target_geometric_gating import (
    TargetGeometricGatingEngine,
    BoxOccupancyConfig,
)
from vlfm.tsp3d_models.grounding_dino import (
    BOTH,
    GD,
    CameraView,
    GdProposalConfig,
    GdProposer,
    Tsp3dCandidate,
    admit_as_suspicious,
    attach_votes,
    is_lockable,
    should_run,
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
            sigma_tar: float = 0.70,
            tau: float = 0.15,
            near_field_dist: float = 1.0,
            near_field_sigma_scale: float = 0.8,
            use_vlfm_nlp: bool = False,
            fusion_style: str = "world",  # "camera" / "world" / "panoramic (scan)" / "none"
            enable_scan: bool = False,    # world + scan
            frontier_trigger: bool = True,
            scan_min_gap: int = 90,
            scan_opportunistic_after: int = 60,
            wm_voxel_size: float = 0.02,
            wm_radius: float = 6.0,
            wm_min_view_disp: float = 0.15,
            wm_min_view_yaw: float = 15.0,
            wm_max_voxels: int = 400000,
            wm_max_frames: Optional[int] = 8, 
            wm_near_refresh_radius: Optional[float] = None,
            wm_near_refresh_value: bool = True,
            panoramic_turn_steps: int = 6,
            panoramic_voxel_size: float = 0.02,
            panoramic_radius: Optional[float] = 6.0,
            panoramic_max_points: int = 200000,
            panoramic_min_move: float = 2.0,
            panoramic_arrive_dist: float = 1.0,
            distance_sample: bool = True,
            near_dist: float = 1.5,
            mid_dist: float = 3.0,
            near_voxel: float = 0.01,
            mid_voxel: float = 0.02,
            far_voxel: float = 0.05,
            pcd_window_size: int = 8,
            fuse_voxel_size: float = 0.02,
            fuse_max_points: int = 200000,
            cam_radius: Optional[float] = None,
            cam_min_view_disp: float = 0.0,
            cam_min_view_yaw: float = 0.0,
            cap_style: str = "random",
            det_seed: int = 0,
            pointnav_stop_radius: float = 0.30,
            enable_fb: bool = True,
            fb_near_radius: float = 2.5,
            fb_hysteresis: int = 5,
            fb_suspicious_hysteresis: int = 2,
            fb_suspicious_conf: float = 0.75,
            enable_s_penalty: bool = True,
            s_penalty_thresh: float = 0.15,
            s_penalty_floor: float = 0.3,
            s_penalty_radius_m: float = 0.5,
            s_penalty_uncovered_w: float = 1.0,
            sx_conflict_gate: bool = True,
            # geometric admission module
            enable_occ_consistency: bool = True,
            occ_min_voxels: int = 8,
            occ_min_ratio: float = 0.001,
            # target-memory manager
            merge_dist_thresh: float = 0.5,
            ema_weight_old: float = 0.8,
            s_penalty_use_surface: bool = True,
            goal_use_surface: bool = True,
            # new GD auxiliary mechanism: independent proposer + frame-level vote
            gdp_enable: bool = True,
            gdp_every_n: int = 5,
            gdp_box_thr: float = 0.4,
            gdp_caption_style: str = "vocab",
            gdp_max_boxes: int = 1,
            gdp_weak_guard: bool = True,
            gdp_merge_dist: float = 0.5,
            gdp_merge_iau: float = 0.3,
            gdp_port: int = 12181,
            lock_min_obs: int = 2,
            lock_exempt: bool = True,
            lock_exempt_identity: bool = True,
            gd_skip_classes: str = "",
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
        self._sigma_tar = sigma_tar
        self._sigma_sce = sigma_sce
        self._tau = tau
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
        self._target_fallback_state: Dict[str, List[Dict[str, Any]]] = {}
        self._last_target_coord: Union[None, np.ndarray] = None

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
        # NOTE: must run BEFORE `_init_memory_manager` (which reads `self._enable_fb` / `self._fb_*`).
        self._init_fallback(
            enable_fb=enable_fb,
            fb_hysteresis=fb_hysteresis,
            fb_suspicious_hysteresis=fb_suspicious_hysteresis,
            fb_suspicious_conf=fb_suspicious_conf,
            max_depth=max_depth,
        )
        self._init_memory_manager(
            merge_dist_thresh=merge_dist_thresh,
            ema_weight_old=ema_weight_old,
            lock_exempt=lock_exempt,
            lock_exempt_identity=lock_exempt_identity,
        )
        self._init_fusion(
            fusion_style=fusion_style,
            enable_scan=enable_scan,
            frontier_trigger=frontier_trigger,
            scan_min_gap=scan_min_gap,
            scan_opportunistic_after=scan_opportunistic_after,
            panoramic_min_move=panoramic_min_move,
            panoramic_arrive_dist=panoramic_arrive_dist,
            panoramic_turn_steps=panoramic_turn_steps,
            panoramic_voxel_size=panoramic_voxel_size,
            panoramic_radius=panoramic_radius,
            panoramic_max_points=panoramic_max_points,
            pcd_window_size=pcd_window_size,
            fuse_voxel_size=fuse_voxel_size,
            fuse_max_points=fuse_max_points,
            wm_voxel_size=wm_voxel_size,
            wm_radius=wm_radius,
            wm_min_view_disp=wm_min_view_disp,
            wm_min_view_yaw=wm_min_view_yaw,
            wm_max_voxels=wm_max_voxels,
            wm_max_frames=wm_max_frames,
            wm_near_refresh_radius=wm_near_refresh_radius,
            wm_near_refresh_value=wm_near_refresh_value,
            cam_radius=cam_radius,
            cam_min_view_disp=cam_min_view_disp,
            cam_min_view_yaw=cam_min_view_yaw,
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
            s_penalty_floor=s_penalty_floor,
            s_penalty_radius_m=s_penalty_radius_m,
            s_penalty_uncovered_w=s_penalty_uncovered_w,
            s_penalty_use_surface=s_penalty_use_surface,
            goal_use_surface=goal_use_surface,
            sx_conflict_gate=sx_conflict_gate,
        )
        self._init_geom_gate(
            enable_occ_consistency=enable_occ_consistency,
            occ_min_voxels=occ_min_voxels,
            occ_min_ratio=occ_min_ratio,
        )
        self._init_gd_proposer(
            gdp_enable=gdp_enable,
            gdp_every_n=gdp_every_n,
            gdp_box_thr=gdp_box_thr,
            gdp_caption_style=gdp_caption_style,
            gdp_max_boxes=gdp_max_boxes,
            gdp_weak_guard=gdp_weak_guard,
            gdp_merge_dist=gdp_merge_dist,
            gdp_merge_iau=gdp_merge_iau,
            gdp_port=gdp_port,
            gd_skip_classes=gd_skip_classes,
            lock_min_obs=lock_min_obs,
        )
        # 通用前置量测器
        self._stats = VlvmStatsLogger(self, enabled=enable_stats)

        # 3D visual grounding and vision-language evaluation clients
        self._tsp3d_client = TSP3DClient(port=int(os.environ.get("TSP3D_PORT", "12186")))
        self._itm_client = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._text_prompt = text_prompt

        self._det_seed = int(det_seed)
        self._rng_step = 0            # monotonic numpy-RNG seed counter (never reset)

    def _reset(self) -> None:
        """Reset memories, step counters, pointnav model, and 3D occupancy map."""
        if self._gd_proposer is not None and self._gd_proposer.cfg.enabled:
            s = self._gd_proposer.summary()
            print(f"[GD2] episode summary: frames={s['frames']} boxes_above_thr={s['boxes_above_thr']} "
                  f"proposals={s['proposals']} both={self._gd_stats['both']} gd={self._gd_stats['gd']} "
                  f"gd_rejected={self._gd_stats['gd_rejected']} both_cleared={self._gd_stats['both_cleared']} "
                  f"tsp3d_new={self._gd_stats['tsp3d_new']} "
                  f"rejects={s['rejects']}")
            self._gd_proposer.reset()
            self._gd_stats = {
                "both": 0, "gd": 0, "gd_rejected": 0, "both_cleared": 0, "tsp3d_new": 0,
            }
        self._target_object = ""
        self._init_step_count = 0
        self._num_steps = 0
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._target_3d_memory.clear()
        self._target_surface_memory.clear()
        self._target_verify_state.clear()
        self._target_fallback_state.clear()
        self._stats.episode_summary()
        self._last_target_coord = None
        self._geom_gate_engine.reset()
        self._scan_in_progress = False
        self._last_pending = []
        self._scan_value_views = []
        self._scan_pos = None
        self._scan_remaining = 0
        self._scan_pcd_frames = []
        self._last_scan_step = -10**9
        self._last_lock_step = 0
        self._pointnav_policy.reset()
        self._obstacle_map3d.reset()
        self._preprocessor.reset()
        self._detect_logs = []
        self._lock_step = -1
        self._stats.reset()
        self._did_reset = True
    
    # ==========================================================================
    # Mechanism initialisation (one `_init_*` per module) 
    # 3D spatial representations / target-memory lifecycle /
    # fusion / fallback / S-penalty / geometric admission / GD source
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
        lock_exempt: bool = True,
        lock_exempt_identity: bool = True,
    ) -> None:
        """
        Target-memory lifecycle manager (EMA merge + fallback + erasure) 
        bound to the policy's memory dicts.
        """
        self._memory_manager = TargetMemoryManager(MemoryManagerConfig(
            enable_fallback=self._enable_fb,
            fb_near_radius=self._fb_near_radius,
            fb_hysteresis=self._fb_hysteresis,
            fb_suspicious_hysteresis=self._fb_suspicious_hysteresis,
            fb_suspicious_conf=self._fb_suspicious_conf,
            merge_dist_thresh=merge_dist_thresh,
            ema_weight_old=ema_weight_old,
        ))
        self._memory_manager.bind(
            self._target_3d_memory, self._target_surface_memory,
            self._target_verify_state, self._target_fallback_state,
        )
        self._lock_exempt = bool(lock_exempt)
        self._lock_exempt_identity = bool(lock_exempt_identity)
        self._lock_step: int = -1

    def _init_fusion(
        self,
        *,
        fusion_style: str,
        enable_scan: bool,
        frontier_trigger: bool,
        scan_min_gap: int,
        scan_opportunistic_after: int,
        panoramic_min_move: float,
        panoramic_arrive_dist: float,
        panoramic_turn_steps: int,
        panoramic_voxel_size: float,
        panoramic_radius: Optional[float],
        panoramic_max_points: int,
        pcd_window_size: int,
        fuse_voxel_size: float,
        fuse_max_points: int,
        wm_voxel_size: float,
        wm_radius: Optional[float],
        wm_min_view_disp: float,
        wm_min_view_yaw: float,
        wm_max_voxels: int,
        wm_max_frames: Optional[int],
        wm_near_refresh_radius: Optional[float],
        wm_near_refresh_value: bool,
        cam_radius: Optional[float],
        cam_min_view_disp: float,
        cam_min_view_yaw: float,
        cap_style: str,
        distance_sample: bool,
        near_dist: float,
        mid_dist: float,
        near_voxel: float,
        mid_voxel: float,
        far_voxel: float,
    ) -> None:
        """Fusion routes (camera / world / world+scan) + the scan state machine."""
        self._scan_turn_action = torch.tensor([[6]], dtype=torch.long)
        self._panoramic_min_move = float(panoramic_min_move)
        self._panoramic_arrive_dist = float(panoramic_arrive_dist)
        self._panoramic_turn_steps = max(int(panoramic_turn_steps), 1)
        self._panoramic_scan_angle_deg = 360.0 / self._panoramic_turn_steps

        # Fusion route selection: "camera" (temporal window) / "world" (incremental map)
        # / "panoramic" (single-position 360° spatial fusion) / "none" (no fusion mechanism).
        self._fusion_style = fusion_style
        self._enable_scan = bool(enable_scan)
        self._frontier_trigger = bool(frontier_trigger)
        self._scan_min_gap = max(int(scan_min_gap), 1)
        self._scan_opportunistic_after = max(int(scan_opportunistic_after), 0)
        self._last_scan_step: int = -10**9
        self._last_lock_step: int = 0
        # Scan state machine: True while an in-place panoramic scan is running
        self._scan_in_progress = False
        # S-penalty (c') candidates of the most recent TSP3D query;
        # the scan-end direction decision ranks them by c'.
        self._last_pending: list = []
        self._scan_value_views: list = []
        self._scan_pos: Optional[np.ndarray] = None      # last scan / init pose (min-move novelty guard)
        self._scan_remaining: int = 0                    # remaining in-place turn frames of a scan
        self._scan_pcd_frames: list = []                 # world-frame pcd slices of the in-progress scan
        self._scan_voxel_size = float(panoramic_voxel_size)
        self._scan_radius = float(panoramic_radius) if panoramic_radius is not None else None
        self._scan_max_points = int(panoramic_max_points)
        self._preprocessor = TSP3DInputPreprocessor(
            fusion_style=fusion_style,
            window_size=pcd_window_size,
            fuse_voxel_size=fuse_voxel_size,
            map_voxel_size=wm_voxel_size,
            map_radius=wm_radius,
            min_view_disp=wm_min_view_disp,
            min_view_yaw=np.deg2rad(wm_min_view_yaw),
            max_map_voxels=wm_max_voxels,
            map_max_frames=wm_max_frames,
            map_near_refresh_radius=wm_near_refresh_radius,
            map_near_refresh_value=wm_near_refresh_value,
            max_points=fuse_max_points,
            cam_radius=cam_radius,
            cam_min_view_disp=cam_min_view_disp,
            cam_min_view_yaw=np.deg2rad(cam_min_view_yaw),
            cap_style=cap_style,
            use_distance_sampling=distance_sample,
            near_dist=near_dist,
            mid_dist=mid_dist,
            near_voxel=near_voxel,
            mid_voxel=mid_voxel,
            far_voxel=far_voxel,
            panoramic_turn_steps=panoramic_turn_steps,
            panoramic_voxel_size=panoramic_voxel_size,
            panoramic_radius=panoramic_radius,
            panoramic_max_points=panoramic_max_points,
            panoramic_min_move=panoramic_min_move,
        )

    def _init_fallback(
        self,
        *,
        enable_fb: bool,
        fb_hysteresis: int,
        fb_suspicious_hysteresis: int,
        fb_suspicious_conf: float,
        max_depth: float,
    ) -> None:
        """Anti-hallucination fallback (near-field hysteresis) parameters.

        The lifecycle manager consumes them; the cone radius is VLFM's `max_depth * 0.5`.
        """
        self._enable_fb = bool(enable_fb)
        self._fb_near_radius = float(max_depth) * 0.5
        self._fb_hysteresis = int(fb_hysteresis)
        self._fb_suspicious_hysteresis = int(fb_suspicious_hysteresis)
        self._fb_suspicious_conf = float(fb_suspicious_conf)

    def _init_s_penalty(
        self,
        *,
        enable_s_penalty: bool,
        s_penalty_thresh: float,
        s_penalty_floor: float,
        s_penalty_radius_m: float,
        s_penalty_uncovered_w: float,
        s_penalty_use_surface: bool,
        goal_use_surface: bool,
        sx_conflict_gate: bool = True,
    ) -> None:
        """
        S-penalty: c' = c * w_S(semantic field), plus the surface-point goal option.
        """
        self._s_penalty_cfg = SPenaltyConfig(
            enable=enable_s_penalty,
            thresh=s_penalty_thresh,
            floor=s_penalty_floor,
            radius_m=s_penalty_radius_m,
            uncovered_w=s_penalty_uncovered_w,
        )
        self._s_penalty_use_surface = s_penalty_use_surface
        self._goal_use_surface = goal_use_surface
        self._sx_conflict_gate = bool(sx_conflict_gate)

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

    def _init_gd_proposer(
        self,
        *,
        gdp_enable: bool,
        gdp_every_n: int,
        gdp_box_thr: float,
        gdp_caption_style: str,
        gdp_max_boxes: int,
        gdp_weak_guard: bool,
        gdp_merge_dist: float,
        gdp_merge_iau: float,
        gdp_port: int,
        gd_skip_classes: str = "",
        lock_min_obs: int = 2,
    ) -> None:
        """
        GD 辅助机制：独立每帧提议源 + 帧级两票（`both` / `gd`）。
        """
        self._gd_proposer = GdProposer(GdProposalConfig(
            enabled=bool(gdp_enable),
            every_n=max(1, int(gdp_every_n)),
            port=int(gdp_port),
            box_thr=float(gdp_box_thr),
            caption_style=str(gdp_caption_style),
            max_boxes=max(1, int(gdp_max_boxes)),
            merge_dist=float(gdp_merge_dist),
            merge_iau=float(gdp_merge_iau),
            weak_guard=bool(gdp_weak_guard),
        ))
        self._gd_weak_guard = bool(gdp_weak_guard)
        # Vote accounting of the GD source.
        self._gd_stats = {
            "both": 0, "gd": 0, "gd_rejected": 0, "both_cleared": 0, "tsp3d_new": 0,
        }
        # GD 类别白名单
        self._gd_skip_classes = {
            c.strip() for c in str(gd_skip_classes).split("|") if c.strip()
        }
        # GD 弱证不锁门槛
        self._lock_min_obs = max(int(lock_min_obs), 1)


    def _gd_geom_admit(
        self,
        proposal: Any,
        target_class: str,
        pcd_world: np.ndarray,
        robot_xyz: np.ndarray,
    ) -> bool:
        """
        Geometric admission of a `gd` single-vote candidate (the SAME gate as TSP3D).

        No new confidence threshold: the GD box only has to survive the existing
        occupancy-consistency gate, exactly like a TSP3D candidate.
        """
        eng = self._geom_gate_engine
        if not eng.enabled:
            return True
        status, _cluster, _diag = eng.process_detection(
            target_class=target_class,
            box_corners=proposal.box8,
            current_frame_pcd=pcd_world,
            obstacle_map_3d=self._obstacle_map3d,
            confidence=float(proposal.conf),
            step=self._num_steps,
            robot_xyz=robot_xyz,
        )
        if status == "REJECTED":
            return False
        return True

    def _gd_apply(
        self,
        image_rgb: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        max_depth: float,
        fx: float,
        fy: float,
        target_classes: List[str],
        robot_xyz: np.ndarray,
        robot_yaw: float,
        pcd_world: np.ndarray,
        pending: List[Any],
    ) -> None:
        """
        Run the GD proposal source on the current frame and consume its votes.

        Frame alignment: GD reads the frame the fusion chain just consumed (`rgb` / `depth` / `tf`), 
        and its point set never enters that chain or TSP3D.

        * `both` -> corroborate the co-located admitted entry (no new entry, the c' scale stays TSP3D's);
        
        * `gd`   -> single vote: passes the SAME geometric gate, 
          then is written as suspicious/pending with the `gd` src tag; 
          c' is set to `sigma_tar` because GD logits are a different scale.
        """
        prog = self._gd_proposer
        if prog is None or not prog.cfg.enabled or not self._done_initializing:
            return
        # 类别白名单：列出的目标类完全跳过 GD 源。
        if self._gd_skip_classes and any(c.strip() in self._gd_skip_classes for c in target_classes):
            return
        if not should_run(self._num_steps, prog.cfg.every_n):
            return
        frames = self._observations_cache.get("object_map_rgbd")
        if image_rgb is None or not frames:
            return
        depth = frames[0][1]
        if depth is None:
            return
        
        depth_arr = np.asarray(depth, dtype=np.float64)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        height, width = int(depth_arr.shape[0]), int(depth_arr.shape[1])
        # The observation depth is normalized in [0, 1].
        depth_m = depth_arr * (max_depth - self._min_depth) + self._min_depth
        cam = CameraView(
            fx=float(fx), fy=float(fy), height=height, width=width,
            tf_camera_to_episodic=tf_camera_to_episodic,
            min_depth=self._min_depth, max_depth=max_depth,
        )
        # Deterministic chain subsampling (own RandomState, never the global stream).
        rng = np.random.RandomState((self._det_seed * 100003 + self._num_steps) & 0x7FFFFFFF)
        proposals = prog.propose(
            image_rgb, depth_m, cam, target_classes,
            np.asarray(robot_xyz, dtype=np.float64)[:2], rng,
        )

        if not proposals:
            return

        # The TSP3D vote = this frame's ADMITTED candidates only, 
        # so a box rejected by the gates can never upgrade a GD proposal to `both`.
        candidates = [
            Tsp3dCandidate(
                centroid=np.asarray(centroid_np, dtype=np.float64).reshape(-1),
                box8=None, conf=float(conf_amp), admitted=True,
            )
            for conf_amp, centroid_np, _surf, _classes in pending
        ]
        for proposal, vote, matched_idx in attach_votes(proposals, candidates, prog.cfg):
            if vote == BOTH and matched_idx is not None:
                conf_amp, centroid_np, _surf, classes = pending[matched_idx]
                # `both` 沿用 TSP3D 的 c'（GD 分数不改变置信度尺度）。
                conf_use = float(np.clip(float(conf_amp), 0.0, 1.0))
                res: Dict[str, Any] = {}
                for cls in classes:
                    res = self._memory_manager.accumulate(
                        cls, np.asarray(centroid_np, dtype=np.float64),
                        confidence=float(conf_use),
                        near_surface=np.asarray(proposal.surface, dtype=np.float64),
                        robot_xyz=robot_xyz, robot_yaw=robot_yaw, max_depth=max_depth,
                        suspicious_override=False, src=BOTH,
                        gd_conf=float(proposal.conf),
                    )
                self._gd_stats["both"] += 1
                if res.get("suspicion_cleared"):
                    self._gd_stats["both_cleared"] += 1
                print(f"[GD2] both -> corroborate {classes} at "
                      f"{np.round(np.asarray(centroid_np, dtype=np.float64)[:3], 2)} "
                      f"(tsp3d={float(conf_amp):.2f} gd={float(proposal.conf):.2f} "
                      f"c''={float(conf_use):.2f} "
                      f"n_obs={res.get('num_obs')} src={res.get('src')} "
                      f"suspicious={res.get('suspicious')} cleared={bool(res.get('suspicion_cleared'))} "
                      f"robot={np.round(np.asarray(robot_xyz, dtype=np.float64)[:2], 2)} "
                      f"yaw={float(robot_yaw):.2f}) -> normal path")
                continue
            if not admit_as_suspicious(vote):
                continue
            cls = proposal.canon if proposal.canon in target_classes else target_classes[0]
            if not self._gd_geom_admit(proposal, cls, pcd_world, robot_xyz):
                self._gd_stats["gd_rejected"] += 1
                print(f"[GD2] gd -> REJECTED by geom gate '{cls}' at "
                      f"{np.round(np.asarray(proposal.centroid, dtype=np.float64)[:3], 2)}")
                continue
            res = self._memory_manager.accumulate(
                cls, np.asarray(proposal.centroid, dtype=np.float64),
                confidence=self._sigma_tar,
                near_surface=np.asarray(proposal.surface, dtype=np.float64),
                robot_xyz=robot_xyz, robot_yaw=robot_yaw, max_depth=max_depth,
                suspicious_override=True, src=GD,
                gd_conf=float(proposal.conf),
            )
            self._gd_stats["gd"] += 1
            print(f"[GD2] gd -> single vote '{cls}' at "
                  f"{np.round(np.asarray(proposal.centroid, dtype=np.float64)[:3], 2)} "
                  f"(gd={float(proposal.conf):.2f} n_obs={res.get('num_obs')} "
                  f"src={res.get('src')} suspicious={res.get('suspicious')} "
                  f"robot={np.round(np.asarray(robot_xyz, dtype=np.float64)[:2], 2)} "
                  f"yaw={float(robot_yaw):.2f}) -> pending")

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
            sigma_tar=self._sigma_tar,
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

    def _geom_gate(
        self,
        detections: ObjectDetections,
        pcd_world: np.ndarray,
        robot_xyz: Optional[np.ndarray] = None,
    ) -> ObjectDetections:
        """Geometric admission on the current candidates.

        REJECTED (empty-air) detections are dropped before the S-penalty gate.
        """
        engine = self._geom_gate_engine
        n = detections.num_detections
        if n == 0:
            return detections
        boxes_np = detections.boxes.detach().cpu().numpy()
        logits_np = detections.logits.detach().cpu().numpy()
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            status, _cluster, _diag = engine.process_detection(
                target_class=detections.phrases[i],
                box_corners=boxes_np[i],
                current_frame_pcd=pcd_world,
                obstacle_map_3d=self._obstacle_map3d,
                confidence=float(logits_np[i]),
                step=self._num_steps,
                robot_xyz=robot_xyz,
            )
            drop = status == "REJECTED"
            keep[i] = not drop
        if not keep.all():
            detections.filter_by_mask(keep)
        return detections


    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """
        Navigation goal among the memorized target candidates.
        Closest centroid with a distance-latch keeps goal switches stable.
        """
        valid_centroids: List[np.ndarray] = []
        valid_goals: List[np.ndarray] = []
        valid_obs: List[int] = []
        valid_src: List[str] = []
        # 各候选条目的原始 GD sigmoid（0.0 = 无 GD 票；量测用）。
        valid_gconf: List[float] = []
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

                    # 弱证不锁：`gd` 单票需 ≥ `lock_min_obs` 次观测才可驱动 explore -> navigate。
                    st = self._memory_manager.state_of(centroid)
                    src = str(st.get("src") or "tsp3d")
                    if not is_lockable(src, obs, self._lock_min_obs, self._gd_weak_guard):
                        continue

                    valid_centroids.append(np.array(centroid))
                    if self._goal_use_surface and i < len(surf_list):
                        valid_goals.append(np.array(surf_list[i]))
                    else:
                        valid_goals.append(np.array(centroid))
                    valid_obs.append(obs)
                    valid_src.append(src)
                    valid_gconf.append(float(st.get("gd_conf", 0.0) or 0.0))
        if len(valid_centroids) == 0:
            return None

        robot_xy = np.asarray(position)[:2]
        centroids = np.array(valid_centroids)
        dists_2d = np.linalg.norm(centroids[:, :2] - robot_xy, axis=1)
        chosen_idx = int(np.argmin(dists_2d))

        chosen_2d = valid_goals[chosen_idx][:2].copy()
        if self._last_target_coord is None:
            self._last_target_coord = chosen_2d
        else:
            # distance-latch: keep the current goal unless the switch is meaningful.
            delta_dist = np.linalg.norm(chosen_2d - self._last_target_coord)
            dist_to_new = float(np.linalg.norm(valid_centroids[chosen_idx][:2] - robot_xy))
            if delta_dist < 0.1:
                pass  # <0.1m from current target: keep
            elif delta_dist < 0.5 and dist_to_new > 2.0:
                pass  # <0.5m and robot >2m away: keep
            else:
                self._last_target_coord = chosen_2d

        self._stats.note_pick(
            valid_centroids, valid_obs, valid_src, chosen_idx, position,
            valid_gconf=valid_gconf,
        )

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

        return self._query_and_process(
            fused_pcd_local, robot_xyz, robot_yaw, pcd, rgb,
            tf_camera_to_episodic, max_depth, fx, fy,
        )

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
        then run the full detection pipeline (filter -> S-penalty -> memory -> fallback).
        """
        target_classes = [c.strip() for c in self._target_object.split("|") if c.strip()]
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
        logits_arr = np.asarray(logits, dtype=np.float64)
        keep = logits_arr >= self._sigma_tar
        if len(keep):
            if not bool(keep.all()):
                detections.filter_by_mask(keep)
        else:
            detections.filter_by_conf(self._sigma_tar)
        detections.filter_by_class(target_classes, use_vlfm_nlp=self._nlp_mode)
        # Skip memory accumulation during initialization turning (repeated surfaces pollute memory).
        if not self._done_initializing:
            return detections

        # REJECTED (empty-air) -> dropped before S-penalty
        if self._geom_gate_engine.enabled:
            detections = self._geom_gate(detections, pcd_world, robot_xyz=robot_xyz)
        # S-penalty gate cross-validation -> write passed detections to memory.
        admitted_meta: List[Any] = []
        pending = apply_s_penalty(
            detections, target_classes, robot_xyz,
            query_semantic=self._query_semantic_at,
            cfg=self._s_penalty_cfg,
            sigma_tar=self._sigma_tar,
            use_surface=self._s_penalty_use_surface,
            goal_use_surface=self._goal_use_surface,
            nlp_mode=self._nlp_mode,
            out_log=self._detect_logs,
            per_det_meta=None,
            meta_out=admitted_meta,
            conflict_gate=bool(self._sx_conflict_gate),
        )
        # Kept for the scan-end direction decision (c' ranking); see `scan_behavior`.
        self._last_pending = list(pending)
        self._stats.note_frame_admitted(pending)

        # With the GD source on, a TSP3D-only write is "single-side" evidence: a NEW
        # entry starts suspicious and a same-frame `both` vote clears the flag again.
        # With GD off there is no second vote at all.
        _gd_on = self._gd_proposer is not None and self._gd_proposer.cfg.enabled
        for (conf_amp, centroid_np, near_surface, active_classes), susp in zip(
            pending, admitted_meta
        ):
            for cls in active_classes:
                res = self._memory_manager.accumulate(
                    cls, centroid_np, confidence=conf_amp, near_surface=near_surface,
                    robot_xyz=robot_xyz, robot_yaw=robot_yaw, max_depth=max_depth,
                    suspicious_override=bool(susp) if susp is not None else None,
                    new_suspicious=_gd_on,
                )
                if _gd_on and not res.get("merged", False):
                    self._gd_stats["tsp3d_new"] += 1
                    print(f"[TSP3D] new entry '{cls}' at "
                          f"{np.round(np.asarray(centroid_np, dtype=np.float64)[:3], 2)} "
                          f"-> suspicious (single-side, c'={float(conf_amp):.2f})")
        # New GD auxiliary mechanism: same-frame second proposer + two-vote
        self._gd_apply(
            image_rgb, tf_camera_to_episodic, max_depth, fx, fy,
            target_classes, robot_xyz, robot_yaw, pcd_world, pending,
        )

        # Target Memory Manager
        camera_pos = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)
        cone_fov = get_fov(self._fx, self._depth_image_shape[1])
        # `lock_exempt`：navigate 模式下 nav goal 条目豁免 near_miss 预算。
        _nav_goal = self._last_target_coord if int(self._lock_step) >= 0 else None
        # `lock_exempt_identity`：用被 pursue 条目质心替代 nav goal 做 0.5 m 匹配。
        _exempt_ref = _nav_goal
        if self._lock_exempt_identity:
            _pxy_ref = (self._stats.last_meta or {}).get("xy")
            if _pxy_ref is not None:
                _exempt_ref = _pxy_ref
        self._memory_manager.step(
            camera_pos, camera_yaw, cone_fov,
            robot_xyz=robot_xyz,
            nav_goal_xy=_exempt_ref,
            exempt_nav_goal=bool(self._lock_exempt),
        )

        return detections

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

        if self._fusion_style == "panoramic":
            rgb, depth, tf, min_depth, max_depth, fx, fy = object_map_rgbd[0]
            pcd = self._get_shared_pcd(0)
            robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
            robot_yaw = self._observations_cache.get("robot_heading", 0.0)
            goal_3d = self._get_target_object_location(robot_xyz)
            det, mode, action = _act_panoramic(
                self, observations, pcd, robot_xyz, robot_yaw, rgb, depth, tf,
                min_depth, max_depth, fx, fy, goal_3d,
            )
            detections.append(det)

        elif self._fusion_style == "world" and self._enable_scan:
            # World temporal fusion + frontier-arrival 360° scan.
            rgb, depth, tf, min_depth, max_depth, fx, fy = object_map_rgbd[0]
            pcd = self._get_shared_pcd(0)
            robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
            robot_yaw = self._observations_cache.get("robot_heading", 0.0)
            goal_3d = self._get_target_object_location(robot_xyz)
            det, mode, action = _act_world_scan(
                self, observations, pcd, robot_xyz, robot_yaw, rgb, depth, tf,
                min_depth, max_depth, fx, fy, goal_3d,
            )
            detections.append(det)

        else:
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
                action = self._pointnav(goal_3d[:2], stop=True)

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
    pointnav_stop_radius: float = 0.30  # Distance (m) at which the agent stops near the goal; larger = stops farther from the target.
    # Phase 6b: Anti-hallucination fallback
    enable_fb: bool = True              # Delete confirmed-false centroids near the FOV cone -> fallback to explore.
    fb_near_radius: float = 2.5         # Near-field confirmation cone radius (m); VLFM uses max_depth*0.5 (2.5m @ max_depth=5). 有意失效（实现内固定）。
    fb_hysteresis: int = 5              # Trusted centroid: consecutive near-field frames without a fresh merge before deletion.
    fb_suspicious_hysteresis: int = 2   # Suspicious centroid (far/low-conf): fewer frames before deletion.
    fb_suspicious_conf: float = 0.75    # Trust boundary。GD 启用时**不作用**：新条目恒被 `new_suspicious=_gd_on` 标可疑、合并写不重算该标志，仅 `both` 合并可清除；现值仅留档（GD-off 工况才生效

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
    sigma_tar: float = 0.70             # Target confidence threshold (admission); higher = stricter/fewer detections, lower = more detections but more false positives.
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
    fusion_style: str = "world"         # Fusion route: "camera" (8-frame temporal window) / "world" (incremental map) / "panoramic" (single-position 360° spatial fusion) / "none" (no fusion, raw single frame straight to TSP3D).
    enable_scan: bool = False           # world+scan integration: frontier-arrival 360° scans are a BYPASS independent of the WorldLocalMap; only active when fusion_style=world. False = pure world baseline.
    scan_min_gap: int = 90              # Sparse-scan cooldown (env steps) between two scans; >= scan_min_gap steps after the last scan end before a new scan may fire.
    scan_opportunistic_after: int = 60  # Opportunistic scan: if >0, also fire a scan when explore runs this many env steps without locking a nav goal. 0 = off.
    frontier_trigger: bool = True       # Frontier-arrival scan on/off; False = opportunistic-only (no frontier scans).
    # Phase 5b: Temporal PCD Sliding Window (Multi-frame Fusion for TSP3D)
    pcd_window_size: int = 8                # Number of frames fused for point-cloud accumulation; larger = more complete geometry but slower/staler.
    fuse_voxel_size: float = 0.02           # Voxel downsample size (m) for window fusion; larger = fewer points, faster, coarser.
    fuse_max_points: int = 200000           # Cap on fused point count; higher = more detail but heavier sparse-conv inference.
    cam_radius: Optional[float] = None      # camera: send-side horizontal radius crop (m), None = no crop (baseline); align with wm_radius=6.0 -> 6.0
    cam_min_view_disp: float = 0.15         # camera: min displacement (m) to accept a frame into fusion, 0 = off; world uses wm_min_view_disp 0.15
    cam_min_view_yaw: float = 15.0          # camera: min yaw change (deg) to accept a frame into fusion, 0 = off; world uses wm_min_view_yaw 15.0
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
    panoramic_turn_steps: int = 6            # Frames (env steps) per 360° scan; rotation/step = 360/turn_steps (6 -> 60°).
    panoramic_voxel_size: float = 0.02       # Local-frame voxel for intra-scan stitching dedup.
    panoramic_radius: Optional[float] = 6.0  # Send-side horizontal radius crop (m) around the scan position.
    panoramic_max_points: int = 200000       # Point cap for the panoramic / single-frame input.
    panoramic_min_move: float = 2.0          # Min distance (m) from the last scan / init pose to allow another scan (new-frontier guard).
    panoramic_arrive_dist: float = 1.0       # Frontier-arrival trigger radius (m): fallback gate when no frontier cluster is being tracked (no-cluster starvation guard).

    # Phase 7: S-penalty (semantic-field cross-validation)
    enable_s_penalty: bool = True           # Master switch: c' = c * w_S(S) — multiply TSP3D confidence by a semantic-field weight (BLIP2 ITM, zero extra queries) to amplify TP/FP discrimination.
    s_penalty_thresh: float = 0.15          # S below this (ITM raw cosine scale ~0.10-0.15; measured min 0.084) -> apply penalty (w_S = floor).
    s_penalty_floor: float = 0.3            # Lower bound of the dynamic penalty weight w_S = max(floor, S/thresh) when S < thresh (non-zero -> keep recall; S->0 -> floor).
    s_penalty_radius_m: float = 0.5         # S query radius (m) around the detection point.
    s_penalty_uncovered_w: float = 1.0      # weight when S=None/<=0 (no coverage): 1.0 = free pass (A1), <1.0 = mild penalty on unverifiable detections.
    sx_conflict_gate: bool = True        # 冲突门：conf≥0.80 ∧ 0<S<thresh 的检测写入口定向拒绝
    # Phase 7b: geometric admission module (occupancy-consistency gate)
    enable_occ_consistency: bool = True     # box must have occupied-voxel support in the 3D grid
    occ_min_voxels: int = 8                 # occ gate: min occupied voxels supporting the box
    occ_min_ratio: float = 0.001            # occ gate: min occupied/(box volume) ratio
    # Phase 7c: target-memory lifecycle
    merge_dist_thresh: float = 0.5          # cross-view EMA merge distance threshold (m)
    ema_weight_old: float = 0.8             # cross-view EMA smoothing (old * w + new * (1 - w))
    s_penalty_use_surface: bool = True      # query S at the bbox near-surface point (facing camera) instead of the centroid.
    goal_use_surface: bool = True           # navigate to the stored near-surface point (first-write fixed) instead of the centroid.
    
    # Phase 7d: new GD auxiliary mechanism — independent proposer + frame-level vote
    gdp_enable: bool = True              # master switch: run the GD proposal source on the current frame
    gdp_every_n: int = 5                 # GD runs on every N-th frame (cost knob 1/3/5/10)
    gdp_box_thr: float = 0.4             # GD logit threshold (single GD detector -> the non-COCO value)
    gdp_caption_style: str = "vocab"     # "vocab" (full class list) / "target" (target classes only)
    gdp_max_boxes: int = 1               # proposals built per frame (cost guard)
    gdp_weak_guard: bool = True          # a `gd` single vote cannot lock before lock_min_obs observations
    gdp_merge_dist: float = 0.5          # frame-level "same object" XY distance (m)
    gdp_merge_iau: float = 0.3           # frame-level "same object" AABB IaU
    gdp_port: int = 12181                # Grounding-DINO server port
    lock_min_obs: int = 2                # a `gd` single-vote entry needs >= this many observations to be lockable
    lock_exempt: bool = True             # 锁保持：navigate 锁定期间 nav goal 条目豁免近场 near_miss 预算
    lock_exempt_identity: bool = True    # 豁免匹配基准：True = 被追踪条目质心（身份）；False = nav goal 近表面点
    gd_skip_classes: str = ""            # object goals that skip the GD source (e.g. "couch")

    enable_stats: bool = True     # 通用前置量测开关（[NAV]/[STOP] 打印与逐集摘要；零行为）


cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
