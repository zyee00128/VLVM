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
from vlfm.tsp3d_models.utils.target_geometric_gating import (
    TargetGeometricGatingEngine,
    BoxOccupancyConfig,
    BoxPointDensityConfig,
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
            fusion_style: str = "world",  # "camera" / "world" / "panoramic" (fusion routes)
            enable_scan: bool = False,
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
            s_penalty_use_surface: bool = True,
            goal_use_surface: bool = True,
            # geometric admission module
            enable_occ_consistency: bool = True,
            occ_min_voxels: int = 8,
            occ_min_ratio: float = 0.001,
            enable_density_gate: bool = True,
            density_occ_support: bool = True,
            density_min_points: int = 150,
            density_min_frames: int = 2,
            density_cluster_merge_dist: float = 0.5,
            density_min_view_span_deg: float = 20.0,
            # target-memory manager
            merge_dist_thresh: float = 0.5,
            ema_weight_old: float = 0.8,
            # new GD auxiliary mechanism (§3.6): independent proposer + frame-level vote
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

            # D2-2 pick ranking: vote-type priority inside a distance band (ranking only).
            d2_pick_src_first: bool = False,
            d2_pick_src_band: float = 0.5,
            # 注：D2-4（单票自洽硬门，= 原 D2-7 的"按类几何门"支）已判负删除（2026-09-16）。
            # D2-8: GD class whitelist — these object goals skip the GD source entirely.
            d2_gd_skip_classes: str = "",

            # 方向 3 机制（§优化方向 3；无法用前置判据确定 ⇒ 直接排档 A/B；默认全关）
            d2_fill_only: bool = False,
            d2_geo_lock: bool = False,
            d2_geo_min: int = 2,
            d2_stop_gate: bool = False,
            d2_stop_gate_src: str = "tsp3d",
            d2_stop_gate_classes: str = "couch|toilet",
            d2_stop_gate_nobs: int = 1,
            d2_stop_gate_win: int = 30,
            d2_stop_gate_release: str = "explore",
            d2_lock_exempt: bool = False,

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
            s_penalty_use_surface=s_penalty_use_surface,
            goal_use_surface=goal_use_surface,
        )
        self._init_geom_gate(
            enable_occ_consistency=enable_occ_consistency,
            occ_min_voxels=occ_min_voxels,
            occ_min_ratio=occ_min_ratio,
            enable_density_gate=enable_density_gate,
            density_occ_support=density_occ_support,
            density_min_points=density_min_points,
            density_min_frames=density_min_frames,
            density_cluster_merge_dist=density_cluster_merge_dist,
            density_min_view_span_deg=density_min_view_span_deg,
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
        )
        self._init_d2(
            d2_pick_src_first=d2_pick_src_first,
            d2_pick_src_band=d2_pick_src_band,
            d2_gd_skip_classes=d2_gd_skip_classes,
            d2_fill_only=d2_fill_only,
            d2_geo_lock=d2_geo_lock,
            d2_geo_min=d2_geo_min,
            d2_stop_gate=d2_stop_gate,
            d2_stop_gate_src=d2_stop_gate_src,
            d2_stop_gate_classes=d2_stop_gate_classes,
            d2_stop_gate_nobs=d2_stop_gate_nobs,
            d2_stop_gate_win=d2_stop_gate_win,
            d2_stop_gate_release=d2_stop_gate_release,
            d2_lock_exempt=d2_lock_exempt,
        )
        self._lock_min_obs = max(int(lock_min_obs), 1)

        # 3D visual grounding and vision-language evaluation clients
        self._tsp3d_client = TSP3DClient(port=int(os.environ.get("TSP3D_PORT", "12186")))
        self._itm_client = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._text_prompt = text_prompt

        self._det_seed = int(det_seed)
        self._rng_step = 0            # monotonic numpy-RNG seed counter (never reset)
        self._nav_pick_key: Optional[tuple] = None   # dedup of the `[NAV] pick` print (M-b measurement)
        # D2-1 / D2-6 measurements: stop-time & approach-time TSP3D co-confirmation.
        self._nav_stats: Dict[str, Any] = {}         # per-episode counters (printed in `_reset`)
        self._frame_admitted: List[Tuple[str, np.ndarray]] = []   # this frame's admitted candidates (cls, XY)
        self._approach_key: Optional[tuple] = None   # dedup of the `[NAV] approach` print (0.5 m band)
        self._last_pick_meta: Dict[str, Any] = {}    # src / n_obs / gd_conf / XY of the current nav pick
        self._lock_step: int = -1                    # step at which the current lock started (-1 = no lock)
    def _reset(self) -> None:
        """Reset memories, step counters, pointnav model, and 3D occupancy map."""
        if self._gd_proposer is not None and self._gd_proposer.cfg.enabled:
            s = self._gd_proposer.summary()
            print(f"[GD2] episode summary: frames={s['frames']} boxes_above_thr={s['boxes_above_thr']} "
                  f"proposals={s['proposals']} both={self._gd_stats['both']} gd={self._gd_stats['gd']} "
                  f"gd_rejected={self._gd_stats['gd_rejected']} both_cleared={self._gd_stats['both_cleared']} "
                  f"tsp3d_new={self._gd_stats['tsp3d_new']} "
                  f"w_new={self._gd_stats['w_new']} w_merge={self._gd_stats['w_merge']} "
                  f"w_new_tsp={self._gd_stats['w_new_tsp']} w_merge_tsp={self._gd_stats['w_merge_tsp']} "
                  f"gd_gated={self._gd_stats.get('gd_gated', 0)} "
                  f"rejects={s['rejects']}")
            self._gd_proposer.reset()
            self._gd_stats = {
                "both": 0, "gd": 0, "gd_rejected": 0, "both_cleared": 0, "tsp3d_new": 0,
                "w_new": 0, "w_merge": 0, "w_new_tsp": 0, "w_merge_tsp": 0,
                "gd_gated": 0,
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
        # D2-1 measurement: episode-level stop / pick read-out of the finished episode.
        self._log_nav_summary()
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
        self._nav_pick_key = None
        self._approach_key = None
        self._last_pick_meta = {}
        self._frame_admitted = []
        self._r0 = {"hit": 0, "miss": 0, "invis": 0}
        self._gself_recs = []
        self._lock_step = -1
        # 3-9 停止闸状态（拦停坐标键 / 累计步数）
        self._stop_gate_key = None
        self._stop_gate_blocked = 0
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
    ) -> None:
        """Target-memory lifecycle manager (EMA merge + fallback + erasure) bound to the
        policy's memory dicts."""
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
        self._scan_pos: Optional[np.ndarray] = None     # last scan / init pose (min-move novelty guard)
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
        s_penalty_use_surface: bool,
        goal_use_surface: bool,
    ) -> None:
        """S-penalty: c' = c * w_S(semantic field), plus the surface-point goal option."""
        self._s_penalty_cfg = SPenaltyConfig(
            enable=enable_s_penalty,
            thresh=s_penalty_thresh,
            floor=s_penalty_floor,
            radius_m=s_penalty_radius_m,
        )
        self._s_penalty_use_surface = s_penalty_use_surface
        self._goal_use_surface = goal_use_surface
    def _init_geom_gate(
        self,
        *,
        enable_occ_consistency: bool,
        occ_min_voxels: int,
        occ_min_ratio: float,
        enable_density_gate: bool,
        density_occ_support: bool,
        density_min_points: int,
        density_min_frames: int,
        density_cluster_merge_dist: float,
        density_min_view_span_deg: float,
    ) -> None:
        """Geometric admission gate (occ-consistency + density). Runs before S-penalty."""
        self._geom_gate_engine = TargetGeometricGatingEngine(
            occ_cfg=BoxOccupancyConfig(
                enable=enable_occ_consistency,
                min_occupied_voxels=occ_min_voxels,
                min_occupancy_ratio=occ_min_ratio,
            ),
            density_cfg=BoxPointDensityConfig(
                enable=enable_density_gate,
                min_points_confirm=density_min_points,
                min_frames_confirm=density_min_frames,
                cluster_merge_dist=density_cluster_merge_dist,
                use_occupancy_support=density_occ_support,
                min_view_span_deg=density_min_view_span_deg,
            ),
        )

    def _init_d2(
        self,
        *,
        d2_pick_src_first: bool,
        d2_pick_src_band: float,
        d2_gd_skip_classes: str,
        d2_fill_only: bool,
        d2_geo_lock: bool,
        d2_geo_min: int,
        d2_stop_gate: bool,
        d2_stop_gate_src: str,
        d2_stop_gate_classes: str,
        d2_stop_gate_nobs: int,
        d2_stop_gate_win: int,
        d2_stop_gate_release: str,
        d2_lock_exempt: bool,
    ) -> None:
        """Direction-2 / 3 mechanisms, one switch per item:

          D2-2 pick ranking          — vote-type priority inside a distance band (`both > gd > tsp3d`);
          D2-5 R0 bookkeeping        — checked & hit / checked & miss counters (always on);
          D2-8 GD class whitelist    — listed object goals skip the GD source;
          3-4 帧级补框（`d2_fill_only`）  — GD 仅在没有 TSP3D 目标类候选的帧新建条目；
          3-5 几何独立票锁门（`d2_geo_lock`）— `gd` 条目以几何独立票数（≥`d2_geo_min`）解锁；
          3-9 停止闸 v2（`d2_stop_gate*`）— 类条件 + 二次证据 + 窗口释放，配套 `d2_lock_exempt`。

        D2-1 / D2-6（旧停止闸）、D2-3（`both` 共位软加权 c''）与 D2-4（单票自洽硬门）
        已判负并删除（2026-09-16，依据 results/VLVM-V8.md §3.15.4 / §3.15.7–§3.15.13）。
        `g_self` 特征记录保留为量测（零行为，判定行 `g_self separability`）。
        """
        self._d2_pick_src_first = bool(d2_pick_src_first)
        self._d2_pick_band = max(float(d2_pick_src_band), 0.0)
        self._d2_gd_skip = {
            c.strip() for c in str(d2_gd_skip_classes).split("|") if c.strip()
        }
        # 3-4 / 3-5 / 3-9（§优化方向 3）
        self._d2_fill_only = bool(d2_fill_only)
        self._d2_geo_lock = bool(d2_geo_lock)
        self._d2_geo_min = max(int(d2_geo_min), 1)
        self._d2_stop_gate = bool(d2_stop_gate)
        self._d2_stop_gate_src = str(d2_stop_gate_src or "tsp3d")
        self._d2_stop_gate_cls = {
            c.strip() for c in str(d2_stop_gate_classes).split("|") if c.strip()
        }
        self._d2_stop_gate_nobs = max(int(d2_stop_gate_nobs), 0)
        self._d2_stop_gate_win = max(int(d2_stop_gate_win), 1)
        self._d2_stop_gate_release = str(d2_stop_gate_release or "explore")
        self._d2_lock_exempt = bool(d2_lock_exempt)
        # 3-9 拦停记账（当前被拦条目的坐标键 / 累计拦停步数）
        self._stop_gate_key: Optional[tuple] = None
        self._stop_gate_blocked: int = 0
        # R0 counters (per episode) + `g_self` 特征量测（零行为）：每个 GD 提案记录
        # `(xy2, density, box_frac, vote)`，供条目级 tp/fp 可分性判定（D2-4 已删，仅留量测）。
        self._r0: Dict[str, int] = {"hit": 0, "miss": 0, "invis": 0}
        self._gself_recs: list = []

    @staticmethod
    def _self_features(proposal: Any, cam: Any) -> tuple:
        """Cluster-level features of a `gd` proposal — **量测用，不影响行为**。

        `density` = cluster points per box pixel, `box_frac` = box area / image area；
        逐提案记录进 `_gself_recs`，供 `g_self separability` 判定（D2-4 门已判负删除）。
        """
        box = np.asarray(getattr(proposal, "box_px", np.zeros(4)), dtype=np.float64).reshape(-1)
        w = max(float(box[2] - box[0]), 1.0) if box.size >= 4 else 1.0
        h = max(float(box[3] - box[1]), 1.0) if box.size >= 4 else 1.0
        area = w * h
        img = max(float(getattr(cam, "width", 1)) * float(getattr(cam, "height", 1)), 1.0)
        density = float(getattr(proposal, "n_points", 0)) / area
        return density, area / img

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
    ) -> None:
        """New GD auxiliary mechanism: an independent per-frame proposer + vote.
        GD only proposes and votes (`both` / `gd`).
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
        # Vote accounting of the GD source（`w_*` = 3-4 帧级补框影子，零行为）。
        self._gd_stats = {
            "both": 0, "gd": 0, "gd_rejected": 0, "both_cleared": 0, "tsp3d_new": 0,
            "w_new": 0, "w_merge": 0, "w_new_tsp": 0, "w_merge_tsp": 0,
            "gd_gated": 0,
        }

    # ---- new GD mechanism runtime ----
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
        occ-consistency / density gate, exactly like a TSP3D candidate.
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
        if eng.density_acc.cfg.enable and status != "HARD_CONFIRMED":
            return False
        return True

    @staticmethod
    def _world_to_pixel(
        point: np.ndarray, cam: Any
    ) -> Union[None, Tuple[float, float, float]]:
        """World point -> (u, v, forward depth) in the current frame; None when not visible.

        R0 visibility test of the R2 negative-side accounting (§3.11 三): the inverse of
        `_project_rgbd_to_3d_point_cloud`, with the camera-base axes `(forward, left, up)`.
        A point behind the camera, outside the image, or beyond the depth range counts as
        "not checked" and therefore **never** as negative evidence.
        """
        tf = np.asarray(cam.tf_camera_to_episodic, dtype=np.float64)
        p = np.asarray(point, dtype=np.float64).reshape(-1)[:3]
        if p.size < 3 or not np.all(np.isfinite(p)):
            return None
        base = tf[:3, :3].T @ (p - tf[:3, 3])
        fwd, left, up = float(base[0]), float(base[1]), float(base[2])
        if not (float(cam.min_depth) < fwd < float(cam.max_depth)):
            return None
        u = (-left) * float(cam.fx) / fwd + float(cam.width) / 2.0
        v = (-up) * float(cam.fy) / fwd + float(cam.height) / 2.0
        if not (0.0 <= u < float(cam.width) and 0.0 <= v < float(cam.height)):
            return None
        return (u, v, fwd)

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
        # D2-8: object-goal classes on the whitelist skip the GD source entirely.
        if self._d2_gd_skip and any(c.strip() in self._d2_gd_skip for c in target_classes):
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

        # D2-5 R0 three-state bookkeeping (checked & hit / checked & miss): live entries
        # visible in this GD frame, split by whether a GD proposal co-locates with them.
        for _cls_name, _centroids in self._target_3d_memory.items():
            for _cen in _centroids:
                _c_full = np.asarray(_cen, dtype=np.float64).reshape(-1)
                if self._world_to_pixel(_c_full, cam) is None:
                    # 三态记账（D2-5）：不可见 = "未检查"，既非命中亦非否定证据。
                    self._r0["invis"] = int(self._r0.get("invis", 0)) + 1
                    continue
                _c2 = _c_full[:2]
                _hit = any(
                    float(np.linalg.norm(
                        _c2 - np.asarray(p.centroid, dtype=np.float64).reshape(-1)[:2]))
                    <= float(prog.cfg.merge_dist)
                    for p in proposals
                )
                _key = "hit" if _hit else "miss"
                self._r0[_key] = int(self._r0.get(_key, 0)) + 1

        if not proposals:
            return

        # 3-4 帧级补框影子（§优化方向 3，零行为）：本帧 TSP3D 是否已有目标类 admitted
        # 候选 —— 判定 "GD 只在该帧无 TSP3D 候选时新建条目" 会砍掉 / 保留多少写入。
        frame_tsp = len(pending) > 0

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
            # `g_self` 特征量测（零行为）：记录每个 GD 提案的特征；逐条目的
            # tp/fp 可分性在 episode 汇总的 "g_self separability" 行判定。
            _den, _frac = self._self_features(proposal, cam)
            self._gself_recs.append((
                np.asarray(proposal.centroid, dtype=np.float64).reshape(-1)[:2].copy(),
                float(_den), float(_frac), str(vote),
            ))
            if vote == BOTH and matched_idx is not None:
                conf_amp, centroid_np, _surf, classes = pending[matched_idx]
                # 注：D2-3 的 `both` 共位软加权（c'' = c' + w * g_geo）已判负删除
                # （2026-09-16，逐集 bit-identical）；此处保持 TSP3D 的 conf_amp。
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
                # 3-4 帧级补框门（§优化方向 3）：带 TSP3D 目标类候选的帧不得新建 GD 条目。
                new_gate=bool(self._d2_fill_only and frame_tsp),
            )
            if res.get("suppressed"):
                self._gd_stats["gd_gated"] = int(self._gd_stats.get("gd_gated", 0)) + 1
                print(f"[GD2] gd -> gated (frame has TSP3D candidates) '{cls}' at "
                      f"{np.round(np.asarray(proposal.centroid, dtype=np.float64)[:3], 2)} "
                      f"(gd={float(proposal.conf):.2f}) -> suppressed")
                continue
            self._gd_stats["gd"] += 1
            # 3-4 帧级补框影子（§优化方向 3，零行为）：本次写入是"新建"还是"重复"，
            # 以及该帧是否已有 TSP3D 目标类候选（→ 只允许干净帧新建时会砍掉哪些）。
            _new = not bool(res.get("merged", False))
            self._gd_stats["w_new" if _new else "w_merge"] += 1
            if frame_tsp:
                self._gd_stats["w_new_tsp" if _new else "w_merge_tsp"] += 1
            print(f"[GD2] gd -> single vote '{cls}' at "
                  f"{np.round(np.asarray(proposal.centroid, dtype=np.float64)[:3], 2)} "
                  f"(gd={float(proposal.conf):.2f} n_obs={res.get('num_obs')} "
                  f"new={_new} frame_tsp={int(frame_tsp)} "
                  f"geo_ind={res.get('gd_geo_ind')} "
                  f"density={_den:.3f} box_frac={_frac:.3f} "
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

        REJECTED (empty-air) and (when the density gate is on) unconfirmed
        SOFT_ACCUMULATING detections are dropped before the S-penalty gate.
        """
        engine = self._geom_gate_engine
        n = detections.num_detections
        if n == 0:
            return detections
        boxes_np = detections.boxes.detach().cpu().numpy()
        logits_np = detections.logits.detach().cpu().numpy()
        keep = np.ones(n, dtype=bool)
        density_on = engine.density_acc.cfg.enable
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
            drop = (status == "REJECTED") or (density_on and status == "SOFT_ACCUMULATING")
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
        # M-a (§五 量测): raw GD sigmoid of each candidate entry (0.0 = no GD vote).
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

                    # Weak evidence does not lock (§3.6.4): a `gd` single vote needs
                    # `lock_min_obs` observations before it may drive explore -> navigate.
                    # 3-5（§优化方向 3，`d2_geo_lock`）：`gd` 条目的解锁判据改用
                    # "几何独立票 >= `d2_geo_min`"（位移 >= 0.5 m 或视角差 >= 30°），
                    # 替代纯计数 `num_obs`；`tsp3d` / `both` 条目不受影响。
                    st = self._memory_manager.state_of(centroid)
                    src = str(st.get("src") or "tsp3d")
                    if self._d2_geo_lock and src == GD:
                        lockable = int(st.get("gd_geo_ind", 0) or 0) >= int(self._d2_geo_min)
                    else:
                        lockable = is_lockable(src, obs, self._lock_min_obs, self._gd_weak_guard)
                    if not lockable:
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
        if self._d2_pick_src_first:
            # D2-2′: inside a ±band of the nearest centroid, prefer the stronger vote type
            # **`both` > `gd` > `tsp3d`**（实测条目正确率 both 63.2% > gd 41.9% >
            # tsp3d 28.8%，原序与实测相反，2026-09-16 修正）；ranking only — `c'` 与阈值不动。
            _rank = {"tsp3d": 0, "gd": 1, "both": 2}
            d_min = float(dists_2d.min())
            in_band = [
                i for i in range(len(dists_2d))
                if float(dists_2d[i]) <= d_min + self._d2_pick_band
            ]
            chosen_idx = int(max(
                in_band,
                key=lambda i: (_rank.get(str(valid_src[i]), 1), -float(dists_2d[i])),
            ))
        else:
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

        # M-b (§五 量测): what the navigation decision actually locked.
        self._log_nav_pick(
            valid_centroids, valid_obs, valid_src, chosen_idx, position,
            valid_gconf=valid_gconf,
        )

        return self._last_target_coord

    def _log_nav_pick(
        self,
        valid_centroids: List[np.ndarray],
        valid_obs: List[int],
        valid_src: List[str],
        chosen_idx: int,
        position: np.ndarray,
        valid_gconf: Optional[List[float]] = None,
    ) -> None:
        """Log the chosen navigation entry: src / num_obs / GD score / distance / pose,
        plus a compact dump of the whole candidate set (M-b §五 量测).

        M-b: `cands=[src:o<n_obs>:g<gd_conf>@<dist>*]` lists every candidate of the same
        decision (nearest first, `*` = the one actually locked), so any alternative
        ranking key (score-first, in-band vote-type priority, ...) can be evaluated
        offline (what-if) instead of by blind online A/B. The dedup key is unchanged,
        so the print volume of earlier arms stays comparable.
        """
        src = str(valid_src[chosen_idx])
        obs = int(valid_obs[chosen_idx])
        robot_xy = np.asarray(position, dtype=np.float64).reshape(-1)[:2]

        def _gconf(j: int) -> float:
            if valid_gconf is None or j >= len(valid_gconf):
                return 0.0
            return float(valid_gconf[j] or 0.0)

        def _dist(j: int) -> float:
            cen = np.asarray(valid_centroids[j], dtype=np.float64).reshape(-1)
            return float(np.linalg.norm(cen[:2] - robot_xy))

        gconf = _gconf(chosen_idx)
        dist = _dist(chosen_idx)
        # 3-5 shadow：被锁定条目的几何独立票数（判据 = 位移 >= 0.5 m 或 视角差 >= 30°）。
        _geo_ind = int(
            self._memory_manager.state_of(valid_centroids[chosen_idx]).get("gd_geo_ind", 0) or 0
        )
        # D2-1 measurement: remember the locked entry (stop-time evidence) and count the
        # vote-type split of the picks, unfiltered by the print dedup below.
        self._last_pick_meta = {
            "src": src, "n_obs": obs, "gd_conf": gconf, "geo_ind": _geo_ind,
            "xy": np.asarray(valid_centroids[chosen_idx], dtype=np.float64).reshape(-1)[:2].copy(),
        }
        pick = self._nav_stats.setdefault("pick", {})
        pick["n"] = int(pick.get("n", 0)) + 1
        pick[src] = int(pick.get(src, 0)) + 1
        key = (src, obs, round(dist / 0.5))
        if key == self._nav_pick_key:
            return
        self._nav_pick_key = key
        cands = sorted(
            (
                (_dist(j), int(j), str(valid_src[j]), int(valid_obs[j]), _gconf(j))
                for j in range(len(valid_centroids))
            ),
            key=lambda t: (t[0], t[1]),
        )
        brief = ", ".join(
            f"{s}:o{o}:g{g:.2f}@{d:.1f}{'*' if j == chosen_idx else ''}"
            for d, j, s, o, g in cands[:4]
        )
        yaw = float(self._observations_cache.get("robot_heading", 0.0))
        print(
            f"[NAV] pick src={src} n_obs={obs} gd_conf={gconf:.2f} geo_ind={_geo_ind} "
            f"dist={dist:.2f} "
            f"robot={np.round(robot_xy, 2)} yaw={yaw:.2f} "
            f"n_cand={len(valid_centroids)} cands=[{brief}]"
        )

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
        )
        # Kept for the scan-end direction decision (c' ranking); see `scan_behavior`.
        self._last_pending = list(pending)
        # D2-1 measurement: keep this frame's admitted candidates (class, XY) so that the
        # stop-time / approach-time co-confirmation can be attributed to TSP3D.
        self._frame_admitted = [
            (str(cls), np.asarray(centroid_np, dtype=np.float64).reshape(-1)[:2])
            for _conf, centroid_np, _surf, classes in pending
            for cls in classes
        ]

        # With the GD source on, a TSP3D-only write is "single-side" evidence: a NEW
        # entry starts suspicious and a same-frame `both` vote clears the flag again.
        # With GD off there is no second vote at all.
        _gd_on = self._gd_proposer is not None and self._gd_proposer.cfg.enabled
        for (conf_amp, centroid_np, near_surface, active_classes), susp in zip(pending, admitted_meta):
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
        # 3-9 配套（`d2_lock_exempt`）：navigate 模式下 nav goal 条目豁免 near_miss 预算。
        _nav_goal = self._last_target_coord if int(self._lock_step) >= 0 else None
        self._memory_manager.step(
            camera_pos, camera_yaw, cone_fov,
            obstacle_map_3d=self._obstacle_map3d,
            robot_xyz=robot_xyz,
            nav_goal_xy=_nav_goal,
            exempt_nav_goal=bool(self._d2_lock_exempt),
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
            # 3-9 停止闸 v2（§优化方向 3，`d2_stop_gate`）：类条件 + 二次证据 + 窗口释放；
            # 关闭时恒 "allow"（= 优化前行为）。
            verdict = self._stop_gate_verdict()
            if verdict == "allow":
                self._log_stop_confirm(goal, allowed=True)
                self._called_stop = True
                return self._stop_action
            nav = self._nav_stats.setdefault("nav", {})
            nav["stop_blocked"] = int(nav.get("stop_blocked", 0)) + 1
            if verdict == "release" and str(self._d2_stop_gate_release) != "explore":
                # 窗口到期：强制放行（veto → delay）
                nav["stop_released"] = int(nav.get("stop_released", 0)) + 1
                self._log_stop_confirm(goal, allowed=True)
                self._called_stop = True
                return self._stop_action
            if verdict == "release":
                # 窗口到期：释放路径 = 放弃该条目并回落 explore（本步先交回 pointnav 动作，
                # 下一步 `_get_target_object_location` 无该候选 ⇒ mode=explore）。
                nav["stop_abandoned"] = int(nav.get("stop_abandoned", 0)) + 1
                self._log_stop_confirm(goal, allowed=False)
                self._stop_gate_abandon()
                return self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
            # 拦停中：本步不停止（交回 pointnav 动作，通常为原地转向/微调）
            self._log_stop_confirm(goal, allowed=False)
            return self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)

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
                self._approach_key = None
                mode = "explore"
                action = self._explore(observations)
            else:
                if self._lock_step < 0:
                    self._lock_step = self._num_steps
                mode = "navigate"
                print(f"[TSP3D Mode] Target '{self._target_object}' located at {goal_3d}. Navigating.")
                self._log_approach(goal_3d, robot_xyz)
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


    # D2-1 / D2-6 measurements: stop-time & approach-time TSP3D co-confirmation
    def _tsp3d_confirms(self, xy: Any, tol: float = 0.5) -> bool:
        """True when this frame's admitted TSP3D candidates co-locate with `xy`."""
        if xy is None:
            return False
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        for _cls, cxy in self._frame_admitted:
            c = np.asarray(cxy, dtype=np.float64).reshape(-1)[:2]
            if float(np.linalg.norm(c - p)) <= float(tol):
                return True
        return False

    def _stop_gate_verdict(self) -> str:
        """3-9 / D2-1″ 停止闸的判定（§优化方向 3）。

        拦停条件（全部满足）：`d2_stop_gate` 开 · 锁定票型 = `d2_stop_gate_src` ·
        目标类 ∈ `d2_stop_gate_classes` · 条目**实时**票型仍为 `d2_stop_gate_src` 且
        `num_obs <= d2_stop_gate_nobs`。二次证据 = 条目被再次写入（`num_obs` 增长）
        或票型升级（both / gd）⇒ 立即放行。

        返回 "allow"（正常停）/ "block"（窗口内拦停，继续接近）/ "release"
        （窗口到期，按 `d2_stop_gate_release` 放行或回落 explore）。
        """
        if not self._d2_stop_gate:
            return "allow"
        meta = self._last_pick_meta or {}
        xy = meta.get("xy")
        if xy is None:
            return "allow"
        if str(meta.get("src", "tsp3d")) != self._d2_stop_gate_src:
            return "allow"
        if self._d2_stop_gate_cls:
            classes = {c.strip() for c in self._target_object.split("|") if c.strip()}
            if not (classes & self._d2_stop_gate_cls):
                return "allow"
        st = self._memory_manager.state_of(xy, tol=0.5)
        if not st:
            return "allow"   # 条目已被回收 ⇒ 不再拦停
        src_live = str(st.get("src") or "tsp3d")
        obs_live = int(st.get("num_obs", int(meta.get("n_obs", 1) or 1)) or 1)
        if src_live != self._d2_stop_gate_src:
            return "allow"   # 二次证据：票型升级（both / gd）
        if obs_live > int(self._d2_stop_gate_nobs):
            return "allow"   # 二次证据：条目被再次写入
        key = (round(float(np.asarray(xy).reshape(-1)[0]), 2),
               round(float(np.asarray(xy).reshape(-1)[1]), 2))
        if self._stop_gate_key != key:
            self._stop_gate_key = key
            self._stop_gate_blocked = 0
        self._stop_gate_blocked = int(self._stop_gate_blocked) + 1
        # 窗口语义：拦停 `d2_stop_gate_win` 步后（第 win+1 次判定）放行。
        if self._stop_gate_blocked > int(self._d2_stop_gate_win):
            return "release"
        return "block"

    def _stop_gate_abandon(self) -> None:
        """3-9 释放路径（release=explore）：丢弃被拦条目并清掉锁存 / 接近记账。"""
        meta = self._last_pick_meta or {}
        self._drop_entry_at(meta.get("xy"))
        self._last_target_coord = None
        self._lock_step = -1
        self._approach_key = None
        self._stop_gate_key = None
        self._stop_gate_blocked = 0

    def _drop_entry_at(self, xy: Any, tol: float = 0.5) -> None:
        """丢弃距离 `xy` 最近的记忆条目（四表同步删除；3-9 释放路径用）。"""
        if xy is None:
            return
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        for cls in list(self._target_3d_memory.keys()):
            recs = self._target_3d_memory.get(cls, [])
            best_i, best_d = -1, float(tol)
            for i, c in enumerate(recs):
                d = float(np.linalg.norm(
                    np.asarray(c, dtype=np.float64).reshape(-1)[:2] - p))
                if d <= best_d:
                    best_d, best_i = d, i
            if best_i < 0:
                continue
            drop_c = np.asarray(recs[best_i], dtype=np.float64).reshape(-1)[:2].copy()
            for tbl in (self._target_3d_memory, self._target_surface_memory,
                        self._target_verify_state, self._target_fallback_state):
                lst = tbl.get(cls)
                if isinstance(lst, list) and best_i < len(lst):
                    lst.pop(best_i)
            print(f"[STOPG] release: dropped '{cls}' entry at {np.round(drop_c, 2)}")
            return

    def _log_stop_confirm(self, goal: np.ndarray, allowed: bool = True) -> None:
        """Stop-time measurement：本次停止（或被拦）的票型 / 单帧共位 / 距锁步数。

        `stop_allowed=False` 表示被 3-9 停止闸拦下（`d2_stop_gate`）。放行的停止计入
        `stops` / 共位分档；拦停尝试计入 `stop_blocked`（在 `_pointnav` 里累加），
        两者分开统计，避免拦停把停止侧口径打散。
        """
        meta = self._last_pick_meta or {}
        confirm = self._tsp3d_confirms(meta.get("xy"))
        steps_since_lock = (
            int(self._num_steps) - int(self._lock_step) if self._lock_step >= 0 else -1
        )
        nav = self._nav_stats.setdefault("nav", {})
        if allowed:
            nav["stops"] = int(nav.get("stops", 0)) + 1
            key = "stop_confirmed" if confirm else "stop_unconfirmed"
            nav[key] = int(nav.get(key, 0)) + 1
            if 0 <= steps_since_lock <= 5:
                nav["stops_within5"] = int(nav.get("stops_within5", 0)) + 1
        print(f"[STOP] step={self._num_steps} src={meta.get('src', '?')} "
              f"n_obs={meta.get('n_obs', 1)} "
              f"gd_conf={float(meta.get('gd_conf', 0.0) or 0.0):.2f} "
              f"geo_ind={int(meta.get('geo_ind', 0) or 0)} "
              f"tsp3d_confirm={confirm} steps_since_lock={steps_since_lock} "
              f"stop_allowed={bool(allowed)}")

    def _log_approach(self, goal: np.ndarray, robot_xyz: np.ndarray) -> None:
        """Approach-time evidence (D2-6): distance to the locked entry + whether this
        frame's TSP3D candidates confirm it, printed when the 0.5 m band changes so the
        confirmation window (distance / frames) can be sized offline."""
        meta = self._last_pick_meta or {}
        g = np.asarray(goal, dtype=np.float64).reshape(-1)[:2]
        robot = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2]
        dist = float(np.linalg.norm(g - robot))
        dedup = (str(meta.get("src", "?")), int(round(dist / 0.5)))
        if dedup == self._approach_key:
            return
        self._approach_key = dedup
        confirm = self._tsp3d_confirms(meta.get("xy"))
        nav = self._nav_stats.setdefault("nav", {})
        key = "approach_confirm" if confirm else "approach_unconfirm"
        nav[key] = int(nav.get(key, 0)) + 1
        print(f"[NAV] approach step={self._num_steps} src={meta.get('src', '?')} "
              f"n_obs={meta.get('n_obs', 1)} dist={dist:.2f} tsp3d_confirm={confirm}")

    def _log_nav_summary(self) -> None:
        """Episode-level D2-1 read-out of the finished episode (printed on `_reset`)."""
        nav = self._nav_stats.get("nav") or {}
        pick = self._nav_stats.get("pick") or {}
        if nav or pick:
            print(f"[NAV] episode summary: stops={nav.get('stops', 0)} "
                  f"stops_within5={nav.get('stops_within5', 0)} "
                  f"stop_confirmed={nav.get('stop_confirmed', 0)} "
                  f"stop_unconfirmed={nav.get('stop_unconfirmed', 0)} "
                  f"approach_confirmed={nav.get('approach_confirm', 0)} "
                  f"approach_unconfirmed={nav.get('approach_unconfirm', 0)} "
                  f"picks={pick.get('n', 0)} pick_gd={pick.get('gd', 0)} "
                  f"pick_both={pick.get('both', 0)} pick_tsp3d={pick.get('tsp3d', 0)} "
                  f"stop_blocked={nav.get('stop_blocked', 0)} "
                  f"stop_released={nav.get('stop_released', 0)} "
                  f"stop_abandoned={nav.get('stop_abandoned', 0)} "
                  f"r0_hit={self._r0.get('hit', 0)} r0_miss={self._r0.get('miss', 0)} "
                  f"r0_invis={self._r0.get('invis', 0)}")
        self._nav_stats = {}

    def _memory_entries_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Compact snapshot of the live memory entries (A4 measurement #2, §3.7.10).

        Read by the evaluation trainer right after an episode finishes (the policy only
        resets its memory on the next `act()`), so the finished episode's entries can be
        labelled tp / fp against the GT target bbox. Plain scalars only, keyed
        `"<class>[i]"` (never injected into `policy_info`: habitat's
        `extract_scalars_from_info` would try `float()` on non-scalar payloads).
        `gd_conf` (M-a §五 量测) is the entry's strongest raw GD sigmoid, kept next to
        `conf` (the entry's `c'`) so the two scales can be compared per entry.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for cls, centroids in self._target_3d_memory.items():
            ver = self._target_verify_state.get(cls, [])
            states = self._target_fallback_state.get(cls, [])
            for i, centroid in enumerate(centroids):
                st = states[i] if i < len(states) else {}
                v = ver[i] if i < len(ver) else (1, np.zeros(2), 0.0, 0.0)
                c = np.asarray(centroid, dtype=np.float64).reshape(-1)
                _gs = self._gself_of(c[:2])
                out[f"{cls}[{i}]"] = {
                    "class": str(cls),
                    "x": float(c[0]),
                    "y": float(c[1]) if c.size > 1 else 0.0,
                    "src": str(st.get("src") or "tsp3d"),
                    "num_obs": int(v[0]) if len(v) >= 1 else 1,
                    "conf": float(v[3]) if len(v) > 3 else 0.0,
                    "gd_conf": float(st.get("gd_conf", 0.0) or 0.0),
                    "suspicious": bool(st.get("suspicious", False)),
                    "gd_geo_obs": int(st.get("gd_geo_obs", 0) or 0),
                    "gd_geo_ind": int(st.get("gd_geo_ind", 0) or 0),
                    "gself_n": int(_gs[0]),
                    "gself_density": float(_gs[1]),
                    "gself_box_frac": float(_gs[2]),
                }
        return out

    def _gself_of(self, xy: np.ndarray) -> tuple:
        """Mean `g_self` features of the GD proposals that touched this entry (量测).

        Matches the per-proposal records within 0.5 m of the entry centroid, so the
        EMA merge (which moves a live centroid) cannot break the attribution.
        """
        p = np.asarray(xy, dtype=np.float64).reshape(-1)[:2]
        n, dsum, fsum = 0, 0.0, 0.0
        for r in self._gself_recs:
            if float(np.linalg.norm(np.asarray(r[0], dtype=np.float64) - p)) <= 0.5:
                n += 1
                dsum += float(r[1])
                fsum += float(r[2])
        if n == 0:
            return (0, 0.0, 0.0)
        return (n, dsum / n, fsum / n)

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
    fb_near_radius: float = 2.5         # Near-field confirmation cone radius (m); VLFM uses max_depth*0.5 (2.5m @ max_depth=5).
    fb_hysteresis: int = 5              # Trusted centroid: consecutive near-field frames without a fresh merge before deletion.
    fb_suspicious_hysteresis: int = 2   # Suspicious centroid (far/low-conf): fewer frames before deletion.
    fb_suspicious_conf: float = 0.75    # Trust boundary (two-tier, must be > sigma_tar to activate): detections in [sigma_tar, fb_suspicious_conf) are flagged suspicious and deleted if not re-detected within 2 near-field frames; 0.75 is the most conservative active tier (V4.1).

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
    s_penalty_use_surface: bool = True      # query S at the bbox near-surface point (facing camera) instead of the centroid.
    goal_use_surface: bool = True           # navigate to the stored near-surface point (first-write fixed) instead of the centroid.
    # Phase 7b: geometric admission module (occ-consistency + density gates)
    enable_occ_consistency: bool = True     # box must have occupied-voxel support in the 3D grid
    occ_min_voxels: int = 8                 # occ gate: min occupied voxels supporting the box
    occ_min_ratio: float = 0.001            # occ gate: min occupied/(box volume) ratio
    enable_density_gate: bool = True        # density x occupancy fusion gate (occ support OR density HARD-confirm)
    density_occ_support: bool = True        # density gate: occupancy support as parallel spatio-temporal evidence
    density_min_points: int = 150           # density gate: min accumulated in-box points to confirm
    density_min_frames: int = 2             # density gate: min observation frames to confirm
    density_cluster_merge_dist: float = 0.5 # density gate: cross-frame cluster association radius (m)
    density_min_view_span_deg: float = 20.0 # density path: required multi-view azimuth span (deg, 0=off)
    # Phase 7c: target-memory lifecycle
    merge_dist_thresh: float = 0.5          # cross-view EMA merge distance threshold (m)
    ema_weight_old: float = 0.8             # cross-view EMA smoothing (old * w + new * (1 - w))
    
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


    # Phase 7h: direction-2 mechanisms (§优化方向 2; all default OFF except the R0 counters)
    # 注：D2-1 / D2-6（停止闸）、D2-3（`both` 软加权）、D2-4（单票自洽硬门）已判负删除（2026-09-16）
    d2_pick_src_first: bool = False      # D2-2: vote-type priority inside a distance band (ranking only)
    d2_pick_src_band: float = 0.5        # D2-2: band width (m)
    d2_gd_skip_classes: str = ""         # D2-8: object goals that skip the GD source (e.g. "couch")

    # Phase 7i: 方向 3 机制（§优化方向 3；无法用前置判据确定 ⇒ 直接排档 A/B；默认全关）
    d2_fill_only: bool = False           # 3-4: GD only creates entries in frames without TSP3D candidates
    d2_geo_lock: bool = False            # 3-5: `gd` unlock needs geometric-independent votes instead of num_obs
    d2_geo_min: int = 2                  # 3-5: required independent votes (>=2)
    d2_stop_gate: bool = False           # 3-9: stop gate v2 (class-conditional + second evidence + window release)
    d2_stop_gate_src: str = "tsp3d"      # 3-9: the vote source being gated
    d2_stop_gate_classes: str = "couch|toilet"  # 3-9: object goals the gate applies to ("" = all)
    d2_stop_gate_nobs: int = 1           # 3-9: gate entries with num_obs <= this
    d2_stop_gate_win: int = 30           # 3-9: blocked steps before release
    d2_stop_gate_release: str = "explore"  # 3-9: "explore" (drop entry & fall back) / "stop" (force allow)
    d2_lock_exempt: bool = False         # 3-9: nav-goal entry exempt from the near-field near_miss budget


cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
