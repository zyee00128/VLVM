import os
from dataclasses import dataclass, fields
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
from vlfm.tsp3d_models.utils.s_penalty import (
    SPenaltyConfig, 
    apply_s_penalty, 
    query_semantic_at,
)
from vlfm.tsp3d_models.utils.scan_behavior import (
    _act_panoramic,
    _act_world_scan,
)
from vlfm.tsp3d_models.utils.target_memory_manager import MemoryManagerConfig, TargetMemoryManager
from vlfm.tsp3d_models.utils.target_geometric_gating import (
    TargetGeometricGatingEngine,
    BoxOccupancyConfig,
    BoxPointDensityConfig,
)
from vlfm.tsp3d_models.utils.near_field_docking import NearFieldDockingConfig, NearFieldDockingEngine


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
            fusion_style: str = "world",  # "camera" / "world" / "panoramic"
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
            # target-memory manager
            merge_dist_thresh: float = 0.5,
            ema_weight_old: float = 0.8,
            enable_near_field_exempt: bool = False,
            exempt_near_radius: float = 1.0,
            exempt_min_merges: int = 1,
            enable_free_space_erasure: bool = False,
            free_erasure_radius: float = 0.3,
            free_erasure_min_explored: int = 5,
            free_erasure_free_ratio: float = 0.85,
            free_erasure_soft_only: bool = True,
            # geometric admission module
            enable_occ_consistency: bool = False,
            occ_min_voxels: int = 8,
            occ_min_ratio: float = 0.001,
            enable_density_gate: bool = False,
            density_occ_support: bool = True,
            density_min_points: int = 150,
            density_min_frames: int = 2,
            density_cluster_merge_dist: float = 0.5,
            density_min_view_span_deg: float = 0.0,
            # near-field camera pitch-down
            pitch: bool = False,
            pitch_trigger_dist: float = 1.5,
            pitch_max_down_steps: int = 1,
            # P3 diagnostics (off = default world behaviour)
            diag_enable: bool = False,
            diag_conf_floor: float = 0.30,
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

        self._scan_turn_action = torch.tensor([[6]], dtype=torch.long)
        self._panoramic_min_move = float(panoramic_min_move)
        self._panoramic_arrive_dist = float(panoramic_arrive_dist)
        self._panoramic_turn_steps = max(int(panoramic_turn_steps), 1)
        self._panoramic_scan_angle_deg = 360.0 / self._panoramic_turn_steps
        _expected_turns = max(1, int(round(self._panoramic_scan_angle_deg / 30.0)))
        _env_turns = max(int(os.environ.get("TURN_LEFT_WIDE_TURNS", str(_expected_turns))), 1)
        _total_deg = self._panoramic_turn_steps * _env_turns * 30.0
        if _env_turns != _expected_turns or abs(_total_deg - 360.0) > 1e-6:
            print(
                f"[Panoramic] WARNING: panoramic_turn_steps={self._panoramic_turn_steps} "
                f"rotates {_total_deg:.0f}° in total (env TURN_LEFT_WIDE_TURNS={_env_turns} "
                f"= {_env_turns * 30}°/step; expected {_expected_turns} turns at "
                f"{self._panoramic_scan_angle_deg:.1f}°/step); coverage is not a full 360°."
            )
        
        self._enable_fb = enable_fb
        self._fb_near_radius = max_depth * 0.5 # fb_near_radius
        self._fb_hysteresis = fb_hysteresis
        self._fb_suspicious_hysteresis = fb_suspicious_hysteresis
        self._fb_suspicious_conf = fb_suspicious_conf
        self._s_penalty_cfg = SPenaltyConfig(
            enable=enable_s_penalty,
            thresh=s_penalty_thresh,
            floor=s_penalty_floor,
            radius_m=s_penalty_radius_m,
        )
        self._s_penalty_use_surface = s_penalty_use_surface
        self._goal_use_surface = goal_use_surface

        # target-memory lifecycle manager (native fallback).
        self._memory_manager = TargetMemoryManager(MemoryManagerConfig(
            enable_fallback=enable_fb,
            fb_near_radius=self._fb_near_radius,
            fb_hysteresis=self._fb_hysteresis,
            fb_suspicious_hysteresis=self._fb_suspicious_hysteresis,
            fb_suspicious_conf=self._fb_suspicious_conf,
            enable_near_field_exempt=enable_near_field_exempt,
            exempt_near_radius=exempt_near_radius,
            exempt_min_merges=exempt_min_merges,
            enable_free_space_erasure=enable_free_space_erasure,
            free_erasure_radius=free_erasure_radius,
            free_erasure_min_explored=free_erasure_min_explored,
            free_erasure_free_ratio=free_erasure_free_ratio,
            free_erasure_soft_only=free_erasure_soft_only,
            merge_dist_thresh=merge_dist_thresh,
            ema_weight_old=ema_weight_old,
        ))
        self._memory_manager.bind(
            self._target_3d_memory, self._target_surface_memory,
            self._target_verify_state, self._target_fallback_state,
        )

        # geometric admission gate.
        # Runs BEFORE the S-penalty gate inside `_query_and_process`.
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

        # near-field pitch-down engine: keeps low targets
        # inside the near-field FOV so TSP3D can re-detect / merge while approaching.
        self._pitch_engine = NearFieldDockingEngine(NearFieldDockingConfig(
            pitch_enable=pitch,
            pitch_trigger_dist=pitch_trigger_dist,
            pitch_max_down_steps=pitch_max_down_steps,
        ))
        self._nav_aim_xyz: Optional[np.ndarray] = None     # 3D aim (incl. z) of the chosen target
        self._pitch_cam_down: int = 0                         # current camera pitch-down steps (module 4)

        # Fusion route selection: "camera" (temporal window) / "world" (incremental map) 
        # / "panoramic" (single-position 360° spatial fusion) / "none" (no fusion mechanism).
        self._fusion_style = fusion_style
        self._enable_scan = bool(enable_scan)
        self._frontier_trigger = bool(frontier_trigger)
        self._det_seed = int(det_seed)
        self._rng_step = 0            # monotonic numpy-RNG seed counter (never reset; P0 determinism)
        self._diag_enable = bool(diag_enable)
        self._diag_conf_floor = float(diag_conf_floor)
        self._diag_logs: List[Dict[str, Any]] = []
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
        self._scan_remaining: int = 0                   # remaining in-place turn frames of a scan
        self._scan_pcd_frames: list = []                # world-frame pcd slices of the in-progress scan
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

        # 3D visual grounding and vision-language evaluation clients
        self._tsp3d_client = TSP3DClient(port=int(os.environ.get("TSP3D_PORT", "12186")))
        self._itm_client = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._text_prompt = text_prompt

        # Core 3D spatial representations
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
        self._target_fallback_state.clear()
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
        self._diag_logs = []
        self._nav_aim_xyz = None
        self._pitch_cam_down = 0
        self._did_reset = True

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
        conf_floor: Optional[float] = None,
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
            conf_floor=conf_floor,
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
        diag_out: Optional[List[Dict[str, Any]]] = None,
    ) -> ObjectDetections:
        """Geometric admission on the current candidates.
        REJECTED (empty-air) and (when the density gate is on) unconfirmed
        SOFT_ACCUMULATING detections are dropped before the S-penalty gate.
        ``diag_out`` (optional, P3 diagnostics) collects dropped candidates as
        {conf, conf_amp, centroid, stage='geom_reject', reason=status}.
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
            if drop and diag_out is not None:
                diag_out.append({
                    "conf": float(logits_np[i]),
                    "conf_amp": float(logits_np[i]),
                    "centroid": np.mean(boxes_np[i], axis=0),
                    "stage": "geom_reject",
                    "reason": status,
                    "admitted": False,
                })
            keep[i] = not drop
        if not keep.all():
            detections.filter_by_mask(keep)
        return detections


    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """
        Navigation goal among the memorized target candidates. 
        Distance-latch keeps goal switches stable.

        world/camera/none: closest centroid with a distance-latch.
        panoramic: rank candidates by their S-penalty score c'; 
                   nearest is the fallback when c' is missing.
        """
        target_classes = self._target_object.split("|")
        valid_centroids: List[np.ndarray] = []
        valid_goals: List[np.ndarray] = []
        valid_confs: List[Optional[float]] = []

        for cls in target_classes:
            if cls in self._target_3d_memory and len(self._target_3d_memory[cls]) > 0:
                surf_list = self._target_surface_memory.get(cls, [])
                verify_list = self._target_verify_state.get(cls, [])
                for i, centroid in enumerate(self._target_3d_memory[cls]):
                    valid_centroids.append(np.array(centroid))
                    if self._goal_use_surface and i < len(surf_list):
                        valid_goals.append(np.array(surf_list[i]))
                    else:
                        valid_goals.append(np.array(centroid))
                    conf = None
                    if i < len(verify_list):
                        v = verify_list[i]
                        conf = float(v[3]) if len(v) >= 4 else None
                    valid_confs.append(conf)

        if len(valid_centroids) == 0:
            self._nav_aim_xyz = None
            return None

        robot_xy = np.asarray(position)[:2]
        if self._fusion_style == "panoramic" and any(c is not None for c in valid_confs):
            # Rank by c' (highest = most target-matching); nearest as a tiebreak.
            def _key(i: int) -> tuple:
                conf_i = valid_confs[i] if valid_confs[i] is not None else -1.0
                return (conf_i, -float(np.linalg.norm(valid_centroids[i][:2] - robot_xy)))
            chosen_idx = int(max(range(len(valid_confs)), key=_key))
        else:
            centroids = np.array(valid_centroids)
            dists_2d = np.linalg.norm(centroids[:, :2] - robot_xy, axis=1)
            chosen_idx = int(np.argmin(dists_2d))

        chosen_2d = valid_goals[chosen_idx][:2].copy()
        # 3D aim (incl. z) for pitch.
        chosen_goal3 = np.asarray(valid_goals[chosen_idx], dtype=np.float64)
        if chosen_goal3.shape[0] < 3:
            chosen_goal3 = np.asarray(valid_centroids[chosen_idx], dtype=np.float64)

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

        # Pitch aim (3D, incl. z) for the near-field camera look-down.
        self._nav_aim_xyz = chosen_goal3

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
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        robot_yaw = self._observations_cache.get("robot_heading", 0.0)

        # Unified input pipeline: camera window / world map -> update -> prepare.
        # (Panoramic route is driven separately by the policy scan state machine.)
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
        raw_detections, _ = self._query_tsp3d_client(
            pcd_local, self._target_object,
            conf_floor=self._diag_conf_floor if self._diag_enable else None,
        )

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

        if self._diag_enable:
            # P3 diagnostics: log raw detections below sigma_tar (the server returns
            # candidates down to diag_conf_floor when enabled). They never enter the
            # decision pipeline (filter_by_conf below still drops them at sigma_tar).
            for det, box_g in zip(valid_raw, boxes_3d_global):
                conf_i = float(det.get("confidence", 0.0))
                if conf_i < self._sigma_tar:
                    self._diag_logs.append({
                        "conf": conf_i,
                        "conf_amp": conf_i,
                        "centroid": np.mean(np.asarray(box_g), axis=0),
                        "stage": "below_sigma",
                        "reason": "below_sigma_tar",
                        "admitted": False,
                    })

        detections.filter_by_conf(self._sigma_tar)
        target_classes = [c.strip() for c in self._target_object.split("|") if c.strip()]
        detections.filter_by_class(target_classes, use_vlfm_nlp=self._nlp_mode)
        # Skip memory accumulation during initialization turning (repeated surfaces pollute memory).
        if not self._done_initializing:
            return detections

        # REJECTED (empty-air) -> dropped before S-penalty
        if self._geom_gate_engine.enabled:
            detections = self._geom_gate(
                detections, pcd_world, robot_xyz=robot_xyz,
                diag_out=self._diag_logs if self._diag_enable else None,
            )

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
        # Keep this query's admitted candidates (c') for the scan-end direction decision.
        self._last_pending = list(pending)
        for (conf_amp, centroid_np, near_surface, active_classes), susp in zip(pending, admitted_meta):
            for cls in active_classes:
                self._memory_manager.accumulate(
                    cls, centroid_np, confidence=conf_amp, near_surface=near_surface,
                    robot_xyz=robot_xyz, robot_yaw=robot_yaw, max_depth=max_depth,
                    suspicious_override=bool(susp) if susp is not None else None,
                )
        # Target Memory Manager
        camera_pos = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)
        cone_fov = get_fov(self._fx, self._depth_image_shape[1])
        self._memory_manager.step(
            camera_pos, camera_yaw, cone_fov,
            obstacle_map_3d=self._obstacle_map3d,
            robot_xyz=robot_xyz,
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
            self._called_stop = True
            return self._stop_action

        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _nav_action(self, robot_xyz: np.ndarray) -> Optional[torch.Tensor]:
        """Pitch: one look_down step while the near-field pitch target is not reached."""
        eng = self._pitch_engine
        aim = self._nav_aim_xyz
        if not eng.config.pitch_enable or aim is None or len(aim) < 3:
            return None
        robot = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)
        target = int(eng.desired_pitch_at(aim, robot, camera_z=self._camera_height)[0])
        if target <= self._pitch_cam_down:
            return None
        self._pitch_cam_down += 1
        return self._look_down_action

    def _restore_action(self) -> Optional[torch.Tensor]:
        """Pitch: one look_up step to level the camera after leaving navigate."""
        eng = self._pitch_engine
        if not eng.config.pitch_enable or self._pitch_cam_down <= 0:
            return None
        self._pitch_cam_down -= 1
        return self._look_up_action

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        # P0 determinism (V7.1): reseed numpy once per env step with a monotonic
        # counter, so every np.random consumer of this step (send-side point cap,
        # density-cluster subsample, ...) is reproducible run-to-run. The counter
        # is never reset across episodes; two identical runs reproduce identical
        # seed sequences -> bit-stable detection totals. Distribution semantics
        # of the random cap are preserved (uniform random subset per step).
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

        else:
            if self._fusion_style == "world" and self._enable_scan:
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
                    # Pitch: level the camera before resuming explore after a down-pitch.
                    pitch_act = self._restore_action()
                    if pitch_act is not None:
                        mode = "pitch_restore"
                        action = pitch_act
                    else:
                        mode = "explore"
                        action = self._explore(observations)
                else:
                    mode = "navigate"
                    print(f"[TSP3D Mode] Target '{self._target_object}' located at {goal_3d}. Navigating.")
                    # Pitch: near-field pitch-down (look down over 1 step while approaching a low target).
                    pitch_act = self._nav_action(robot_xyz)
                    if pitch_act is not None:
                        mode = "pitch_down"
                        action = pitch_act
                    else:
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
    use_vlfm_nlp: bool = False           # Use raw NLP prompt formatting; True = multi-class synonym merging, False = use only the primary class.

    cap_style: str = "random"           # send-side point cap (shared): "random" (baseline) / "near_first" (near-field priority)
    det_seed: int = 0                   # P0 determinism (V7.1): base of the per-step numpy reseed; also set TSP3D_SEED for the model RNG. Same config -> bit-stable double-run.
    distance_sample: bool = True        # distance-adaptive sampling (dense near / sparse far, shared post-processing)
    near_dist: float = 1.5              # distance sampling: near/mid band boundary (m)
    mid_dist: float = 3.0               # distance sampling: mid/far band boundary (m)
    near_voxel: float = 0.01            # distance sampling: near-band voxel (m)
    mid_voxel: float = 0.02             # distance sampling: mid-band voxel (m)
    far_voxel: float = 0.05             # distance sampling: far-band voxel (m)
    fusion_style: str = "world"         # Fusion route: "camera" (8-frame temporal window) / "world" (incremental map) / "panoramic" (single-position 360° spatial fusion) / "none" (no fusion, raw single frame straight to TSP3D). Replaces the old use_world_map bool.
    enable_scan: bool = False           # world+scan integration: frontier-arrival 360° scans are a BYPASS independent of the WorldLocalMap; only active when fusion_style=world. False = pure world baseline.
    scan_min_gap: int = 90              # Sparse-scan cooldown (env steps) between two frontier scans (last-resort, 09-04); >= scan_min_gap steps after the last scan end before a new scan may fire.
    scan_opportunistic_after: int = 60   # Opportunistic scan: if >0, also fire a scan when explore runs this many env steps without locking a nav goal (hunts the target on long empty travels). 0 = off.
    frontier_trigger: bool = True       # Frontier-arrival scan on/off; False = opportunistic-only (no frontier scans).

    # Phase 5b: Temporal PCD Sliding Window (Multi-frame Fusion for TSP3D)
    pcd_window_size: int = 8                # Number of frames fused for point-cloud accumulation; larger = more complete geometry but slower/staler.
    fuse_voxel_size: float = 0.02           # Voxel downsample size (m) for window fusion; larger = fewer points, faster, coarser.
    fuse_max_points: int = 200000           # Cap on fused point count; higher = more detail but heavier sparse-conv inference.
    cam_radius: Optional[float] = None      # camera: send-side horizontal radius crop (m), None = no crop (baseline); align with wm_radius=6.0 -> 6.0
    cam_min_view_disp: float = 0.15         # camera: min displacement (m) to accept a frame into fusion, 0 = off; world uses wm_min_view_disp 0.15
    cam_min_view_yaw: float = 15.0          # camera: min yaw change (deg) to accept a frame into fusion, 0 = off; world uses wm_min_view_yaw 15.0

    # Phase 5c: World-frame Local Map (TSP3DInputPreprocessor)
    wm_max_frames: Optional[int] = 8  # world: frame-window cap (keep voxels of the most recent N frames); None = all history
    wm_voxel_size: float = 0.02       # world: local map voxel size (m) for fixed-grid dedup
    wm_max_voxels: int = 400000       # world: hard cap on map voxels
    wm_radius: Optional[float] = 6.0  # world: local map radius (m), None = no crop (only max_voxels hard cap); cam_radius=None is the symmetric case
    wm_min_view_disp: float = 0.15    # world: min displacement (m) to accept a frame into fusion
    wm_min_view_yaw: float = 15.0     # world: min yaw change (deg) to accept a frame into fusion
    wm_near_refresh_radius: Optional[float] = 3.0  # world: near-field refresh radius (m). None=off (first-observation accounting); >0: re-observed near-field voxels are re-owned by the current frame (cam-style accounting, survives frame-window slide-out).
    wm_near_refresh_value: bool = False  # world: also overwrite the stored point of refreshed near-field voxels with the current observation (cam-style sliding refresh of the value layer). False = value layer stays first.

    # Phase 5d: Panoramic Fusion
    panoramic_turn_steps: int = 6            # Frames (env steps) per 360° scan; rotation/step = 360/turn_steps (6 -> 60°).
    panoramic_voxel_size: float = 0.02       # Local-frame voxel for intra-scan stitching dedup.
    panoramic_radius: Optional[float] = 6.0  # Send-side horizontal radius crop (m) around the scan position.
    panoramic_max_points: int = 200000       # Point cap for the panoramic / single-frame input.
    panoramic_min_move: float = 2.0          # Min distance (m) from the last scan / init pose to allow another scan (new-frontier guard).
    panoramic_arrive_dist: float = 1.0       # Frontier-arrival trigger radius (m): fallback gate when no frontier cluster is being tracked (no-cluster starvation guard). Main trigger = pursued-cluster disappearance (fixed "and" mechanism, 09-03).

    # Phase 7: S-penalty (semantic-field cross-validation)
    enable_s_penalty: bool = True           # Master switch: c' = c * w_S(S) — multiply TSP3D confidence by a semantic-field weight (BLIP2 ITM, zero extra queries) to amplify TP/FP discrimination.
    s_penalty_thresh: float = 0.15          # S below this (ITM raw cosine scale ~0.10-0.15; measured min 0.084) -> apply penalty (w_S = floor).
    s_penalty_floor: float = 0.3            # Lower bound of the dynamic penalty weight w_S = max(floor, S/thresh) when S < thresh (non-zero -> keep recall; S->0 -> floor).
    s_penalty_radius_m: float = 0.5         # S query radius (m) around the detection point.
    s_penalty_use_surface: bool = True      # query S at the bbox near-surface point (facing camera) instead of the centroid.
    goal_use_surface: bool = True           # navigate to the stored near-surface point (first-write fixed) instead of the centroid.

    # Phase 7b: target-memory lifecycle (all default OFF = world baseline)
    merge_dist_thresh: float = 0.5          # cross-view EMA merge distance threshold (m)
    ema_weight_old: float = 0.8             # cross-view EMA smoothing (old * w + new * (1 - w))
    enable_near_field_exempt: bool = False  # keep a trusted target parked next to the robot from deletion
    exempt_near_radius: float = 1.0         # robot-target distance below which a trusted target is exempt (m)
    exempt_min_merges: int = 1              # min merges of a trusted target to be exempt
    enable_free_space_erasure: bool = False # single-step ghost removal via the occupancy grid
    free_erasure_radius: float = 0.3        # box half-side (m) around the centroid to inspect
    free_erasure_min_explored: int = 5      # min explored voxels in the box to trust "this space was seen"
    free_erasure_free_ratio: float = 0.85   # free-voxel ratio above which the box is judged "pure air"
    free_erasure_soft_only: bool = True     # True = erase suspicious targets only (conservative)

    # Phase 7c: geometric admission module (occ-consistency + density gates; default OFF)
    enable_occ_consistency: bool = False    # box must have occupied-voxel support in the 3D grid
    occ_min_voxels: int = 8                 # occ gate: min occupied voxels supporting the box
    occ_min_ratio: float = 0.001            # occ gate: min occupied/(box volume) ratio
    enable_density_gate: bool = False       # density x occupancy fusion gate (occ support OR density HARD-confirm)
    density_occ_support: bool = True        # density gate: occupancy support as parallel spatio-temporal evidence
    density_min_points: int = 150           # density gate: min accumulated in-box points to confirm
    density_min_frames: int = 2             # density gate: min observation frames to confirm
    density_cluster_merge_dist: float = 0.5 # density gate: cross-frame cluster association radius (m)
    density_min_view_span_deg: float = 0.0  # density path: required multi-view azimuth span (deg, 0=off)

    # Phase 7e: V7 near-field camera pitch-down (module 4; default OFF = world baseline)
    pitch: bool = False                  # Near-field camera pitch-down when approaching a low target.
    pitch_trigger_dist: float = 1.5      # Pitch: only look down within this 2D distance (m) to the aim point.
    pitch_max_down_steps: int = 1        # Pitch: max look_down steps (tilt 30 deg each).

    # Phase 7f: P3 recall-side diagnostics (off = default world behaviour; never changes decisions)
    diag_enable: bool = False            # record below-sigma_tar + geom-gate-dropped detections for fn (bed) cause analysis
    diag_conf_floor: float = 0.30        # server return floor (diagnostics only); client still filters at sigma_tar

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        return [f.name for f in fields(VLVMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
