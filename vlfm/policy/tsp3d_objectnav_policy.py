import os
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from hydra.core.config_store import ConfigStore
try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.base_policy import BasePolicy
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import rho_theta, extract_yaw, within_fov_cone, get_fov
from vlfm.mapping.obstacle_map import ObstacleMap3D, ProbabilisticGrid
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.tsp3d import TSP3DClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.tsp3d_models.utils.pipeline import TSP3DInputPreprocessor
from vlfm.tsp3d_models.utils.vqa_confirmation import vqa_confirm_detections
from vlfm.tsp3d_models.utils.s_penalty import (SPenaltyConfig, apply_s_penalty, 
                                            near_surface_point, query_semantic_at)

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
            use_raw_nlp: bool = False,
            use_world_map: bool = False,
            wm_voxel_size: float = 0.02,
            wm_radius: float = 6.0,
            wm_min_view_disp: float = 0.15,
            wm_min_view_yaw: float = 15.0,
            wm_max_voxels: int = 400000,
            wm_max_frames: Optional[int] = 8, 
            wm_near_refresh_radius: Optional[float] = None,
            wm_near_refresh_value: bool = False,
            distance_sample: bool = False,
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
            use_vqa: bool = False,
            vqa_prompt: str = "Is this ",
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
        self._nlp_mode = use_raw_nlp
        self._pointnav_stop_radius = pointnav_stop_radius
        self._init_step_count = 0
        self._num_steps = 0
        self._last_goal = np.zeros(2)  # Current 3D goal coordinate [x, y, z]
        self._done_initializing = False
        self._called_stop = False
        self._did_reset = False
        self._stop_action = torch.tensor([[0]], dtype=torch.long)
        self._turn_left_action = torch.tensor([[2]], dtype=torch.long)
        self._target_3d_memory: Dict[str, List[np.ndarray]] = {}
        self._target_surface_memory: Dict[str, List[np.ndarray]] = {}  # per-centroid near-surface point
        self._target_verify_state: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self._target_fallback_state: Dict[str, List[Dict[str, Any]]] = {}
        self._last_target_coord: Union[None, np.ndarray] = None
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
      
        # camera 8-frame window / world-frame accumulation, switched by use_world_map
        self._use_world_map = use_world_map
        self._preprocessor = TSP3DInputPreprocessor(
            fusion_style="world" if use_world_map else "camera",
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
        )

        # 3D visual grounding and vision-language evaluation clients
        self._tsp3d_client = TSP3DClient(port=int(os.environ.get("TSP3D_PORT", "12186")))
        self._itm_client = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._vqa_client = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185"))) if use_vqa else None
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._text_prompt = text_prompt
        self._vqa_prompt = vqa_prompt
        self._use_vqa = use_vqa
        self._vqa_confirm_detections = partial(
            vqa_confirm_detections,
            use_vqa=self._use_vqa,
            vqa_client=self._vqa_client,
            vqa_prompt=self._vqa_prompt,
        )
  
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
        self._pointnav_policy.reset()
        self._obstacle_map3d.reset()
        self._preprocessor.reset()
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

    def _anti_hallucination_fallback(self, tf_camera_to_episodic: np.ndarray, max_depth: float) -> None:
        """VLFM-aligned anti-hallucination fallback (centroid version).

        Increment the miss counter of centroids inside the near-field FOV cone that got
        no fresh re-detection; delete past the hysteresis threshold. 
        Fresh merges reset the counter, so only drift-only (hallucinated)
        boxes are ever deleted -> fall back to explore.
        """
        camera_pos = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)
        # hFOV from sensor-derived focal length so the cone matches the actual frustum.
        cone_fov = get_fov(self._fx, self._depth_image_shape[1])
        near_radius = self._fb_near_radius # max_depth * 0.5

        for target_class in list(self._target_3d_memory.keys()):
            centroids = self._target_3d_memory[target_class]
            states = self._target_fallback_state.get(target_class, [])
            if len(states) != len(centroids):
                # Defensive: keep state in sync with centroids
                states = [{"suspicious": False, "near_miss": 0} for _ in centroids]
                self._target_fallback_state[target_class] = states

            keep_c, keep_v, keep_s = [], [], []
            surfaces = self._target_surface_memory.get(target_class, [])
            keep_surf = []

            for idx, centroid in enumerate(centroids):
                # Inside the near-field confirmation cone?
                if centroid.shape[0] >= 3:
                    query_pt = centroid.reshape(1, 3)
                else:
                    query_pt = np.append(centroid, 0.5).reshape(1, 3)
                in_cone = within_fov_cone(camera_pos, camera_yaw, cone_fov, near_radius, query_pt)
                if len(in_cone) > 0:
                    states[idx]["near_miss"] += 1
                    thr = (
                        self._fb_suspicious_hysteresis
                        if states[idx]["suspicious"]
                        else self._fb_hysteresis
                    )
                    if states[idx]["near_miss"] >= thr:
                        print(f"[AntiHallucination] Deleted {target_class} centroid "
                              f"{np.round(centroid, 3)} (suspicious={states[idx]['suspicious']}, "
                              f"near_miss={states[idx]['near_miss']}) -> fallback to explore")
                        continue  # drop this centroid
                keep_c.append(centroid)
                keep_surf.append(surfaces[idx] if idx < len(surfaces) else centroid)
                keep_v.append(self._target_verify_state[target_class][idx])
                keep_s.append(states[idx])

            self._target_3d_memory[target_class] = keep_c
            self._target_surface_memory[target_class] = keep_surf
            self._target_verify_state[target_class] = keep_v
            self._target_fallback_state[target_class] = keep_s

        # Drop emptied classes (no target -> explore)
        for target_class in list(self._target_3d_memory.keys()):
            if len(self._target_3d_memory[target_class]) == 0:
                del self._target_3d_memory[target_class]
                self._target_surface_memory.pop(target_class, None)
                self._target_verify_state.pop(target_class, None)
                self._target_fallback_state.pop(target_class, None)

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

    def _query_tsp3d_client(
        self,
        aligned_pcd: np.ndarray,
        target_query: str
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
            use_raw_nlp=self._nlp_mode,
        )

        return raw_preds, diagnostics

    def _accumulate_3d_target_memory(self, target_class: str, centroid: np.ndarray, confidence: float = 1.0, near_surface: Optional[np.ndarray] = None) -> None:
        """Write detection into memory; EMA-merge within 0.5m; 
        track suspicious (far/low-conf) + near-miss counter — fresh merge resets it, 
        so drift-only hallucinated boxes get deleted near the FOV cone -> explore.
        """

        robot_xy = self._observations_cache.get("robot_xy", np.zeros(2))
        robot_yaw = self._observations_cache.get("robot_heading", 0.0)
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))

        # Suspicious = far or low-conf (VLFM range_id != 1 analogue); two-tier gating:
        # [sigma_tar, fb_suspicious_conf) deleted after 2 near-field misses, >= conf trusted (5).
        if centroid.shape[0] >= 3:
            detect_dist = float(np.linalg.norm(centroid - robot_xyz))
        else:
            detect_dist = float(np.linalg.norm(centroid - robot_xyz[:2]))
        suspicious = (confidence < self._fb_suspicious_conf) or (detect_dist > self._max_depth * 0.95)

        if target_class not in self._target_3d_memory:
            self._target_3d_memory[target_class] = [centroid]
            self._target_surface_memory[target_class] = [near_surface if near_surface is not None else centroid]
            self._target_verify_state[target_class] = [(1, robot_xy.copy(), robot_yaw)]
            self._target_fallback_state[target_class] = [{"suspicious": suspicious, "near_miss": 0}]
            return

        existing_centroids = np.array(self._target_3d_memory[target_class])
        dists = np.linalg.norm(existing_centroids - centroid, axis=1)
        closest_idx = np.argmin(dists)

        # EMA-merge observations within 0.5m
        if dists[closest_idx] < 0.5:
            self._target_3d_memory[target_class][closest_idx] = (
                0.8 * self._target_3d_memory[target_class][closest_idx] + 0.2 * centroid
            )
            # Cross-frame consensus is strong evidence; hallucinated boxes drift and can't merge.
            num_obs, _, _ = self._target_verify_state[target_class][closest_idx]
            num_obs += 1
            self._target_verify_state[target_class][closest_idx] = (
                num_obs, robot_xy.copy(), robot_yaw
            )
            # Fresh merge = real target: reset near-miss counter (drift-only boxes can't).
            self._target_fallback_state[target_class][closest_idx]["near_miss"] = 0
        else:
            self._target_3d_memory[target_class].append(centroid)
            self._target_surface_memory[target_class].append(near_surface if near_surface is not None else centroid)
            self._target_verify_state[target_class].append((1, robot_xy.copy(), robot_yaw))
            self._target_fallback_state[target_class].append({"suspicious": suspicious, "near_miss": 0})

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """Closest target centroid with hysteresis latch against switching:
        keep current if new candidate is <0.1m away, or <0.5m while robot >2.0m away.
        """
        target_classes = self._target_object.split("|")
        valid_centroids = []
        valid_goals = []

        for cls in target_classes:
            if cls in self._target_3d_memory and len(self._target_3d_memory[cls]) > 0:
                surf_list = self._target_surface_memory.get(cls, [])
                for i, centroid in enumerate(self._target_3d_memory[cls]):
                    valid_centroids.append(np.array(centroid))
                    if self._goal_use_surface and i < len(surf_list):
                        valid_goals.append(np.array(surf_list[i]))
                    else:
                        valid_goals.append(np.array(centroid))

        if len(valid_centroids) == 0:
            return None

        centroids = np.array(valid_centroids)
        goals = np.array(valid_goals)
        robot_xy = np.asarray(position)[:2]
        dists_2d = np.linalg.norm(centroids[:, :2] - robot_xy, axis=1)

        closest_idx = np.argmin(dists_2d)
        closest_2d = goals[closest_idx][:2].copy()

        if self._last_target_coord is None:
            self._last_target_coord = closest_2d
            return self._last_target_coord

        delta_dist = np.linalg.norm(closest_2d - self._last_target_coord)
        dist_to_new = dists_2d[closest_idx]
        if delta_dist < 0.1:
            pass  # <0.1m from current target: keep
        elif delta_dist < 0.5 and dist_to_new > 2.0:
            pass  # <0.5m and robot >2m away: keep
        else:
            self._last_target_coord = closest_2d
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
        self._preprocessor.update(pcd, robot_xyz, robot_yaw)
        fused_pcd_local = self._preprocessor.prepare(
            robot_xyz,
            robot_yaw,
            camera_height=self._camera_height,
        )
        raw_detections, _ = self._query_tsp3d_client(fused_pcd_local, self._target_object)

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
            pcd_source=pcd,
            image_source=rgb,
            fx=fx,
            fy=fy,
            tf_camera_to_episodic=tf_camera_to_episodic
        )

        detections.filter_by_conf(self._sigma_tar)
        target_classes = [c.strip() for c in self._target_object.split("|") if c.strip()]
        detections.filter_by_class(target_classes, use_raw_nlp=self._nlp_mode)
        # Skip memory accumulation during initialization turning (repeated surfaces pollute memory).
        if not self._done_initializing:
            return detections

        # BLIP2 VQA false-positive confirmation (VLFM-aligned): drop detections
        # whose visual answer does not start with 'yes' before they reach memory.
        if self._use_vqa and self._vqa_client is not None:
            self._vqa_confirm_detections(detections)

        # S-penalty gate cross-validation -> write passed detections to memory.
        pending = apply_s_penalty(
            detections, target_classes, robot_xyz,
            query_semantic=self._query_semantic_at,
            cfg=self._s_penalty_cfg,
            sigma_tar=self._sigma_tar,
            use_surface=self._s_penalty_use_surface,
            goal_use_surface=self._goal_use_surface,
            nlp_mode=self._nlp_mode,
        )
        for conf_amp, centroid_np, near_surface, active_classes in pending:
            for cls in active_classes:
                self._accumulate_3d_target_memory(cls, centroid_np, confidence=conf_amp, near_surface=near_surface)

        # VLFM-aligned: run anti-hallucination fallback every step (delete confirmed-false -> explore).
        if self._enable_fb:
            self._anti_hallucination_fallback(tf_camera_to_episodic, max_depth)

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

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
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

        # Exploration via habitat frontier_sensor (same 2D frontiers as VLFM).
        if not self._done_initializing:
            mode = "initialize"
            action = self._initialize()
        elif goal_3d is None:
            mode = "explore"
            action = self._explore(observations)
        else:
            mode = "navigate"
            print(f"[TSP3D Mode] Target '{self._target_object}' located at {goal_3d}. Navigating.")
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
    use_raw_nlp: bool = False           # Use raw NLP prompt formatting; True = multi-class synonym merging, False = use only the primary class.
    cap_style: str = "random"           # send-side point cap (shared): "random" (baseline) / "near_first" (near-field priority)
    distance_sample: bool = False       # distance-adaptive sampling (dense near / sparse far, shared post-processing)
    near_dist: float = 1.5              # distance sampling: near/mid band boundary (m)
    mid_dist: float = 3.0               # distance sampling: mid/far band boundary (m)
    near_voxel: float = 0.01            # distance sampling: near-band voxel (m)
    mid_voxel: float = 0.02             # distance sampling: mid-band voxel (m)
    far_voxel: float = 0.05             # distance sampling: far-band voxel (m)
    use_world_map: bool = True          # True = world-frame accumulation / False = camera 8-frame window (baseline)

    # Phase 5b: Temporal PCD Sliding Window (Multi-frame Fusion for TSP3D)
    pcd_window_size: int = 8                # Number of frames fused for point-cloud accumulation; larger = more complete geometry but slower/staler.
    fuse_voxel_size: float = 0.02           # Voxel downsample size (m) for window fusion; larger = fewer points, faster, coarser.
    fuse_max_points: int = 200000           # Cap on fused point count; higher = more detail but heavier sparse-conv inference.
    cam_radius: Optional[float] = None      # camera: send-side horizontal radius crop (m), None = no crop (baseline); align with wm_radius=6.0 -> 6.0
    cam_min_view_disp: float = 0.15         # camera: min displacement (m) to accept a frame into fusion, 0 = off; world uses wm_min_view_disp 0.15
    cam_min_view_yaw: float = 15.0           # camera: min yaw change (deg) to accept a frame into fusion, 0 = off; world uses wm_min_view_yaw 15.0

    # Phase 5c: World-frame Local Map (TSP3DInputPreprocessor)
    wm_max_frames: Optional[int] = 8  # world: frame-window cap (keep voxels of the most recent N frames); None = all history
    wm_voxel_size: float = 0.02       # world: local map voxel size (m) for fixed-grid dedup
    wm_max_voxels: int = 400000       # world: hard cap on map voxels
    wm_radius: Optional[float] = 6.0  # world: local map radius (m), None = no crop (only max_voxels hard cap); cam_radius=None is the symmetric case
    wm_min_view_disp: float = 0.15    # world: min displacement (m) to accept a frame into fusion
    wm_min_view_yaw: float = 15.0     # world: min yaw change (deg) to accept a frame into fusion
    wm_near_refresh_radius: Optional[float] = 3.0  # world: near-field refresh radius (m). None=off (first-observation accounting); >0: re-observed near-field voxels are re-owned by the current frame (cam-style accounting, survives frame-window slide-out).
    wm_near_refresh_value: bool = False  # world: also overwrite the stored point of refreshed near-field voxels with the current observation (cam-style sliding refresh of the value layer). False = value layer stays first.
    
    # Phase 5d: S-penalty (semantic-field cross-validation)
    enable_s_penalty: bool = True           # Master switch: c' = c * w_S(S) — multiply TSP3D confidence by a semantic-field weight (BLIP2 ITM, zero extra queries) to amplify TP/FP discrimination.
    s_penalty_thresh: float = 0.15          # S below this (ITM raw cosine scale ~0.10-0.15; measured min 0.084) -> apply penalty (w_S = floor).
    s_penalty_floor: float = 0.3            # Lower bound of the dynamic penalty weight w_S = max(floor, S/thresh) when S < thresh (non-zero -> keep recall; S->0 -> floor).
    s_penalty_radius_m: float = 0.5         # S query radius (m) around the detection point.
    s_penalty_use_surface: bool = True      # query S at the bbox near-surface point (facing camera) instead of the centroid.
    goal_use_surface: bool = True           # navigate to the stored near-surface point (first-write fixed) instead of the centroid.

    # Phase 5e: BLIP2 VQA false-positive confirmation (VLFM-aligned)
    use_vqa: bool = False                   # Ask BLIP2 'Is this a {obj}?' on each TSP3D detection; drop answers not starting with 'yes'. Requires the BLIP2 VQA server (vlfm.vlm.blip2) on BLIP2_PORT.
    vqa_prompt: str = "Is this "            # VQA question prefix; full prompt = 'Question: {vqa_prompt}[a ]{obj}? Answer:'.

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        return [f.name for f in fields(VLVMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
