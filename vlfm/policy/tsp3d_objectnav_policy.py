import os
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union
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
from vlfm.utils.geometry_utils import rho_theta
from vlfm.mapping.obstacle_map import ObstacleMap3D, ProbabilisticGrid
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.tsp3d import TSP3DClient
from vlfm.vlm.detections import ObjectDetections

PROMPT_SEPARATOR = "|"

class TSP3DObjectNavPolicy(BasePolicy):
    """
    3D Active Semantic Target Navigation Policy utilizing TSP3D.
    Localizes target objects directly on 3D point clouds and sparse voxels,
    while building geometric and semantic representations to guide active exploration.
    """
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
            sigma_tar: float = 0.25,
            tau: float = 0.10,
            near_field_dist: float = 0.8,
            near_field_sigma_scale: float = 0.3,
            use_raw_nlp: bool = True,
            enable_retry: bool = False,
            pcd_window_size: int = 8,
            fuse_voxel_size: float = 0.02,
            fuse_max_points: int = 200000,
            pointnav_stop_radius: float = 0.25,
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
        self._enable_retry = enable_retry
        # Temporal PCD sliding window (multi-frame fusion for TSP3D)
        self._pcd_window: deque = deque(maxlen=max(pcd_window_size, 1))
        self._fuse_voxel_size = fuse_voxel_size
        self._fuse_max_points = fuse_max_points
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
        self._target_verify_state: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self._last_target_coord: Union[None, np.ndarray] = None

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
        """
        Resets target memories, step counters, pointnav models, and 3D occupancy maps.
        """
        self._target_object = ""
        self._init_step_count = 0
        self._num_steps = 0
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._target_3d_memory.clear()
        self._target_verify_state.clear()
        self._last_target_coord = None
        self._pointnav_policy.reset()
        self._obstacle_map3d.reset()
        self._pcd_window.clear()
        self._did_reset = True

    # ==========================================================================
    # === Input & Mapping Module ===
    # ==========================================================================
    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        """
        Pre-step initialization. Triggers reset on episode termination,
        caches observation data, and resets policy logging dict.
        """
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
        """
        Extracts and normalizes rgb, depth, and camera extrinsics from raw observations.
        Must be implemented by environment-specific wrappers or subclasses.
        """
        raise NotImplementedError

    def _accumulate_3d_target_memory(self, target_class: str, centroid: np.ndarray) -> None:
        """
        Registers detected 3D target centroids into persistent memory.
        A detection is written into memory immediately (locking it in for navigation);
        repeated observations within 0.5m are merged via EMA spatial consensus.
        """
        robot_xy = self._observations_cache.get("robot_xy", np.zeros(2))
        robot_yaw = self._observations_cache.get("robot_heading", 0.0)

        if target_class not in self._target_3d_memory:
            self._target_3d_memory[target_class] = [centroid]
            self._target_verify_state[target_class] = [(1, robot_xy.copy(), robot_yaw)]
            return

        existing_centroids = np.array(self._target_3d_memory[target_class])
        dists = np.linalg.norm(existing_centroids - centroid, axis=1)
        closest_idx = np.argmin(dists)

        # Merge observations if within 0.5m using EMA smoothing
        if dists[closest_idx] < 0.5:
            self._target_3d_memory[target_class][closest_idx] = (
                0.8 * self._target_3d_memory[target_class][closest_idx] + 0.2 * centroid
            )
            # Repeated consensus of the same world-frame location across frames is
            # itself strong evidence: hallucinated boxes drift and cannot merge.
            num_obs, _, _ = self._target_verify_state[target_class][closest_idx]
            num_obs += 1
            self._target_verify_state[target_class][closest_idx] = (
                num_obs, robot_xy.copy(), robot_yaw
            )
        else:
            self._target_3d_memory[target_class].append(centroid)
            self._target_verify_state[target_class].append((1, robot_xy.copy(), robot_yaw))

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
        """
        Projects raw RGB-D arrays into colored 3D point cloud coordinates in the
        episodic world frame, returning an array of (N, 6) [x, y, z, r, g, b].
        """
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
        """
        Returns the i-th shared back-projected point cloud (world frame, (N, 6)).
        """
        shared_pcds = self._observations_cache.get("shared_pcds")
        if shared_pcds is not None and i < len(shared_pcds):
            return shared_pcds[i]
        rgb, depth, tf, min_depth, max_depth, fx, fy = self._observations_cache["object_map_rgbd"][i]
        return self._project_rgbd_to_3d_point_cloud(
            rgb, depth, fx, fy, tf, min_depth, max_depth
        )

    def _fuse_temporal_pcd_window(self, pcd_world: np.ndarray) -> np.ndarray:
        """
        Fuses a temporal sliding window of world-frame colored point clouds
        into the current camera-canonical frame for TSP3D inference.

        Accumulates world-frame point clouds over a sliding window, re-projects the
        whole window into the current camera-canonical coordinate frame, then
        voxel-downsamples to bound the point count. This gives TSP3D a much more
        complete local geometry than a single-frame partial shell.
        """
        # Voxel-downsample the current frame BEFORE pushing into the window
        if len(pcd_world) > 0:
            vox = np.floor(pcd_world[:, :3] / self._fuse_voxel_size).astype(np.int64)
            _, idx = np.unique(vox, axis=0, return_index=True)
            pcd_world = pcd_world[np.sort(idx)]
            self._pcd_window.append(pcd_world)

        if len(self._pcd_window) == 0:
            return np.empty((0, 6))

        fused = np.concatenate(list(self._pcd_window), axis=0)

        # Transform the entire window into the current camera-canonical frame
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        robot_yaw = self._observations_cache.get("robot_heading", 0.0)

        fused_local = fused.copy()
        fused_local[:, :2] -= robot_xyz[:2]
        cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        xy = fused_local[:, :2]
        fused_local[:, 0] = cos_yaw * xy[:, 0] + sin_yaw * xy[:, 1]
        fused_local[:, 1] = -sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]

        # Voxel downsample to deduplicate overlapping frames
        voxel_coords = np.round(fused_local[:, :3] / self._fuse_voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        fused_local = fused_local[np.sort(unique_idx)]

        # Point-count bound (sparse conv is sensitive to extreme density)
        if len(fused_local) > self._fuse_max_points:
            idx = np.random.choice(len(fused_local), self._fuse_max_points, replace=False)
            fused_local = fused_local[idx]

        return fused_local

    def _query_tsp3d_client(
        self,
        aligned_pcd: np.ndarray,
        target_query: str
    ) -> List[Dict[str, Any]]:
        """
        Queries predictions from the multi-modal TSP3D visual grounding client,
        adaptively adjusting the text-guided pruning threshold when extremely
        close to surfaces.
        """
        if len(aligned_pcd) == 0:
            return []
        
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

        raw_preds = self._tsp3d_client.predict(
            pcd=aligned_pcd,
            text=target_query,
            sigma_tar=self._sigma_tar,
            sigma_sce=dynamic_sigma_sce,
            tau=self._tau,
            use_raw_nlp=self._nlp_mode
        )

        # Retry on empty results is disabled by default: the measured recovery
        # rate of a sigma_sce-reduced retry is only 0.05% while it accounts for
        # ~48% of queries.
        if self._enable_retry and len(raw_preds) == 0 and dynamic_sigma_sce > 0.02:
            fallback_sce = max(0.01, dynamic_sigma_sce * 0.3)
            raw_preds = self._tsp3d_client.predict(
                pcd=aligned_pcd,
                text=target_query,
                sigma_tar=self._sigma_tar,
                sigma_sce=fallback_sce,
                tau=self._tau,
                use_raw_nlp=self._nlp_mode
            )

        return raw_preds

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """
        Returns the closest observed target centroid with a hysteresis latch to
        suppress target switching (avoiding circling / back-and-forth):
        - New closest candidate 2D offset < 0.1m from current target  -> keep current target
        - Offset 0.1~0.5m and robot > 2.0m from the new candidate     -> keep current target
        - Otherwise -> switch target
        """
        target_classes = self._target_object.split("|")
        valid_centroids = []

        for cls in target_classes:
            if cls in self._target_3d_memory and len(self._target_3d_memory[cls]) > 0:
                for centroid in self._target_3d_memory[cls]:
                    valid_centroids.append(np.array(centroid))

        if len(valid_centroids) == 0:
            return None

        centroids = np.array(valid_centroids)
        robot_xy = np.asarray(position)[:2]
        dists_2d = np.linalg.norm(centroids[:, :2] - robot_xy, axis=1)

        closest_idx = np.argmin(dists_2d)
        closest_2d = centroids[closest_idx][:2].copy()

        if self._last_target_coord is None:
            self._last_target_coord = closest_2d
            return self._last_target_coord

        delta_dist = np.linalg.norm(closest_2d - self._last_target_coord)
        dist_to_new = dists_2d[closest_idx]
        if delta_dist < 0.1:
            pass  # Small offset from the current target -> keep current target
        elif delta_dist < 0.5 and dist_to_new > 2.0:
            pass  # Small offset and robot still far from the new candidate -> keep current target
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
        """
        Updates the 3D occupancy map, projects RGB-D to a 3D point cloud, and
        queries the TSP3D client for grounded target detections.
        """
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

        # Multi-frame sliding-window fusion: transform historical world-frame point
        # clouds into the current camera-canonical coordinate frame
        fused_pcd_local = self._fuse_temporal_pcd_window(pcd)
        raw_detections = self._query_tsp3d_client(fused_pcd_local, self._target_object)

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
        # Skip detection accumulation during the initialization turning phase:
        # in-place rotation sees the same nearby surfaces repeatedly and pollutes
        # target memory with duplicate spurious detections.
        if not self._done_initializing:
            return detections
        
        # Any detection that passes the confidence and class filters is written
        # into target memory immediately and locked in.
        for centroid, _, _ in zip(detections.centroids, detections.phrases, detections.boxes):
            centroid_np = centroid.cpu().numpy()
            if self._nlp_mode:
                active_classes = target_classes
            else:
                active_classes = [target_classes[0]] if len(target_classes) > 0 else []
            if not active_classes:
                continue
            for cls in active_classes:
                self._accumulate_3d_target_memory(cls, centroid_np)

        return detections

    # ==========================================================================
    # === Plan & Do Module ===
    # ==========================================================================
    def _initialize(self) -> Tensor:
        raise NotImplementedError
    
    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Computes rho/theta from the agent's current position to the goal and
        drives the pre-trained PointNav policy. Supports 2D (x, y) and 3D
        (x, y, z) goal inputs.
        """
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

        # Exploration uses the habitat `frontier_sensor` (same 2D
        # detect_frontier_waypoints frontiers as VLFM).
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

    # Phase 3: 3D Mapping & Occupancy Grid
    om_style: str = "obstacle"
    voxel_size: float = 0.01                 # Voxel grid resolution (m); smaller = finer map but more memory/compute.
    min_obstacle_height: float = 0.15        # Lower height bound (m) for obstacle voxels; too low includes floor noise, too high misses low obstacles.
    max_obstacle_height: float = 1.50        # Upper height bound (m) for obstacle voxels; too low ignores tall obstacles.
    agent_radius: float = 0.18               # Robot physical radius (m) used for collision inflation; larger = more conservative navigation.
    nav_slice_height: float = 0.35           # Reference navigation height (m) for frontier slicing and visualization.
    agent_height: float = 0.88               # Robot height (m) for the 3D cylinder structuring element (dilation / collision).
    hole_area_thresh: int = 100000           # Hole area threshold (px) for depth gap filling; -1 disables filling (assume max depth).

    # Phase 3.5: ProbabilisticGrid log-odds inverse-sensor model (om_style=probabilistic only)
    log_odds_occ: float = 2.0                # Occupied endpoint evidence weight (default symmetric: |free|==occ)
    log_odds_free: float = -2.0              # Free ray-body evidence weight; |free| < occ -> conservative bias (offset not cancelled to 0)
    occ_threshold: float = 0.0               # Occupied decision threshold (l > occ_thr); >0 -> hysteresis buffer (Unknown band)
    free_threshold: float = 0.0              # Free decision threshold (l < free_thr); <0 -> hysteresis buffer (Unknown band)
    obstacle_map_area_threshold: float = 1.5 # Frontier area threshold (m^2) to filter small isolated regions; higher = fewer, larger frontiers.

    # Phase 4: TSP3D Perception & Visual Grounding
    sigma_tar: float = 0.25           # Target confidence threshold; higher = stricter/fewer detections, lower = more detections but more false positives.
    sigma_sce: float = 0.15           # TGP scene voxel retention threshold; higher = more aggressive pruning (fewer hallucinations, may miss targets).
    tau: float = 0.10                 # Soft-pruning temperature; higher = smoother/looser pruning, lower = harder thresholding.
    near_field_dist: float = 0.8      # Distance (m) below which near-field adaptive sigma scaling activates; larger = scaling kicks in earlier.
    near_field_sigma_scale: float = 0.3  # Minimum voxel retention ratio near surfaces; higher = keep more voxels near obstacles.
    use_raw_nlp: bool = True          # Use raw NLP prompt formatting; True = multi-class synonym merging, False = use only the primary class.
    enable_retry: bool = False        # Retry once on empty TSP3D results with reduced sigma_sce; disabled by default (measured recovery 0.05%).

    # Phase 4.5: Temporal PCD Sliding Window (Multi-frame Fusion for TSP3D)
    pcd_window_size: int = 8          # Number of frames fused for point-cloud accumulation; larger = more complete geometry but slower/staler.
    fuse_voxel_size: float = 0.02     # Voxel downsample size (m) for window fusion; larger = fewer points, faster, coarser.
    fuse_max_points: int = 200000     # Cap on fused point count; higher = more detail but heavier sparse-conv inference.

    # Phase 6: Navigation Execution & Termination
    pointnav_stop_radius: float = 0.25  # Distance (m) at which the agent stops near the goal; larger = stops farther from the target.

    # Phase 7: 2.5D Semantic Value Plane (BEV)
    vm_style: str = "region"          # Semantic value mapping mode: "region" (V1) / "surface" (V2)
    h_lam: float = 0.3                # Bonus weight lambda (lambda=0 reduces to VLFM baseline)
    h_norm_max: float = 1.0           # Upper bound for normalizing H (locks the value score range)
    h_z_min: float = 0.15             # Lower bound of the H1 passable band (m)
    h_z_max: float = 0.88             # Upper bound of the H1 passable band (m, robot height)
    query_radius_m: float = 0.5       # Horizontal query radius r_h for route 2 (includes dilation semantics)
    query_z_min: float = 0.15         # Lower bound of the fixed query height band (m, surface landing)
    query_z_max: float = 1.50         # Upper bound of the fixed query height band (m)

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        return [f.name for f in fields(VLVMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
