import os
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import DBSCAN
from hydra.core.config_store import ConfigStore
try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.base_policy import BasePolicy
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import rho_theta
from vlfm.mapping.obstacle_map import ObstacleMap3D, OctoMap
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
            # Phase 1: System & Base Camera Configuration
            pointnav_policy_path: str = "data/pointnav_weights.pth",
            depth_image_shape: Tuple[int, int] = (224, 224),
            fov_angle: float = 79.0,
            camera_height: float = 0.88,
            text_prompt: str = "Seems like there is a target_object ahead.",
            visualize: bool = False,
            init_turn_steps: int = 12,
            # Phase 2: Depth Sensor Filtering
            min_depth: float = 0.5,
            max_depth: float = 5.0,
            # Phase 3: 3D Mapping & Occupancy Grid
            use_octomap: bool = True,
            voxel_size: float = 0.01,
            min_obstacle_height: float = 0.05,
            max_obstacle_height: float = 1.50,
            agent_radius: float = 0.18,
            nav_slice_height: float = 0.35,
            hole_area_thresh: int = 100000,
            obstacle_map_area_threshold: float = 3,
            # Phase 4: TSP3D Perception & Visual Grounding
            sigma_sce: float = 0.15,
            sigma_tar: float = 0.25,
            tau: float = 0.10,
            near_field_dist: float = 0.8,         # Distance (m) below which near-field adaptive pruning scaling activates
            near_field_sigma_scale: float = 0.3,  # Minimum voxel retention ratio when pruning near surfaces
            use_raw_nlp: bool = True,
            enable_retry: bool = False,           # 08-18: retry on empty results with reduced sigma_sce (measured recovery 0.05%, yet ~48% of queries); True keeps old retry for A/B
            # Phase 4.5: Temporal PCD Sliding Window (Multi-frame Fusion for TSP3D)
            pcd_window_size: int = 8,             # Number of frames in the fusion window (recommended 5~10)
            fuse_voxel_size: float = 0.02,        # Voxel downsample size (m) for window fusion
            fuse_max_points: int = 200000,        # Cap on the number of fused points
            # Phase 5: 3D Exploration & Frontier Clustering
            compute_frontiers: bool = True,
            dbscan_eps: float = 0.15,
            dbscan_min_samples: int = 5,
            min_dists_2d: float = 0.5,
            # Phase 6: Navigation Execution & Termination
            pointnav_stop_radius: float = 0.25,
            *args: Any,
            **kwargs: Any,
        ) -> None:
        super().__init__()
        # Phase 1: System & Base Camera Configuration
        self._depth_image_shape = tuple(depth_image_shape)
        self._fov_angle = fov_angle
        self._camera_height = camera_height
        self._visualize = visualize
        self._init_turn_steps = init_turn_steps
        self._cached_grid_size = None
        self._cached_u_flat = None
        self._cached_v_flat = None
        # Calculate focal length parameters based on FOV and camera resolution
        fov_rad = np.deg2rad(fov_angle)
        self._fx = self._fy = depth_image_shape[1] / (2 * np.tan(fov_rad / 2))
        # Phase 2: Depth Sensor Filtering
        self._min_depth = min_depth
        self._max_depth = max_depth
        # Phase 3: 3D Mapping & Occupancy Grid
        self._use_octomap = use_octomap
        self._voxel_size = voxel_size
        # Phase 4: TSP3D Perception & Visual Grounding
        self._sigma_tar = sigma_tar  # Target validation activation threshold
        self._sigma_sce = sigma_sce  # Voxel-pruning scenario conservation threshold
        self._tau = tau              # Soft-pruning temperature coefficient
        self._near_field_dist = near_field_dist
        self._near_field_sigma_scale = near_field_sigma_scale
        self._nlp_mode = use_raw_nlp  # Whether to use raw NLP processing for text prompts
        self._enable_retry = enable_retry  # 08-18: retry on empty TSP3D results with reduced sigma_sce (default off; eliminates ~48% useless queries)
        # Temporal PCD sliding window state (multi-frame fusion for TSP3D)
        self._pcd_window: deque = deque(maxlen=max(pcd_window_size, 1))
        self._fuse_voxel_size = fuse_voxel_size
        self._fuse_max_points = fuse_max_points
        # Phase 5: 3D Exploration & Frontier Clustering
        self._compute_frontiers = compute_frontiers
        self._dbscan_eps = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples
        self._min_dists_2d = min_dists_2d
        # Phase 6: Navigation Execution & Internal Policy States
        self._pointnav_stop_radius = pointnav_stop_radius
        self._init_step_count = 0
        self._num_steps = 0
        self._last_goal = np.zeros(2)  # Stores current 3D goal coordinate [x, y, z]
        self._done_initializing = False
        self._called_stop = False
        self._did_reset = False
        self._stop_action = torch.tensor([[0]], dtype=torch.long)
        self._turn_left_action = torch.tensor([[2]], dtype=torch.long)
        self._target_3d_memory: Dict[str, List[np.ndarray]] = {}
        self._target_verify_state: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self._last_target_coord: Union[None, np.ndarray] = None
        
        # Initialize clients for 3D visual grounding and vision-language evaluation
        self._tsp3d_client = TSP3DClient(port=int(os.environ.get("TSP3D_PORT", "12186")))
        self._itm_client = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._text_prompt = text_prompt

        # Instantiate core 3D spatial representations
        height_range = max_obstacle_height - min_obstacle_height
        height_size = int(height_range / voxel_size) + 1
        pixels_per_meter = int(1.0 / voxel_size)
        size = 400

        if self._use_octomap:
            self._obstacle_map3d = OctoMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
                size=size,
                pixels_per_meter=pixels_per_meter,
                voxel_size=voxel_size,
                height_size=height_size,
                nav_slice_height=nav_slice_height,
                visualize=visualize,
            )
        else:
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
        A detection is written into memory immediately (locking it in for navigation).
        Repeated observations within 0.5m are merged using EMA spatial consensus to
        suppress detection noise; the observation count is only used for logging and
        does not gate locking.
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
            # Increment the cumulative observation count: repeated consensus of
            # the same world-frame location across frames is itself strong
            # evidence. Hallucinated boxes drift and cannot merge repeatedly.
            num_obs, last_pos, last_yaw = self._target_verify_state[target_class][closest_idx]
            num_obs += 1
            dist_moved = np.linalg.norm(robot_xy[:2] - last_pos[:2])
            yaw_changed = abs(robot_yaw - last_yaw)
            self._target_verify_state[target_class][closest_idx] = (
                num_obs, robot_xy.copy(), robot_yaw
            )
            print(f"[DEBUG VERIFY] '{target_class}' idx={closest_idx} obs={num_obs}, dist_moved={dist_moved:.2f}, yaw_changed={np.rad2deg(yaw_changed):.0f}°, centroid={self._target_3d_memory[target_class][closest_idx]}")
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
        Projects raw RGB-D arrays into colored 3D point cloud coordinates in the episodic world frame.

        Args:
            rgb (np.ndarray): Color image frame of shape (H, W, 3).
            depth (np.ndarray): Depth image map of shape (H, W).
            fx (float): Horizontal focal length.
            fy (float): Vertical focal length.
            tf_camera_to_episodic (np.ndarray): Extrinsic transformation matrix (4x4).
            min_depth (float): Minimum valid depth distance.
            max_depth (float): Maximum valid depth distance.

        Returns:
            np.ndarray: Array of 3D point coordinates aligned with RGB values, shape (N, 6).
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

    def _fuse_temporal_pcd_window(self, pcd_world: np.ndarray) -> np.ndarray:
        """
        Fuses a temporal sliding window of world-frame colored point clouds
        into the current camera-canonical frame for TSP3D inference.

        Accumulates world-frame point clouds over a sliding window and
        re-projects the whole window into the current camera-canonical
        coordinate frame (zero-centered XY + de-rotated world yaw), then
        voxel-downsamples to bound the point count. This gives TSP3D a much
        more complete local geometry than a single-frame "partial shell",
        mitigating TGP mis-pruning hallucinations on online partial clouds.

        Args:
            pcd_world (np.ndarray): Current frame colored point cloud in the
                episodic world frame, shape (N, 6).

        Returns:
            np.ndarray: Fused, camera-canonical colored point cloud, shape (M, 6).
        """
        # 1. Voxel-downsample the current frame BEFORE pushing into the window,
        #    so the window memory and the final np.unique stay small.
        if len(pcd_world) > 0:
            vox = np.floor(pcd_world[:, :3] / self._fuse_voxel_size).astype(np.int64)
            _, idx = np.unique(vox, axis=0, return_index=True)
            pcd_world = pcd_world[np.sort(idx)]
            self._pcd_window.append(pcd_world)

        if len(self._pcd_window) == 0:
            return np.empty((0, 6))

        fused = np.concatenate(list(self._pcd_window), axis=0)

        # 2. Transform the entire window into the current camera-canonical frame
        #    (consistent with the prior single-frame de-yaw/de-center logic).
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        robot_yaw = self._observations_cache.get("robot_heading", 0.0)

        fused_local = fused.copy()
        fused_local[:, :2] -= robot_xyz[:2]
        cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        xy = fused_local[:, :2]
        fused_local[:, 0] = cos_yaw * xy[:, 0] + sin_yaw * xy[:, 1]
        fused_local[:, 1] = -sin_yaw * xy[:, 0] + cos_yaw * xy[:, 1]

        # 3. Voxel downsample to deduplicate overlapping frames
        #    (keep the first-encountered color per voxel).
        voxel_coords = np.round(fused_local[:, :3] / self._fuse_voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
        fused_local = fused_local[np.sort(unique_idx)]

        # 4. Point-count bound (sparse conv is sensitive to extreme density).
        if len(fused_local) > self._fuse_max_points:
            idx = np.random.choice(len(fused_local), self._fuse_max_points, replace=False)
            fused_local = fused_local[idx]

        print(
            f"[DEBUG FUSE] Window={len(self._pcd_window)} frames | "
            f"Fused pts: {len(fused)} -> {len(fused_local)} "
            f"(voxel={self._fuse_voxel_size}m)"
        )
        return fused_local

    def _query_tsp3d_client(
        self,
        aligned_pcd: np.ndarray,
        target_query: str
    ) -> List[Dict[str, Any]]:
        """
        Queries predictions from the multi-modal TSP3D visual grounding client.
        Dynamically adjusts text-guided pruning threshold when extremely close to surfaces.

        Args:
            aligned_pcd (np.ndarray): 3D aligned colored point cloud of shape (N, 6).
            target_query (str): Language target object class query.

        Returns:
            List[Dict[str, Any]]: Predicted 3D bounding boxes and confidence scores.
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
        print(f"[DEBUG TSP3D] Query '{target_query}' | PCD: {len(aligned_pcd)} pts | MinDist: {min_dist:.2f}m | Yaw: {np.rad2deg(self._observations_cache.get('robot_heading', 0)):.0f}° | sigma_sce: {dynamic_sigma_sce:.2f} | sigma_tar: {self._sigma_tar}")

        raw_preds = self._tsp3d_client.predict(
            pcd=aligned_pcd,
            text=target_query,
            sigma_tar=self._sigma_tar,
            sigma_sce=dynamic_sigma_sce,
            tau=self._tau,
            use_raw_nlp=self._nlp_mode
        )

        # Retry on empty results is disabled by default: measured recovery rate of a
        # sigma_sce-reduced retry is only 0.05% while it accounts for ~48% of queries.
        if self._enable_retry and len(raw_preds) == 0 and dynamic_sigma_sce > 0.02:
            fallback_sce = max(0.01, dynamic_sigma_sce * 0.3)
            print(f"[DEBUG TSP3D] Empty result → retry sigma_sce={fallback_sce:.3f}")
            raw_preds = self._tsp3d_client.predict(
                pcd=aligned_pcd,
                text=target_query,
                sigma_tar=self._sigma_tar,
                sigma_sce=fallback_sce,
                tau=self._tau,
                use_raw_nlp=self._nlp_mode
            )

        for i, det in enumerate(raw_preds):
            print(f"  - Det {i}: conf={det.get('confidence', 0.0):.3f}")

        return raw_preds

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        """
        Mirrors the hysteresis latch semantics of the original vlfm
        ObjectPointCloudMap.get_best_object. Each step recomputes the closest
        observed target centroid, but uses the last_target_coord hysteresis band
        to suppress target switching (avoiding circling / back-and-forth), while
        still allowing corrections once the robot approaches the new candidate.

        - New closest candidate 2D offset < 0.1m from current target  -> keep current target
        - Offset 0.1~0.5m and robot > 2.0m from the new candidate     -> keep current target
        - Otherwise (offset large enough and robot close to the new candidate) -> switch target
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
            print(
                f"[DEBUG LATCH] First lock target '{self._target_object}' at {closest_2d}. "
                f"Will latch unless a closer candidate is confirmed."
            )
            return self._last_target_coord

        delta_dist = np.linalg.norm(closest_2d - self._last_target_coord)
        dist_to_new = dists_2d[closest_idx]
        if delta_dist < 0.1:
            pass  # Small offset from the current target -> keep current target
        elif delta_dist < 0.5 and dist_to_new > 2.0:
            pass  # Small offset and robot still far from the new candidate -> keep current target
        else:
            self._last_target_coord = closest_2d
            print(
                f"[DEBUG LATCH] Switch target '{self._target_object}': "
                f"delta={delta_dist:.2f}m, robot→new={dist_to_new:.2f}m, new={closest_2d}"
            )
        return self._last_target_coord

    def _project_box_to_image_crop(
        self,
        box_np: np.ndarray,
        rgb: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        fx: float,
        fy: float,
    ) -> Union[np.ndarray, None]:
        """
        Projects a 3D box (8 corners in episodic world frame) onto the current
        RGB image and returns the cropped bounding-rect region.

        Args:
            box_np (np.ndarray): Box corners of shape (8, 3) in episodic world frame.
            rgb (np.ndarray): RGB image of shape (H, W, 3).
            tf_camera_to_episodic (np.ndarray): 4x4 camera-to-episodic transform.
            fx, fy: focal lengths in pixels.

        Returns:
            np.ndarray | None: Cropped image region, or None if the box is
                fully outside the frame / behind the camera.
        """
        H, W = rgb.shape[:2]
        try:
            tf_episodic_to_camera = np.linalg.inv(tf_camera_to_episodic)
        except np.linalg.LinAlgError:
            return None
        # camera base frame = [forward, left, up]; zc=forward, xc=-left, yc=-up
        pts_homo = np.hstack([box_np, np.ones((len(box_np), 1))])
        pts_base = (tf_episodic_to_camera @ pts_homo.T).T[:, :3]
        zc = pts_base[:, 0]
        if np.all(zc <= 0.1):
            return None
        xc = -pts_base[:, 1]
        yc = -pts_base[:, 2]
        u = xc / np.maximum(zc, 0.1) * fx + W / 2.0
        v = yc / np.maximum(zc, 0.1) * fy + H / 2.0
        u_min, u_max = int(np.clip(np.min(u), 0, W)), int(np.clip(np.max(u), 0, W))
        v_min, v_max = int(np.clip(np.min(v), 0, H)), int(np.clip(np.max(v), 0, H))
        if (u_max - u_min) < 20 or (v_max - v_min) < 20:
            return None
        return rgb[v_min:v_max, u_min:u_max]

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
        Updates 3D occupancy map, projects RGB-D to 3D point cloud, and queries TSP3D client.

        Returns:
            ObjectDetections: Grounded target detection bounding boxes and confidence logits.
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

        # Multi-frame sliding-window fusion: transform all historical world-frame
        # point clouds into the current camera-canonical coordinate frame
        fused_pcd_local = self._fuse_temporal_pcd_window(pcd)
        raw_detections = self._query_tsp3d_client(fused_pcd_local, self._target_object)

        # Restore predicted boxes to global coordinates (inverse rotation + translation)
        # Keep only detections that carry a valid 3D box
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
        # in-place rotation sees the same nearby surfaces repeatedly and
        # pollutes target memory with duplicate spurious detections.
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
            print(f"[DEBUG DET] Lock '{self._target_object}': centroid={centroid_np}")
            for cls in active_classes:
                self._accumulate_3d_target_memory(cls, centroid_np)

        return detections

    # ==========================================================================
    # === Plan & Do Module ===
    # ==========================================================================
    def _extract_3d_frontiers_bfs(self) -> np.ndarray:
        """Extracts 3D frontier voxels (K x 3) directly from the obstacle map backend."""
        if not self._compute_frontiers:
            return np.empty((0, 3))
        return self._obstacle_map3d.frontiers
    
    def _cluster_and_extract_centroids_dbscan(self, frontier_voxels: np.ndarray) -> np.ndarray:
        """
        Clusters discrete 3D frontier voxels using DBSCAN density clustering,
        filters out noisy voxels, and calculates cluster centroids (M x 3).
        """
        if len(frontier_voxels) == 0:
            return np.empty((0, 3))

        if len(frontier_voxels) > 1000:
            step = len(frontier_voxels) // 1000 + 1
            frontier_voxels = frontier_voxels[::step]

        db = DBSCAN(eps=self._dbscan_eps, min_samples=self._dbscan_min_samples).fit(frontier_voxels)
        labels = db.labels_  # Label -1 indicates noise; 0..C-1 indicate cluster IDs
        unique_labels = set(labels)
        centroids = []

        for label in unique_labels:
            if label == -1:
                continue  # Filter out noise points
            cluster_mask = (labels == label)
            cluster_points = frontier_voxels[cluster_mask]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

        if len(centroids) == 0:
            return np.empty((0, 3))

        return np.array(centroids)

    def _rank_3d_frontier_waypoints(self, candidate_waypoints: np.ndarray) -> np.ndarray:
        """
        Filters and ranks candidate 3D waypoints using collision checking and 2D planar distances.
        """
        if len(candidate_waypoints) == 0:
            return np.empty((0, 3))

        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        # Compute 2D planar distances to align with lower-level PointNav control
        dists_2d = np.linalg.norm(candidate_waypoints[:, :2] - robot_xyz[:2], axis=1)

        # Filter out waypoints that are too close to the agent's current position
        far_mask = dists_2d > self._min_dists_2d
        if np.any(far_mask):
            candidate_waypoints = candidate_waypoints[far_mask]
            dists_2d = dists_2d[far_mask]

        # Collision filtering using 3D obstacle map backend
        colliding_mask = self._obstacle_map3d.check_collision(candidate_waypoints)
        safe_indices = np.where(~colliding_mask)[0]
        if len(safe_indices) == 0:
            # Fallback to all candidate waypoints if all collide
            safe_indices = np.arange(len(candidate_waypoints))

        safe_waypoints = candidate_waypoints[safe_indices]
        safe_dists_2d = dists_2d[safe_indices]

        # Rank by ascending 2D planar distance
        sorted_idx = np.argsort(safe_dists_2d)
        return safe_waypoints[sorted_idx]

    def _initialize(self) -> Tensor:
        raise NotImplementedError
    
    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the agent's current position to the goal,
        and drives the pre-trained PointNav policy. Supports both 2D (x, y) and 3D (x, y, z) goal inputs.

        Args:
            goal (np.ndarray): Goal location coordinates (2D or 3D).
            stop (bool): Whether to trigger stop action when reaching goal radius.
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
                # interpolation_mode="nearest",
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
        shared_pcds = self._observations_cache.get("shared_pcds")
        detections = []
        for i, (rgb, depth, tf, min_depth, max_depth, fx, fy) in enumerate(object_map_rgbd):
            pcd = shared_pcds[i] if shared_pcds is not None and i < len(shared_pcds) else None
            detections.append(
                self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy, pcd=pcd)
            )
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        goal_3d = self._get_target_object_location(robot_xyz)

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

        if self._compute_frontiers:
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
    use_octomap: bool = True                 # Use the sparse OctoMap backend; True saves memory, False uses the dense grid (faster on small maps).
    voxel_size: float = 0.01                 # Voxel grid resolution (m); smaller = finer map but more memory/compute.
    min_obstacle_height: float = 0.15        # Lower height bound (m) for obstacle voxels; too low includes floor noise, too high misses low obstacles.
    max_obstacle_height: float = 1.50        # Upper height bound (m) for obstacle voxels; too low ignores tall obstacles.
    agent_radius: float = 0.18               # Robot physical radius (m) used for collision inflation; larger = more conservative navigation.
    nav_slice_height: float = 0.35           # Reference navigation height (m) for frontier slicing and visualization.
    hole_area_thresh: int = 100000           # Hole area threshold (px) for depth gap filling; -1 disables filling (assume max depth).
    obstacle_map_area_threshold: float = 1.5 # Frontier area threshold (m^2) to filter small isolated regions; higher = fewer, larger frontiers.

    # Phase 4: TSP3D Perception & Visual Grounding
    sigma_tar: float = 0.25           # Target confidence threshold; higher = stricter/fewer detections, lower = more detections but more false positives.
    sigma_sce: float = 0.15           # TGP scene voxel retention threshold; higher = more aggressive pruning (fewer hallucinations, may miss targets).
    tau: float = 0.10                 # Soft-pruning temperature; higher = smoother/looser pruning, lower = harder thresholding.
    near_field_dist: float = 0.8      # Distance (m) below which near-field adaptive sigma scaling activates; larger = scaling kicks in earlier.
    near_field_sigma_scale: float = 0.3  # Minimum voxel retention ratio near surfaces; higher = keep more voxels near obstacles.
    use_raw_nlp: bool = True          # Use raw NLP prompt formatting; True = multi-class synonym merging, False = use only the primary class.
    enable_retry: bool = False        # 08-18: retry on empty results with reduced sigma_sce (measured recovery 0.05%, ~48% of queries); True keeps old retry for A/B

    # Phase 4.5: Temporal PCD Sliding Window (Multi-frame Fusion for TSP3D)
    pcd_window_size: int = 8          # Number of frames fused for point-cloud accumulation; larger = more complete geometry but slower/staler.
    fuse_voxel_size: float = 0.02     # Voxel downsample size (m) for window fusion; larger = fewer points, faster, coarser.
    fuse_max_points: int = 200000     # Cap on fused point count; higher = more detail but heavier sparse-conv inference.

    # Phase 5: 3D Exploration & Frontier Clustering
    compute_frontiers: bool = True    # Compute frontiers from the 3D map for exploration; False disables frontier-based exploration.
    dbscan_eps: float = 0.15          # DBSCAN neighborhood radius (m); larger = coarser clusters, smaller = more fragmented frontiers.
    dbscan_min_samples: int = 5       # Min points for a DBSCAN core; higher = filters noise more aggressively.
    min_dists_2d: float = 0.5         # Min 2D distance (m) a waypoint must be from the robot; higher = avoids close targets.

    # Phase 6: Navigation Execution & Termination
    pointnav_stop_radius: float = 0.25  # Distance (m) at which the agent stops near the goal; larger = stops farther from the target.

    # Phase 7: ITM3D Semantic Value Mapping & Space Carving (used by HabitatITM3DPolicy)
    use_max_confidence: bool = True          # Project voxel similarity using global max confidence; False uses mean/average aggregation.
    sync_explored_areas: bool = False        # Synchronize the physically explored area into the semantic map; True improves map consistency at some cost.
    exploration_thresh: float = 0.15         # Min similarity to switch into target-chasing; higher = more conservative chasing.
    carving_noise_tolerance: float = 0.2     # Depth error tolerance (m) to classify voxels as ghosts/false positives; higher = less aggressive carving.
    min_carving_conf: float = 0.05           # Confidence floor; voxels below this are removed entirely; higher = cleaner map, may erase weak targets.
    carving_decay_factor: float = 0.5        # Multiplicative confidence decay per frame inside free space; lower = faster decay/cleaner map.
    pruning_min_conf: float = 0.01           # Global voxel low-confidence pruning threshold; higher = removes more voxels.
    max_voxel_dist: float = 15.0             # Max distance (m) to keep a voxel; larger = bigger map memory, smaller = less far-field noise.
    downsampling_step: int = 8               # Downsampling stride for similarity projection; larger = faster but coarser value map.
    min_valid_conf: float = 1e-4             # Floor for valid back-projection weights; higher = filters weaker contributions.
    cylinder_radius: float = 1.0             # Physical radius (m) of the 3D scoring cylinder around a boundary point.
    cylinder_height: float = 1.5             # Physical height (m) of the 3D scoring cylinder; taller = scores more vertical neighbors.
    query_radius: float = 0.5                # Query radius (m) for 2D collision pre-filtering; larger = more conservative.

    # Phase 8: 2.5D Semantic Value Plane (BEV) -- used by itm_policy (HabitatITMPolicyV1/V2)
    value_map_style: str = "region"          # Semantic value mapping mode: "region" (V1) / "surface" (V2)
    h_lam: float = 0.0                       # Bonus weight lambda (lambda=0 reduces to VLFM baseline)
    h_norm_max: float = 1.0                  # Upper bound for normalizing H (locks the value score range)
    h_z_min: float = 0.15                    # Lower bound of the H1 passable band (m)
    h_z_max: float = 0.88                    # Upper bound of the H1 passable band (m, robot height)
    query_radius_m: float = 0.5              # Horizontal query radius r_h for route 2 (includes dilation semantics)
    query_z_min: float = 0.15                # Lower bound of the fixed query height band (m, surface landing)
    query_z_max: float = 1.50                # Upper bound of the fixed query height band (m)

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        return [f.name for f in fields(VLVMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlvm_config_base", node=VLVMConfig())
