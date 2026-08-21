import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.policy.tsp3d_objectnav_policy import TSP3DObjectNavPolicy

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"

class BaseITM3DPolicy(TSP3DObjectNavPolicy):
    """
    Manages 3D semantic mapping, space carving, DBSCAN clustering, and waypoint scoring.
    """
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(3)  # Shape (3,) for 3D compatibility

    def __init__(
        self,
        text_prompt: str = "Seems like there is a target_object ahead.",
        voxel_size: float = 0.05,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        exploration_thresh: float = 0.15,
        carving_noise_tolerance: float = 0.2,
        min_carving_conf: float = 0.05,
        carving_decay_factor: float = 0.5,
        pruning_min_conf: float = 0.01,
        max_voxel_dist: float = 15.0,
        downsampling_step: int = 8,
        min_valid_conf: float = 1e-4,
        cylinder_radius: float = 1.0,
        cylinder_height: float = 1.5,
        query_radius: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            text_prompt=text_prompt,
            voxel_size=voxel_size,
            *args,
            **kwargs
        )
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_channels = len(text_prompt.split(PROMPT_SEPARATOR))
        self._acyclic_enforcer = AcyclicEnforcer()
        
        # 3D Semantic Map configurations
        self._voxel_size = voxel_size 
        self._3d_semantic_voxel_map: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._did_reset = True

        self._map_cache_dirty = True
        self._cached_keys: List[Tuple[int, int, int]] = []
        self._cached_coords = np.empty((0, 3))
        self._cached_scores = np.empty((0, 1))
        # Exposure of key parameters
        self._exploration_thresh = exploration_thresh
        self._carving_noise_tolerance = carving_noise_tolerance
        self._min_carving_conf = min_carving_conf
        self._carving_decay_factor = carving_decay_factor
        self._pruning_min_conf = pruning_min_conf
        self._max_voxel_dist = max_voxel_dist
        self._downsampling_step = downsampling_step
        self._min_valid_conf = min_valid_conf
        self._cylinder_radius = cylinder_radius
        self._cylinder_height = cylinder_height
        self._query_radius = query_radius

    def _reset(self) -> None:
        super()._reset()
        self._map_cache_dirty = True
        self._cached_keys = []
        self._cached_coords = np.empty((0, 3))
        self._cached_scores = np.empty((0, self._value_channels))

        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(3)
        self._3d_semantic_voxel_map.clear()
        self._did_reset = True

    def _rebuild_map_cache(self) -> None:
        """
        Rebuilds cached numpy arrays (keys, coordinates, scores) if the map state has been marked dirty.
        """
        if not self._map_cache_dirty:
            return
        self._map_cache_dirty = False
        if not self._3d_semantic_voxel_map:
            self._cached_keys = []
            self._cached_coords = np.empty((0, 3))
            self._cached_scores = np.empty((0, self._value_channels))
        else:
            self._cached_keys = list(self._3d_semantic_voxel_map.keys())
            self._cached_coords = np.array(self._cached_keys) * self._voxel_size
            self._cached_scores = np.array([v[0] for v in self._3d_semantic_voxel_map.values()])
        self._map_cache_dirty = False

    def _carve_free_space(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        fov: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """
        Reprojects saved 3D voxels back into the current camera frustum,
        reducing confidence values of historical voxels located in the observed free space.
        """
        if not self._3d_semantic_voxel_map:
            return

        self._rebuild_map_cache()
        voxel_keys = self._cached_keys
        pts_w = self._cached_coords
        
        try:
            tf_episodic_to_camera = np.linalg.inv(tf_camera_to_episodic)
        except np.linalg.LinAlgError:
            return
        R_T = tf_episodic_to_camera[:3, :3].T
        t = tf_episodic_to_camera[:3, 3]
        pts_c = pts_w @ R_T + t
        # camera base frame is [forward, left, up] (see _project_rgbd_to_3d_point_cloud)
        forward = pts_c[:, 0]  # depth axis
        left = pts_c[:, 1]
        up = pts_c[:, 2]

        valid_depth = (forward > min_depth) & (forward < max_depth)
        if not np.any(valid_depth):
            return

        H, W = depth.shape[:2]
        fx = W / (2 * np.tan(fov / 2))
        fy = H / (2 * np.tan(fov / 2))
        
        # pinhole back-projection: forward=zc, left=-xc, up=-yc
        u = np.round(-left * fx / (forward + 1e-6) + W / 2.0).astype(int)
        v = np.round(-up * fy / (forward + 1e-6) + H / 2.0).astype(int)
        
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        active_mask = valid_depth & in_bounds
        if not np.any(active_mask):
            return
            
        measured_depth_norm = depth[v[active_mask], u[active_mask]]
        measured_depth_m = measured_depth_norm * (max_depth - min_depth) + min_depth
        voxel_depth_m = forward[active_mask]
        
        free_space_indices = np.where(voxel_depth_m < (measured_depth_m - self._carving_noise_tolerance))[0]        
        if len(free_space_indices) == 0:
            return
            
        original_indices = np.where(active_mask)[0][free_space_indices]
        did_modify = False
        for idx in original_indices:
            key = voxel_keys[idx]
            v_prev, c_prev = self._3d_semantic_voxel_map[key]
            c_new = c_prev * self._carving_decay_factor
            if c_new < self._min_carving_conf:
                self._3d_semantic_voxel_map.pop(key, None)
                did_modify = True
            else:
                self._3d_semantic_voxel_map[key] = (v_prev, c_new)
                did_modify = True
        if did_modify:
            self._map_cache_dirty = True

    def _prune_distant_voxels(self) -> None:
        """
        Prunes historical voxels that are located too far from the agent or possess negligible confidence values.
        """
        robot_xyz = self._observations_cache.get("robot_xy_z")
        if robot_xyz is None:
            robot_xy = self._observations_cache.get("robot_xy", np.zeros(2))
            robot_xyz = np.array([robot_xy[0], robot_xy[1], getattr(self, "_camera_height", 0.88)])
        
        keys_to_remove = []
        for key, (v, c) in self._3d_semantic_voxel_map.items():
            if c < self._pruning_min_conf:
                keys_to_remove.append(key)
                continue
            pt_w = np.array(key) * self._voxel_size
            if np.linalg.norm(pt_w - robot_xyz) > self._max_voxel_dist:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            self._3d_semantic_voxel_map.pop(key, None)
        if len(keys_to_remove) > 0:
            self._map_cache_dirty = True

    def _project_value_to_map3d(
        self,
        scores: np.ndarray,
        pcd: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> None:
        """
        Projects 2D similarity scores into the sparse 3D semantic voxel grid with space-carving.
        """
        if len(pcd) == 0:
            return
        
        step = self._downsampling_step ** 2
        pcd_sub = pcd[::step]
        pts_w = pcd_sub[:, :3]
        try:
            tf_episodic_to_camera = np.linalg.inv(tf_camera_to_episodic)
        except np.linalg.LinAlgError:
            return
        R_T = tf_episodic_to_camera[:3, :3].T
        t = tf_episodic_to_camera[:3, 3]
        pts_c = pts_w @ R_T + t
        # camera base frame is [forward, left, up] (see _project_rgbd_to_3d_point_cloud)
        forward = pts_c[:, 0]  # optical / depth axis
        left = pts_c[:, 1]
        up = pts_c[:, 2]

        dist_3d = np.sqrt(forward**2 + left**2 + up**2)
        cos_theta = np.clip(forward / (dist_3d + 1e-6), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        c_angular = np.zeros_like(theta)
        in_fov = theta <= (fov / 2.0)
        c_angular[in_fov] = np.cos(theta[in_fov] / (fov / 2.0) * (np.pi / 2.0)) ** 2
        c_distance = np.maximum(0.0, 1.0 - dist_3d / max_depth)
        C_3D = c_angular * c_distance

        valid_conf = C_3D > self._min_valid_conf
        if not np.any(valid_conf):
            return
        pts_w_valid = pts_w[valid_conf]
        C_3D = C_3D[valid_conf]
        voxel_coords = np.round(pts_w_valid / self._voxel_size).astype(int)
        norm_scores = np.clip(scores.copy(), 0.0, 1.0)

        sort_idx = np.argsort(-C_3D)
        voxel_coords_sorted = voxel_coords[sort_idx]
        C_3D_sorted = C_3D[sort_idx]
        unique_coords, unique_indices = np.unique(voxel_coords_sorted, axis=0, return_index=True)
        unique_C_3D = C_3D_sorted[unique_indices]
        
        for coord, c_curr in zip(unique_coords, unique_C_3D):
            voxel_key = tuple(coord)
            v_curr = norm_scores
            if voxel_key in self._3d_semantic_voxel_map:
                v_prev, c_prev = self._3d_semantic_voxel_map[voxel_key]
                denom = c_curr + c_prev
                if denom > 1e-6:
                    v_new = (c_curr * v_curr + c_prev * v_prev) / denom
                    c_new = (c_curr**2 + c_prev**2) / denom
                else:
                    v_new = v_prev
                    c_new = c_prev
                self._3d_semantic_voxel_map[voxel_key] = (v_new, c_new)
            else:
                self._3d_semantic_voxel_map[voxel_key] = (v_curr, c_curr)
        self._map_cache_dirty = True

        # Free space carving using exact positions (correct signature without norm_scores)
        self._carve_free_space(depth, tf_camera_to_episodic, fov, min_depth, max_depth)
        self._prune_distant_voxels()
        self._map_cache_dirty = True

    def _update_value_map(
        self, 
        pcds: Union[List[np.ndarray], None] = None,
    ) -> None:
        """
        Computes 2D image-text similarities using BLIP-2 ITM and projects them to the 3D map.
        Uses cached observation buffers.
        """
        if self._target_object == "":
            return

        all_rgb = [item[0] for item in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    frame,
                    prompt.replace("target_object", self._target_object.replace("|", "/")),
                )
                for prompt in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for frame in all_rgb
        ]
        if pcds is None:
            pcds = [
                self._project_rgbd_to_3d_point_cloud(rgb, depth, fx, fy, tf, min_depth, max_depth)
                for (rgb, depth, tf, min_depth, max_depth, fx, fy)
                in self._observations_cache["object_map_rgbd"]
            ]
        shared_pcds = pcds if pcds is not None else [np.empty((0, 6))] * len(cosines)

        for cosine, pcd, (frame, depth, tf, min_depth, max_depth, fov_rad) in zip(
            cosines, shared_pcds, self._observations_cache["value_map_rgbd"]):

            self._project_value_to_map3d(
                scores=np.array(cosine),
                pcd=pcd,
                depth=depth,
                tf_camera_to_episodic=tf,
                min_depth=min_depth,
                max_depth=max_depth,
                fov=fov_rad)
            
            if getattr(self, "_visualize", False) and getattr(self, "_value_map", None) is not None:
                self._value_map.update_map(
                    np.array(cosine), depth, tf, min_depth, max_depth, fov_rad)


    def _sort_waypoints_3d(
        self, waypoints: np.ndarray, radius: float, reduce_fn: Any = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Sorts 3D waypoints based on integrated local cylinder scores within a given radius.
        Mirrors the behavior of ValueMap.sort_waypoints() but operates natively in 3D space.
        """
        if len(waypoints) == 0:
            return np.empty((0, 3)), []
        
        robot_xyz = self._observations_cache.get("robot_xy_z")
        current_z = robot_xyz[2] if robot_xyz is not None else getattr(self, "_camera_height", 0.88)

        # Batch fast collision pre-filtering
        is_free_batch = np.ones(len(waypoints), dtype=bool)
        if self._obstacle_map3d is not None:
            grid = getattr(self._obstacle_map3d, "_grid", None)
            if grid is not None:
                px_coords = self._obstacle_map3d._xy_to_px(waypoints[:, :2]).astype(int)
                H, W = grid.shape[:2]
                in_bounds = (px_coords[:, 1] >= 0) & (px_coords[:, 1] < H) & \
                            (px_coords[:, 0] >= 0) & (px_coords[:, 0] < W)
                
                valid_px = px_coords[in_bounds]
                if len(grid.shape) == 3:
                    is_free_batch[in_bounds] = ~np.any(grid[valid_px[:, 1], valid_px[:, 0], :] == 2, axis=1)
                else:
                    is_free_batch[in_bounds] = (grid[valid_px[:, 1], valid_px[:, 0]] != 2)
                is_free_batch[~in_bounds] = False
        
        waypoint_raw_vectors = []
        for idx, wp in enumerate(waypoints):
            if not is_free_batch[idx]:
                waypoint_raw_vectors.append(np.full(self._value_channels, -1.0))
                continue

            wp_aligned = wp.copy()
            if len(wp_aligned) == 2:
                wp_aligned = np.append(wp_aligned, current_z)
            elif len(wp_aligned) >= 3 and np.abs(wp_aligned[2]) < 1e-4:
                wp_aligned[2] = current_z
                
            raw_vector = self._query_local_3d_semantic_score(wp_aligned, radius)
            waypoint_raw_vectors.append(raw_vector)

        if self._value_channels > 1:
            assert reduce_fn is not None, "Must provide a reduction function when using multiple value channels."
            values = reduce_fn(waypoint_raw_vectors)
        else:
            values = [float(v[0]) for v in waypoint_raw_vectors]

        sorted_indices = np.argsort(values)[::-1]
        sorted_waypoints_3d = []
        for i in sorted_indices:
            wp = waypoints[i]
            wp_aligned = wp.copy()
            if len(wp_aligned) == 2:
                wp_aligned = np.append(wp_aligned, current_z)
            elif len(wp_aligned) >= 3 and np.abs(wp_aligned[2]) < 1e-4:
                wp_aligned[2] = current_z
            sorted_waypoints_3d.append(wp_aligned)
        
        sorted_waypoints = np.array(sorted_waypoints_3d)
        sorted_values = [values[i] for i in sorted_indices]
        return sorted_waypoints, sorted_values

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        """
        Extracts the optimal target coordinate from the algorithmic base 
        and pipes it to PointNav.
        """
        result = self._get_exploration_goal(observations)
        # _get_exploration_goal returns None (rather than an empty tuple) when no
        # valid frontier exists, so the return value must be explicitly checked
        if result is None:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = result
        if len(best_frontier) == 0 or np.array_equal(best_frontier, np.zeros(3)):
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        # Convert the high-level 3D target coordinate to low-level control actions
        pointnav_action = self._pointnav(best_frontier[:2], stop=False)
        return pointnav_action


    def _is_obstacle_free(self, x: float, y: float) -> bool:
        """
        Helper method to project world points to the 2D obstacle map grid.
        """
        if self._obstacle_map3d is None:
            return True
        try:
            pts = np.array([[x, y]])
            px_coord = self._obstacle_map3d._xy_to_px(pts)[0]
            py, px = px_coord[0], px_coord[1]

            grid = getattr(self._obstacle_map3d, "_grid", None)
            if grid is None:
                return True
            
            H, W = grid.shape[:2]
            if 0 <= px < H and 0 <= py < W:
                if len(grid.shape) == 3:
                    # Passable as long as the column contains no value 2 (occupied obstacle)
                    return bool(not np.any(grid[px, py, :] == 2))
                else:
                    # Passable as long as the cell does not equal value 2 (occupied obstacle)
                    return bool(grid[px, py] != 2)
        except Exception:
            pass
        return True

    def _query_local_3d_semantic_score(self, waypoint: np.ndarray, radius: float) -> float:
        """
        Vectorized 3D Cylinder query that aggregates voxel scores around a waypoint.
        """
        if not self._is_obstacle_free(waypoint[0], waypoint[1]):
            return np.full(self._value_channels, -1.0)
        
        if not self._3d_semantic_voxel_map:
            return np.full(self._value_channels, -1.0)

        self._rebuild_map_cache()
        voxel_coords = self._cached_coords
        scores_arr = self._cached_scores

        horizontal_r = max(radius, self._cylinder_radius)
        vertical_h = self._cylinder_height
        dx_dy = voxel_coords[:, :2] - waypoint[:2]
        dist_horizontal = np.linalg.norm(dx_dy, axis=1)
        dist_vertical = np.abs(voxel_coords[:, 2] - waypoint[2])
        
        in_cylinder = (dist_horizontal <= horizontal_r) & (dist_vertical <= vertical_h)
        if not np.any(in_cylinder):
            return np.full(self._value_channels, -1.0)

        median_score = np.median(scores_arr[in_cylinder], axis=0)
        return median_score

    def _filter_candidate_waypoints(self, candidate_waypoints: np.ndarray) -> np.ndarray:
        """
        Filters out candidate 3D waypoints that are too close to the agent or collide with obstacles.
        """
        if len(candidate_waypoints) == 0:
            return np.empty((0, 3))
            
        robot_xyz = self._observations_cache.get("robot_xy_z", np.zeros(3))
        waypoints_3d = []
        for wp in candidate_waypoints:
            wp_3d = wp.copy()
            if len(wp_3d) == 2:
                wp_3d = np.append(wp_3d, robot_xyz[2])
            elif len(wp_3d) >= 3 and np.abs(wp_3d[2]) < 1e-4:
                wp_3d[2] = robot_xyz[2]
            waypoints_3d.append(wp_3d)
        waypoints_3d = np.array(waypoints_3d)

        dists_2d = np.linalg.norm(waypoints_3d[:, :2] - robot_xyz[:2], axis=1)
        far_mask = dists_2d > 0.5  # Legacy 3D waypoint min-distance filter (original default 0.5m)
        if np.any(far_mask):
            waypoints_3d = waypoints_3d[far_mask]

        if len(waypoints_3d) == 0:
            return np.empty((0, 3))

        colliding_mask = self._obstacle_map3d.check_collision(waypoints_3d)
        safe_indices = np.where(~colliding_mask)[0]
        if len(safe_indices) == 0:
            return waypoints_3d

        return waypoints_3d[safe_indices]

    def _get_best_frontier(
        self,
        observations: Any,
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Selects the best 3D frontier waypoint based on the 3D semantic value map.
        """
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        if len(sorted_pts) == 0:
            return np.zeros(3), -1.0

        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2]) if len(sorted_values) >= 2 else (sorted_values[0] if len(sorted_values) > 0 else 0.0, 0.0)

        os.environ["DEBUG_INFO"] = ""
        if not np.array_equal(self._last_frontier, np.zeros(3)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)
                if closest_index != -1:
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                value_flat = abs(curr_value - self._last_value) < 0.02
                if not value_flat and curr_value + 0.01 > self._last_value:
                    print(f"Sticking to last point. (Index: {curr_index})")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier[:2], top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = min(
                range(len(sorted_pts)),
                key=lambda i: np.linalg.norm(sorted_pts[i][:2] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]

        if np.array_equal(best_frontier, self._last_frontier):
            # If very close to the previous goal point, the agent has arrived or is
            # stuck; force a switch to the next highest-scoring frontier.
            if np.linalg.norm(best_frontier[:2] - robot_xy) < 0.3 and len(sorted_pts) > 1:
                print("Agent is close to last frontier, forcing next best frontier to avoid stalling.")
                best_frontier_idx = 1 if len(sorted_pts) > 1 else 0
                best_frontier = sorted_pts[best_frontier_idx]
                best_value = sorted_values[best_frontier_idx]
        
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier[:2], top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_exploration_goal(self, observations: Any) -> Union[np.ndarray, None]:
        """
        Core path planner.
        Extracts frontiers, clusters them, performs collision checking, 
        and selects the highest scoring 3D coordinate.
        """
        candidate_waypoints = np.empty((0, 3))
        fallback = self._observations_cache.get("frontier_sensor_3d")
        if fallback is None:
            fallback = self._observations_cache.get("frontier_sensor")
        if fallback is not None and len(fallback) > 0:
            fallback_arr = np.array(fallback)
            if fallback_arr.ndim == 2 and fallback_arr.shape[1] >= 2 and not np.all(fallback_arr == 0):
                candidate_waypoints = fallback_arr

        if len(candidate_waypoints) == 0:
            return None
            
        filtered_waypoints = self._filter_candidate_waypoints(candidate_waypoints)
        if len(filtered_waypoints) == 0:
            filtered_waypoints = candidate_waypoints
            
        best_frontier, best_value = self._get_best_frontier(observations, filtered_waypoints)
        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []
        frontiers = self._observations_cache.get("frontier_sensor", np.empty((0, 2)))
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(3)):
            if any(np.array_equal(self._last_goal[:2], frontier[:2]) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal[:2], marker_kwargs))
            
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info


class ITM3DPolicyV1(BaseITM3DPolicy):
    """
    Habitat-specific active navigation strategy wrapper. 
    """
    def __init__(
        self,
        text_prompt: str = "Seems like there is a target_object ahead.",
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        voxel_size: float = 0.05,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            text_prompt=text_prompt,
            voxel_size=voxel_size,
            *args,
            **kwargs
        )
        self._value_map = ValueMap(
            value_channels=self._value_channels,
            size=400,
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map3d if sync_explored_areas else None,
        )
        self._vis_reduce_fn = lambda i: np.max(i, axis=-1)

    def _reset(self) -> None:
        super()._reset()
        if hasattr(self, "_value_map") and self._value_map is not None:
            self._value_map.reset()

    def _sort_frontiers_by_value(self, observations: "TensorDict", frontiers: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        return self._sort_waypoints_3d(frontiers, 0.5)

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        shared_pcds = [
            self._project_rgbd_to_3d_point_cloud(rgb, depth, fx, fy, tf, min_depth, max_depth)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy)
            in self._observations_cache["object_map_rgbd"]
        ]
        self._observations_cache["shared_pcds"] = shared_pcds
        self._update_value_map(pcds=shared_pcds)
        return super().act(observations, rnn_hidden_states, 
                        prev_actions, masks, deterministic)


class ITM3DPolicyV2(BaseITM3DPolicy):
    """
    Habitat-specific active navigation strategy wrapper. 
    """
    def __init__(
        self,
        text_prompt: str = "Seems like there is a target_object ahead.",
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        voxel_size: float = 0.05,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(voxel_size=voxel_size, *args, **kwargs)
        self._value_map = ValueMap(
            value_channels=self._value_channels,
            size=400,
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map3d if sync_explored_areas else None,
        )

        exploration_thresh = self._exploration_thresh
        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            if arr.shape[2] == 1:
                return arr[:, :, 0]
            first_channel = arr[:, :, 0]
            max_values = np.max(arr, axis=2)
            mask = first_channel > exploration_thresh
            return np.where(mask, first_channel, max_values)

        self._vis_reduce_fn = visualize_value_map

    def _reset(self) -> None:
        super()._reset()
        if hasattr(self, "_value_map") and self._value_map is not None:
            self._value_map.reset()

    def _reduce_values(self, values: List[np.ndarray]) -> List[float]:
        """
        Uses the maximum target confidence among all candidate points.
        """
        if len(values) == 0:
            return []
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        # Global mode selection: if target confidence is low, run exploration mode
        if max_target_value < self._exploration_thresh:
            # Return exploration channel score
            explore_values = [float(v[1]) if self._value_channels > 1 else float(v[0]) for v in values]
            return explore_values
        else:
            # Target detected: switch to pursuit mode and sort by target confidence
            return [float(v[0]) for v in values]

    def _sort_frontiers_by_value(self, observations: "TensorDict", frontiers: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        return self._sort_waypoints_3d(frontiers, 0.5, reduce_fn=self._reduce_values)

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, 
                        prev_actions, masks, deterministic)


## To use ITM3D V1/V2, add the following parameters to class VLVMConfig and the YAML config file for registration.
    # use_max_confidence: bool = True
    # sync_explored_areas: bool = False
    # carving_noise_tolerance: float = 0.2
    # min_carving_conf: float = 0.05
    # carving_decay_factor: float = 0.5
    # pruning_min_conf: float = 0.01
    # max_voxel_dist: float = 15.0
    # downsampling_step: int = 8
    # min_valid_conf: float = 1e-4
    # cylinder_radius: float = 1.0
    # cylinder_height: float = 1.5
    # query_radius: float = 0.5
    # exploration_thresh: float = 0.15
