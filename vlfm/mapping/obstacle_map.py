# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Union, Optional

import cv2
import numpy as np

from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes

class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([])  # Stores the frontiers in pixel coordinates
    frontiers: np.ndarray = np.array([])      # Stores the frontiers in world coordinates (meters)
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,                  # Lower height bound (m) for obstacle points; too low includes floor noise, too high misses low obstacles
        max_height: float,                  # Upper height bound (m) for obstacle points; too low ignores tall obstacles
        agent_radius: float,                # Robot physical radius (m) used for inflation; larger = more conservative navigation
        area_thresh: float = 3.0,           # Frontier area threshold (m^2); higher = fewer, larger frontiers
        hole_area_thresh: int = 100000,     # Depth hole filling threshold (px^2); higher = fills larger holes
        size: int = 1000,                   # Square map size in pixels; larger = bigger map memory
        pixels_per_meter: int = 20,         # Map resolution (px/m); higher = finer but more memory
    ):
        super().__init__(size, pixels_per_meter)
        self.explored_area = np.zeros((size, size), dtype=bool)     # Explored space mask
        self._map = np.zeros((size, size), dtype=bool)              # Raw obstacle map (1 for obstacles)
        self._navigable_map = np.zeros((size, size), dtype=bool)    # Navigable space mask

        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2

        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def reset(self) -> None:
        super().reset()
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).

            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if update_obstacles:
            # Fill small holes or depth sensor blind spots (where depth is 0)
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0  # Assume blind spots are at maximum depth range
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh) # Connected-component filtering

            # Scale normalized depth [0, 1] back to metric range [min_depth, max_depth]
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth  # Filter points beyond maximum range
            
            # Back-project pixels to 3D coordinates and transform to world frame
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
            
            # Filter point cloud by height limits [min_height, max_height] to detect obstacles
            obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            # Project 3D obstacle points to the 2D grid map
            # Populate topdown map with obstacle locations
            xy_points = obstacle_cloud[:, :2]  # Discard Z axis
            pixel_points = self._xy_to_px(xy_points)  # Map coordinates to pixel grid using BaseMap scaling
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1  # Mark obstacle pixels

            # Update the navigable area, which is an inverse of the obstacle map 
            # after a dilation operation to accommodate the robot's radius.
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]  # Extract current agent 2D position
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0  # Ensure known obstacles are never marked as free/explored
        
        # Keep the largest connected component of the explored area containing the agent
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

    def visualize(self) -> np.ndarray:
        """Visualizes the map."""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # Draw unnavigable areas in gray
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # Draw obstacles in black
        vis_img[self._map == 1] = (0, 0, 0)
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img

def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]


from scipy import ndimage
class ObstacleMap3D(BaseMap):
    """
    3D Occupancy Grid Map based on a dense NumPy 3D array.
    """
    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([]) # Stores pixel-coordinates of the frontiers
    frontiers: np.ndarray = np.array([]) # Stores world-coordinates (meters) of the frontiers
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,                  # Lower height bound (m) for obstacle points; too low includes floor noise, too high misses low obstacles
        max_height: float,                  # Upper height bound (m) for obstacle points; too low ignores tall obstacles
        agent_radius: float,                # Robot physical radius (m) used for collision checking; larger = more conservative navigation
        area_thresh: float = 3.0,           # Frontier area threshold (m^2); higher = fewer, larger frontiers
        hole_area_thresh: int = 100000,     # Depth hole filling area threshold (px^2); higher = fills larger holes
        size: int = 1000,                   # Square map size in pixels; larger = bigger map memory
        pixels_per_meter: int = 20,         # Map resolution (px/m); higher = finer but more memory
        voxel_size: float = 0.05,           # 3D voxel size (m); smaller = finer grid but more memory/compute
        height_size: int = 40,              # Number of Z-axis slices; higher = taller vertical coverage
        nav_slice_height: float = 0.35,     # Reference navigation height (m) for frontier slicing and visualization
    ):
        super().__init__(size, pixels_per_meter)
        self._min_height = min_height
        self._max_height = max_height
        self._voxel_size = voxel_size
        self._height_size = height_size
        self._agent_radius = agent_radius
        self._hole_area_thresh = hole_area_thresh
        self._nav_slice_height = nav_slice_height

       # 3D State Grid: 0=Unexplored, 1=Free, 2=Occupied
        self._grid = np.zeros((size, size, height_size), dtype=np.uint8)
        # Compatibility properties maintained for downstream policies and visualizers
        self.explored_area = np.zeros((size, size, height_size), dtype=bool)     # Explored space mask
        self._navigable_map = np.zeros((size, size, height_size), dtype=bool)    # Navigable space mask
        self._map = np.zeros((size, size, height_size), dtype=bool)              # Raw obstacle mask
        
    def reset(self) -> None:
        super().reset()
        self._grid.fill(0)
        self.explored_area.fill(0)
        self._navigable_map.fill(0)
        self._map.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def _xyz_to_grid_index(self, xyz: np.ndarray) -> np.ndarray:
        """Maps world coordinates (x, y, z) to 3D grid index (cx, cy, cz)"""
        indices = np.empty((xyz.shape[0], 3), dtype=np.int32)
        indices[:, :2] = self._xy_to_px(xyz[:, [0, 1]])
        indices[:, 2] = ((xyz[:, 2] - self._min_height) / self._voxel_size).astype(np.int32)
        return indices

    def _grid_index_to_xyz(self, indices: np.ndarray) -> np.ndarray:
        """Maps 3D grid index (cy, cx, cz) back to world coordinates (x, y, z)"""
        # Note: indices input from _get_frontiers is [cy, cx, cz] (row, col, z)
        # _px_to_xy expects [cx, cy] (col, row), so we swap the first two columns.
        xy_world = self._px_to_xy(indices[:, [1, 0]])
        z_world = indices[:, 2] * self._voxel_size + self._min_height
        return np.stack([xy_world[:, 0], xy_world[:, 1], z_world], axis=-1)

    def _mark_free_space(self, camera_pos: np.ndarray, points: np.ndarray) -> None:
        if len(points) == 0:
            return
        
        # Subsample point clouds to maintain real-time performance
        step_size = max(1, len(points) // 1000)
        sampled_points = points[::step_size]
        # Calculate vectorized ray directions and lengths
        directions = sampled_points - camera_pos
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        valid_mask = distances.flatten() > 0
        directions = directions[valid_mask]
        distances = distances[valid_mask]
        if len(directions) == 0:
            return
        unit_directions = directions / distances
        max_dist = np.max(distances)

        # Calculate maximum interpolation steps clamped between 2 and 50
        num_steps = int(max_dist / self._voxel_size) + 1
        num_steps = max(2, min(num_steps, 50))

        # Interpolate rays up to 0.95 length to avoid clearing the obstacle surface itself
        steps = np.linspace(0.0, 0.95, num_steps)
        ray_lengths = steps[:, None] * distances.T
        ray_pts = camera_pos[None, None, :] + ray_lengths[:, :, None] * unit_directions[None, :, :]
        # Reshape to coordinate list and retrieve corresponding grid indices
        ray_pts_flat = ray_pts.reshape(-1, 3)
        free_indices = self._xyz_to_grid_index(ray_pts_flat)
        free_indices = self._filter_in_bounds_indices(free_indices)
        if len(free_indices) == 0:
            return

        # Prune duplicate spatial voxels to reduce overhead on grid operations
        flat_indices = free_indices[:, 0] * (self.size * self._height_size) + free_indices[:, 1] * self._height_size + free_indices[:, 2]
        _, unique_idx = np.unique(flat_indices, return_index=True)
        free_indices = free_indices[unique_idx]
        
        # Only update Unexplored (0) states to Free (1) to prevent flickering of obstacles
        current_states = self._grid[free_indices[:, 1], free_indices[:, 0], free_indices[:, 2]]
        unexplored_mask = (current_states == 0)
        self._grid[free_indices[unexplored_mask, 1], free_indices[unexplored_mask, 0], free_indices[unexplored_mask, 2]] = 1

    def _get_frontiers(self) -> np.ndarray:
        """Finds voxels that are Free (1) and have at least one Unknown (0) voxel in their 26-neighborhood"""

        # Vertical height-range constraint: slice a local band around the robot's current vertical slice cz
        nav_target_z = self._nav_slice_height  # Standard height (m) for robot navigation interaction
        nav_cz = int((nav_target_z - self._min_height) / self._voxel_size)
        cz = np.clip(nav_cz, 0, self._height_size - 1)
        z_min = max(0, cz - 3)
        z_max = min(self._height_size - 1, cz + 3)
        # Horizontal downsampling (stride=2) to extract a local grid slice (500, 500, 7)
        grid_slice = self._grid[::2, ::2, z_min:z_max+1]

        is_free = (grid_slice == 1)
        if not np.any(is_free):
            return np.array([])

        is_unknown = (grid_slice == 0)
        struct = np.ones((3, 3, 3), dtype=bool)
        dilated_unknown = ndimage.binary_dilation(is_unknown, structure=struct)

        frontier_mask = is_free & dilated_unknown
        cy_sub, cx_sub, cz_sub = np.where(frontier_mask)
        if len(cy_sub) == 0:
            return np.array([])
        cy = cy_sub * 2
        cx = cx_sub * 2
        cz = cz_sub + z_min

        frontiers = np.stack([cy, cx, cz], axis=-1)
        step = max(1, len(frontiers) // 500)
        return frontiers[::step]

    def _filter_in_bounds_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Filter out all indices that lie within the bounds of the 3D grid map.
        Indices have shape (N, 3): column 0 is px/cx, column 1 is py/cy, column 2 is cz.
        """
        if len(indices) == 0:
            return indices
        in_bounds = (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.size) &
            (indices[:, 1] >= 0) & (indices[:, 1] < self.size) &
            (indices[:, 2] >= 0) & (indices[:, 2] < self._height_size)
        )
        return indices[in_bounds]


    def update_map(
        self,
        tf_camera_to_episodic: np.ndarray,
        depth: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float = 0.0,
        pcd: Optional[np.ndarray] = None,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        camera_pos = tf_camera_to_episodic[:3, 3]
        agent_index = self._xyz_to_grid_index(camera_pos.reshape(1, 3))[0]
        self._last_agent_cz = np.clip(agent_index[2], 0, self._height_size - 1)

        if update_obstacles:
            if pcd is None or len(pcd) == 0:
                if self._hole_area_thresh == -1:
                    filled_depth = depth.copy()
                    filled_depth[depth == 0] = 1.0
                else:
                    filled_depth = fill_small_holes(depth, self._hole_area_thresh)
                scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
                mask = (scaled_depth > min_depth) & (scaled_depth < max_depth)
                point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
                if len(point_cloud_camera_frame) > 0:
                    pcd = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
                else:
                    pcd = np.empty((0, 3))
            if len(pcd) > 0:
                point_cloud_episodic_frame = pcd[:, :3]
                height_mask = (point_cloud_episodic_frame[:, 2] >= self._min_height) & \
                            (point_cloud_episodic_frame[:, 2] <= self._max_height)
                point_cloud_episodic_frame = point_cloud_episodic_frame[height_mask]

                # Update Occupied state (2)
                if len(point_cloud_episodic_frame) > 0:
                    voxel_indices = self._xyz_to_grid_index(point_cloud_episodic_frame)
                    voxel_indices = self._filter_in_bounds_indices(voxel_indices)
                    if len(voxel_indices) > 0:
                            self._grid[voxel_indices[:, 1], voxel_indices[:, 0], voxel_indices[:, 2]] = 2
                # Update Free state (1) along the camera rays
                self._mark_free_space(camera_pos, point_cloud_episodic_frame)

            # Fill raw occupied and basic navigable states directly without explicit dilation
            self._map = (self._grid == 2)
            self._navigable_map = (self._grid == 1) # Navigable maps directly represent Free voxels; collision is handled implicitly
        
        if not explore:
            return
        
        # Explored area is represented by any voxel that has been seen (not Unknown/0)
        self.explored_area = (self._grid > 0)
         # Keep only the largest explored connected component containing the agent
        labeled_array, num_features = ndimage.label(self.explored_area)
        if num_features > 1:
            px = np.clip(agent_index[0], 0, self.size - 1)
            py = np.clip(agent_index[1], 0, self.size - 1)
            cz = np.clip(agent_index[2], 0, self._height_size - 1)
            agent_label = labeled_array[py, px, cz]
            if agent_label > 0:
                self.explored_area = (labeled_array == agent_label)
        
        # Compute 3D frontiers
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._grid_index_to_xyz(self._frontiers_px)

    def check_collision(self, xyz: np.ndarray) -> np.ndarray:
        """
        Performs low-cost local-neighborhood collision checking against the built dense grid.
        """
    
        indices = self._xyz_to_grid_index(xyz)
        rx = int(np.ceil(self._agent_radius * self.pixels_per_meter))
        ry = int(np.ceil(self._agent_radius * self.pixels_per_meter))
        rz = int(np.ceil(self._agent_radius / self._voxel_size))
        
        colliding = np.zeros(len(xyz), dtype=bool)
        for i, idx in enumerate(indices):
            px, py, cz = idx  # Unpack [px, py, cz]

            zmin_raw = cz - rz
            zmax_raw = cz + rz
            # Vertical SAT culling: if the collision region is fully outside the grid's
            # vertical extent, skip it (it cannot possibly collide)
            if zmin_raw >= self._height_size or zmax_raw < 0:
                continue
            
            ymin = max(0, min(self.size - 1, py - ry))
            ymax = max(0, min(self.size - 1, py + ry))
            xmin = max(0, min(self.size - 1, px - rx))
            xmax = max(0, min(self.size - 1, px + rx))
            zmin = max(0, min(self._height_size - 1, cz - rz))
            zmax = max(0, min(self._height_size - 1, cz + rz))

            patch = self._grid[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1]
            if np.any(patch == 2):
                colliding[i] = True

        return colliding

    def visualize_slice(self) -> np.ndarray:
        """Visualizes a 2D slice of the 3D map at the robot's height"""

        # Lock onto the effective navigation height layer for robot-base/obstacle interaction
        nav_target_z = self._nav_slice_height  # Standard height (m) for robot navigation interaction
        nav_cz = int((nav_target_z - self._min_height) / self._voxel_size)
        cz = np.clip(nav_cz, 0, self._height_size - 1)
        
        explored_2d = self.explored_area[:, :, cz]
        navigable_2d = self._navigable_map[:, :, cz]
        map_2d = self._map[:, :, cz]
        
        vis_img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        vis_img[explored_2d == 1] = (200, 255, 200)
        vis_img[navigable_2d == 0] = self.radius_padding_color
        vis_img[map_2d == 1] = (0, 0, 0)
        
        if len(self._frontiers_px) > 0:
            for frontier in self._frontiers_px:
                if abs(frontier[2] - cz) <= 3:
                    cv2.circle(vis_img, (int(frontier[1]), int(frontier[0])), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


class ProbabilisticGrid(BaseMap):
    """
    3D Probabilistic Occupancy Grid Map backed by a dense int8 log-odds array.

    Unlike the ternary hard-decision ObstacleMap3D (0/1/2 overwrite), each voxel
    stores a log-odds occupancy score accumulated by Bayesian addition:

        ray endpoint  (observed surface):  l(v) <- clamp(l(v) + l_occ,  l_min, l_max)
        ray body      (free space):        l(v) <- clamp(l(v) + l_free, l_min, l_max)

    A single erroneous observation is therefore corrected by later consistent
    ones (no permanent "pinning", no state flipping), and repeated same-direction
    observations drive the voxel to a confident state.

    States are projected on query via thresholds:
        l(v) >  occ_thr  -> Occupied (2)
        l(v) <  free_thr -> Free     (1)
        otherwise        -> Unknown  (0)
    `_states` (uint8 0/1/2) is refreshed at the end of each update_map so all
    downstream consumers (frontier extraction, collision checking, 2.5D H1 mask
    aggregation) keep the exact same interface as ObstacleMap3D.
    """

    _map_dtype: np.dtype = np.dtype(bool)
    _frontiers_px: np.ndarray = np.array([])  # Stores pixel-coordinates of the frontiers
    frontiers: np.ndarray = np.array([])  # Stores world-coordinates (meters) of the frontiers
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,                  # Lower height bound (m) for obstacle points; too low includes floor noise, too high misses low obstacles
        max_height: float,                  # Upper height bound (m) for obstacle points; too low ignores tall obstacles
        agent_radius: float,                # Robot physical radius (m) used for collision checking; larger = more conservative navigation
        area_thresh: float = 3.0,           # Frontier area threshold (m^2); higher = fewer, larger frontiers
        hole_area_thresh: int = 100000,     # Depth hole filling area threshold (px^2); higher = fills larger holes
        size: int = 1000,                   # Square map size in pixels; larger = bigger map memory
        pixels_per_meter: int = 20,         # Map resolution (px/m); higher = finer but more memory
        voxel_size: float = 0.05,           # 3D voxel size (m); smaller = finer grid but more memory/compute
        height_size: int = 40,              # Number of Z-axis slices; higher = taller vertical coverage
        nav_slice_height: float = 0.35,     # Reference navigation height (m) for frontier slicing and visualization
        # ---- log-odds inverse sensor model ----
        log_odds_occ: float = 2.0,          # + increment on ray endpoint voxels (observed surface)
        log_odds_free: float = -2.0,        # - increment on ray-body voxels (free space)
        log_odds_min: float = -6.0,         # clamp lower bound (~0.003 occupancy probability)
        log_odds_max: float = 6.0,          # clamp upper bound (~0.997 occupancy probability)
        occ_threshold: float = 0.0,         # l(v) > occ_thr  -> Occupied
        free_threshold: float = 0.0,        # l(v) < free_thr -> Free
    ):
        super().__init__(size, pixels_per_meter)
        self._min_height = min_height
        self._max_height = max_height
        self._voxel_size = voxel_size
        self._height_size = height_size
        self._agent_radius = agent_radius
        self._hole_area_thresh = hole_area_thresh
        self._nav_slice_height = nav_slice_height

        self._l_occ = int(log_odds_occ)
        self._l_free = int(log_odds_free)
        self._l_min = int(log_odds_min)
        self._l_max = int(log_odds_max)
        self._occ_thr = occ_threshold
        self._free_thr = free_threshold

        # Dense int8 log-odds grid (one byte covers ~0.003~0.997 occupancy prob)
        self._grid = np.zeros((size, size, height_size), dtype=np.int8)
        # Projected ternary states (0=Unknown, 1=Free, 2=Occupied), refreshed per update
        self._states = np.zeros((size, size, height_size), dtype=np.uint8)
        # Compatibility properties maintained for downstream policies and visualizers
        self.explored_area = np.zeros((size, size, height_size), dtype=bool)     # Explored space mask
        self._navigable_map = np.zeros((size, size, height_size), dtype=bool)    # Navigable space mask
        self._map = np.zeros((size, size, height_size), dtype=bool)              # Raw obstacle mask

    def reset(self) -> None:
        super().reset()
        self._grid.fill(0)
        self._states.fill(0)
        self.explored_area.fill(0)
        self._navigable_map.fill(0)
        self._map.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def _xyz_to_grid_index(self, xyz: np.ndarray) -> np.ndarray:
        """Maps world coordinates (x, y, z) to 3D grid index (cx, cy, cz)"""
        indices = np.empty((xyz.shape[0], 3), dtype=np.int32)
        indices[:, :2] = self._xy_to_px(xyz[:, [0, 1]])
        indices[:, 2] = ((xyz[:, 2] - self._min_height) / self._voxel_size).astype(np.int32)
        return indices

    def _grid_index_to_xyz(self, indices: np.ndarray) -> np.ndarray:
        """Maps 3D grid index (cy, cx, cz) back to world coordinates (x, y, z)"""
        # Note: indices input from _get_frontiers is [cy, cx, cz] (row, col, z)
        # _px_to_xy expects [cx, cy] (col, row), so we swap the first two columns.
        xy_world = self._px_to_xy(indices[:, [1, 0]])
        z_world = indices[:, 2] * self._voxel_size + self._min_height
        return np.stack([xy_world[:, 0], xy_world[:, 1], z_world], axis=-1)

    def _filter_in_bounds_indices(self, indices: np.ndarray) -> np.ndarray:
        """
        Filter out all indices that lie within the bounds of the 3D grid map.
        Indices have shape (N, 3): column 0 is px/cx, column 1 is py/cy, column 2 is cz.
        """
        if len(indices) == 0:
            return indices
        in_bounds = (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.size) &
            (indices[:, 1] >= 0) & (indices[:, 1] < self.size) &
            (indices[:, 2] >= 0) & (indices[:, 2] < self._height_size)
        )
        return indices[in_bounds]

    def _accumulate(self, indices: np.ndarray, delta: int) -> None:
        """Bayesian log-odds accumulation with clamp (int16 intermediate to avoid int8 overflow)."""
        if len(indices) == 0:
            return
        idx = (indices[:, 1], indices[:, 0], indices[:, 2])
        updated = np.clip(
            self._grid[idx].astype(np.int16) + delta,
            self._l_min,
            self._l_max,
        ).astype(np.int8)
        self._grid[idx] = updated

    def _project_states(self) -> None:
        """Project the int8 log-odds grid into ternary states (0/1/2) via thresholds."""
        states = np.zeros_like(self._grid, dtype=np.uint8)
        states[self._grid > self._occ_thr] = 2
        states[self._grid < self._free_thr] = 1
        self._states = states

    def _mark_free_space(self, camera_pos: np.ndarray, points: np.ndarray) -> None:
        if len(points) == 0:
            return

        # Subsample point clouds to maintain real-time performance
        step_size = max(1, len(points) // 1000)
        sampled_points = points[::step_size]
        # Calculate vectorized ray directions and lengths
        directions = sampled_points - camera_pos
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        valid_mask = distances.flatten() > 0
        directions = directions[valid_mask]
        distances = distances[valid_mask]
        if len(directions) == 0:
            return
        unit_directions = directions / distances
        max_dist = np.max(distances)

        # Calculate maximum interpolation steps clamped between 2 and 50
        num_steps = int(max_dist / self._voxel_size) + 1
        num_steps = max(2, min(num_steps, 50))

        # Interpolate rays up to 0.95 length to avoid clearing the obstacle surface itself
        steps = np.linspace(0.0, 0.95, num_steps)
        ray_lengths = steps[:, None] * distances.T
        ray_pts = camera_pos[None, None, :] + ray_lengths[:, :, None] * unit_directions[None, :, :]
        # Reshape to coordinate list and retrieve corresponding grid indices
        ray_pts_flat = ray_pts.reshape(-1, 3)
        free_indices = self._xyz_to_grid_index(ray_pts_flat)
        free_indices = self._filter_in_bounds_indices(free_indices)
        if len(free_indices) == 0:
            return

        # Prune duplicate spatial voxels to reduce overhead on grid operations
        flat_indices = free_indices[:, 0] * (self.size * self._height_size) + free_indices[:, 1] * self._height_size + free_indices[:, 2]
        _, unique_idx = np.unique(flat_indices, return_index=True)
        free_indices = free_indices[unique_idx]

        # Bayesian free-space evidence: accumulate l_free and clamp (corrects stale occupied votes)
        self._accumulate(free_indices, self._l_free)

    def _get_frontiers(self) -> np.ndarray:
        """Finds voxels that are Free (1) and have at least one Unknown (0) voxel in their 26-neighborhood"""

        # Vertical height-range constraint: slice a local band around the robot's current vertical slice cz
        nav_target_z = self._nav_slice_height  # Standard height (m) for robot navigation interaction
        nav_cz = int((nav_target_z - self._min_height) / self._voxel_size)
        cz = np.clip(nav_cz, 0, self._height_size - 1)
        z_min = max(0, cz - 3)
        z_max = min(self._height_size - 1, cz + 3)
        # Horizontal downsampling (stride=2) to extract a local grid slice (500, 500, 7)
        grid_slice = self._states[::2, ::2, z_min:z_max+1]

        is_free = (grid_slice == 1)
        if not np.any(is_free):
            return np.array([])

        is_unknown = (grid_slice == 0)
        struct = np.ones((3, 3, 3), dtype=bool)
        dilated_unknown = ndimage.binary_dilation(is_unknown, structure=struct)

        frontier_mask = is_free & dilated_unknown
        cy_sub, cx_sub, cz_sub = np.where(frontier_mask)
        if len(cy_sub) == 0:
            return np.array([])
        cy = cy_sub * 2
        cx = cx_sub * 2
        cz = cz_sub + z_min

        frontiers = np.stack([cy, cx, cz], axis=-1)
        step = max(1, len(frontiers) // 500)
        return frontiers[::step]

    def update_map(
        self,
        tf_camera_to_episodic: np.ndarray,
        depth: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float = 0.0,
        pcd: Optional[np.ndarray] = None,
        explore: bool = True,
        update_obstacles: bool = True,
    ) -> None:
        camera_pos = tf_camera_to_episodic[:3, 3]
        agent_index = self._xyz_to_grid_index(camera_pos.reshape(1, 3))[0]
        self._last_agent_cz = np.clip(agent_index[2], 0, self._height_size - 1)

        if update_obstacles:
            if pcd is None or len(pcd) == 0:
                if self._hole_area_thresh == -1:
                    filled_depth = depth.copy()
                    filled_depth[depth == 0] = 1.0
                else:
                    filled_depth = fill_small_holes(depth, self._hole_area_thresh)
                scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
                mask = (scaled_depth > min_depth) & (scaled_depth < max_depth)
                point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
                if len(point_cloud_camera_frame) > 0:
                    pcd = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
                else:
                    pcd = np.empty((0, 3))
            if len(pcd) > 0:
                point_cloud_episodic_frame = pcd[:, :3]
                height_mask = (point_cloud_episodic_frame[:, 2] >= self._min_height) & \
                            (point_cloud_episodic_frame[:, 2] <= self._max_height)
                point_cloud_episodic_frame = point_cloud_episodic_frame[height_mask]

                # Occupied evidence: accumulate l_occ on ray endpoint voxels
                if len(point_cloud_episodic_frame) > 0:
                    voxel_indices = self._xyz_to_grid_index(point_cloud_episodic_frame)
                    voxel_indices = self._filter_in_bounds_indices(voxel_indices)
                    if len(voxel_indices) > 0:
                        self._accumulate(voxel_indices, self._l_occ)
                # Free evidence: accumulate l_free along the camera rays
                self._mark_free_space(camera_pos, point_cloud_episodic_frame)

            # Project log-odds to ternary states and refresh compatibility matrices
            self._project_states()
            self._map = (self._states == 2)
            self._navigable_map = (self._states == 1)  # Navigable maps directly represent Free voxels; collision is handled implicitly

        if not explore:
            return

        # Explored area is any voxel that has reached a non-Unknown state
        self.explored_area = (self._states != 0)
        # Keep only the largest explored connected component containing the agent
        labeled_array, num_features = ndimage.label(self.explored_area)
        if num_features > 1:
            px = np.clip(agent_index[0], 0, self.size - 1)
            py = np.clip(agent_index[1], 0, self.size - 1)
            cz = np.clip(agent_index[2], 0, self._height_size - 1)
            agent_label = labeled_array[py, px, cz]
            if agent_label > 0:
                self.explored_area = (labeled_array == agent_label)

        # Compute 3D frontiers
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._grid_index_to_xyz(self._frontiers_px)

    def check_collision(self, xyz: np.ndarray) -> np.ndarray:
        """
        Performs low-cost local-neighborhood collision checking against the projected ternary states.
        """

        indices = self._xyz_to_grid_index(xyz)
        rx = int(np.ceil(self._agent_radius * self.pixels_per_meter))
        ry = int(np.ceil(self._agent_radius * self.pixels_per_meter))
        rz = int(np.ceil(self._agent_radius / self._voxel_size))

        colliding = np.zeros(len(xyz), dtype=bool)
        for i, idx in enumerate(indices):
            px, py, cz = idx  # Unpack [px, py, cz]

            zmin_raw = cz - rz
            zmax_raw = cz + rz
            # Vertical SAT culling: if the collision region is fully outside the grid's
            # vertical extent, skip it (it cannot possibly collide)
            if zmin_raw >= self._height_size or zmax_raw < 0:
                continue

            ymin = max(0, min(self.size - 1, py - ry))
            ymax = max(0, min(self.size - 1, py + ry))
            xmin = max(0, min(self.size - 1, px - rx))
            xmax = max(0, min(self.size - 1, px + rx))
            zmin = max(0, min(self._height_size - 1, cz - rz))
            zmax = max(0, min(self._height_size - 1, cz + rz))

            patch = self._states[ymin:ymax+1, xmin:xmax+1, zmin:zmax+1]
            if np.any(patch == 2):
                colliding[i] = True

        return colliding

    def visualize_slice(self) -> np.ndarray:
        """Visualizes a 2D slice of the 3D map at the robot's height"""

        # Lock onto the effective navigation height layer for robot-base/obstacle interaction
        nav_target_z = self._nav_slice_height  # Standard height (m) for robot navigation interaction
        nav_cz = int((nav_target_z - self._min_height) / self._voxel_size)
        cz = np.clip(nav_cz, 0, self._height_size - 1)

        explored_2d = self.explored_area[:, :, cz]
        navigable_2d = self._navigable_map[:, :, cz]
        map_2d = self._map[:, :, cz]

        vis_img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        vis_img[explored_2d == 1] = (200, 255, 200)
        vis_img[navigable_2d == 0] = self.radius_padding_color
        vis_img[map_2d == 1] = (0, 0, 0)

        if len(self._frontiers_px) > 0:
            for frontier in self._frontiers_px:
                if abs(frontier[2] - cz) <= 3:
                    cv2.circle(vis_img, (int(frontier[1]), int(frontier[0])), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img
