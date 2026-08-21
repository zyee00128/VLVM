# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Tuple, Union, Optional

import cv2
import numpy as np
from scipy import ndimage

from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes

class _BaseObstacleMap3D(BaseMap):
    """
    Shared 3D machinery for the two geometry fields:
      - ObstacleMap3D     (binary 0/1 occupancy grid)
      - ProbabilisticGrid (int8 log-odds occupancy grid)

    Common services: 3D cylindrical dilation, 3D explored mask (VLFM "known"
    semantics), and ray-body/endpoint observed-voxel updates via per-class hooks.
    """

    _map_dtype: np.dtype = np.dtype(bool)
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,
        hole_area_thresh: int = 100000,
        size: int = 1000,
        pixels_per_meter: int = 20,
        voxel_size: float = 0.05,
        height_size: int = 40,
        nav_slice_height: float = 0.35,
        agent_height: float = 0.88,
        compute_navigable: bool = False,
    ):
        super().__init__(size, pixels_per_meter)
        self._min_height = min_height
        self._max_height = max_height
        self._voxel_size = voxel_size
        self._height_size = height_size
        self._agent_radius = agent_radius
        self._agent_height = agent_height
        self._hole_area_thresh = hole_area_thresh
        self._nav_slice_height = nav_slice_height

        # 3D explored mask + VLFM-aligned compatibility matrix `_map`
        self.explored_area = np.zeros((size, size, height_size), dtype=bool)
        self._map = np.zeros((size, size, height_size), dtype=bool)
        # Cylinder-dilated `_navigable_map` is lazily allocated only when
        # compute_navigable=True (visualization); skipped on the 2.5D inference path.
        self._compute_navigable = compute_navigable


    def _xyz_to_grid_index(self, xyz: np.ndarray) -> np.ndarray:
        """Maps world coordinates (x, y, z) to 3D grid index (cx, cy, cz)"""
        indices = np.empty((xyz.shape[0], 3), dtype=np.int32)
        indices[:, :2] = self._xy_to_px(xyz[:, [0, 1]])
        indices[:, 2] = ((xyz[:, 2] - self._min_height) / self._voxel_size).astype(np.int32)
        return indices

    def _grid_index_to_xyz(self, indices: np.ndarray) -> np.ndarray:
        """Maps 3D grid index (cy, cx, cz) back to world coordinates (x, y, z)"""
        xy_world = self._px_to_xy(indices[:, [1, 0]])
        z_world = indices[:, 2] * self._voxel_size + self._min_height
        return np.stack([xy_world[:, 0], xy_world[:, 1], z_world], axis=-1)

    def _filter_in_bounds_indices(self, indices: np.ndarray) -> np.ndarray:
        """Filter out indices outside the 3D grid (cols: px/cx, py/cy, cz)."""
        if len(indices) == 0:
            return indices
        in_bounds = (
            (indices[:, 0] >= 0) & (indices[:, 0] < self.size) &
            (indices[:, 1] >= 0) & (indices[:, 1] < self.size) &
            (indices[:, 2] >= 0) & (indices[:, 2] < self._height_size)
        )
        return indices[in_bounds]


    def _reset_grid(self) -> None:
        """Reset the per-class occupancy grid (and any extra state)."""
        raise NotImplementedError

    def _mark_occupied_voxels(self, idx: np.ndarray) -> None:
        """Update occupancy evidence on ray-endpoint voxels (idx: px, py, cz)."""
        raise NotImplementedError

    def _mark_ray_voxels(self, idx: np.ndarray) -> None:
        """Update explored/free evidence on ray-body voxels (idx: px, py, cz)."""
        raise NotImplementedError


    def reset(self) -> None:
        super().reset()
        self.explored_area.fill(0)
        self._map.fill(0)
        if getattr(self, "_navigable_map", None) is not None:
            self._navigable_map.fill(0)
        self._reset_grid()

    def _mark_explored_along_rays(self, camera_pos: np.ndarray, points: np.ndarray) -> None:
        """Mark explored voxels along camera->point rays (interpolate to 0.95 x distance)."""
        if len(points) == 0:
            return
        step_size = max(1, len(points) // 1000)
        sampled = points[::step_size]
        directions = sampled - camera_pos
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        valid_mask = distances.flatten() > 0
        directions = directions[valid_mask]
        distances = distances[valid_mask]
        if len(directions) == 0:
            return
        unit = directions / distances
        max_dist = np.max(distances)
        num_steps = int(max_dist / self._voxel_size) + 1
        num_steps = max(2, min(num_steps, 50))
        steps = np.linspace(0.0, 0.95, num_steps)
        ray_lengths = steps[:, None] * distances.T
        ray_pts = camera_pos[None, None, :] + ray_lengths[:, :, None] * unit[None, :, :]
        ray_pts_flat = ray_pts.reshape(-1, 3)
        idx = self._xyz_to_grid_index(ray_pts_flat)
        idx = self._filter_in_bounds_indices(idx)
        if len(idx) == 0:
            return
        flat = idx[:, 0] * (self.size * self._height_size) + idx[:, 1] * self._height_size + idx[:, 2]
        _, u = np.unique(flat, return_index=True)
        self._mark_ray_voxels(idx[u])

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
                pts = pcd[:, :3]
                height_mask = (pts[:, 2] >= self._min_height) & (pts[:, 2] <= self._max_height)
                pts = pts[height_mask]
                if len(pts) > 0:
                    idx = self._xyz_to_grid_index(pts)
                    idx = self._filter_in_bounds_indices(idx)
                    if len(idx) > 0:
                        self._mark_occupied_voxels(idx)
                # Ray bodies are observed (explored)
                self._mark_explored_along_rays(camera_pos, pts)

            # Refresh the VLFM-aligned compatibility matrices from the grid
            self._refresh_compat()

        if not explore:
            return

        # Keep only the largest explored connected component containing the agent
        labeled_array, num_features = ndimage.label(self.explored_area)
        if num_features > 1:
            px = np.clip(agent_index[0], 0, self.size - 1)
            py = np.clip(agent_index[1], 0, self.size - 1)
            cz = np.clip(agent_index[2], 0, self._height_size - 1)
            agent_label = labeled_array[py, px, cz]
            if agent_label > 0:
                self.explored_area = (labeled_array == agent_label)


    # ------------------------------------------------------------------ #
    # Dilation / structuring element                                      #
    # ------------------------------------------------------------------ #
    def _ensure_dilation_tools(self) -> None:
        """Lazily build the 3D cylinder kernel, cached disk/rz, and `_navigable_map`."""
        if getattr(self, "_cylinder_kernel", None) is not None:
            return
        rx = max(1, int(np.ceil(self._agent_radius * self.pixels_per_meter)))
        rz = max(1, int(np.ceil(self._agent_height / self._voxel_size)))
        kernel = np.zeros((2 * rx + 1, 2 * rx + 1, 2 * rz + 1), dtype=bool)
        yy, xx = np.mgrid[-rx:rx + 1, -rx:rx + 1]
        kernel[(xx ** 2 + yy ** 2) <= rx ** 2] = True
        disk = np.zeros((2 * rx + 1, 2 * rx + 1), dtype=np.uint8)
        disk[(xx ** 2 + yy ** 2) <= rx ** 2] = 1
        self._cylinder_kernel = kernel
        self._dilate_disk_2d = disk
        self._cylinder_rz = rz
        self._navigable_map = np.zeros((self.size, self.size, self._height_size), dtype=bool)
    def _refresh_compat(self) -> None:
        """Refresh `_map` (and `_navigable_map` when compute_navigable)."""
        raise NotImplementedError

    # Visualization / parity / full-3D utilities (not on the 2.5D inference path)
    def dilate_3d_cylinder(self, occupied: np.ndarray, rz: int, disk: np.ndarray) -> np.ndarray:
        """Fast 3D cylinder dilation (agent_radius x agent_height).

        cylinder_dilate(O)[x,y,z] = OR_{|dz|<=rz} dilate_2d(O[:,:,z+dz], disk)[x,y]
        => z-window OR via a 1D maximum_filter, then per-layer 2D disk dilation via
        OpenCV. Mathematically identical to ndimage.binary_dilation with the cylinder
        kernel but ~100x faster on structured scenes.
        """
        occ = occupied.astype(np.uint8)
        occ_z = ndimage.maximum_filter(occ, size=(1, 1, 2 * rz + 1))
        out = np.empty(occ.shape, dtype=np.uint8)
        for z in range(occ.shape[2]):
            out[:, :, z] = cv2.dilate(occ_z[:, :, z], disk)
        return out.astype(bool)
    def visualize_slice(self) -> np.ndarray:
        """Visualizes a 2D slice of the 3D map at the robot's height."""
        if not self._compute_navigable:
            self._ensure_dilation_tools()  # ensure navigable slice exists when called
        nav_cz = int((self._nav_slice_height - self._min_height) / self._voxel_size)
        cz = np.clip(nav_cz, 0, self._height_size - 1)

        explored_2d = self.explored_area[:, :, cz]
        navigable_2d = self._navigable_map[:, :, cz]
        map_2d = self._map[:, :, cz]

        vis_img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        vis_img[explored_2d == 1] = (200, 255, 200)
        vis_img[navigable_2d == 0] = self.radius_padding_color
        vis_img[map_2d == 1] = (0, 0, 0)

        vis_img = cv2.flip(vis_img, 0)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )
        return vis_img
    def check_collision(self, xyz: np.ndarray) -> np.ndarray:
        """3D cylindrical patch collision query (agent_radius x agent_height)."""
        if len(xyz) == 0:
            return np.zeros(0, bool)
        indices = self._xyz_to_grid_index(xyz)
        rx = max(1, int(np.ceil(self._agent_radius * self.pixels_per_meter)))
        ry = rx
        rz = max(1, int(np.ceil(self._agent_height / self._voxel_size)))
        colliding = np.zeros(len(xyz), bool)
        for i, idx in enumerate(indices):
            px, py, cz = idx
            zmin_raw = cz - rz
            zmax_raw = cz + rz
            if zmin_raw >= self._height_size or zmax_raw < 0:
                continue
            ymin = max(0, min(self.size - 1, py - ry))
            ymax = max(0, min(self.size - 1, py + ry))
            xmin = max(0, min(self.size - 1, px - rx))
            xmax = max(0, min(self.size - 1, px + rx))
            zmin = max(0, min(self._height_size - 1, cz - rz))
            zmax = max(0, min(self._height_size - 1, cz + rz))
            patch = self._map[ymin:ymax + 1, xmin:xmax + 1, zmin:zmax + 1]
            if np.any(patch):
                colliding[i] = True
        return colliding


class ObstacleMap3D(_BaseObstacleMap3D):
    """
    3D Occupancy Grid Map: binary occupancy (0/1) + 3D explored mask, with
    VLFM-aligned 2D behavior lifted to 3D.

    - Occupied: hard-overwrite, monotonic (aligned with VLFM conflict handling; never flips)
    - explored_area (3D): voxels observed by ray bodies + ray endpoints
      (VLFM explored_area "known" semantics; Free = ~occupied & explored)
    - 3D cylindrical dilation (agent_radius x agent_height) for navigable / collision
    """

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,
        hole_area_thresh: int = 100000,
        size: int = 1000,
        pixels_per_meter: int = 20,
        voxel_size: float = 0.05,
        height_size: int = 40,
        nav_slice_height: float = 0.35,
        agent_height: float = 0.88,
        compute_navigable: bool = False,
    ):
        super().__init__(
            min_height, max_height, agent_radius, area_thresh, hole_area_thresh,
            size, pixels_per_meter, voxel_size, height_size, nav_slice_height,
            agent_height, compute_navigable,
        )
        # Binary occupancy grid: 0 = non-occupied (incl. unknown), 1 = occupied
        self._grid = np.zeros((size, size, height_size), dtype=np.uint8)

    def _reset_grid(self) -> None:
        self._grid.fill(0)

    def _mark_occupied_voxels(self, idx: np.ndarray) -> None:
        """Hard-overwrite occupied (monotonic; aligned with VLFM); endpoints observed."""
        self._grid[idx[:, 1], idx[:, 0], idx[:, 2]] = 1
        self.explored_area[idx[:, 1], idx[:, 0], idx[:, 2]] = True

    def _mark_ray_voxels(self, idx: np.ndarray) -> None:
        """Ray bodies are observed (explored)."""
        self.explored_area[idx[:, 1], idx[:, 0], idx[:, 2]] = True

    def _refresh_compat(self) -> None:
        self._map = (self._grid == 1)
        if self._compute_navigable:
            self._ensure_dilation_tools()
            self._navigable_map = ~self.dilate_3d_cylinder(self._map, self._cylinder_rz, self._dilate_disk_2d)


class ProbabilisticGrid(_BaseObstacleMap3D):
    """
    3D Probabilistic Occupancy Grid Map backed by a dense int8 log-odds array.

    Keeps its core Bayesian mechanism (accumulate + clamp + threshold projection),
    while sharing the VLFM-aligned 3D machinery of ObstacleMap3D:
    - 3D cylindrical dilation (agent_radius x agent_height)
    - 3D explored mask = any voxel with a non-zero log-odds (observed)
    Occupied/Free are projected from log-odds via thresholds; single erroneous
    observations are corrected by later consistent ones (no permanent pinning).
    """

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,
        hole_area_thresh: int = 100000,
        size: int = 1000,
        pixels_per_meter: int = 20,
        voxel_size: float = 0.05,
        height_size: int = 40,
        nav_slice_height: float = 0.35,
        agent_height: float = 0.88,
        compute_navigable: bool = False,
        # ---- log-odds inverse sensor model ----
        log_odds_occ: float = 2.0,
        log_odds_free: float = -2.0,
        log_odds_min: float = -6.0,
        log_odds_max: float = 6.0,
        occ_threshold: float = 0.0,
        free_threshold: float = 0.0,
    ):
        super().__init__(
            min_height, max_height, agent_radius, area_thresh, hole_area_thresh,
            size, pixels_per_meter, voxel_size, height_size, nav_slice_height,
            agent_height, compute_navigable,
        )
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

    def _reset_grid(self) -> None:
        self._grid.fill(0)
        self._states.fill(0)

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

    def _mark_occupied_voxels(self, idx: np.ndarray) -> None:
        """Occupied evidence: accumulate l_occ on ray endpoint voxels."""
        self._accumulate(idx, self._l_occ)
        self.explored_area[idx[:, 1], idx[:, 0], idx[:, 2]] = True

    def _mark_ray_voxels(self, idx: np.ndarray) -> None:
        """Free evidence: accumulate l_free on ray-body voxels (marks explored)."""
        self._accumulate(idx, self._l_free)
        self.explored_area[idx[:, 1], idx[:, 0], idx[:, 2]] = True

    def _refresh_compat(self) -> None:
        self._project_states()
        self._map = (self._states == 2)
        if self._compute_navigable:
            self._ensure_dilation_tools()
            self._navigable_map = ~self.dilate_3d_cylinder(self._map, self._cylinder_rz, self._dilate_disk_2d)
