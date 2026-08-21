# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.utils.img_utils import pixel_value_within_radius
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.policy.tsp3d_objectnav_policy import TSP3DObjectNavPolicy
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class _ExploredAreaAdapter:
    """
    Thin adapter exposing the 3D->2D explored projection (passable band) of the
    reconstructed 3D obstacle map to the 2D `ValueMap` (which expects an object
    with an `explored_area` attribute). The projection itself lives in the
    value-field policy (`BaseITMPolicy._get_explored_2d`).
    """

    def __init__(self, policy: "BaseITMPolicy", z_min: float, z_max: float) -> None:
        self.pixels_per_meter = policy._obstacle_map3d.pixels_per_meter
        self.size = policy._obstacle_map3d.size
        self._policy = policy
        self._z_min = z_min
        self._z_max = z_max

    @property
    def explored_area(self) -> np.ndarray:
        return self._policy._get_explored_2d(self._z_min, self._z_max)


class BaseITMPolicy(TSP3DObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        vm_style: str = "region",
        h_lam: float = 0.3,
        h_norm_max: float = 1.0,
        h_z_min: float = 0.15,
        h_z_max: float = 0.88,
        query_radius_m: float = 0.5,
        query_z_min: float = 0.15,
        query_z_max: float = 1.50,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self._value_channels = len(text_prompt.split(PROMPT_SEPARATOR))
        self._vm_style = vm_style
        self._h_lam = h_lam
        self._h_norm_max = h_norm_max
        self._h_z_min = h_z_min
        self._h_z_max = h_z_max
        self._query_radius_m = query_radius_m
        self._query_z_min = query_z_min
        self._query_z_max = query_z_max
        self._value_map: ValueMap = ValueMap(
            value_channels=self._value_channels,
            size=self._obstacle_map3d.size,  # Same grid as the 3D map (400x400, ppm=20, origin=200)
            use_max_confidence=use_max_confidence,
            obstacle_map=_ExploredAreaAdapter(self, self._h_z_min, self._h_z_max),
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _query_2d_map_radius(self, arr_2d: np.ndarray, x: float, y: float) -> float:
        """
        Queries a 2D map with a disk of radius query_radius_m centered at world (x, y).
        Max-pooling.
        """
        ret = self._obstacle_map3d._xy_to_px(np.array([[x, y]]))[0]
        row, col = int(ret[1]), int(ret[0])
        r_px = int(self._query_radius_m * self._obstacle_map3d.pixels_per_meter)
        H, W = arr_2d.shape[:2]
        if not (0 <= row < H and 0 <= col < W):
            return 0.0
        top, bot = max(0, row - r_px), min(H, row + r_px + 1)
        left, right = max(0, col - r_px), min(W, col + r_px + 1)
        patch = arr_2d[top:bot, left:right]
        if patch.size == 0:
            return 0.0
        return float(patch.max())


    def _z_layer_range(self, z_min: float, z_max: float) -> Tuple[int, int]:
        """Maps a world height range [z_min, z_max] to the 3D layer interval [cz0, cz1)."""
        om = self._obstacle_map3d
        cz0 = int((z_min - om._min_height) / om._voxel_size)
        cz1 = int((z_max - om._min_height) / om._voxel_size) + 1
        return max(0, cz0), min(om._height_size, cz1)

    def _get_explored_2d(self, z_min: float, z_max: float) -> np.ndarray:
        """2D projection of the 3D explored mask over [z_min, z_max] (for the 2D ValueMap).
        Obstacle voxels are excluded so only passable explored cells are considered free.
        """
        om = self._obstacle_map3d
        cz0, cz1 = self._z_layer_range(z_min, z_max)
        if cz1 <= cz0:
            return np.ones((om.size, om.size), bool)
        observed = om.explored_area[:, :, cz0:cz1] & ~om._map[:, :, cz0:cz1]
        return np.any(observed, axis=2)

    def _free_layers_2d(self, z_min: float, z_max: float) -> Union[np.ndarray, None]:
        """H1: fraction of (non-occupied & explored) layers within [z_min, z_max]."""
        om = self._obstacle_map3d
        cz0, cz1 = self._z_layer_range(z_min, z_max)
        if cz1 <= cz0:
            return np.ones((om.size, om.size), np.float32)
        n = cz1 - cz0
        free = ~om._map[:, :, cz0:cz1] & om.explored_area[:, :, cz0:cz1]
        return np.minimum(free.sum(axis=2).astype(np.float32) / n, 1.0)

    def _compute_h1(self) -> Union[np.ndarray, None]:
        """H1 = fraction of (non-occupied & explored) layers within the passable band,
        computed directly on the reconstructed 3D obstacle map."""
        return self._free_layers_2d(self._h_z_min, self._h_z_max)


    def _update_value_map_impl(self, cosines: List[List[float]]) -> None:
        raise NotImplementedError

    def _update_value_map(self) -> None:
        if self._target_object == "":
            return
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]

        self._update_value_map_impl(cosines)

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
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []
        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))

        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info


    def _score_frontiers(self, frontiers: np.ndarray) -> List[float]:
        raise NotImplementedError

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        if len(frontiers) == 0:
            return np.empty((0, 2)), []
        values = self._score_frontiers(frontiers)
        sorted_inds = np.argsort([-v for v in values])
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([frontiers[i] for i in sorted_inds])
        return sorted_frontiers, sorted_values

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on the 2.5D value map."""
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action


class ITMPolicyV1(BaseITMPolicy):
    """
    Route 1 (region style): fills the 2D ValueMap with the VLFM visible cone (S)
    + vertical passability bonus (H1).
    """

    def __init__(
        self,
        vm_style: str = "region",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(vm_style=vm_style, *args, **kwargs)

    def _update_value_map_impl(self, cosines: List[List[float]]) -> None:
        """Fills the 2D ValueMap with the VLFM visible cone and draws the trajectory."""
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)
        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _score_frontiers(self, frontiers: np.ndarray) -> List[float]:
        """S (region ValueMap radius query) + lambda * H1 (vertical passability)."""
        h1 = self._compute_h1()
        values = []
        for (x, y) in frontiers:
            s = self._query_2d_value(x, y)
            h = self._query_2d_map_radius(h1, x, y) if h1 is not None else 0.0
            values.append(s + self._h_lam * h)
        return values

    def _query_2d_value(self, x: float, y: float) -> float:
        """
        Radius query on the 2D ValueMap (reuses the VLFM sort_waypoints pixel convention).
        """
        vm = self._value_map
        ppm = vm.pixels_per_meter
        px = int(-x * ppm) + vm._episode_pixel_origin[0]
        py = int(-y * ppm) + vm._episode_pixel_origin[1]
        point_px = (vm._value_map.shape[0] - px, py)
        r_px = int(self._query_radius_m * ppm)
        H, W = vm._value_map.shape[:2]
        if not (0 <= point_px[0] < H and 0 <= point_px[1] < W):
            return 0.0
        best = 0.0
        for c in range(vm._value_channels):
            v = pixel_value_within_radius(vm._value_map[..., c], point_px, r_px)
            best = max(best, float(v))
        return best if best > 0.0 else 0.0


class ITMPolicyV2(BaseITMPolicy):
    """Route 2 (surface style): lands scores from 3D surface points into 2D (x, y)
    buckets (S: confidence-gated max, consistent with VLFM) + height-axis
    semantic value (H1)."""

    _min_valid_conf: float = 1e-4   # Per-point confidence floor (same as itm3d min_valid_conf)

    def __init__(
        self,
        vm_style: str = "surface",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(vm_style=vm_style, *args, **kwargs)
        # V2 surface-style state: 2D (x, y) buckets for surface points in the fixed height band
        size = self._obstacle_map3d.size
        self._surface_max_map = np.zeros((size, size), np.float32)   # Per-bucket semantic value score
        self._surface_conf_map = np.zeros((size, size), np.float32)  # Per-bucket confidence

    def _reset(self) -> None:
        super()._reset()
        self._surface_max_map.fill(0.0)
        self._surface_conf_map.fill(0.0)

    def _update_value_map_impl(self, cosines: List[List[float]]) -> None:
        """
        Lands cosine scores into 2D (x, y) buckets (S: confidence-gated max).
        Reuses the single shared back-projected point cloud (see `_get_shared_pcd`).
        """
        value_rgbd = self._observations_cache["value_map_rgbd"]
        for i, (cosine, (rgb, depth, tf, min_depth, max_depth, fov)) in enumerate(
            zip(cosines, value_rgbd)
        ):
            pcd = self._get_shared_pcd(i)
            self._project_value_to_surface(np.array(cosine), pcd, tf, max_depth, fov)

    def _score_frontiers(self, frontiers: np.ndarray) -> List[float]:
        """
        S (surface-bucket max radius query) + lambda * H1 (height-axis semantic value, 0/1 simplified form).
        """
        h1 = self._compute_h1()
        values = []
        for (x, y) in frontiers:
            s = self._query_2d_map_radius(self._surface_max_map, x, y)  # S: bucket max radius query
            h = self._query_2d_map_radius(h1, x, y) if h1 is not None else 0.0  # H1: free-layer ratio in the passable band
            values.append(s + self._h_lam * h)
        return values

    def _project_value_to_surface(
        self,
        scores: np.ndarray,
        pcd: Union[np.ndarray, None],
        tf_camera_to_episodic: np.ndarray,
        max_depth: float,
        fov: float,
    ) -> None:
        """
        Lands cosine scores into 2D (x, y) buckets under confidence gating (surface
        points within the fixed height band).

        - S (`_surface_max_map`): per-bucket semantic value score; a bucket is
          overwritten only when the current observation's confidence is higher
          (VLFM `use_max_confidence=True` fusion).
        - Confidence (`_surface_conf_map`): per-point angular distance confidence
          C_3D = c_angular * c_distance (VLFM cos^2 angular term + itm3d distance term).
        """
        if pcd is None or len(pcd) == 0:
            return
        om = self._obstacle_map3d
        step = 8 ** 2  # Downsampling stride (same as itm3d downsampling_step)
        pts_w = pcd[::step, :3]
        if len(pts_w) == 0:
            return
        z_mask = (pts_w[:, 2] >= self._query_z_min) & (pts_w[:, 2] <= self._query_z_max)
        pts_w = pts_w[z_mask]
        if len(pts_w) == 0:
            return
        score = float(np.clip(np.mean(scores), 0.0, 1.0))  # Single-frame semantic value (mean over channels)

        # Per-point confidence (VLFM angular confidence; distance term follows itm3d)
        try:
            tf_episodic_to_camera = np.linalg.inv(tf_camera_to_episodic)
        except np.linalg.LinAlgError:
            return
        R_T = tf_episodic_to_camera[:3, :3].T
        t = tf_episodic_to_camera[:3, 3]
        pts_c = pts_w @ R_T + t
        # Camera base frame is [forward, left, up] (see _project_rgbd_to_3d_point_cloud)
        forward = pts_c[:, 0]
        left = pts_c[:, 1]
        up = pts_c[:, 2]
        dist_3d = np.sqrt(forward ** 2 + left ** 2 + up ** 2)
        cos_theta = np.clip(forward / (dist_3d + 1e-6), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        c_angular = np.zeros_like(theta)
        in_fov = theta <= (fov / 2.0)
        c_angular[in_fov] = np.cos(theta[in_fov] / (fov / 2.0) * (np.pi / 2.0)) ** 2
        conf = c_angular
        valid = conf > self._min_valid_conf
        if not np.any(valid):
            return
        pts_w = pts_w[valid]
        conf = conf[valid]

        # Land surface points into (x, y) pixel buckets (vectorized, no Python loop)
        px_py = om._xy_to_px(pts_w[:, :2]).astype(int)
        in_b = (
            (px_py[:, 0] >= 0) & (px_py[:, 0] < self._surface_max_map.shape[1]) &
            (px_py[:, 1] >= 0) & (px_py[:, 1] < self._surface_max_map.shape[0])
        )
        if not in_b.any():
            return
        rows, cols = px_py[in_b, 1], px_py[in_b, 0]
        pt_conf = conf[in_b]

        # Per-frame max confidence per bucket (this bucket's observation quality)
        frame_conf = np.zeros_like(self._surface_conf_map)
        np.maximum.at(frame_conf, (rows, cols), pt_conf)
        b_rows, b_cols = np.nonzero(frame_conf)
        if len(b_rows) == 0:
            return

        # Confidence-gated fusion: overwrite S and confidence only when this frame's
        # confidence is higher (VLFM `use_max_confidence=True` path).
        improve = frame_conf[b_rows, b_cols] > self._surface_conf_map[b_rows, b_cols]
        if improve.any():
            idx_r, idx_c = b_rows[improve], b_cols[improve]
            self._surface_conf_map[idx_r, idx_c] = frame_conf[idx_r, idx_c]
            self._surface_max_map[idx_r, idx_c] = score
