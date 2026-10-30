"""Shadow-only online field exploration coverage.

The class in this module is deliberately side-channel:

* it owns its own incremental region grid;
* it never imports or mutates policy memory, nav goals, STOP state or actions;
* every exception is the caller's responsibility to catch/log.

The policy calls :meth:`observe` after it has already selected an action.  The
returned row is written to ``detections/field_shadow.jsonl`` and is intended
for offline analysis only.  ``region_semantic`` is intentionally not computed
here; it remains a separate experiment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .inference import FieldInference, FieldInferenceConfig, field_config_from_mapping


@dataclass(frozen=True)
class FieldShadowConfig:
    enabled: bool = False
    config_path: Optional[str] = None
    cell_size_m: float = 1.0
    top_k: int = 5
    max_regions: int = 5000


class _OnlineRegionBuilder:
    """Small online mirror of the geometry used by ``offline_data``."""

    def __init__(self, cell_size_m: float) -> None:
        self.cell_size_m = float(cell_size_m)
        self.keys: set = set()
        self.visits: Dict[Tuple[int, int], int] = {}
        self.candidates: Dict[Tuple[int, int], int] = {}

    def _key(self, xy: np.ndarray) -> Tuple[int, int]:
        return (
            int(math.floor(float(xy[0]) / self.cell_size_m)),
            int(math.floor(float(xy[1]) / self.cell_size_m)),
        )

    def add_robot(self, xy: np.ndarray) -> None:
        key = self._key(np.asarray(xy, dtype=np.float64).reshape(-1)[:2])
        self.keys.add(key)
        self.visits[key] = int(self.visits.get(key, 0)) + 1

    def add_points(self, points: Any, *, count_candidates: bool = False) -> None:
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for row in arr:
            if row.size < 2 or not np.all(np.isfinite(row[:2])):
                continue
            key = self._key(row[:2])
            self.keys.add(key)
            if count_candidates:
                self.candidates[key] = int(self.candidates.get(key, 0)) + 1

    def snapshot(self) -> Tuple[Tuple[Tuple[int, int], ...], np.ndarray, Tuple[str, ...]]:
        keys = tuple(sorted(self.keys))
        centers = np.asarray(
            [
                ((ix + 0.5) * self.cell_size_m, (iy + 0.5) * self.cell_size_m)
                for ix, iy in keys
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        ids = tuple(f"g:{ix}:{iy}" for ix, iy in keys)
        return keys, centers, ids


class FieldExplorationShadow:
    """Compute and return one log-only exploration row per policy step."""

    def __init__(self, config: Optional[FieldShadowConfig] = None) -> None:
        self.config = config or FieldShadowConfig()
        self._builder = _OnlineRegionBuilder(self.config.cell_size_m)
        self._inference, self._raw_config = self._load_inference(self.config.config_path)
        self._feature_names = tuple(self._raw_config.get("feature_names", ()) or ())
        self._weight_count = len(self._inference.config.weights)
        self._steps = 0

    @staticmethod
    def _load_inference(
        path: Optional[str],
    ) -> Tuple[FieldInference, Dict[str, Any]]:
        if not path:
            return FieldInference(FieldInferenceConfig()), {}
        try:
            with open(str(path), encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError("field inference config must be a JSON object")
            return FieldInference(field_config_from_mapping(raw)), raw
        except Exception:
            # A missing/invalid config must not disable the shadow or affect
            # navigation.  Uniform posterior remains a valid diagnostic.
            return FieldInference(FieldInferenceConfig()), {}

    def reset(self) -> None:
        self._builder = _OnlineRegionBuilder(self.config.cell_size_m)
        self._steps = 0

    def observe(
        self,
        *,
        step: int,
        robot_xy: np.ndarray,
        frontiers_3d: Any,
        frontiers_2d: Any,
        candidate_points: Any,
        nav_goal_xy: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return one shadow row; never writes policy state.

        The caller must not use the return value for action selection.
        """
        if not self.config.enabled:
            return None
        robot = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
        if robot.size < 2 or not np.all(np.isfinite(robot)):
            return None
        self._builder.add_robot(robot)
        self._builder.add_points(frontiers_3d)
        self._builder.add_points(frontiers_2d)
        self._builder.add_points(candidate_points, count_candidates=True)
        keys, centers, region_ids = self._builder.snapshot()
        if centers.shape[0] == 0:
            return None
        if centers.shape[0] > int(self.config.max_regions):
            # Keep the shadow bounded.  The truncation is explicit in the log;
            # navigation is unaffected because this branch only builds a row.
            nearest = np.argsort(
                np.linalg.norm(centers - robot[None, :], axis=1)
            )[: int(self.config.max_regions)]
            keys = tuple(keys[int(i)] for i in nearest)
            centers = centers[nearest]
            region_ids = tuple(region_ids[int(i)] for i in nearest)

        frontier_arrays: List[np.ndarray] = []
        for value in (frontiers_3d, frontiers_2d):
            arr = np.asarray(value, dtype=np.float64)
            if arr.size:
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                frontier_arrays.append(arr[:, :2])
        frontiers_xy = (
            np.concatenate(frontier_arrays, axis=0)
            if frontier_arrays
            else np.empty((0, 2), dtype=np.float64)
        )

        features = np.zeros(
            (centers.shape[0], max(1, self._weight_count)), dtype=np.float64
        )
        for i, center in enumerate(centers):
            key = keys[i]
            dist = float(np.linalg.norm(center - robot))
            if frontiers_xy.size:
                frontier_dist = float(np.min(np.linalg.norm(frontiers_xy - center, axis=1)))
                frontier_prox = float(math.exp(-frontier_dist / 2.0))
            else:
                frontier_prox = 0.0
            visits = int(self._builder.visits.get(key, 0))
            candidates = int(self._builder.candidates.get(key, 0))
            geom = (
                float(math.log1p(dist)),
                float(frontier_prox),
                float(math.log1p(visits)),
                float(math.log1p(candidates)),
                1.0 if visits > 0 else 0.0,
            )
            width = max(1, self._weight_count)
            features[i, : min(width, len(geom))] = geom[:width]

        posterior = np.asarray(self._inference.posterior(features), dtype=np.float64)
        known = posterior[:-1]
        if known.size == 0:
            return None
        k = min(int(self.config.top_k), int(known.size))
        order = np.argsort(-known)[:k]
        top = set(int(i) for i in order.tolist())
        selected_centers = centers[order]
        rel = selected_centers - robot[None, :]
        distances = np.linalg.norm(rel, axis=1)
        if rel.shape[0] >= 2:
            angles = np.arctan2(rel[:, 1], rel[:, 0])
            resultant = float(abs(np.mean(np.exp(1j * angles))))
            angular_dispersion = float(1.0 - resultant)
        else:
            angular_dispersion = 0.0

        candidate_regions = set()
        cand = np.asarray(candidate_points, dtype=np.float64)
        if cand.size:
            if cand.ndim == 1:
                cand = cand.reshape(1, -1)
            for point in cand:
                if point.size >= 2 and np.all(np.isfinite(point[:2])):
                    idx = _nearest_region(centers, point[:2], self.config.cell_size_m)
                    if idx is not None:
                        candidate_regions.add(idx)
        frontier_regions = set()
        for point in frontiers_xy:
            idx = _nearest_region(centers, point[:2], self.config.cell_size_m)
            if idx is not None:
                frontier_regions.add(idx)

        total = float(known.sum())
        if total > 0.0:
            probs = np.clip(known / total, 1e-12, 1.0)
            entropy = float(-(probs * np.log(probs)).sum() / max(np.log(float(known.size)), 1e-12))
            entropy = max(0.0, min(1.0, entropy))
        else:
            entropy = 0.0
        sorted_known = np.sort(known)[::-1]
        margin = float(sorted_known[0] - sorted_known[1]) if sorted_known.size >= 2 else 0.0
        top1_idx = int(order[0]) if k else None
        top_regions = []
        for idx in order.tolist():
            idx = int(idx)
            top_regions.append(
                {
                    "region_id": region_ids[idx],
                    "center_xy": centers[idx].tolist(),
                    "posterior": float(known[idx]),
                    "visits": int(self._builder.visits.get(keys[idx], 0)),
                    "candidates": int(self._builder.candidates.get(keys[idx], 0)),
                }
            )
        nav_goal = None if nav_goal_xy is None else np.asarray(nav_goal_xy, dtype=np.float64).reshape(-1)[:2]
        nav_goal_distance = None
        if nav_goal is not None and nav_goal.size >= 2 and top1_idx is not None:
            nav_goal_distance = float(np.linalg.norm(centers[top1_idx] - nav_goal))

        self._steps += 1
        return {
            "event": "field_exploration_shadow",
            "shadow_only": True,
            "step": int(step),
            "mode": mode,
            "robot_xy": robot.tolist(),
            "nav_goal_xy": None if nav_goal is None else nav_goal.tolist(),
            "n_regions": int(centers.shape[0]),
            "n_frontiers": int(frontiers_xy.shape[0]),
            "n_candidates": int(cand.shape[0]) if cand.size else 0,
            "top_k": int(k),
            "weights_loaded": bool(self._weight_count),
            "config_path": self.config.config_path,
            "feature_names": list(self._feature_names),
            "top1_region_id": None if top1_idx is None else region_ids[top1_idx],
            "top1_xy": None if top1_idx is None else centers[top1_idx].tolist(),
            "top1_to_nav_goal_m": nav_goal_distance,
            "top_regions": top_regions,
            "candidate_novelty_rate": float(len(top - candidate_regions) / max(1, len(top))),
            "frontier_overlap_rate": float(len(top & frontier_regions) / max(1, len(top))),
            "angular_dispersion": float(angular_dispersion),
            "mean_topk_distance_m": float(distances.mean()) if distances.size else 0.0,
            "posterior_entropy": float(entropy),
            "top1_top2_margin": float(margin),
        }


def _nearest_region(
    centers: np.ndarray, xy: np.ndarray, cell_size_m: float
) -> Optional[int]:
    if centers.size == 0:
        return None
    d = np.linalg.norm(centers - np.asarray(xy, dtype=np.float64).reshape(-1)[:2], axis=1)
    idx = int(np.argmin(d))
    if float(d[idx]) > 1.5 * float(cell_size_m):
        return None
    return idx
