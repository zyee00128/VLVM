"""Offline readers and feature builders for the semantic-spatial field.

The field logger stores one ZIP per episode:

  <episode>.zip
    episode.json
    frames/000000.npz ...
    ground_truth/top_down_map.npz

Each frame contains filtered RGB-D (not needed for this first analyzer),
camera/robot pose, frontier points, and map projection parameters.  The
analyzer below uses the pose/frontier/GT subset and separately joins the
ordinary navigation JSONL logs for candidate/track labels.  It never imports
or calls the policy.

Optional semantic sidecar
-------------------------
A caller can provide ``region_semantics.jsonl`` with one JSON object per line::

    {
      "scene": "00800-TEEsavR23oF",
      "episode_id": 12,
      "step": 37,
      "region_id": "g:4:-3",
      "context_scores": [0.7, 0.2, ...],
      "context_reliability": [0.9, 0.8, ...],
      "relation_scores": [0.4, ...]
    }

The sidecar is deliberately model-agnostic: the VLM/context scorer can be run
separately, and the analyzer only consumes its already-computed scores.  If no
sidecar is supplied the analyzer still builds a geometry-only diagnostic model
and marks the semantic channel as unavailable.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import zipfile
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vlfm.utils.geometry_utils import transform_points
from vlfm.utils.habitat_visualizer import sim_xy_to_grid_xy

try:  # Habitat is available in the project environment.
    from frontier_exploration.utils.general_utils import xyz_to_habitat
except Exception:  # pragma: no cover - only for non-Habitat offline installs
    xyz_to_habitat = None


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
GEOMETRY_FEATURE_NAMES: Tuple[str, ...] = (
    "geom_log_distance",
    "geom_frontier_proximity",
    "geom_log_visits",
    "geom_log_candidates",
    "geom_observed",
)

DISTANCE_EDGES: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, float("inf"))


def scene_key(scene: Any) -> str:
    return os.path.basename(str(scene)).split(".")[0]


def step_key(scene: Any, episode_id: Any, step: Any) -> Tuple[str, int, int]:
    return scene_key(scene), int(episode_id), int(step)


def episode_key(scene: Any, episode_id: Any) -> Tuple[str, int]:
    return scene_key(scene), int(episode_id)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _as_xy(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return arr[:2]


# ---------------------------------------------------------------------------
# field archive reader
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldFrame:
    step: int
    view_index: int
    robot_xy: np.ndarray
    robot_xyz: np.ndarray
    robot_yaw_rad: float
    frontiers_3d: np.ndarray
    frontiers_2d: np.ndarray


@dataclass
class FieldEpisode:
    path: str
    metadata: Dict[str, Any]
    frames: List[FieldFrame]
    ground_truth: Dict[str, np.ndarray]

    @property
    def scene(self) -> str:
        return scene_key(self.metadata.get("scene_id", ""))

    @property
    def episode_id(self) -> int:
        return int(self.metadata.get("episode_id", -1))


def _load_frame(npz_path: str, archive: zipfile.ZipFile) -> Optional[FieldFrame]:
    try:
        with archive.open(npz_path) as handle:
            with np.load(handle, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
    except Exception:
        return None
    step = int(arrays.get("step", -1))
    view_index = int(arrays.get("view_index", 0))
    robot_xyz = np.asarray(arrays.get("robot_xyz", np.zeros(3)), dtype=np.float64).reshape(-1)
    robot_xy = arrays.get("robot_xy")
    if robot_xy is None:
        robot_xy = robot_xyz[:2]
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(-1)
    if robot_xy.size < 2 or not np.all(np.isfinite(robot_xy[:2])):
        return None
    frontiers_3d = np.asarray(
        arrays.get("frontiers_3d", np.empty((0, 3))), dtype=np.float64
    )
    if frontiers_3d.size == 0:
        frontiers_3d = np.empty((0, 3), dtype=np.float64)
    else:
        frontiers_3d = frontiers_3d.reshape(-1, 3)
    frontiers_2d = np.asarray(
        arrays.get("frontiers_2d", np.empty((0, 0))), dtype=np.float64
    )
    return FieldFrame(
        step=step,
        view_index=view_index,
        robot_xy=robot_xy[:2],
        robot_xyz=robot_xyz[:3],
        robot_yaw_rad=float(arrays.get("robot_yaw_rad", 0.0)),
        frontiers_3d=frontiers_3d,
        frontiers_2d=frontiers_2d,
    )


def load_field_episodes(
    field_root: str,
    *,
    max_episodes: Optional[int] = None,
    episode_ids: Optional[Iterable[int]] = None,
    verbose: bool = False,
) -> Tuple[List[FieldEpisode], List[str]]:
    """Read all completed field-observation ZIPs below ``field_root``."""
    episodes: List[FieldEpisode] = []
    errors: List[str] = []
    if not field_root or not os.path.isdir(field_root):
        return episodes, [f"field observation directory not found: {field_root}"]
    paths = sorted(glob.glob(os.path.join(field_root, "**", "*.zip"), recursive=True))
    if episode_ids is not None:
        wanted = {int(value) for value in episode_ids}
        filtered: List[str] = []
        for path in paths:
            try:
                episode_id = int(os.path.basename(path).split("_", 1)[0])
            except (TypeError, ValueError):
                continue
            if episode_id in wanted:
                filtered.append(path)
        paths = filtered
    if max_episodes is not None:
        paths = paths[: max(0, int(max_episodes))]
    for path in paths:
        if os.path.basename(path).startswith(".pending_field_"):
            continue
        try:
            with zipfile.ZipFile(path, "r") as archive:
                names = set(archive.namelist())
                if "episode.json" not in names:
                    errors.append(f"{path}: missing episode.json")
                    continue
                metadata = json.loads(archive.read("episode.json").decode("utf-8"))
                frame_names = sorted(
                    name for name in names if name.startswith("frames/") and name.endswith(".npz")
                )
                frames = [
                    frame
                    for name in frame_names
                    if (frame := _load_frame(name, archive)) is not None
                ]
                ground_truth: Dict[str, np.ndarray] = {}
                gt_name = "ground_truth/top_down_map.npz"
                if gt_name in names:
                    with archive.open(gt_name) as handle:
                        with np.load(handle, allow_pickle=False) as data:
                            ground_truth = {key: np.asarray(data[key]) for key in data.files}
                episodes.append(
                    FieldEpisode(
                        path=path,
                        metadata=metadata,
                        frames=frames,
                        ground_truth=ground_truth,
                    )
                )
                if verbose:
                    print(f"[field] loaded {path}: {len(frames)} frames")
        except Exception as exc:  # corrupt archives must not stop the batch
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return episodes, errors


# ---------------------------------------------------------------------------
# navigation JSONL joins
# ---------------------------------------------------------------------------
@dataclass
class NavigationLogs:
    candidates: List[Dict[str, Any]]
    tracks: List[Dict[str, Any]]
    nav_goals: List[Dict[str, Any]]
    episodes: List[Dict[str, Any]]
    _step_cache: Dict[str, Dict[Tuple[str, int, int], List[Dict[str, Any]]]] = dc_field(
        default_factory=dict, init=False, repr=False
    )
    _episode_cache: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]] = dc_field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def load(cls, detections_dir: str) -> "NavigationLogs":
        return cls(
            candidates=load_jsonl(os.path.join(detections_dir, "candidates.jsonl")),
            tracks=load_jsonl(os.path.join(detections_dir, "track_frames.jsonl")),
            nav_goals=load_jsonl(os.path.join(detections_dir, "nav_goals.jsonl")),
            episodes=load_jsonl(os.path.join(detections_dir, "episodes.jsonl")),
        )

    def by_step(self, field: str) -> Dict[Tuple[str, int, int], List[Dict[str, Any]]]:
        if field not in self._step_cache:
            rows = getattr(self, field)
            grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = collections.defaultdict(list)
            for row in rows:
                try:
                    key = step_key(row.get("scene"), row.get("episode_id"), row.get("step"))
                except (TypeError, ValueError):
                    continue
                grouped[key].append(row)
            self._step_cache[field] = dict(grouped)
        return self._step_cache[field]

    def by_episode(self, field: str) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
        if field not in self._episode_cache:
            rows = getattr(self, field)
            grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
            for row in rows:
                try:
                    key = episode_key(row.get("scene"), row.get("episode_id"))
                except (TypeError, ValueError):
                    continue
                grouped[key].append(row)
            self._episode_cache[field] = dict(grouped)
        return self._episode_cache[field]

    def episode_row(self, scene: str, episode_id: int) -> Optional[Dict[str, Any]]:
        for row in self.episodes:
            try:
                if episode_key(row.get("scene"), row.get("episode_id")) == (
                    scene_key(scene),
                    int(episode_id),
                ):
                    return row
            except (TypeError, ValueError):
                continue
        return None


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionSet:
    cell_size_m: float
    keys: Tuple[Tuple[int, int], ...]
    centers_xy: np.ndarray
    region_ids: Tuple[str, ...]
    index: Mapping[Tuple[int, int], int]

    def key_for_xy(self, xy: np.ndarray) -> Tuple[int, int]:
        return (
            int(math.floor(float(xy[0]) / self.cell_size_m)),
            int(math.floor(float(xy[1]) / self.cell_size_m)),
        )

    def index_for_xy(self, xy: np.ndarray) -> Optional[int]:
        return self.index.get(self.key_for_xy(xy))

    def nearest_index(
        self, xy: np.ndarray, *, max_distance_m: Optional[float] = None
    ) -> Optional[int]:
        if not self.region_ids:
            return None
        distances = np.linalg.norm(self.centers_xy - np.asarray(xy, dtype=np.float64)[:2], axis=1)
        idx = int(np.argmin(distances))
        if max_distance_m is not None and float(distances[idx]) > float(max_distance_m):
            return None
        return idx


class IncrementalRegionBuilder:
    """Grow a grid from observations without looking into the future."""

    def __init__(self, cell_size_m: float):
        if not np.isfinite(cell_size_m) or cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        self.cell_size_m = float(cell_size_m)
        self.keys: set = set()
        self.visit_counts: collections.Counter = collections.Counter()
        self.candidate_counts: collections.Counter = collections.Counter()

    def _key(self, xy: np.ndarray) -> Tuple[int, int]:
        return (
            int(math.floor(float(xy[0]) / self.cell_size_m)),
            int(math.floor(float(xy[1]) / self.cell_size_m)),
        )

    def add_robot(self, xy: np.ndarray) -> Tuple[int, int]:
        key = self._key(xy)
        self.keys.add(key)
        self.visit_counts[key] += 1
        return key

    def add_points(self, points: Any, *, count_candidates: bool = False) -> None:
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return
        arr = arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1)
        for row in arr:
            if row.size < 2 or not np.all(np.isfinite(row[:2])):
                continue
            key = self._key(row[:2])
            self.keys.add(key)
            if count_candidates:
                self.candidate_counts[key] += 1

    def snapshot(self) -> RegionSet:
        keys = tuple(sorted(self.keys))
        centers = np.asarray(
            [[(ix + 0.5) * self.cell_size_m, (iy + 0.5) * self.cell_size_m] for ix, iy in keys],
            dtype=np.float64,
        ).reshape(-1, 2)
        ids = tuple(f"g:{ix}:{iy}" for ix, iy in keys)
        index = {key: i for i, key in enumerate(keys)}
        return RegionSet(self.cell_size_m, keys, centers, ids, index)


# ---------------------------------------------------------------------------
# ground-truth labels
# ---------------------------------------------------------------------------
class GtLabeler:
    """Map an episodic XY point to the static Habitat target mask."""

    def __init__(self, ground_truth: Mapping[str, np.ndarray], cell_size_m: float):
        self.cell_size_m = float(cell_size_m)
        self.mask = ground_truth.get("target_bboxes_mask")
        self.upper = ground_truth.get("upper_bound")
        self.lower = ground_truth.get("lower_bound")
        self.grid = ground_truth.get("grid_resolution")
        self.transform = ground_truth.get("tf_episodic_to_global")
        self.available = all(
            value is not None
            for value in (self.mask, self.upper, self.lower, self.grid, self.transform)
        ) and xyz_to_habitat is not None
        self._cache: Dict[Tuple[float, float], Optional[int]] = {}

    def label(self, xy: np.ndarray) -> Optional[int]:
        if not self.available:
            return None
        center = (round(float(xy[0]), 3), round(float(xy[1]), 3))
        if center in self._cache:
            return self._cache[center]
        values: List[int] = []
        # Sample the full cell, including corners.  A 1 m cell whose center is
        # up to 0.5 m away from a small target bbox can otherwise be labelled
        # negative even when the target touches that cell.
        offset = 0.5 * self.cell_size_m
        offsets = (-offset, 0.0, offset)
        for dx in offsets:
            for dy in offsets:
                point = np.asarray(
                    [[float(xy[0]) + dx, float(xy[1]) + dy, 0.0]],
                    dtype=np.float64,
                )
                try:
                    global_xyz = transform_points(np.asarray(self.transform), point)
                    habitat_xyz = xyz_to_habitat(global_xyz)  # type: ignore[misc]
                    habitat_xy = np.asarray(habitat_xyz)[:, [2, 0]]
                    grid_xy = sim_xy_to_grid_xy(
                        self.upper,
                        self.lower,
                        self.grid,
                        habitat_xy,
                        remove_duplicates=True,
                    )
                    for row_raw, col_raw in np.asarray(grid_xy).reshape(-1, 2):
                        row, col = int(row_raw), int(col_raw)
                        if 0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1]:  # type: ignore[union-attr]
                            values.append(int(self.mask[row, col] != 0))  # type: ignore[index]
                except Exception:
                    continue
        label = 1 if any(values) else (0 if values else None)
        self._cache[center] = label
        return label


# ---------------------------------------------------------------------------
# optional semantic sidecar
# ---------------------------------------------------------------------------
class SemanticSidecar:
    """Already-computed per-region context/relation scores."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.context_dim = max(
            (len(row.get("context_scores", ()) or ()) for row in rows), default=0
        )
        self.relation_dim = max(
            (len(row.get("relation_scores", ()) or ()) for row in rows), default=0
        )
        self.target_dim = int(
            any(row.get("target_context_score") is not None for row in rows)
        )
        self.rows: Dict[Tuple[str, int, int, str], Mapping[str, Any]] = {}
        self.context_prompts: Tuple[str, ...] = ()
        for row in rows:
            try:
                key = (
                    scene_key(row.get("scene")),
                    int(row.get("episode_id")),
                    int(row.get("step")),
                    str(row.get("region_id")),
                )
            except (TypeError, ValueError):
                continue
            self.rows[key] = row
            if not self.context_prompts:
                prompts = row.get("context_prompts") or ()
                if isinstance(prompts, (list, tuple)) and prompts:
                    self.context_prompts = tuple(str(p) for p in prompts)[
                        : self.context_dim
                    ]
        names: List[str] = []
        if self.context_dim:
            names.extend(("sem_context_mean", "sem_context_coverage"))
            # Keep each prompt score as a separate feature.  Collapsing all
            # contexts to a mean destroys the target-specific signal (for
            # example "a bed" vs "a television") and makes the semantic field
            # no more expressive than a single generic saliency score.
            for i in range(self.context_dim):
                prompt = (
                    self.context_prompts[i]
                    if i < len(self.context_prompts)
                    else None
                )
                if prompt:
                    slug = "".join(
                        char if char.isalnum() else "_" for char in prompt.lower()
                    ).strip("_")
                    names.append(f"sem_context_{slug or 'prompt'}")
                else:
                    names.append(f"sem_context_{i}")
        if self.target_dim:
            names.extend(("sem_target_context", "sem_target_margin"))
        if self.relation_dim:
            names.extend(f"sem_relation_{i}" for i in range(self.relation_dim))
        if self.rows:
            names.append("sem_available")
        self.feature_names: Tuple[str, ...] = tuple(names)

    @classmethod
    def load(cls, path: Optional[str]) -> "SemanticSidecar":
        return cls(load_jsonl(path) if path else [])

    @property
    def available(self) -> bool:
        return bool(self.rows)

    def features(
        self,
        scene: str,
        episode_id: int,
        step: int,
        region_id: str,
    ) -> np.ndarray:
        if not self.feature_names:
            return np.empty((0,), dtype=np.float64)
        row = self.rows.get((scene_key(scene), int(episode_id), int(step), str(region_id)))
        if row is None:
            return np.concatenate((np.zeros(len(self.feature_names) - 1), [0.0]))
        context = np.asarray(row.get("context_scores", ()), dtype=np.float64).reshape(-1)
        if context.size < self.context_dim:
            context = np.pad(context, (0, self.context_dim - context.size))
        else:
            context = context[: self.context_dim]
        reliability = np.asarray(
            row.get("context_reliability", np.ones_like(context)), dtype=np.float64
        ).reshape(-1)
        if reliability.size < context.size:
            reliability = np.pad(reliability, (0, context.size - reliability.size), constant_values=1.0)
        else:
            reliability = reliability[: context.size]
        reliability = np.clip(reliability, 0.0, 1.0)
        # Reorder prompt scores into the sidecar's canonical prompt order so
        # feature indices stay comparable across rows and episodes.
        if self.context_prompts:
            row_prompts = row.get("context_prompts") or ()
            if row_prompts:
                prompt_index = {str(p): i for i, p in enumerate(row_prompts)}
                reordered = np.zeros(self.context_dim, dtype=np.float64)
                reordered_reliability = np.ones(self.context_dim, dtype=np.float64)
                for i, prompt in enumerate(self.context_prompts):
                    source = prompt_index.get(prompt)
                    if source is None or source >= context.size:
                        continue
                    reordered[i] = context[source]
                    if source < reliability.size:
                        reordered_reliability[i] = reliability[source]
                context = reordered
                reliability = reordered_reliability
        weighted = np.divide(
            float((context * reliability).sum()),
            float(reliability.sum()),
            out=np.zeros((), dtype=np.float64),
            where=float(reliability.sum()) > 0,
        )
        coverage = float(reliability.sum() / max(1, context.size))
        blocks: List[np.ndarray] = []
        if self.context_dim:
            blocks.append(np.asarray([weighted, coverage], dtype=np.float64))
            blocks.append(context)
        if self.target_dim:
            target_score = row.get("target_context_score")
            target_margin = row.get("target_context_margin")
            try:
                target_score = float(target_score)
            except (TypeError, ValueError):
                target_score = 0.0
            try:
                target_margin = float(target_margin)
            except (TypeError, ValueError):
                target_margin = 0.0
            blocks.append(
                np.asarray([target_score, target_margin], dtype=np.float64)
            )
        relation = np.asarray(row.get("relation_scores", ()), dtype=np.float64).reshape(-1)
        if self.relation_dim:
            if relation.size < self.relation_dim:
                relation = np.pad(relation, (0, self.relation_dim - relation.size))
            else:
                relation = relation[: self.relation_dim]
            blocks.append(relation)
        if self.rows:
            blocks.append(np.asarray([1.0], dtype=np.float64))
        return np.concatenate(blocks) if blocks else np.empty((0,), dtype=np.float64)


# ---------------------------------------------------------------------------
# frame samples
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionSample:
    scene: str
    episode_id: int
    step: int
    region_id: str
    center_xy: Tuple[float, float]
    features: Tuple[float, ...]
    label: Optional[int]


@dataclass(frozen=True)
class FrameSample:
    scene: str
    episode_id: int
    step: int
    regions: RegionSet
    features: np.ndarray
    labels: Tuple[Optional[int], ...]
    target_indices: Tuple[int, ...]
    robot_xy: np.ndarray
    frontiers_xy: np.ndarray
    candidate_count: int


def build_episode_samples(
    episode: FieldEpisode,
    logs: NavigationLogs,
    *,
    cell_size_m: float = 1.0,
    semantic_sidecar: Optional[SemanticSidecar] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[List[FrameSample], List[RegionSample], List[str]]:
    """Build per-step region samples without using future observations."""
    warnings: List[str] = []
    scene = episode.scene
    eid = episode.episode_id
    if not episode.frames:
        return [], [], [f"{scene}:{eid}: no frames"]
    semantic_sidecar = semantic_sidecar or SemanticSidecar([])
    names = tuple(feature_names or (GEOMETRY_FEATURE_NAMES + semantic_sidecar.feature_names))
    builder = IncrementalRegionBuilder(cell_size_m)
    labeler = GtLabeler(episode.ground_truth, cell_size_m)
    if not labeler.available:
        warnings.append(f"{scene}:{eid}: GT target mask unavailable; region labels are None")
    candidates_by_step = logs.by_step("candidates")
    tracks_by_step = logs.by_step("tracks")
    goals_by_step = logs.by_step("nav_goals")

    # Multiple camera views share a step; the field operates once per step.
    by_step: Dict[int, List[FieldFrame]] = collections.defaultdict(list)
    for frame in episode.frames:
        by_step[int(frame.step)].append(frame)

    frame_samples: List[FrameSample] = []
    region_samples: List[RegionSample] = []
    for step in sorted(by_step):
        frames = by_step[step]
        first = frames[0]
        builder.add_robot(first.robot_xy)
        for frame in frames:
            builder.add_points(frame.frontiers_3d[:, :2] if frame.frontiers_3d.size else (), count_candidates=False)
            if frame.frontiers_2d.size:
                builder.add_points(frame.frontiers_2d[:, :2], count_candidates=False)

        key3 = (scene, eid, step)
        candidate_rows = candidates_by_step.get(key3, []) + tracks_by_step.get(key3, []) + goals_by_step.get(key3, [])
        candidate_points = []
        for row in candidate_rows:
            point = _as_xy(row.get("centroid"))
            if point is not None:
                candidate_points.append(point)
        if candidate_points:
            builder.add_points(np.asarray(candidate_points), count_candidates=True)

        regions = builder.snapshot()
        # The observation archive stores 2D frontier cells even when 3D
        # frontier points are absent.  Use both so the frontier feature and
        # exploration replay see the same frontier set the policy used.
        frontier_arrays = [
            frame.frontiers_2d[:, :2]
            for frame in frames
            if frame.frontiers_2d.size
        ]
        frontier_arrays.extend(
            frame.frontiers_3d[:, :2]
            for frame in frames
            if frame.frontiers_3d.size
        )
        frontiers_xy = (
            np.concatenate(frontier_arrays, axis=0)
            if frontier_arrays
            else np.empty((0, 2))
        )

        rows: List[List[float]] = []
        labels: List[Optional[int]] = []
        for i, center in enumerate(regions.centers_xy):
            key = regions.keys[i]
            dist = float(np.linalg.norm(center - first.robot_xy[:2]))
            if frontiers_xy.size:
                frontier_dist = float(np.min(np.linalg.norm(frontiers_xy - center, axis=1)))
                frontier_prox = float(math.exp(-frontier_dist / 2.0))
            else:
                frontier_dist = float("inf")
                frontier_prox = 0.0
            visits = int(builder.visit_counts.get(key, 0))
            candidates = int(builder.candidate_counts.get(key, 0))
            geometry = (
                float(math.log1p(dist)),
                frontier_prox,
                float(math.log1p(visits)),
                float(math.log1p(candidates)),
                1.0 if visits > 0 else 0.0,
            )
            semantic = semantic_sidecar.features(scene, eid, step, regions.region_ids[i])
            features = geometry + tuple(float(v) for v in semantic)
            if len(features) != len(names):
                raise ValueError(
                    f"feature width mismatch: got {len(features)}, expected {len(names)}"
                )
            label = labeler.label(center)
            rows.append(list(features))
            labels.append(label)
            if label is not None:
                region_samples.append(
                    RegionSample(
                        scene=scene,
                        episode_id=eid,
                        step=step,
                        region_id=regions.region_ids[i],
                        center_xy=(float(center[0]), float(center[1])),
                        features=tuple(float(v) for v in features),
                        label=int(label),
                    )
                )
        feature_matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names))
        target_indices = tuple(i for i, value in enumerate(labels) if value == 1)
        frame_samples.append(
            FrameSample(
                scene=scene,
                episode_id=eid,
                step=step,
                regions=regions,
                features=feature_matrix,
                labels=tuple(labels),
                target_indices=target_indices,
                robot_xy=first.robot_xy[:2].copy(),
                frontiers_xy=frontiers_xy,
                candidate_count=len(candidate_points),
            )
        )
    return frame_samples, region_samples, warnings


# ---------------------------------------------------------------------------
# observation-rate calibration from track_frames.jsonl
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObservationRateBin:
    key: str
    min_distance_m: float
    max_distance_m: float
    n_target: int
    n_nontarget: int
    positive_target: int
    positive_nontarget: int
    hit_probability: float
    false_alarm_probability: float


@dataclass(frozen=True)
class ObservationRates:
    hit_probability: float
    false_alarm_probability: float
    ambiguous_probability: float
    source: str


class ObservationRateTable:
    """Visibility-stratified Beta-smoothed rates from track-frame labels."""

    def __init__(self, bins: Sequence[ObservationRateBin], global_bin: ObservationRateBin):
        self.bins = tuple(bins)
        self.global_bin = global_bin

    @classmethod
    def from_tracks(
        cls,
        tracks: Sequence[Mapping[str, Any]],
        *,
        distance_edges: Sequence[float] = DISTANCE_EDGES,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> "ObservationRateTable":
        aggregates: Dict[int, Dict[str, int]] = collections.defaultdict(
            lambda: {"n_target": 0, "n_nontarget": 0, "pos_target": 0, "pos_nontarget": 0}
        )
        total = {"n_target": 0, "n_nontarget": 0, "pos_target": 0, "pos_nontarget": 0}
        for row in tracks:
            if "matched" not in row:
                continue
            in_fov = bool(row.get("in_fov", True))
            depth_ok = bool(row.get("depth_ok", True))
            if not (in_fov and depth_ok):
                continue
            distance = _as_float(row.get("distance"))
            if distance is None:
                centroid = _as_xy(row.get("centroid"))
                robot = _as_xy(row.get("robot_xy"))
                if centroid is None or robot is None:
                    continue
                distance = float(np.linalg.norm(centroid - robot))
            bin_idx = _distance_bin_index(float(distance), distance_edges)
            target = bool(row.get("tp", False))
            positive = bool(row.get("matched", False))
            aggregates[bin_idx]["n_target" if target else "n_nontarget"] += 1
            total["n_target" if target else "n_nontarget"] += 1
            if positive:
                aggregates[bin_idx]["pos_target" if target else "pos_nontarget"] += 1
                total["pos_target" if target else "pos_nontarget"] += 1

        def make_bin(key: str, lo: float, hi: float, counts: Mapping[str, int]) -> ObservationRateBin:
            n_t = int(counts["n_target"])
            n_n = int(counts["n_nontarget"])
            pos_t = int(counts["pos_target"])
            pos_n = int(counts["pos_nontarget"])
            # Missing-class fallbacks are conservative defaults, not the
            # Beta(1,1) midpoint, which would imply 50% false alarms.
            hit = (pos_t + alpha) / (n_t + alpha + beta) if n_t > 0 else 0.65
            false_alarm = (pos_n + alpha) / (n_n + alpha + beta) if n_n > 0 else 0.10
            return ObservationRateBin(
                key=key,
                min_distance_m=float(lo),
                max_distance_m=float(hi),
                n_target=n_t,
                n_nontarget=n_n,
                positive_target=pos_t,
                positive_nontarget=pos_n,
                hit_probability=float(hit),
                false_alarm_probability=float(false_alarm),
            )

        bins = [
            make_bin(
                _distance_bin_label(lo, hi),
                lo,
                hi,
                aggregates[idx],
            )
            for idx, (lo, hi) in enumerate(zip(distance_edges[:-1], distance_edges[1:]))
            if idx in aggregates
        ]
        global_bin = make_bin("global", 0.0, float("inf"), total)
        return cls(bins, global_bin)

    def lookup(self, distance_m: float, visibility: float = 1.0) -> ObservationRates:
        if not np.isfinite(visibility) or visibility <= 0.0:
            return ObservationRates(0.0, 0.0, 1.0, "unobservable")
        distance = float(distance_m) if np.isfinite(distance_m) else float("inf")
        chosen = self.global_bin
        for candidate in self.bins:
            if candidate.min_distance_m <= distance < candidate.max_distance_m:
                if candidate.n_target + candidate.n_nontarget > 0:
                    chosen = candidate
                break
        scale = float(np.clip(visibility, 0.0, 1.0))
        hit = float(np.clip(chosen.hit_probability * scale, 0.0, 1.0))
        false_alarm = float(np.clip(chosen.false_alarm_probability * scale, 0.0, 1.0))
        ambiguous = float(np.clip(max(1.0 - scale, 0.0), 0.0, 1.0))
        if hit + ambiguous > 1.0:
            ambiguous = max(0.0, 1.0 - hit)
        if false_alarm + ambiguous > 1.0:
            false_alarm = max(0.0, 1.0 - ambiguous)
        return ObservationRates(
            hit_probability=hit,
            false_alarm_probability=false_alarm,
            ambiguous_probability=ambiguous,
            source=chosen.key if scale >= 1.0 else f"{chosen.key}+partial",
        )

    @classmethod
    def default(cls) -> "ObservationRateTable":
        empty = ObservationRateBin("global", 0.0, float("inf"), 0, 0, 0, 0, 0.65, 0.10)
        return cls((), empty)


def _distance_bin_index(distance: float, edges: Sequence[float]) -> int:
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo <= distance < hi:
            return idx
    return len(edges) - 2


def _distance_bin_label(lo: float, hi: float) -> str:
    end = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
    return f"[{lo:.1f},{end})"
