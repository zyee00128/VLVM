"""Generate the optional ``region_semantics.jsonl`` sidecar for the field.

The field analyzer deliberately keeps semantic scoring outside the policy.  This
module bridges that gap without importing the navigation policy:

    field archive (RGB + pose + intrinsics)
      -> per-step region IDs from the same IncrementalRegionBuilder used offline
      -> project each region centre into the current camera image
      -> crop a region patch
      -> BLIP-2 image/text cosine scores for context prompts
      -> region_semantics.jsonl

The sidecar is only an *input* to ``offline_analysis``.  It never modifies the
field, the policy, or TSP3D.  Regions without a usable projection/crop are
omitted; ``SemanticSidecar`` treats missing rows as unavailable rather than as
negative evidence.

Example (one episode, then inspect the output)::

    python -m vlfm.semantic_spatial_field.region_semantics \
      --run-dir outputs/run \
      --out /tmp/region_semantics.jsonl \
      --max-episodes 1 --max-frames 20 --max-regions-per-frame 4

The full three-scene archive is intentionally opt-in: BLIP-2 scoring every
region is expensive.  Use ``--max-episodes``/``--every-n`` for a smoke run and
remove them only after checking the semantic rows.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .offline_data import (
    FieldEpisode,
    FrameSample,
    NavigationLogs,
    SemanticSidecar,
    build_episode_samples,
    load_field_episodes,
    scene_key,
)


DEFAULT_CONTEXTS: Tuple[str, ...] = (
    "a bed",
    "a chair",
    "a couch",
    "a toilet",
    "a television",
    "a room with furniture",
    "a doorway",
    "a corridor",
    "a table",
    "an open area",
)


@dataclass(frozen=True)
class RegionSemanticConfig:
    """Runtime options for sidecar generation."""

    cell_size_m: float = 1.0
    region_z_m: float = 0.6
    region_z_levels: Tuple[float, ...] = (0.4, 0.8, 1.2)
    crop_radius_px: int = 80
    max_regions_per_frame: int = 0  # 0 = all regions
    every_n: int = 1
    max_episodes: Optional[int] = None
    max_frames: Optional[int] = None
    scenes: Optional[Tuple[str, ...]] = None
    episode_ids: Optional[Tuple[int, ...]] = None
    normalize_context: bool = True
    port: int = 12182
    contexts: Tuple[str, ...] = DEFAULT_CONTEXTS


def _load_image_frame(
    archive: zipfile.ZipFile,
    name: str,
) -> Optional[Dict[str, Any]]:
    """Read the RGB/pose/intrinsics subset needed for projection."""
    try:
        with archive.open(name) as handle:
            with np.load(handle, allow_pickle=False) as data:
                rgb = np.asarray(data["rgb"], dtype=np.uint8)
                camera_to_episodic = np.asarray(
                    data["camera_to_episodic"], dtype=np.float64
                ).reshape(4, 4)
                intrinsics = np.asarray(
                    data["depth_range_intrinsics"], dtype=np.float64
                ).reshape(-1)
                robot_xy = np.asarray(data["robot_xy"], dtype=np.float64).reshape(-1)
                robot_yaw = float(np.asarray(data["robot_yaw_rad"]).reshape(-1)[0])
                step = int(np.asarray(data["step"]).reshape(-1)[0])
                view_index = int(np.asarray(data["view_index"]).reshape(-1)[0])
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        if intrinsics.size < 4:
            return None
        return {
            "rgb": rgb[:, :, :3],
            "tf_camera_to_episodic": camera_to_episodic,
            "fx": float(intrinsics[2]),
            "fy": float(intrinsics[3]),
            "min_depth": float(intrinsics[0]),
            "max_depth": float(intrinsics[1]),
            "robot_xy": robot_xy[:2],
            "robot_yaw": robot_yaw,
            "step": step,
            "view_index": view_index,
        }
    except Exception:
        return None


def _read_episode_images(episode: FieldEpisode) -> Dict[int, Dict[str, Any]]:
    """Load one RGB view per step from a field ZIP.

    The field archive already contains pose-only frames in ``episode``; this
    helper only reads the image subset for the current episode and keeps it
    local so the full archive is never resident in memory.
    """
    images: Dict[int, Dict[str, Any]] = {}
    try:
        with zipfile.ZipFile(episode.path, "r") as archive:
            frame_names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("frames/") and name.endswith(".npz")
            )
            for name in frame_names:
                item = _load_image_frame(archive, name)
                if item is None:
                    continue
                step = int(item["step"])
                view = int(item["view_index"])
                if step not in images or view < int(images[step].get("view_index", 1 << 30)):
                    images[step] = item
    except Exception:
        return {}
    return images


def _project_region_to_image(
    xy: np.ndarray,
    *,
    image: Dict[str, Any],
    region_z_m: float,
) -> Optional[Tuple[float, float, float]]:
    """Project a world XY region centre to pixel coordinates.

    Returns ``(u, v, depth)`` or ``None``.  The point/cloud convention in
    ``vlfm.utils.geometry_utils.get_point_cloud`` is ``(forward, -right, -up)``;
    the inverse projection below follows that convention.
    """
    point_world = np.asarray(
        [float(xy[0]), float(xy[1]), float(region_z_m), 1.0], dtype=np.float64
    )
    tf = np.asarray(image["tf_camera_to_episodic"], dtype=np.float64).reshape(4, 4)
    point_camera = tf[:3, :3].T @ (point_world[:3] - tf[:3, 3])
    depth = float(point_camera[0])
    if not np.isfinite(depth) or depth <= 0.0:
        return None
    if depth < float(image["min_depth"]) or depth > float(image["max_depth"]):
        return None
    height, width = image["rgb"].shape[:2]
    u = width / 2.0 - float(image["fx"]) * float(point_camera[1]) / depth
    v = height / 2.0 - float(image["fy"]) * float(point_camera[2]) / depth
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    if u < 0.0 or u >= width or v < 0.0 or v >= height:
        return None
    return float(u), float(v), depth


def _crop_region(
    rgb: np.ndarray,
    u: float,
    v: float,
    radius: int,
) -> Optional[np.ndarray]:
    """Crop a square RGB patch around a projected region centre."""
    height, width = rgb.shape[:2]
    radius = max(4, int(radius))
    x0 = max(0, int(round(u)) - radius)
    x1 = min(width, int(round(u)) + radius + 1)
    y0 = max(0, int(round(v)) - radius)
    y1 = min(height, int(round(v)) + radius + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8)


class Blip2ContextScorer:
    """Thin wrapper around the existing BLIP-2 ITM client."""

    def __init__(self, port: int = 12182) -> None:
        # Import lazily so geometry-only tests do not require the VLM client.
        from vlfm.vlm.blip2itm import BLIP2ITMClient

        self.client = BLIP2ITMClient(port=port)
        self.contexts: Tuple[str, ...] = ()

    def score(self, crop: np.ndarray, contexts: Sequence[str]) -> List[float]:
        scores: List[float] = []
        for prompt in contexts:
            try:
                value = float(self.client.cosine(crop, str(prompt)))
            except Exception:
                value = float("nan")
            scores.append(value)
        return scores


def _score_region_at_heights(
    *,
    center: np.ndarray,
    image: Dict[str, Any],
    z_levels: Sequence[float],
    crop_radius: int,
    scorer: "Blip2ContextScorer",
    contexts: Sequence[str],
    normalize: bool,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """Project one region at several heights and keep the best visible crop.

    A 2D region has no height.  Projecting only ``z=0.6`` sends wall-mounted
    objects (TVs, cabinets, wall art) to the wall base and produces a crop that
    contains no object evidence.  Trying a small height ladder lets the
    semantic channel see the object while keeping one row per region.
    """
    counters: Dict[str, int] = {
        "regions_considered": 0,
        "regions_not_visible": 0,
        "regions_crop_failed": 0,
        "score_failed": 0,
    }
    best: Optional[Dict[str, Any]] = None
    for z_m in z_levels:
        counters["regions_considered"] += 1
        projection = _project_region_to_image(
            center, image=image, region_z_m=float(z_m)
        )
        if projection is None:
            counters["regions_not_visible"] += 1
            continue
        u, v, depth = projection
        crop = _crop_region(image["rgb"], u, v, int(crop_radius))
        if crop is None:
            counters["regions_crop_failed"] += 1
            continue
        raw_scores = scorer.score(crop, contexts)
        if not any(np.isfinite(raw) for raw in raw_scores):
            counters["score_failed"] += 1
            continue
        scores: List[float] = []
        for raw in raw_scores:
            if not np.isfinite(raw):
                scores.append(0.0)
            elif normalize:
                scores.append(float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0)))
            else:
                scores.append(float(raw))
        candidate = {
            "region_z_m": float(z_m),
            "depth_m": float(depth),
            "pixel_uv": [float(u), float(v)],
            "context_scores": scores,
            "score_key": float(max(scores)),
        }
        if best is None or candidate["score_key"] > best["score_key"]:
            best = candidate
    if best is None:
        return None, counters
    best.pop("score_key", None)
    return best, counters


def _target_context_index(
    target_object: Any, contexts: Sequence[str]
) -> Optional[int]:
    """Map an episode target object to a context prompt index.

    ``tv``/``television`` and ``couch``/``sofa`` are common aliases in
    HM3D/ObjectNav; all other targets use a case-insensitive substring match.
    Returning ``None`` means the row has no target-conditioned semantic score;
    the sidecar treats it as unavailable, not negative evidence.
    """
    target = str(target_object or "").strip().lower().replace("_", " ")
    if not target:
        return None
    aliases = {
        "tv": ("tv", "television"),
        "television": ("television", "tv"),
        "couch": ("couch", "sofa"),
        "sofa": ("sofa", "couch"),
    }
    keys = aliases.get(target, (target,))
    for i, prompt in enumerate(contexts):
        text = str(prompt).lower()
        if any(key in text for key in keys):
            return i
    return None


def _select_regions(
    sample: FrameSample,
    max_regions: int,
) -> List[int]:
    """Choose regions to score, prioritising candidates/frontiers/robot.

    The field sidecar is allowed to be sparse.  This ordering keeps a smoke run
    useful without pretending that unobserved regions have semantic evidence.
    """
    n = int(sample.features.shape[0])
    if n == 0:
        return []
    # Feature layout: log distance, frontier proximity, log visits,
    # log candidates, observed.
    priority = (
        sample.features[:, 3]
        + sample.features[:, 1]
        + sample.features[:, 4]
        - 0.05 * sample.features[:, 0]
    )
    order = np.argsort(-priority)
    if max_regions > 0:
        order = order[: int(max_regions)]
    return [int(i) for i in order.tolist()]


def generate_region_semantics(
    *,
    run_dir: str,
    out_path: str,
    config: RegionSemanticConfig,
    scorer: Optional[Blip2ContextScorer] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate sidecar rows for completed field-observation archives.

    Returns a compact generation summary.  Rows with no visible crop are
    skipped; the caller can verify coverage with the returned counters.
    """
    field_root = os.path.join(run_dir, "detections", "field_observations")
    detections_dir = os.path.join(run_dir, "detections")
    if not os.path.isdir(field_root):
        raise FileNotFoundError(f"field_observations not found: {field_root}")

    logs = NavigationLogs.load(detections_dir)
    semantic_empty = SemanticSidecar([])
    scorer = scorer or Blip2ContextScorer(port=config.port)
    context_names = tuple(str(c) for c in config.contexts)
    if not context_names:
        raise ValueError("at least one context prompt is required")
    z_levels = tuple(float(z) for z in (config.region_z_levels or (config.region_z_m,)))

    scene_dirs = sorted(
        path for path in glob.glob(os.path.join(field_root, "*")) if os.path.isdir(path)
    )
    if config.scenes:
        wanted_scenes = {str(scene) for scene in config.scenes}
        scene_dirs = [
            path for path in scene_dirs if os.path.basename(path) in wanted_scenes
        ]
    wanted_episodes = (
        {int(episode_id) for episode_id in config.episode_ids}
        if config.episode_ids
        else None
    )
    counters: Counter = Counter()
    generated = 0
    tmp_path = out_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(tmp_path, "w", encoding="utf-8") as out:
        for scene_dir in scene_dirs:
            episodes, load_errors = load_field_episodes(
                scene_dir,
                max_episodes=config.max_episodes,
                episode_ids=config.episode_ids,
                verbose=False,
            )
            counters["load_errors"] += len(load_errors)
            for episode in episodes:
                if wanted_episodes is not None and int(episode.episode_id) not in wanted_episodes:
                    continue
                if config.max_episodes is not None and counters["episodes"] >= int(
                    config.max_episodes
                ):
                    break
                counters["episodes"] += 1
                try:
                    frames, _regions, build_warnings = build_episode_samples(
                        episode,
                        logs,
                        cell_size_m=float(config.cell_size_m),
                        semantic_sidecar=semantic_empty,
                    )
                except Exception as exc:
                    counters["build_errors"] += 1
                    if verbose:
                        print(
                            f"[region-semantics] build failed {episode.path}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                    continue
                counters["build_warnings"] += len(build_warnings)
                images = _read_episode_images(episode)
                if not images:
                    counters["image_missing_episodes"] += 1
                    continue
                for sample in frames:
                    if config.max_frames is not None and counters["frames"] >= int(
                        config.max_frames
                    ):
                        break
                    if int(sample.step) % max(1, int(config.every_n)) != 0:
                        continue
                    image = images.get(int(sample.step))
                    if image is None:
                        counters["frame_missing"] += 1
                        continue
                    counters["frames"] += 1
                    for index in _select_regions(
                        sample, int(config.max_regions_per_frame)
                    ):
                        region_id = str(sample.regions.region_ids[index])
                        center = sample.regions.centers_xy[index]
                        scored, trial_counts = _score_region_at_heights(
                            center=center,
                            image=image,
                            z_levels=z_levels,
                            crop_radius=int(config.crop_radius_px),
                            scorer=scorer,
                            contexts=context_names,
                            normalize=bool(config.normalize_context),
                        )
                        for name, value in trial_counts.items():
                            counters[name] += int(value)
                        if scored is None:
                            continue
                        target_object = str(episode.metadata.get("target_object", ""))
                        target_index = _target_context_index(
                            target_object, context_names
                        )
                        target_score: Optional[float] = None
                        target_margin: Optional[float] = None
                        target_prompt = ""
                        if target_index is not None:
                            target_prompt = context_names[target_index]
                            target_score = float(
                                scored["context_scores"][target_index]
                            )
                            others = [
                                float(value)
                                for i, value in enumerate(scored["context_scores"])
                                if i != target_index
                            ]
                            target_margin = float(
                                target_score - (sum(others) / max(1, len(others)))
                            )
                        row = {
                            "schema": "region_semantics_v1",
                            "scene": scene_key(episode.scene),
                            "episode_id": int(episode.episode_id),
                            "step": int(sample.step),
                            "region_id": region_id,
                            "center_xy": [float(center[0]), float(center[1])],
                            "region_z_m": float(scored["region_z_m"]),
                            "target_object": target_object,
                            "target_context_prompt": target_prompt,
                            "target_context_score": target_score,
                            "target_context_margin": target_margin,
                            "context_prompts": list(context_names),
                            "context_scores": scored["context_scores"],
                            "context_reliability": [1.0] * len(context_names),
                            "relation_scores": [],
                            "visibility": 1.0,
                            "crop_radius_px": int(config.crop_radius_px),
                            "depth_m": float(scored["depth_m"]),
                            "pixel_uv": scored["pixel_uv"],
                            "scorer": "blip2_itm",
                            "source": "field_archive",
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        generated += 1
                        counters["rows"] += 1
                    if config.max_frames is not None and counters["frames"] >= int(
                        config.max_frames
                    ):
                        break
                if config.max_frames is not None and counters["frames"] >= int(
                    config.max_frames
                ):
                    break
            if config.max_frames is not None and counters["frames"] >= int(
                config.max_frames
            ):
                break

    if generated == 0:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            "no region-semantic rows generated; check field archives, BLIP-2 "
            "service, projection parameters, and max/episode limits"
        )
    os.replace(tmp_path, out_path)
    return {
        "out_path": os.path.abspath(out_path),
        "contexts": list(context_names),
        "generated_rows": generated,
        "counters": dict(counters),
        "config": {
            "cell_size_m": config.cell_size_m,
            "region_z_m": config.region_z_m,
            "region_z_levels": list(config.region_z_levels),
            "crop_radius_px": config.crop_radius_px,
            "max_regions_per_frame": config.max_regions_per_frame,
            "every_n": config.every_n,
            "max_episodes": config.max_episodes,
            "max_frames": config.max_frames,
            "scenes": list(config.scenes) if config.scenes else None,
            "episode_ids": list(config.episode_ids) if config.episode_ids else None,
            "normalize_context": config.normalize_context,
            "port": config.port,
        },
    }


def _parse_contexts(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate region_semantics.jsonl for the semantic-spatial field"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--region-z", type=float, default=0.6)
    parser.add_argument(
        "--region-z-levels",
        default="0.4,0.8,1.2",
        help="comma-separated projection heights, e.g. 0.4,0.8,1.2; "
        "highest-scoring visible height is kept",
    )
    parser.add_argument("--crop-radius", type=int, default=80)
    parser.add_argument("--max-regions-per-frame", type=int, default=0)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--scenes", default=None, help="comma-separated scene names")
    parser.add_argument("--episode-ids", default=None, help="comma-separated episode ids")
    parser.add_argument("--contexts", default=",".join(DEFAULT_CONTEXTS))
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--no-normalize-context", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    scenes = _parse_contexts(args.scenes) if args.scenes else None
    episode_ids = (
        tuple(int(part) for part in _parse_contexts(args.episode_ids))
        if args.episode_ids
        else None
    )

    run_dir = os.path.abspath(args.run_dir)
    out_path = args.out or os.path.join(
        run_dir, "detections", "region_semantics.jsonl"
    )
    region_z_levels = (
        tuple(float(part) for part in _parse_contexts(args.region_z_levels))
        if args.region_z_levels
        else ()
    )
    cfg = RegionSemanticConfig(
        cell_size_m=float(args.cell_size),
        region_z_m=float(args.region_z),
        region_z_levels=region_z_levels,
        crop_radius_px=int(args.crop_radius),
        max_regions_per_frame=max(0, int(args.max_regions_per_frame)),
        every_n=max(1, int(args.every_n)),
        max_episodes=args.max_episodes,
        max_frames=args.max_frames,
        scenes=scenes,
        episode_ids=episode_ids,
        normalize_context=not bool(args.no_normalize_context),
        port=int(args.port),
        contexts=_parse_contexts(args.contexts),
    )
    summary = generate_region_semantics(
        run_dir=run_dir,
        out_path=out_path,
        config=cfg,
        verbose=bool(args.verbose),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
