"""Offline fitting, replay, and reporting for the semantic-spatial field.

Usage
-----
::

    python -m vlfm.semantic_spatial_field.offline_analysis \
        --run-dir outputs/<run> \
        --out outputs/<run>/field_offline

The command reads ``<run>/detections/field_observations`` and the ordinary
navigation JSONL logs.  It never starts Habitat, TSP3D, a VLM server, or the
policy.  Outputs are a JSON report and a compact Markdown summary.

The analyzer intentionally separates three things:

1. region samples and GT labels (data layer);
2. fitted region weights / observation rates (calibration layer);
3. candidate/coverage replay (evaluation layer).

No fitted value is written back to the policy or to any YAML file.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core import fit_logistic_weights
from .inference import (
    CandidateFieldScore,
    FieldInference,
    FieldInferenceConfig,
    InspectionView,
)
from .offline_data import (
    GEOMETRY_FEATURE_NAMES,
    FrameSample,
    NavigationLogs,
    ObservationRateBin,
    ObservationRateTable,
    RegionSample,
    SemanticSidecar,
    build_episode_samples,
    episode_key,
    load_field_episodes,
    scene_key,
)


# ---------------------------------------------------------------------------
# configuration / small utilities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OfflineAnalysisConfig:
    cell_size_m: float = 1.0
    top_k: int = 5
    l2: float = 1.0
    min_visibility: float = 0.15
    max_episodes: Optional[int] = None
    scene_cv: bool = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    pairs = [
        (float(score), bool(label))
        for score, label in zip(scores, labels)
        if np.isfinite(float(score))
    ]
    n_pos = sum(1 for _, label in pairs if label)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = np.zeros(len(pairs), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    pos_rank_sum = float(sum(ranks[i] for i, (_, label) in enumerate(pairs) if label))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probs = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    if labels.size == 0:
        return {"n": 0}
    nll = float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))
    brier = float(np.mean((probs - labels) ** 2))
    return {
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "n_negative": int(labels.size - labels.sum()),
        "auc": _auc(labels.astype(bool).tolist(), probs.tolist()),
        "nll": nll,
        "brier": brier,
        "mean_probability": float(probs.mean()),
        "base_rate": float(labels.mean()),
    }


def _group_by_step(
    rows: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[str, int, int], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        try:
            key = (
                scene_key(row.get("scene")),
                int(row.get("episode_id")),
                int(row.get("step")),
            )
        except (TypeError, ValueError):
            continue
        groups[key].append(row)
    return groups


def _row_distance(row: Mapping[str, Any], fallback_xy: np.ndarray) -> float:
    value = row.get("distance")
    try:
        distance = float(value)
        if np.isfinite(distance) and distance >= 0:
            return distance
    except (TypeError, ValueError):
        pass
    centroid = np.asarray(row.get("centroid", ()), dtype=np.float64).reshape(-1)
    if centroid.size >= 2 and np.all(np.isfinite(centroid[:2])):
        return float(np.linalg.norm(centroid[:2] - fallback_xy[:2]))
    return float("inf")


def _row_visibility(row: Mapping[str, Any]) -> float:
    """Conservative visibility proxy from existing detection/track fields."""
    if row.get("occ_passed") is False:
        return 0.0
    if row.get("in_fov") is False:
        return 0.0
    if row.get("depth_ok") is False:
        return 0.0
    if str(row.get("geom_status") or "") == "geom_unknown":
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def fit_region_model(
    samples: Sequence[RegionSample],
    feature_names: Sequence[str],
    *,
    l2: float = 1.0,
    scene_cv: bool = True,
) -> Dict[str, Any]:
    """Fit a binary region-presence model and optional scene-grouped CV."""
    usable = [sample for sample in samples if sample.label in (0, 1)]
    if not usable:
        return {
            "status": "no_labels",
            "weights": [],
            "region_bias": 0.0,
            "temperature": 1.0,
            "feature_names": list(feature_names),
            "n": 0,
        }
    x = np.asarray([sample.features for sample in usable], dtype=np.float64)
    y = np.asarray([int(sample.label) for sample in usable], dtype=np.float64)
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names do not match region samples")
    if len(np.unique(y)) < 2:
        return {
            "status": "one_class",
            "weights": [0.0] * len(feature_names),
            "region_bias": 0.0,
            "temperature": 1.0,
            "feature_names": list(feature_names),
            "n": int(len(y)),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        }
    try:
        weights, bias = fit_logistic_weights(x, y, l2=l2)
    except Exception as exc:
        return {
            "status": "fit_error",
            "error": f"{type(exc).__name__}: {exc}",
            "weights": [0.0] * len(feature_names),
            "region_bias": 0.0,
            "temperature": 1.0,
            "feature_names": list(feature_names),
            "n": int(len(y)),
        }
    probabilities = _sigmoid(x @ weights + bias)
    result: Dict[str, Any] = {
        "status": "ok",
        "weights": [float(v) for v in weights],
        "region_bias": float(bias),
        "temperature": 1.0,
        "feature_names": list(feature_names),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "train_metrics": _binary_metrics(y, probabilities),
        "scene_cv": [],
    }
    if scene_cv:
        scenes = sorted({sample.scene for sample in usable})
        for held_out in scenes:
            train = [sample for sample in usable if sample.scene != held_out]
            test = [sample for sample in usable if sample.scene == held_out]
            if not train or not test:
                continue
            xt = np.asarray([sample.features for sample in train], dtype=np.float64)
            yt = np.asarray([int(sample.label) for sample in train], dtype=np.float64)
            xv = np.asarray([sample.features for sample in test], dtype=np.float64)
            yv = np.asarray([int(sample.label) for sample in test], dtype=np.float64)
            if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
                continue
            try:
                w_fold, b_fold = fit_logistic_weights(xt, yt, l2=l2)
                probs = _sigmoid(xv @ w_fold + b_fold)
                metrics = _binary_metrics(yv, probs)
            except Exception as exc:  # keep one bad fold from killing the report
                metrics = {"error": f"{type(exc).__name__}: {exc}"}
            result["scene_cv"].append(
                {
                    "held_out_scene": held_out,
                    "n_train": int(len(yt)),
                    "n_test": int(len(yv)),
                    **metrics,
                }
            )
    return result


def observation_rate_report(table: ObservationRateTable) -> Dict[str, Any]:
    def row(bin_: ObservationRateBin) -> Dict[str, Any]:
        return {
            "key": bin_.key,
            "min_distance_m": bin_.min_distance_m,
            "max_distance_m": bin_.max_distance_m,
            "n_target": bin_.n_target,
            "n_nontarget": bin_.n_nontarget,
            "positive_target": bin_.positive_target,
            "positive_nontarget": bin_.positive_nontarget,
            "hit_probability": bin_.hit_probability,
            "false_alarm_probability": bin_.false_alarm_probability,
        }

    return {
        "bins": [row(bin_) for bin_ in table.bins],
        "global": row(table.global_bin),
        "source": "track_frames.jsonl; visible rows only; Beta(1,1) smoothing",
    }


# ---------------------------------------------------------------------------
# candidate replay
# ---------------------------------------------------------------------------
def _candidate_id(row: Mapping[str, Any], fallback: str) -> str:
    for key in ("candidate_uid", "memory_uid", "candidate_id", "memory_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def score_candidate_rows(
    frame_samples: Mapping[Tuple[str, int, int], FrameSample],
    rows: Sequence[Mapping[str, Any]],
    inference: FieldInference,
    rate_table: ObservationRateTable,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Attach field and baseline scores to candidate/track rows."""
    warnings: List[str] = []
    scored: List[Dict[str, Any]] = []
    for key, group in _group_by_step(rows).items():
        sample = frame_samples.get(key)
        if sample is None or sample.features.shape[0] == 0:
            warnings.append(f"{key}: no matching field frame")
            continue
        candidates = []
        row_by_id: Dict[str, Mapping[str, Any]] = {}
        for i, row in enumerate(group):
            centroid = np.asarray(row.get("centroid", ()), dtype=np.float64).reshape(-1)
            if centroid.size < 2 or not np.all(np.isfinite(centroid[:2])):
                continue
            region_index = sample.regions.index_for_xy(centroid[:2])
            if region_index is None:
                region_index = sample.regions.nearest_index(
                    centroid[:2], max_distance_m=1.5 * sample.regions.cell_size_m
                )
            if region_index is None:
                continue
            distance = _row_distance(row, sample.robot_xy)
            if not np.isfinite(distance):
                continue
            visibility = _row_visibility(row)
            rates = rate_table.lookup(distance, visibility)
            cid = _candidate_id(row, f"{key[0]}:{key[1]}:{key[2]}:{i}")
            # Make IDs unique within a step while preserving the logged ID.
            unique_cid = cid if cid not in row_by_id else f"{cid}:{i}"
            row_by_id[unique_cid] = row
            view = InspectionView(
                view_id=f"{unique_cid}:view",
                route_cost=float(distance),
                target_region=int(region_index),
                visibility=float(visibility),
                hit_probability=float(rates.hit_probability),
                false_alarm_probability=float(rates.false_alarm_probability),
                ambiguous_probability=float(rates.ambiguous_probability),
            )
            candidates.append((unique_cid, int(region_index), (view,)))
        if not candidates:
            continue
        try:
            scores = inference.score_candidates(sample.features, candidates)
        except Exception as exc:
            warnings.append(f"{key}: field scoring failed: {type(exc).__name__}: {exc}")
            continue
        by_id = {score.candidate_id: score for score in scores}
        for cid, row in row_by_id.items():
            score = by_id.get(cid)
            if score is None:
                continue
            baseline = row.get("sigma_c", row.get("conf", 0.0))
            try:
                baseline = float(baseline)
            except (TypeError, ValueError):
                baseline = 0.0
            scored.append(
                {
                    "key": key,
                    "candidate_id": cid,
                    "tp": bool(row.get("tp", False)),
                    "field_score": float(score.field_score),
                    "field_valid": bool(score.valid),
                    "field_reason": score.reason,
                    "field_entropy_bits": float(score.entropy_bits)
                    if np.isfinite(score.entropy_bits)
                    else None,
                    "field_gain_bits": float(score.expected_gain_bits),
                    "baseline_score": baseline,
                    "distance": float(_row_distance(row, sample.robot_xy)),
                    "visibility": float(_row_visibility(row)),
                    "region_index": int(score.region_index or 0),
                }
            )
    return scored, warnings


def _ranking_metrics(rows: Sequence[Mapping[str, Any]], score_key: str) -> Dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get(score_key) is not None and np.isfinite(float(row[score_key]))
    ]
    labels = [bool(row.get("tp", False)) for row in usable]
    scores = [float(row[score_key]) for row in usable]
    by_step: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in usable:
        by_step[row["key"]].append(row)
    top1_hits = 0
    top1_positive_groups = 0
    top1_groups_with_tp = 0
    recalls: Dict[int, int] = collections.Counter()
    positive_groups = 0
    for group in by_step.values():
        ranked = sorted(group, key=lambda row: float(row[score_key]), reverse=True)
        has_tp = any(bool(row.get("tp", False)) for row in ranked)
        if not ranked:
            continue
        if has_tp:
            positive_groups += 1
            if bool(ranked[0].get("tp", False)):
                top1_positive_groups += 1
        for k in (1, 3, 5):
            if any(bool(row.get("tp", False)) for row in ranked[:k]):
                recalls[k] += 1
    n_groups = len(by_step)
    return {
        "n_rows": len(usable),
        "n_positive_rows": sum(labels),
        "n_groups": n_groups,
        "n_positive_groups": positive_groups,
        "pairwise_auc": _auc(labels, scores),
        "top1_tp_rate_all_groups": float(top1_positive_groups / n_groups) if n_groups else 0.0,
        "top1_tp_rate_positive_groups": float(top1_positive_groups / positive_groups)
        if positive_groups
        else 0.0,
        "recall_at_1": float(recalls[1] / positive_groups) if positive_groups else 0.0,
        "recall_at_3": float(recalls[3] / positive_groups) if positive_groups else 0.0,
        "recall_at_5": float(recalls[5] / positive_groups) if positive_groups else 0.0,
    }


# ---------------------------------------------------------------------------
# coverage replay
# ---------------------------------------------------------------------------
def _coverage_for_frames(
    samples: Sequence[FrameSample],
    inference: FieldInference,
    top_k: int,
) -> Dict[str, Any]:
    hit_frames = 0
    evaluated_frames = 0
    frames_with_target = 0
    target_unavailable_frames = 0
    first_hits: List[int] = []
    for sample in samples:
        if sample.features.shape[0] == 0:
            continue
        evaluated_frames += 1
        if not sample.target_indices:
            target_unavailable_frames += 1
            continue
        frames_with_target += 1
        posterior = inference.posterior(sample.features)
        known = posterior[:-1]
        k = min(int(top_k), max(0, known.size))
        if k <= 0:
            continue
        top = set(np.argsort(-known)[:k].tolist())
        if top.intersection(sample.target_indices):
            hit_frames += 1
            if not first_hits or sample.step < first_hits[-1]:
                first_hits.append(int(sample.step))
    return {
        "evaluated_frames": evaluated_frames,
        "frames_with_target_region": frames_with_target,
        "frames_target_not_in_region_set": target_unavailable_frames,
        "top_k": int(top_k),
        "hit_frames": hit_frames,
        "frame_hit_rate": float(hit_frames / frames_with_target) if frames_with_target else 0.0,
        "first_hit_step": min(first_hits) if first_hits else None,
    }


def _episode_has_tp(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int], bool]:
    out: Dict[Tuple[str, int], bool] = collections.defaultdict(bool)
    for row in rows:
        if not row.get("tp"):
            continue
        try:
            out[
                (
                    scene_key(row.get("scene")),
                    int(row.get("episode_id", -1)),
                )
            ] = True
        except (TypeError, ValueError):
            continue
    return out


def coverage_report(
    frame_samples: Sequence[FrameSample],
    inferred_samples: Sequence[FrameSample],
    inference: FieldInference,
    candidate_rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    by_episode: Dict[Tuple[str, int], List[FrameSample]] = collections.defaultdict(list)
    for sample in frame_samples:
        by_episode[(sample.scene, sample.episode_id)].append(sample)
    has_tp = _episode_has_tp(candidate_rows)
    no_tp = [
        sample
        for sample in inferred_samples
        if not has_tp.get((sample.scene, sample.episode_id), False)
    ]
    return {
        "all": _coverage_for_frames(frame_samples, inference, top_k),
        "no_tp_candidate_episodes": _coverage_for_frames(no_tp, inference, top_k),
        "episodes": len(by_episode),
        "episodes_with_tp_candidate": sum(1 for v in has_tp.values() if v),
    }


# ---------------------------------------------------------------------------
# exploration replay
# ---------------------------------------------------------------------------
def _region_index_for_xy(sample: FrameSample, xy: Any) -> Optional[int]:
    point = np.asarray(xy, dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    index = sample.regions.index_for_xy(point[:2])
    if index is not None:
        return int(index)
    return sample.regions.nearest_index(
        point[:2], max_distance_m=1.5 * sample.regions.cell_size_m
    )


def _exploration_for_samples(
    samples: Sequence[FrameSample],
    inference: FieldInference,
    candidate_rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    """Score exploration properties that candidate ranking does not capture.

    The metrics answer three different questions:

    * coverage: does the field Top-K contain the GT target region?
    * direction: how spread out are the selected regions around the robot?
    * diversity: are selected regions outside the current TSP3D candidate set
      and how much do they overlap frontier cells?
    """
    candidate_by_step = _group_by_step(candidate_rows)
    evaluated = 0
    target_frames = 0
    target_hits = 0
    target_top1_hits = 0
    target_hits_outside_candidates = 0
    target_missing = 0
    novelty_sum = 0.0
    frontier_overlap_sum = 0.0
    angular_dispersion_sum = 0.0
    mean_distance_sum = 0.0
    entropy_sum = 0.0
    margin_sum = 0.0
    first_hits: List[int] = []

    for sample in samples:
        if sample.features.shape[0] == 0:
            continue
        posterior = inference.posterior(sample.features)
        known = np.asarray(posterior[:-1], dtype=np.float64)
        if known.size == 0 or not np.all(np.isfinite(known)):
            continue
        k = min(int(top_k), int(known.size))
        if k <= 0:
            continue
        evaluated += 1
        order = np.argsort(-known)[:k]
        top = set(int(i) for i in order.tolist())
        top1 = int(order[0])

        key = (sample.scene, sample.episode_id, sample.step)
        candidate_regions = set()
        for row in candidate_by_step.get(key, []):
            index = _region_index_for_xy(sample, row.get("centroid"))
            if index is not None:
                candidate_regions.add(index)

        if sample.target_indices:
            target_frames += 1
            hit_targets = top.intersection(sample.target_indices)
            if hit_targets:
                target_hits += 1
                first_hits.append(int(sample.step))
                if top1 in sample.target_indices:
                    target_top1_hits += 1
                if not hit_targets.intersection(candidate_regions):
                    target_hits_outside_candidates += 1
        else:
            target_missing += 1

        novelty_sum += float(len(top - candidate_regions) / max(1, len(top)))

        frontier_regions = set()
        for xy in np.asarray(sample.frontiers_xy, dtype=np.float64).reshape(-1, 2):
            index = _region_index_for_xy(sample, xy)
            if index is not None:
                frontier_regions.add(index)
        frontier_overlap_sum += float(len(top & frontier_regions) / max(1, len(top)))

        centers = np.asarray(sample.regions.centers_xy, dtype=np.float64)
        robot = np.asarray(sample.robot_xy, dtype=np.float64).reshape(-1)[:2]
        rel = centers[list(order)] - robot
        dists = np.linalg.norm(rel, axis=1)
        mean_distance_sum += float(dists.mean()) if dists.size else 0.0
        if rel.shape[0] >= 2:
            angles = np.arctan2(rel[:, 1], rel[:, 0])
            resultant = abs(np.mean(np.exp(1j * angles)))
            angular_dispersion_sum += float(1.0 - resultant)
        # Posterior entropy over known regions, normalized by log(n).
        total = float(known.sum())
        if total > 0.0:
            probs = np.clip(known / total, 1e-12, 1.0)
            entropy = float(-(probs * np.log(probs)).sum() / max(np.log(float(known.size)), 1e-12))
            entropy_sum += max(0.0, min(1.0, entropy))
        sorted_top = np.sort(known)[::-1]
        if sorted_top.size >= 2:
            margin_sum += float(sorted_top[0] - sorted_top[1])

    denom = max(1, evaluated)
    return {
        "top_k": int(top_k),
        "evaluated_frames": evaluated,
        "frames_with_target_region": target_frames,
        "frames_target_not_in_region_set": target_missing,
        "target_hit_frames": target_hits,
        "target_hit_rate": float(target_hits / target_frames) if target_frames else 0.0,
        "target_top1_frames": target_top1_hits,
        "target_top1_rate": float(target_top1_hits / target_frames) if target_frames else 0.0,
        "target_hit_outside_candidates_frames": target_hits_outside_candidates,
        "target_hit_outside_candidates_rate": (
            float(target_hits_outside_candidates / target_hits)
            if target_hits
            else 0.0
        ),
        "first_hit_step": min(first_hits) if first_hits else None,
        "candidate_novelty_rate": float(novelty_sum / denom),
        "frontier_overlap_rate": float(frontier_overlap_sum / denom),
        "angular_dispersion": float(angular_dispersion_sum / denom),
        "mean_topk_distance_m": float(mean_distance_sum / denom),
        "posterior_entropy": float(entropy_sum / denom),
        "top1_top2_margin": float(margin_sum / denom),
    }


def exploration_report(
    frame_samples: Sequence[FrameSample],
    inferred_samples: Sequence[FrameSample],
    inference: FieldInference,
    candidate_rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    """Report coverage/direction/diversity beyond candidate ranking."""
    has_tp = _episode_has_tp(candidate_rows)
    no_tp = [
        sample
        for sample in inferred_samples
        if not has_tp.get((sample.scene, sample.episode_id), False)
    ]
    return {
        "all": _exploration_for_samples(
            frame_samples, inference, candidate_rows, top_k
        ),
        "no_tp_candidate_episodes": _exploration_for_samples(
            no_tp, inference, candidate_rows, top_k
        ),
    }


# ---------------------------------------------------------------------------
# orchestration / report
# ---------------------------------------------------------------------------
def analyze(
    *,
    detections_dir: str,
    field_dir: Optional[str],
    out_dir: str,
    region_semantics_path: Optional[str],
    config: OfflineAnalysisConfig,
    verbose: bool = False,
) -> Dict[str, Any]:
    detections_dir = os.path.abspath(detections_dir)
    field_dir = field_dir or os.path.join(detections_dir, "field_observations")
    if region_semantics_path is None:
        candidate = os.path.join(detections_dir, "region_semantics.jsonl")
        if os.path.isfile(candidate):
            region_semantics_path = candidate
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logs = NavigationLogs.load(detections_dir)
    semantic = SemanticSidecar.load(region_semantics_path)
    warnings: List[str] = []
    if not semantic.available:
        warnings.append(
            "semantic region sidecar unavailable; fitting/evaluating geometry-only baseline"
        )
    feature_names = tuple(GEOMETRY_FEATURE_NAMES) + tuple(semantic.feature_names)
    field_episodes, load_errors = load_field_episodes(
        field_dir, max_episodes=config.max_episodes, verbose=verbose
    )
    warnings.extend(load_errors)
    frame_samples: List[FrameSample] = []
    region_samples: List[RegionSample] = []
    for episode in field_episodes:
        try:
            frames, regions, episode_warnings = build_episode_samples(
                episode,
                logs,
                cell_size_m=config.cell_size_m,
                semantic_sidecar=semantic,
                feature_names=feature_names,
            )
        except Exception as exc:
            warnings.append(
                f"{episode.scene}:{episode.episode_id}: feature build failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        frame_samples.extend(frames)
        region_samples.extend(regions)
        warnings.extend(episode_warnings)
        if verbose:
            print(
                f"[field] {episode.scene}:{episode.episode_id} "
                f"frames={len(frames)} regions={len(regions)}"
            )

    fit = fit_region_model(
        region_samples,
        feature_names,
        l2=config.l2,
        scene_cv=config.scene_cv,
    )
    if fit.get("status") != "ok":
        warnings.append(f"region model status: {fit.get('status')}")
    rate_table = ObservationRateTable.from_tracks(logs.tracks)
    if rate_table.global_bin.n_target + rate_table.global_bin.n_nontarget == 0:
        warnings.append("no visible track_frames; using conservative default observation rates")
        rate_table = ObservationRateTable.default()
    inference = FieldInference(
        FieldInferenceConfig(
            weights=tuple(float(v) for v in fit.get("weights", ())),
            region_bias=float(fit.get("region_bias", 0.0) or 0.0),
            unknown_logit=0.0,
            temperature=float(fit.get("temperature", 1.0) or 1.0),
            min_visibility=config.min_visibility,
            feature_names=feature_names,
        )
    )

    candidate_rows: Sequence[Mapping[str, Any]] = logs.candidates or logs.nav_goals
    scored_candidates: List[Dict[str, Any]] = []
    inferred_samples: List[FrameSample] = []
    if fit.get("status") == "ok" and frame_samples:
        inferred_samples = [
            sample
            for sample in frame_samples
            if sample.features.shape[0] > 0 and sample.target_indices
        ]
        scored_candidates, score_warnings = score_candidate_rows(
            {(s.scene, s.episode_id, s.step): s for s in frame_samples},
            candidate_rows,
            inference,
            rate_table,
        )
        warnings.extend(score_warnings)
        field_rows = [row for row in scored_candidates if row.get("field_valid", False)]
        candidate_metrics = {
            "field": _ranking_metrics(field_rows, "field_score"),
            "baseline_sigma_c": _ranking_metrics(field_rows, "baseline_score"),
            "n_scored_rows": len(scored_candidates),
            "n_field_valid_rows": len(field_rows),
            "n_invalid_field_rows": len(scored_candidates) - len(field_rows),
        }
        coverage = coverage_report(
            frame_samples,
            inferred_samples,
            inference,
            candidate_rows,
            config.top_k,
        )
        exploration = exploration_report(
            frame_samples,
            inferred_samples,
            inference,
            candidate_rows,
            config.top_k,
        )
    else:
        candidate_metrics = {
            "status": "skipped",
            "reason": f"region model {fit.get('status', 'unavailable')}",
        }
        coverage = {"status": "skipped", "reason": "region model unavailable"}
        exploration = {"status": "skipped", "reason": "region model unavailable"}

    inference_config = {
        "weights": list(fit.get("weights", ())),
        "region_bias": float(fit.get("region_bias", 0.0) or 0.0),
        "unknown_logit": 0.0,
        "temperature": float(fit.get("temperature", 1.0) or 1.0),
        "min_visibility": float(config.min_visibility),
        "feature_names": list(feature_names),
        "observation_rates": observation_rate_report(rate_table),
    }
    inference_config_path = os.path.join(out_dir, "field_inference_config.json")
    with open(inference_config_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(inference_config), handle, ensure_ascii=False, indent=2)
    candidate_scores_path = os.path.join(out_dir, "field_candidate_scores.jsonl")
    with open(candidate_scores_path, "w", encoding="utf-8") as handle:
        for row in scored_candidates:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    report: Dict[str, Any] = {
        "status": "ok" if field_episodes else "no_field_episodes",
        "inputs": {
            "detections_dir": detections_dir,
            "field_dir": field_dir,
            "region_semantics_path": region_semantics_path,
            "out_dir": out_dir,
        },
        "config": {
            "cell_size_m": config.cell_size_m,
            "top_k": config.top_k,
            "l2": config.l2,
            "min_visibility": config.min_visibility,
            "max_episodes": config.max_episodes,
            "scene_cv": config.scene_cv,
        },
        "data": {
            "field_episodes": len(field_episodes),
            "field_frames": len(frame_samples),
            "region_samples": len(region_samples),
            "region_labels_positive": sum(1 for s in region_samples if s.label == 1),
            "region_labels_negative": sum(1 for s in region_samples if s.label == 0),
            "semantic_sidecar_available": semantic.available,
            "semantic_sidecar_rows": len(semantic.rows),
            "candidate_rows": len(logs.candidates),
            "track_rows": len(logs.tracks),
            "nav_goal_rows": len(logs.nav_goals),
        },
        "feature_names": list(feature_names),
        "region_model": fit,
        "observation_rates": observation_rate_report(rate_table),
        "inference_config_path": inference_config_path,
        "candidate_scores_path": candidate_scores_path,
        "candidate_ranking": candidate_metrics,
        "coverage": coverage,
        "exploration": exploration,
        "warnings": warnings[:200],
    }
    report = _jsonable(report)
    report_path = os.path.join(out_dir, "field_offline_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary_path = os.path.join(out_dir, "field_offline_summary.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(_markdown_summary(report))
    report["report_path"] = report_path
    report["summary_path"] = summary_path
    return report


def _markdown_summary(report: Mapping[str, Any]) -> str:
    lines = ["# Semantic-spatial field offline analysis", ""]
    data = report.get("data", {})
    lines.append(f"- status: `{report.get('status')}`")
    lines.append(f"- field episodes: `{data.get('field_episodes')}`")
    lines.append(f"- field frames: `{data.get('field_frames')}`")
    lines.append(f"- region samples: `{data.get('region_samples')}`")
    lines.append(
        f"- labels: `+{data.get('region_labels_positive')}` / "
        f"`-{data.get('region_labels_negative')}`"
    )
    lines.append(
        f"- semantic sidecar: `{data.get('semantic_sidecar_available')}` "
        f"({data.get('semantic_sidecar_rows')} rows)"
    )
    if report.get("inference_config_path"):
        lines.append(f"- inference config: `{report.get('inference_config_path')}`")
    if report.get("candidate_scores_path"):
        lines.append(f"- candidate scores: `{report.get('candidate_scores_path')}`")
    fit = report.get("region_model", {})
    lines.extend(["", "## Region model", ""])
    lines.append(f"- status: `{fit.get('status')}`")
    if fit.get("status") == "ok":
        lines.append(f"- weights: `{fit.get('weights')}`")
        lines.append(f"- intercept: `{fit.get('region_bias')}`")
        train = fit.get("train_metrics", {})
        lines.append(
            f"- train AUC: `{train.get('auc')}`, Brier: `{train.get('brier')}`, "
            f"NLL: `{train.get('nll')}`"
        )
        cv = fit.get("scene_cv", [])
        if cv:
            lines.append(f"- scene-grouped folds: `{len(cv)}`")
            for fold in cv:
                lines.append(
                    f"  - `{fold.get('held_out_scene')}`: "
                    f"AUC=`{fold.get('auc')}`, n=`{fold.get('n_test')}`"
                )
    ranking = report.get("candidate_ranking", {})
    lines.extend(["", "## Candidate ranking", ""])
    if ranking.get("status") == "skipped":
        lines.append(f"- skipped: {ranking.get('reason')}")
    else:
        for name in ("field", "baseline_sigma_c"):
            table = ranking.get(name, {})
            lines.append(
                f"- {name}: AUC=`{table.get('pairwise_auc')}`, "
                f"Top1=`{table.get('top1_tp_rate_positive_groups')}`, "
                f"R@1=`{table.get('recall_at_1')}`, R@3=`{table.get('recall_at_3')}`, "
                f"R@5=`{table.get('recall_at_5')}`"
            )
    coverage = report.get("coverage", {})
    lines.extend(["", "## Coverage", ""])
    if coverage.get("status") == "skipped":
        lines.append(f"- skipped: {coverage.get('reason')}")
    else:
        for name in ("all", "no_tp_candidate_episodes"):
            table = coverage.get(name, {})
            lines.append(
                f"- {name}: hit=`{table.get('frame_hit_rate')}`, "
                f"frames=`{table.get('frames_with_target_region')}`, "
                f"target_missing=`{table.get('frames_target_not_in_region_set')}`"
            )
    exploration = report.get("exploration", {})
    lines.extend(["", "## Exploration", ""])
    if exploration.get("status") == "skipped":
        lines.append(f"- skipped: {exploration.get('reason')}")
    else:
        for name in ("all", "no_tp_candidate_episodes"):
            table = exploration.get(name, {})
            lines.append(
                f"- {name}: target_hit=`{table.get('target_hit_rate')}`, "
                f"target_top1=`{table.get('target_top1_rate')}`, "
                f"outside_candidates=`{table.get('target_hit_outside_candidates_rate')}`, "
                f"novelty=`{table.get('candidate_novelty_rate')}`, "
                f"frontier_overlap=`{table.get('frontier_overlap_rate')}`, "
                f"angular_dispersion=`{table.get('angular_dispersion')}`, "
                f"mean_distance=`{table.get('mean_topk_distance_m')}`"
            )
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:30])
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline semantic-spatial field fitting and replay "
        "(does not start navigation or modify the policy)."
    )
    parser.add_argument("--run-dir", default=None, help="run directory containing detections/")
    parser.add_argument("--detections-dir", default=None, help="directory containing JSONL logs")
    parser.add_argument(
        "--field-dir",
        default=None,
        help="field_observations directory (default: <detections>/field_observations)",
    )
    parser.add_argument("--region-semantics", default=None, help="optional region_semantics.jsonl")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--min-visibility", type=float, default=0.15)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-scene-cv", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.detections_dir:
        detections_dir = args.detections_dir
    elif args.run_dir:
        detections_dir = os.path.join(args.run_dir, "detections")
    else:
        parser.error("provide --detections-dir or --run-dir")
        return 2
    out_dir = args.out or os.path.join(detections_dir, "field_offline")
    config = OfflineAnalysisConfig(
        cell_size_m=args.cell_size,
        top_k=args.top_k,
        l2=args.l2,
        min_visibility=args.min_visibility,
        max_episodes=args.max_episodes,
        scene_cv=not args.no_scene_cv,
    )
    report = analyze(
        detections_dir=detections_dir,
        field_dir=args.field_dir,
        out_dir=out_dir,
        region_semantics_path=args.region_semantics,
        config=config,
        verbose=args.verbose,
    )
    print(f"field offline report: {report['report_path']}")
    print(f"field offline summary: {report['summary_path']}")
    if report.get("status") == "no_field_episodes":
        print("no field_observations found; report contains diagnostics only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
