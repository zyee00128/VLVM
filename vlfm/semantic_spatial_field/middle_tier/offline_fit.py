"""Offline fit for the S7 middle-tier promotion-probability ranker.

This module is intentionally NOT imported by ``middle_tier/__init__.py`` or by
the policy.  Run it as a CLI after a shadow run::

    python -m vlfm.semantic_spatial_field.middle_tier.offline_fit \
        --run-dir outputs/<run> --K 10

It reads ``detections/middle_rank_shadow.jsonl`` plus
``detections/memory_events.jsonl`` / ``episodes.jsonl`` and writes a small
JSON model.  It never starts Habitat, TSP3D, a VLM server, or the policy.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import fit_logistic_weights
from .features import FEATURE_GROUPS, MIDDLE_TIER_TYPES
from .ranker import MODEL_FORMAT, logit, submodel_predict

_EPS = 1e-6


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _key(scene: Any, episode: Any, memory: Any) -> Optional[Tuple[str, str, str]]:
    if memory in (None, ""):
        return None
    return (
        str(scene or ""),
        str(episode if episode is not None else ""),
        str(memory),
    )


def _episode_key(scene: Any, episode: Any) -> Tuple[str, str]:
    return (
        str(scene or ""),
        str(episode if episode is not None else ""),
    )


def _numeric(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_promotion_event(event: Dict[str, Any], type_name: str) -> bool:
    """Return True for a middle-tier promotion event.

    ``gd-only`` promotes via an ``upgrade`` event whose ``upgrade_reason`` is
    ``gd_only_tsp_merge``.  ``tsp-confirmed`` promotes via the
    ``gd_tsp_merge`` event.  Accept both event-name forms for forward
    compatibility with older/newer logs.
    """
    name = str(event.get("event") or "")
    reason = str(event.get("upgrade_reason") or "")
    if type_name == "gd-only":
        return name == "gd_only_tsp_merge" or (
            name == "upgrade" and reason == "gd_only_tsp_merge"
        )
    return name == "gd_tsp_merge" or (
        name == "upgrade" and reason == "gd_tsp"
    )


def _collect_samples(
    run_dir: Path, window: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    det = run_dir / "detections"
    shadow = _read_jsonl(det / "middle_rank_shadow.jsonl")
    events = _read_jsonl(det / "memory_events.jsonl")
    episodes = _read_jsonl(det / "episodes.jsonl")

    step_by_episode: Dict[Tuple[str, str], int] = {}
    for row in episodes:
        key = _episode_key(row.get("scene"), row.get("episode_id"))
        try:
            step_by_episode[key] = int(row.get("steps_count") or 0)
        except (TypeError, ValueError):
            pass

    events_by_target: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for event in events:
        key = _key(
            event.get("scene"), event.get("episode_id"), event.get("memory_id")
        )
        if key is not None:
            events_by_target[key].append(event)

    targets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in shadow:
        if str(row.get("type") or "") not in MIDDLE_TIER_TYPES:
            continue
        key = _key(row.get("scene"), row.get("episode_id"), row.get("memory_id"))
        if key is not None:
            targets[key].append(row)

    samples: List[Dict[str, Any]] = []
    skipped = collections.Counter()
    for key, rows in targets.items():
        rows.sort(key=lambda row: int(row.get("step", -1)))
        first = next((row for row in rows if bool(row.get("selected"))), None)
        if first is None:
            skipped["never_selected"] += 1
            continue
        type_name = str(first.get("type") or "")
        t0 = int(first.get("step", -1))
        future = [
            event
            for event in events_by_target.get(key, [])
            if t0 < int(event.get("step", -1)) <= t0 + int(window)
            and _is_promotion_event(event, type_name)
        ]
        if not future:
            end = step_by_episode.get(_episode_key(key[0], key[1]))
            if end is not None and end < t0 + int(window):
                skipped["censored"] += 1
                continue
        features = first.get("features") or {}
        if not isinstance(features, dict):
            skipped["missing_features"] += 1
            continue
        samples.append(
            {
                "type": type_name,
                "target_class": str(first.get("target_class") or ""),
                "scene": key[0],
                "episode_id": key[1],
                "memory_id": key[2],
                "step": t0,
                "features": features,
                "label": 1 if future else 0,
            }
        )
    return samples, dict(skipped)


def _auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positive = [score for label, score in zip(labels, scores) if int(label) == 1]
    negative = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not positive or not negative:
        return None
    pairs = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                pairs += 1.0
            elif pos == neg:
                pairs += 0.5
    return pairs / (len(positive) * len(negative))


def _fit_group(
    samples: Sequence[Dict[str, Any]], group: str, l2: float
) -> Optional[Dict[str, Any]]:
    raw_names: List[str] = []
    seen = set()
    for sample in samples:
        group_values = (sample.get("features") or {}).get(group) or {}
        for name in group_values:
            if name not in seen:
                seen.add(name)
                raw_names.append(str(name))
    raw_names.sort()
    if not raw_names:
        return None

    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.float64)
    if labels.min() == labels.max():
        return None

    raw_matrix = np.full((len(samples), len(raw_names)), np.nan, dtype=np.float64)
    for row_index, sample in enumerate(samples):
        group_values = (sample.get("features") or {}).get(group) or {}
        for col_index, name in enumerate(raw_names):
            value = _numeric(group_values.get(name))
            if value is not None:
                raw_matrix[row_index, col_index] = value

    medians = np.zeros(len(raw_names), dtype=np.float64)
    for col_index in range(len(raw_names)):
        values = raw_matrix[:, col_index]
        finite = values[np.isfinite(values)]
        medians[col_index] = float(np.median(finite)) if finite.size else 0.0
    missing_mask = ~np.isfinite(raw_matrix)
    filled = np.where(missing_mask, medians.reshape(1, -1), raw_matrix)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales[scales < 1e-9] = 1.0
    standardized = (filled - means) / scales

    missing_names = [
        f"__missing__.{group}.{name}"
        for name in raw_names
        if bool(missing_mask[:, raw_names.index(name)].any())
    ]
    missing_columns = np.zeros((len(samples), len(missing_names)), dtype=np.float64)
    for col, name in enumerate(missing_names):
        base = name[len(f"__missing__.{group}.") :]
        base_index = raw_names.index(base)
        missing_columns[:, col] = missing_mask[:, base_index].astype(np.float64)

    design = np.concatenate([standardized, missing_columns], axis=1)
    coefficients, intercept = fit_logistic_weights(design, labels, l2=float(l2))
    names = [f"{group}.{name}" for name in raw_names] + missing_names
    return {
        "features": names,
        "mean": list(map(float, means)) + [0.0] * len(missing_names),
        "scale": list(map(float, scales)) + [1.0] * len(missing_names),
        "coef": list(map(float, coefficients)),
        "intercept": float(intercept),
    }


def _fit_type(
    samples: Sequence[Dict[str, Any]], type_name: str, l2: float
) -> Optional[Dict[str, Any]]:
    if len(samples) < 4:
        return None
    labels = [int(sample["label"]) for sample in samples]
    if min(labels) == max(labels):
        return None

    groups: Dict[str, Any] = {}
    group_probabilities: Dict[str, List[float]] = {}
    for group in FEATURE_GROUPS:
        submodel = _fit_group(samples, group, l2)
        if submodel is None:
            group_probabilities[group] = [0.5] * len(samples)
            continue
        groups[group] = submodel
        group_probabilities[group] = [
            submodel_predict(submodel, sample.get("features") or {})
            for sample in samples
        ]

    combine_rows = np.asarray(
        [
            [logit(group_probabilities[group][index]) for group in FEATURE_GROUPS]
            for index in range(len(samples))
        ],
        dtype=np.float64,
    )
    coefficients, intercept = fit_logistic_weights(
        combine_rows, np.asarray(labels, dtype=np.float64), l2=0.1
    )
    combine = {
        "coef": {
            group: float(coefficients[index])
            for index, group in enumerate(FEATURE_GROUPS)
        },
        "intercept": float(intercept),
    }
    probabilities = [
        1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, float(combine_rows[index] @ coefficients + intercept)))))
        for index in range(len(samples))
    ]
    return {
        "groups": groups,
        "combine": combine,
        "metadata": {
            "n_samples": len(samples),
            "positives": int(sum(labels)),
            "fit_mode": "in_sample",
            "auc_in_sample": _auc(labels, probabilities),
            "target_class_counts": dict(
                collections.Counter(sample.get("target_class") for sample in samples)
            ),
            "scene_counts": dict(
                collections.Counter(sample.get("scene") for sample in samples)
            ),
        },
    }


def fit_model(
    run_dir: Path,
    *,
    window: int = 10,
    l2: float = 1.0,
    version: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    samples, skipped = _collect_samples(run_dir, window)
    by_type: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for sample in samples:
        by_type[str(sample["type"])].append(sample)

    model: Dict[str, Any] = {
        "format": MODEL_FORMAT,
        "metadata": {
            "K": int(window),
            "version": str(version or ""),
            "fit_mode": "in_sample",
            "n_samples": len(samples),
            "skipped": skipped,
        },
        "types": {},
    }
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "K": int(window),
        "n_samples": len(samples),
        "skipped": skipped,
        "types": {},
    }
    for type_name in MIDDLE_TIER_TYPES:
        type_samples = by_type.get(type_name, [])
        fitted = _fit_type(type_samples, type_name, l2)
        if fitted is None:
            summary["types"][type_name] = {
                "status": "skipped",
                "n_samples": len(type_samples),
            }
            continue
        model["types"][type_name] = {
            "groups": fitted["groups"],
            "combine": fitted["combine"],
        }
        summary["types"][type_name] = {
            "status": "fitted",
            **fitted["metadata"],
        }
    return model, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the S7 middle-tier promotion-probability ranker."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--version", default="")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "middle_tier" / "middle_tier_model.json"
    summary_out = (
        Path(args.summary_out)
        if args.summary_out
        else run_dir / "middle_tier" / "middle_tier_fit_summary.json"
    )
    model, summary = fit_model(
        run_dir, window=int(args.K), l2=float(args.l2), version=str(args.version)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, ensure_ascii=False)
    with summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
