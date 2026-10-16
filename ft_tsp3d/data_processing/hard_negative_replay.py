#!/usr/bin/env python3
"""训练场景 online hard-negative replay 数据处理。

本文件支持两步：
用已完成的 R1 checkpoint 在 train local-view 上逐 record 挖掘在线形态 hard negative capture；
再做纯 CPU 去重/配平/导出。它不启动 Habitat 导航或训练。

原始 capture schema（每个在线 view/step 一个 ``.npz``）
-------------------------------------------------------
```
points:    float16/float32 (N, >=6)，TSP3D canonical 局部点云；或省略并在 meta 中提供
           ``points_source={"npz": ..., "view": ...}``，引用原 local-view 切片
meta_json: JSON 字符串：
{
  "schema_version": 1,
  "sample_id": "scene/episode/step",       # 可省略，默认相对文件名
  "split": "train",
  "scene": "00006-HkseAnWCgqk",
  "episode": "12",
  "step": 37,
  "poses": [...],
  "records": [
    {
      "query": "chair",
      "label": 0 | 1,
      "gt_box": null | [cx, cy, cz, lx, ly, lz],
      "candidates": [
        {"score": 0.91, "box": [cx, cy, cz, lx, ly, lz], "reason": "optional"}
      ]
    }
  ]
}
```

``box``/``gt_box`` 也接受 ``{"center": [...], "size": [...]}``、
``{"box": ...}`` 或 8×3 corners。分数键兼容 ``score/raw_score/confidence``。

筛选与输出
----------
1. 仅接收 ``splits.json`` 的 train scene；val/评测/未知 scene 默认全部拒绝，防止泄漏。
2. ``score >= min_score`` 且：target-absent query，或正 query 的候选 IoU < ``max_fp_iou``。
3. 同一 sample/query 只保留最危险候选；同 scene/query/episode 的近邻重复误框按中心距离/3D IoU 去重。
4. 先限制 scene×class，再按 scene 轮询生成 class 队列，最后按 class 轮询；
   避免单场景、单类别或长轨迹支配 replay。
5. 输出 ``train/<scene>.npz``，schema 与 ``LocalViewDataset`` 主数据完全兼容；
   每条 record 保留 ``hard_negative`` 审计元数据。
   另写 ``selection.jsonl`` / ``stats.json`` / ``splits.json``。

示例
----
```
python ft_tsp3d/data_processing/hard_negative_replay.py \
  --mine-checkpoint data/tsp3d_ft/rank/r1_5080_ckpt_last.pth \
  --input-dir data/ft_datasets/r2_capture \
  --output-dir data/ft_datasets/hard_negative_replay \
  --splits-json data/ft_datasets/local_view/splits.json

python ft_tsp3d/data_processing/hard_negative_replay.py --self-test
```
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_HERE = Path(__file__).resolve().parent
_FT_ROOT = _HERE.parent
_REPO_ROOT = _FT_ROOT.parent
for _path in (str(_FT_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


DEFAULT_CLASSES: Tuple[str, ...] = (
    "chair", "bed", "potted plant", "toilet", "tv", "couch",
)
SCHEMA_VERSION = 1
_SAFE_SCENE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ReplayConfig:
    min_score: float = 0.50
    max_fp_iou: float = 0.25
    min_points: int = 64
    dedup_center_m: float = 0.50
    dedup_iou: float = 0.50
    dedup_step_window: int = 8
    max_total: int = 6000
    max_per_class: int = 1000
    max_per_scene_class: int = 25
    max_per_scene: int = 100
    replay_weight: float = 1.0
    classes: Tuple[str, ...] = DEFAULT_CLASSES

    def validate(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if not 0.0 <= self.max_fp_iou <= 1.0:
            raise ValueError("max_fp_iou must be in [0, 1]")
        if self.min_points < 1:
            raise ValueError("min_points must be positive")
        if self.dedup_center_m < 0.0 or not 0.0 <= self.dedup_iou <= 1.0:
            raise ValueError("invalid dedup thresholds")
        for name in ("max_total", "max_per_class", "max_per_scene_class", "max_per_scene"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0 (0 means unlimited)")
        if self.replay_weight <= 0.0:
            raise ValueError("replay_weight must be positive")
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be non-empty and unique")


@dataclass
class HardCandidate:
    source: Path
    sample_id: str
    scene: str
    episode: str
    step: int
    query: str
    label: int
    gt_box: Optional[np.ndarray]
    pred_box: np.ndarray
    score: float
    pred_iou: float
    reason: str
    poses: List[Dict[str, Any]]
    n_points: int
    candidate_rank: int

    @property
    def hardness_key(self) -> Tuple[float, float, int, str]:
        # 高分、低 IoU 优先；其余字段只负责确定性 tie-break。
        return (-self.score, self.pred_iou, self.step, self.sample_id)

    def audit_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": str(self.source),
            "scene": self.scene,
            "episode": self.episode,
            "step": self.step,
            "query": self.query,
            "label": self.label,
            "score": self.score,
            "pred_iou": self.pred_iou,
            "pred_box": self.pred_box.tolist(),
            "gt_box": None if self.gt_box is None else self.gt_box.tolist(),
            "reason": self.reason,
            "n_points": self.n_points,
            "candidate_rank": self.candidate_rank,
        }


def _first(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def canonical_box(value: Any) -> Optional[np.ndarray]:
    """常见 box 表示转 canonical ``[cx,cy,cz,lx,ly,lz]``。"""
    if value is None:
        return None
    if isinstance(value, dict):
        if "box" in value:
            return canonical_box(value["box"])
        if "center" in value and "size" in value:
            value = list(value["center"]) + list(value["size"])
        elif "corners" in value:
            value = value["corners"]
        else:
            return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (6,):
        box = arr
    elif arr.shape == (8, 3):
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        box = np.concatenate(((lo + hi) * 0.5, hi - lo))
    else:
        return None
    if not np.isfinite(box).all() or np.any(box[3:] <= 0.0):
        return None
    return box.astype(np.float32)


def box_iou3d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(6)
    b = np.asarray(b, dtype=np.float64).reshape(6)
    alo, ahi = a[:3] - a[3:] * 0.5, a[:3] + a[3:] * 0.5
    blo, bhi = b[:3] - b[3:] * 0.5, b[:3] + b[3:] * 0.5
    inter = np.maximum(0.0, np.minimum(ahi, bhi) - np.maximum(alo, blo))
    vi = float(np.prod(inter))
    va, vb = float(np.prod(a[3:])), float(np.prod(b[3:]))
    return vi / max(va + vb - vi, 1e-12)


def normalize_scene(value: Any) -> str:
    scene = Path(str(value or "")).name
    for suffix in (".basis.glb", ".semantic.glb", ".glb", ".json"):
        if scene.endswith(suffix):
            scene = scene[:-len(suffix)]
            break
    if not scene or not _SAFE_SCENE.fullmatch(scene):
        raise ValueError(f"unsafe/empty scene id: {value!r}")
    return scene


def _json_scalar(value: Any) -> Dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    data = json.loads(str(value))
    if not isinstance(data, dict):
        raise ValueError("meta_json must decode to an object")
    return data


def load_capture(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" not in data:
            raise ValueError("capture requires meta_json")
        meta = _json_scalar(data["meta_json"])
        points = np.asarray(data["points"]) if "points" in data else None
    if points is None:
        source = meta.get("points_source")
        if not isinstance(source, dict) or "npz" not in source or "view" not in source:
            raise ValueError("capture without points requires points_source {npz, view}")
        source_path = Path(str(source["npz"])).expanduser()
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        view = int(source["view"])
        declared_view = meta.get("source_view_index")
        if declared_view is not None and int(declared_view) != view:
            raise ValueError(
                f"points_source view mismatch: {view} != source_view_index {declared_view}"
            )
        with np.load(source_path, allow_pickle=False) as data:
            source_points = np.asarray(data["points"])
            offsets = np.asarray(data["offsets"], dtype=np.int64)
        if view < 0 or view + 1 >= len(offsets):
            raise ValueError(f"points_source view out of range: {view}")
        points = source_points[offsets[view]:offsets[view + 1]]
    if points.ndim != 2 or points.shape[1] < 6:
        raise ValueError(f"points must be (N,>=6), got {points.shape}")
    points = points[:, :6]
    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite values")
    return points, meta


def write_capture(path: Path, points: np.ndarray, meta: Dict[str, Any]) -> None:
    """供 rollout 采集端复用的原子落盘 API。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 6 or not np.isfinite(pts[:, :6]).all():
        raise ValueError("points must be finite (N,>=6)")
    payload = dict(meta)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_name,
            points=np.ascontiguousarray(pts[:, :6], dtype=np.float16),
            meta_json=np.asarray(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        )
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_reference_capture(
    path: Path,
    meta: Dict[str, Any],
    source_npz: Path,
    source_view: int,
) -> None:
    """落盘轻量 capture：只保存候选元数据并引用原 local-view 点云切片。"""
    path = Path(path)
    source_npz = Path(source_npz).resolve()
    if not source_npz.is_file():
        raise FileNotFoundError(f"source local-view npz not found: {source_npz}")
    payload = dict(meta)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["source_view_index"] = int(source_view)
    payload["points_source"] = {"npz": str(source_npz), "view": int(source_view)}
    _atomic_npz(
        path,
        meta_json=np.asarray(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    )


def load_train_scenes(path: Path) -> set:
    data = json.loads(path.read_text())
    scenes = {normalize_scene(item) for item in data.get("train", [])}
    if not scenes:
        raise ValueError(f"no train scenes in {path}")
    return scenes


def extract_candidates(
    path: Path,
    input_root: Path,
    train_scenes: set,
    cfg: ReplayConfig,
    rejected: Counter,
) -> List[HardCandidate]:
    try:
        points, meta = load_capture(path)
    except Exception as exc:  # noqa: BLE001
        rejected[f"capture_invalid:{type(exc).__name__}"] += 1
        return []
    if int(meta.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        rejected["schema_version"] += 1
        return []
    try:
        scene = normalize_scene(meta.get("scene"))
    except ValueError:
        rejected["scene_invalid"] += 1
        return []
    if str(meta.get("split", "train")) != "train":
        rejected["non_train_split"] += 1
        return []
    if scene not in train_scenes:
        rejected["scene_not_in_train_split"] += 1
        return []
    if len(points) < cfg.min_points:
        rejected["too_few_points"] += 1
        return []

    rel = path.relative_to(input_root).as_posix()
    sample_id = str(meta.get("sample_id") or rel)
    episode = str(meta.get("episode", ""))
    step = int(meta.get("step", -1))
    poses = list(meta.get("poses") or [])
    records = meta.get("records")
    if not isinstance(records, list):
        rejected["records_invalid"] += 1
        return []

    out: List[HardCandidate] = []
    allowed = set(cfg.classes)
    for record in records:
        if not isinstance(record, dict):
            rejected["record_invalid"] += 1
            continue
        query = str(record.get("query", "")).strip().lower()
        if query not in allowed:
            rejected["query_not_allowed"] += 1
            continue
        label = int(record.get("label", 0))
        if label not in (0, 1):
            rejected["label_invalid"] += 1
            continue
        gt_raw = _first(record, ("gt_box", "gt", "target"))
        gt_box = canonical_box(gt_raw)
        if label == 1 and gt_box is None:
            rejected["positive_without_gt"] += 1
            continue
        if label == 0:
            gt_box = None
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            rejected["candidates_invalid"] += 1
            continue
        for rank, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                rejected["candidate_invalid"] += 1
                continue
            score_raw = _first(candidate, ("score", "raw_score", "confidence", "conf"))
            box_raw = _first(candidate, ("box", "pred_box", "box_3d", "corners"))
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                rejected["score_invalid"] += 1
                continue
            pred_box = canonical_box(box_raw)
            if pred_box is None or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                rejected["prediction_invalid"] += 1
                continue
            if score < cfg.min_score:
                rejected["below_min_score"] += 1
                continue
            pred_iou = 0.0 if gt_box is None else box_iou3d(pred_box, gt_box)
            if label == 1 and pred_iou >= cfg.max_fp_iou:
                rejected["not_false_positive"] += 1
                continue
            reason = str(candidate.get("reason") or (
                "target_absent" if label == 0 else "low_iou"))
            out.append(HardCandidate(
                source=path,
                sample_id=sample_id,
                scene=scene,
                episode=episode,
                step=step,
                query=query,
                label=label,
                gt_box=gt_box,
                pred_box=pred_box,
                score=score,
                pred_iou=pred_iou,
                reason=reason,
                poses=poses,
                n_points=int(len(points)),
                candidate_rank=int(candidate.get("candidate_rank", rank)),
            ))
    return out


def _is_duplicate(a: HardCandidate, b: HardCandidate, cfg: ReplayConfig) -> bool:
    if a.scene != b.scene or a.query != b.query or a.episode != b.episode:
        return False
    # 缺 episode id 时无法证明来自同一条轨迹；同 sample/query 已在上一层折叠，
    # 这里宁可保留不同 capture，也不跨未知轨迹误删。
    if not a.episode:
        return False
    if a.step >= 0 and b.step >= 0 and abs(a.step - b.step) > cfg.dedup_step_window:
        return False
    center_dist = float(np.linalg.norm(a.pred_box[:3] - b.pred_box[:3]))
    return center_dist <= cfg.dedup_center_m or box_iou3d(a.pred_box, b.pred_box) >= cfg.dedup_iou


def deduplicate(candidates: Sequence[HardCandidate], cfg: ReplayConfig) -> List[HardCandidate]:
    # 同一个模型调用的同 query 只需回放一次；保留最高分/最低 IoU 候选作为审计代表。
    per_sample: Dict[Tuple[str, str], HardCandidate] = {}
    for item in sorted(candidates, key=lambda x: x.hardness_key):
        per_sample.setdefault((item.sample_id, item.query), item)

    kept: List[HardCandidate] = []
    groups: Dict[Tuple[str, str, str], List[HardCandidate]] = defaultdict(list)
    for item in per_sample.values():
        groups[(item.scene, item.query, item.episode)].append(item)
    for key in sorted(groups):
        group_kept: List[HardCandidate] = []
        for item in sorted(groups[key], key=lambda x: x.hardness_key):
            if not any(_is_duplicate(item, old, cfg) for old in group_kept):
                group_kept.append(item)
        kept.extend(group_kept)
    return kept


def _round_robin(buckets: Dict[str, List[HardCandidate]]) -> Iterable[HardCandidate]:
    queues = {key: list(values) for key, values in sorted(buckets.items()) if values}
    while queues:
        for key in list(queues):
            values = queues[key]
            if values:
                yield values.pop(0)
            if not values:
                queues.pop(key, None)


def balance(candidates: Sequence[HardCandidate], cfg: ReplayConfig) -> List[HardCandidate]:
    by_class_scene: Dict[str, Dict[str, List[HardCandidate]]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        by_class_scene[item.query][item.scene].append(item)
    class_queues: Dict[str, List[HardCandidate]] = {}
    for query in cfg.classes:
        scene_buckets: Dict[str, List[HardCandidate]] = {}
        for scene, items in by_class_scene.get(query, {}).items():
            ordered = sorted(items, key=lambda x: x.hardness_key)
            if cfg.max_per_scene_class:
                ordered = ordered[:cfg.max_per_scene_class]
            scene_buckets[scene] = ordered
        merged = list(_round_robin(scene_buckets))
        if cfg.max_per_class:
            merged = merged[:cfg.max_per_class]
        class_queues[query] = merged

    selected: List[HardCandidate] = []
    scene_counts: Counter = Counter()
    for item in _round_robin(class_queues):
        if cfg.max_per_scene and scene_counts[item.scene] >= cfg.max_per_scene:
            continue
        selected.append(item)
        scene_counts[item.scene] += 1
        if cfg.max_total and len(selected) >= cfg.max_total:
            break
    return selected


def _target_json(box: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if box is None:
        return None
    return {"box": {"center": box[:3].tolist(), "size": box[3:].tolist()}}


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_dataset(
    selected: Sequence[HardCandidate],
    output_dir: Path,
    cfg: ReplayConfig,
    rejected: Counter,
    raw_count: int,
    dedup_count: int,
    overwrite: bool = False,
) -> Dict[str, Any]:
    train_dir = output_dir / "train"
    output_files = [train_dir / f"{scene}.npz" for scene in sorted({x.scene for x in selected})]
    fixed_files = [output_dir / "selection.jsonl", output_dir / "stats.json", output_dir / "splits.json"]
    conflicts = [path for path in output_files + fixed_files if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {conflicts[0]}")

    by_scene_sample: Dict[str, Dict[str, List[HardCandidate]]] = defaultdict(lambda: defaultdict(list))
    for item in selected:
        by_scene_sample[item.scene][item.sample_id].append(item)

    n_views = n_records = n_points = 0
    for scene in sorted(by_scene_sample):
        point_blocks: List[np.ndarray] = []
        offsets = [0]
        records_json: List[Dict[str, Any]] = []
        for view_idx, sample_id in enumerate(sorted(by_scene_sample[scene])):
            items = sorted(by_scene_sample[scene][sample_id], key=lambda x: (x.query, x.hardness_key))
            points, _meta = load_capture(items[0].source)
            point_blocks.append(np.ascontiguousarray(points[:, :6], dtype=np.float16))
            offsets.append(offsets[-1] + len(points))
            for item in items:
                records_json.append({
                    "view": view_idx,
                    "k": len(item.poses),
                    "poses": item.poses,
                    "query": item.query,
                    "label": item.label,
                    "weight": float(cfg.replay_weight),
                    "target": _target_json(item.gt_box),
                    "hard_negative": {
                        "source_sample_id": item.sample_id,
                        "source_episode": item.episode,
                        "source_step": item.step,
                        "mined_score": item.score,
                        "mined_iou": item.pred_iou,
                        "mined_box": item.pred_box.tolist(),
                        "reason": item.reason,
                        "candidate_rank": item.candidate_rank,
                    },
                })
                n_records += 1
            n_views += 1
            n_points += len(points)
        points_cat = (np.concatenate(point_blocks, axis=0) if point_blocks
                      else np.zeros((0, 6), dtype=np.float16))
        _atomic_npz(
            train_dir / f"{scene}.npz",
            points=points_cat,
            offsets=np.asarray(offsets, dtype=np.int64),
            records_json=np.asarray(json.dumps(records_json, ensure_ascii=False, separators=(",", ":"))),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.jsonl"
    selection_path.write_text(
        "".join(json.dumps(item.audit_dict(), ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    scenes = sorted(by_scene_sample)
    (output_dir / "splits.json").write_text(
        json.dumps({"train": scenes, "val": []}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scores = np.asarray([x.score for x in selected], dtype=np.float64)
    stats: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "raw_hard_candidates": int(raw_count),
        "after_dedup": int(dedup_count),
        "selected": len(selected),
        "scenes": len(scenes),
        "views": n_views,
        "records": n_records,
        "points": n_points,
        "by_class": dict(sorted(Counter(x.query for x in selected).items())),
        "by_scene": dict(sorted(Counter(x.scene for x in selected).items())),
        "by_label": {str(k): v for k, v in sorted(Counter(x.label for x in selected).items())},
        "by_reason": dict(sorted(Counter(x.reason for x in selected).items())),
        "score": {
            "min": None if not len(scores) else float(scores.min()),
            "mean": None if not len(scores) else float(scores.mean()),
            "max": None if not len(scores) else float(scores.max()),
        },
        "rejected": dict(sorted(rejected.items())),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def build_replay(
    input_dir: Path,
    output_dir: Path,
    splits_json: Path,
    cfg: ReplayConfig,
    overwrite: bool = False,
) -> Dict[str, Any]:
    cfg.validate()
    input_dir, output_dir, splits_json = Path(input_dir), Path(output_dir), Path(splits_json)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not splits_json.is_file():
        raise FileNotFoundError(f"splits json not found: {splits_json}")
    files = sorted(path for path in input_dir.rglob("*.npz") if output_dir not in path.parents)
    if not files:
        raise RuntimeError(f"no capture .npz under {input_dir}")
    train_scenes = load_train_scenes(splits_json)
    rejected: Counter = Counter()
    raw: List[HardCandidate] = []
    for path in files:
        raw.extend(extract_candidates(path, input_dir, train_scenes, cfg, rejected))
    deduped = deduplicate(raw, cfg)
    selected = balance(deduped, cfg)
    if not selected:
        raise RuntimeError(
            f"no hard negatives selected; valid={len(raw)} rejected={dict(rejected)}")
    return write_dataset(
        selected, output_dir, cfg, rejected, len(raw), len(deduped), overwrite=overwrite)


def mine_captures(
    checkpoint: Path,
    data_dir: Path,
    capture_dir: Path,
    cfg: ReplayConfig,
    device_name: str = "cuda:0",
    scenes: Optional[str] = None,
    limit_views: int = 0,
    overwrite: bool = False,
    log_every: int = 100,
) -> Dict[str, Any]:
    """用 R1 对 train local-view 前向，落盘达到 R2 条件的候选。

    使用训练前向的 soft feature path 取得 R1v1 交付 Top-1，但全程
    ``no_grad`` 且关闭 BN/dropout 随机性。每次只保留一个 record 的计算图/输出，显存规模
    与单条推理一致。输出仍经过 ``build_replay`` 的第二次独立校验。
    """
    import torch

    here = Path(__file__).resolve().parent
    repo = here.parent
    for item in (str(here), str(repo)):
        if item not in sys.path:
            sys.path.insert(0, item)
    from data_loader import LoaderConfig, LocalViewDataset
    from init_from_pretrain import build_model, load_pretrained
    from losses import delivery_tensors
    from pipeline import forward_train, move_sample_points, set_stochastic_eval

    checkpoint, data_dir, capture_dir = Path(checkpoint), Path(data_dir), Path(capture_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"mining checkpoint not found: {checkpoint}")
    if not (data_dir / "train").is_dir():
        raise FileNotFoundError(f"train data not found: {data_dir / 'train'}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA mining requested but CUDA is unavailable")

    dataset = LocalViewDataset(LoaderConfig(
        data_dir=str(data_dir), split="train",
        scenes=scenes, limit_views=limit_views, cache_scenes=2,
    ))
    model = build_model(ft_config={"enable_gate": False})
    report = load_pretrained(model, str(checkpoint))
    if report["n_unexpected"]:
        raise RuntimeError(
            f"checkpoint has unexpected model keys: {report['unexpected'][:8]}")
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # BeaUTyDETR 只有 training 分支导出 ft_outputs；随后单独关闭所有统计/随机层。
    model.train(True)
    set_stochastic_eval(model)

    capture_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stats: Counter = Counter()
    score_sum = 0.0
    for index in range(len(dataset)):
        sample = dataset[index]
        scene, view = str(sample["scene"]), int(sample["view"])
        # ``view`` 是原始 record ID，可能稀疏且远大于该场景实际保存的 view 数；
        # local-view 点云的 offsets 使用 0..V-1 连续序号，二者不能混用。
        source_view = int(sample["view_index"])
        indexed_scene, indexed_view = dataset.index[index]
        if scene != str(indexed_scene) or source_view != int(indexed_view):
            raise RuntimeError(
                "local-view index mismatch: "
                f"sample=({scene}, {source_view}) dataset=({indexed_scene}, {indexed_view})"
            )
        destination = capture_dir / scene / f"v{view:06d}.npz"
        if destination.exists() and not overwrite:
            stats["views_skipped_existing"] += 1
            continue
        points_t = move_sample_points(sample, device)
        output_records: List[Dict[str, Any]] = []
        for record in sample["records"]:
            inputs = {
                "point_clouds": [points_t], "text": [record["query"]],
                "return_outputs": True,
            }
            with torch.no_grad():
                pred = forward_train(model, inputs)
                boxes_t, logits_t = delivery_tensors(pred, 1e-6)
                scores_t = logits_t.sigmoid()
            boxes = boxes_t.detach().float().cpu().numpy()
            scores = scores_t.detach().float().cpu().numpy()
            gt = record.get("gt")
            gt_box = None if gt is None else np.asarray(gt, dtype=np.float32).reshape(6)
            label = int(record.get("label", 0))
            candidates: List[Dict[str, Any]] = []
            for rank, (box, score_value) in enumerate(zip(boxes, scores)):
                score = float(score_value)
                if score < cfg.min_score:
                    stats["candidate_below_score"] += 1
                    continue
                pred_iou = 0.0 if gt_box is None else box_iou3d(box, gt_box)
                if label == 1 and pred_iou >= cfg.max_fp_iou:
                    stats["candidate_not_fp"] += 1
                    continue
                reason = "target_absent" if label == 0 else "low_iou"
                candidates.append({
                    "score": score,
                    "box": np.asarray(box, dtype=np.float32).tolist(),
                    "candidate_rank": rank,
                    "reason": reason,
                })
                stats["hard_candidates"] += 1
                stats[f"hard_class:{record['query']}"] += 1
                score_sum += score
            stats["records_seen"] += 1
            if candidates:
                output_records.append({
                    "query": str(record["query"]),
                    "label": label,
                    "gt_box": None if gt_box is None else gt_box.tolist(),
                    "candidates": candidates,
                })
        stats["views_seen"] += 1
        if output_records:
            poses = list(sample["records"][0].get("poses") or []) if sample["records"] else []
            source_npz = data_dir / "train" / f"{scene}.npz"
            write_reference_capture(destination, {
                "schema_version": SCHEMA_VERSION,
                "sample_id": f"{scene}/v{view:06d}",
                "split": "train",
                "scene": scene,
                "episode": f"local_view:{view}",
                "step": view,
                "poses": poses,
                "records": output_records,
                "mined_with": str(checkpoint.resolve()),
            }, source_npz=source_npz, source_view=source_view)
            stats["views_written"] += 1
        if log_every > 0 and (index + 1) % log_every == 0:
            elapsed = time.time() - started
            rate = elapsed / (index + 1)
            remain_h = (len(dataset) - index - 1) * rate / 3600.0
            print(
                f"[mine {index + 1}/{len(dataset)}] "
                f"written={stats['views_written']} hard={stats['hard_candidates']} "
                f"{rate:.2f}s/view ETA={remain_h:.1f}h",
                flush=True,
            )

    elapsed = time.time() - started
    result: Dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "data_dir": str(data_dir.resolve()),
        "capture_dir": str(capture_dir.resolve()),
        "device": str(device),
        "elapsed_s": elapsed,
        "score_mean": score_sum / max(1, stats["hard_candidates"]),
        "counts": dict(sorted(stats.items())),
        "model_load": {
            "checkpoint_keys": report["ckpt_keys"],
            "missing": report["missing"],
            "unexpected": report["unexpected"],
        },
    }
    (capture_dir / "mining_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="r2_replay_test_") as tmp:
        root = Path(tmp)
        capture, output = root / "capture", root / "output"
        capture.mkdir()
        splits = root / "splits.json"
        splits.write_text(json.dumps({"train": ["sceneA", "sceneB"], "val": ["sceneV"]}))
        points = np.zeros((128, 6), dtype=np.float32)
        source_npz = root / "source_scene.npz"
        np.savez_compressed(
            source_npz,
            points=np.arange(256 * 6, dtype=np.float32).reshape(256, 6),
            offsets=np.asarray([0, 128, 256], dtype=np.int64),
        )
        reference_probe = capture / "reference_probe.npz"
        write_reference_capture(
            reference_probe,
            {"scene": "sceneA", "split": "train", "records": []},
            source_npz=source_npz,
            source_view=1,
        )
        reference_points, reference_meta = load_capture(reference_probe)
        assert reference_points.shape == (128, 6)
        assert reference_meta["points_source"]["view"] == 1
        reference_probe.unlink()

        def emit(name: str, scene: str, episode: str, step: int, record: Dict[str, Any]) -> None:
            write_capture(capture / f"{name}.npz", points, {
                "scene": scene, "split": "train", "episode": episode, "step": step,
                "sample_id": name, "poses": [], "records": [record],
            })

        gt = [0, 0, 0, 1, 1, 1]
        fp = [3, 3, 3, 1, 1, 1]
        emit("a0", "sceneA", "ep0", 10, {
            "query": "chair", "label": 1, "gt_box": gt,
            "candidates": [{"score": 0.90, "box": fp}, {"score": 0.80, "box": gt}],
        })
        emit("a1_duplicate", "sceneA", "ep0", 11, {
            "query": "chair", "label": 1, "gt_box": gt,
            "candidates": [{"score": 0.85, "box": [3.05, 3, 3, 1, 1, 1]}],
        })
        emit("b0", "sceneB", "ep1", 4, {
            "query": "tv", "label": 0, "gt_box": None,
            "candidates": [{"score": 0.88, "box": [2, 0, 0, 1, 1, 1]}],
        })
        emit("leak", "sceneV", "ep2", 1, {
            "query": "bed", "label": 0, "gt_box": None,
            "candidates": [{"score": 0.99, "box": fp}],
        })
        stats = build_replay(capture, output, splits, ReplayConfig(), overwrite=False)
        assert stats["raw_hard_candidates"] == 3, stats
        assert stats["after_dedup"] == 2, stats
        assert stats["selected"] == 2, stats
        assert stats["rejected"]["scene_not_in_train_split"] == 1, stats
        for scene in ("sceneA", "sceneB"):
            with np.load(output / "train" / f"{scene}.npz", allow_pickle=False) as data:
                assert data["points"].shape == (128, 6)
                assert data["offsets"].tolist() == [0, 128]
                records = json.loads(str(data["records_json"]))
                assert len(records) == 1 and "hard_negative" in records[0]
        
        # 生产训练 loader 必须能无转换读取 replay 输出。
        from data_loader import LoaderConfig, LocalViewDataset
        dataset = LocalViewDataset(LoaderConfig(
            data_dir=str(output), split="train", cache_scenes=1))
        assert len(dataset) == 2
        loaded = [dataset[index] for index in range(len(dataset))]
        assert sum(len(item["records"]) for item in loaded) == 2
        assert all(item["points"].shape == (128, 6) for item in loaded)

        # 原始 view ID 可以稀疏；点云引用必须使用 offsets 的连续 view_index。
        sparse_root = root / "sparse_local_view"
        sparse_train = sparse_root / "train"
        sparse_train.mkdir(parents=True)
        sparse_scene = sparse_train / "sceneSparse.npz"
        sparse_points = np.arange(256 * 6, dtype=np.float32).reshape(256, 6)
        sparse_records = [
            {"view": 10, "query": "chair", "label": 0, "poses": []},
            {"view": 42, "query": "chair", "label": 0, "poses": []},
        ]
        np.savez_compressed(
            sparse_scene,
            points=sparse_points,
            offsets=np.asarray([0, 128, 256], dtype=np.int64),
            records_json=np.asarray(json.dumps(sparse_records)),
        )
        (sparse_root / "splits.json").write_text(
            json.dumps({"train": ["sceneSparse"], "val": []}))
        sparse_dataset = LocalViewDataset(LoaderConfig(
            data_dir=str(sparse_root), split="train", cache_scenes=1))
        sparse_sample = sparse_dataset[1]
        assert sparse_sample["view"] == 42
        assert sparse_sample["view_index"] == 1
        sparse_capture = capture / "sparse_reference.npz"
        write_reference_capture(
            sparse_capture,
            {"scene": "sceneSparse", "split": "train", "records": []},
            source_npz=sparse_scene,
            source_view=sparse_sample["view_index"],
        )
        sparse_loaded, _ = load_capture(sparse_capture)
        assert np.array_equal(sparse_loaded, sparse_sample["points"])
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print("R2 hard-negative replay self-test: PASS")


def parse_args() -> argparse.Namespace:
    repo = _REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path,
                        default=repo / "data" / "ft_datasets" / "r2_capture")
    parser.add_argument("--output-dir", type=Path,
                        default=repo / "data" / "ft_datasets" / "hard_negative_replay")
    parser.add_argument("--splits-json", type=Path,
                        default=repo / "data" / "ft_datasets" / "local_view" / "splits.json")
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--max-fp-iou", type=float, default=0.25)
    parser.add_argument("--min-points", type=int, default=64)
    parser.add_argument("--dedup-center-m", type=float, default=0.50)
    parser.add_argument("--dedup-iou", type=float, default=0.50)
    parser.add_argument("--dedup-step-window", type=int, default=8)
    parser.add_argument("--max-total", type=int, default=6000)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--max-per-scene-class", type=int, default=25)
    parser.add_argument("--max-per-scene", type=int, default=100)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mine-checkpoint", type=Path, default=None,
                        help="先用该 R1 checkpoint 在 train local-view 上生成 capture")
    parser.add_argument("--mine-data-dir", type=Path,
                        default=repo / "data" / "ft_datasets" / "local_view")
    parser.add_argument("--mine-scenes", default=None,
                        help="仅挖掘逗号分隔的 train scene（调试用）")
    parser.add_argument("--mine-limit-views", type=int, default=0,
                        help="每场景最多挖掘的 view 数（0=全部）")
    parser.add_argument("--mine-device", default="cuda:0")
    parser.add_argument("--mine-log-every", type=int, default=100)
    parser.add_argument("--mine-overwrite", action="store_true")
    parser.add_argument("--mine-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    
    classes = tuple(item.strip().lower() for item in args.classes.split(",") if item.strip())
    cfg = ReplayConfig(
        min_score=args.min_score,
        max_fp_iou=args.max_fp_iou,
        min_points=args.min_points,
        dedup_center_m=args.dedup_center_m,
        dedup_iou=args.dedup_iou,
        dedup_step_window=args.dedup_step_window,
        max_total=args.max_total,
        max_per_class=args.max_per_class,
        max_per_scene_class=args.max_per_scene_class,
        max_per_scene=args.max_per_scene,
        replay_weight=args.replay_weight,
        classes=classes,
    )
    if args.mine_checkpoint is not None:
        mine_captures(
            checkpoint=args.mine_checkpoint,
            data_dir=args.mine_data_dir,
            capture_dir=args.input_dir,
            cfg=cfg,
            device_name=args.mine_device,
            scenes=args.mine_scenes,
            limit_views=args.mine_limit_views,
            overwrite=args.mine_overwrite,
            log_every=args.mine_log_every,
        )
        if args.mine_only:
            return
    elif args.mine_only:
        raise ValueError("--mine-only requires --mine-checkpoint")
    stats = build_replay(
        args.input_dir, args.output_dir, args.splits_json, cfg, overwrite=args.overwrite)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
