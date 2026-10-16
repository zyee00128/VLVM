#!/usr/bin/env python3
"""微调管道 · 数据加载管道。

作用
----
1. 读取主数据集 `local_view/{split}/<scene>.npz`（`points` float16 (M, 6) +
   `offsets` (V+1,) + `records_json`），**采样粒度 = view**：同一视图的正/负样本
   共享一份点云（同一 npz 仅加载一次，场景级 LRU 缓存，重复 epoch 零磁盘开销）。
2. 文本 prompt = record 的 `query`（规范类名，与在线 `use_vlfm_nlp=False` 的主词口径一致）。

输出（每个 item = 一个 view）
-----------------------------
```
{
  "scene": str,
  "view": int,              # 原始 record view ID（审计/展示）
  "view_index": int,        # 场景内连续序号（points offsets 索引）
  "n_points": int,
  "points": float32 (N, 6),
  "records": [
      {"query": str, "label": 0|1, "weight": float,
       "gt": None | float32 (6,)            # 正样本：canonical (x, y, z, l, w, h)
       "poses": [{"x","y","floor","yaw_deg"}, ...],   # 原始帧位姿（审计用）
      }, ...
  ],
}
```

步数口径：`steps_per_epoch = len(dataset)`（每步 1 个 view）。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class LoaderConfig:
    data_dir: str = str(_REPO / "data" / "ft_datasets" / "local_view")
    split: str = "train"
    cache_scenes: int = 3                   # 场景级 LRU（每场景 ~80-200 MB float16）
    scenes: Optional[str] = None            # 逗号分隔 scene_id 过滤（冒烟）
    limit_views: int = 0                    # 每场景最多 N 个 view（冒烟）


class _SceneBundle:
    """
    单场景缓存：点云 + 按 view 聚合的 records。
    """

    __slots__ = ("scene_id", "points", "offsets", "views")

    def __init__(self, scene_id: str, points: np.ndarray, offsets: np.ndarray,
                 views: List[List[dict]]) -> None:
        self.scene_id = scene_id
        self.points = points
        self.offsets = offsets
        self.views = views


def _group_views(records: List[dict], n_views: int) -> List[List[dict]]:
    """records（view 连续）-> 每 view 一组的列表；校验与 offsets 分块一致。"""
    views: List[List[dict]] = []
    for r in records:
        v = int(r["view"])
        if not views or int(views[-1][-1]["view"]) != v:
            views.append([])
        views[-1].append(r)
    if len(views) != n_views:
        raise ValueError(f"view groups({len(views)}) != offsets-1({n_views}) —— 数据集不一致")
    return views


def _gt_from_record(r: dict) -> Optional[np.ndarray]:
    """record -> canonical (x, y, z, l, w, h) float32；负样本 None。"""
    tgt = r.get("target")
    if not tgt:
        return None
    box = tgt.get("box")
    if not box:
        return None
    return np.asarray(list(box["center"]) + list(box["size"]), dtype=np.float32)


class LocalViewDataset(Dataset):
    """view 粒度数据集（每 item = 一个 view 的全部 records）。"""

    def __init__(self, cfg: LoaderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.split_dir = self.data_dir / cfg.split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"split dir not found: {self.split_dir}")
        scenes = self._scene_list()
        self.scene_ids: List[str] = scenes
        self._cache: "Dict[str, _SceneBundle]" = {}
        self._cache_order: List[str] = []
        self.warnings: List[str] = []

        # view 索引（按场景顺序展平；场景内 view 序号 0..V-1）
        self.index: List[Tuple[str, int]] = []
        for sid in scenes:
            npz = self.split_dir / f"{sid}.npz"
            with np.load(npz, allow_pickle=False) as z:
                n_views = int(np.asarray(z["offsets"]).shape[0]) - 1
            n = n_views if cfg.limit_views <= 0 else min(n_views, int(cfg.limit_views))
            self.index.extend((sid, v) for v in range(n))

    # ---- scene list ----
    def _scene_list(self) -> List[str]:
        splits_json = self.data_dir / "splits.json"
        scenes: List[str] = []
        if splits_json.is_file():
            plan = json.loads(splits_json.read_text())
            scenes = [str(s) for s in plan.get(self.cfg.split, [])]
        if not scenes:
            scenes = sorted(p.stem for p in self.split_dir.glob("*.npz"))
        if self.cfg.scenes:
            want = {s.strip() for s in self.cfg.scenes.split(",") if s.strip()}
            scenes = [s for s in scenes if s in want]
        return scenes

    # ---- scene loading (LRU) ----
    def _load_scene(self, scene_id: str) -> _SceneBundle:
        cached = self._cache.get(scene_id)
        if cached is not None:
            return cached
        npz = self.split_dir / f"{scene_id}.npz"
        with np.load(npz, allow_pickle=False) as z:
            points = np.asarray(z["points"])          # float16 (M, 6)
            offsets = np.asarray(z["offsets"], dtype=np.int64)
            records = json.loads(str(z["records_json"]))
        views = _group_views(records, int(offsets.shape[0]) - 1)

        bundle = _SceneBundle(scene_id, points, offsets, views)
        self._cache[scene_id] = bundle
        self._cache_order.append(scene_id)
        while len(self._cache_order) > max(1, int(self.cfg.cache_scenes)):
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return bundle

    # ---- Dataset API ----
    def __len__(self) -> int:
        return len(self.index)

    def iter_view_metadata(self) -> Iterator[Tuple[int, str, int, List[dict]]]:
        """按全局 dataset index 读取 view 元数据，不加载点云数组。"""
        global_index = 0
        for scene_id in self.scene_ids:
            npz = self.split_dir / f"{scene_id}.npz"
            with np.load(npz, allow_pickle=False) as data:
                n_views = int(np.asarray(data["offsets"]).shape[0]) - 1
                records = json.loads(str(data["records_json"]))
            views = _group_views(records, n_views)
            n = n_views if self.cfg.limit_views <= 0 else min(n_views, int(self.cfg.limit_views))
            for view_index in range(n):
                yield global_index, scene_id, view_index, views[view_index]
                global_index += 1
        if global_index != len(self.index):
            raise RuntimeError(
                f"metadata index size mismatch: metadata={global_index} dataset={len(self.index)}")

    def __getitem__(self, i: int) -> Dict[str, Any]:
        scene_id, v = self.index[i]
        bundle = self._load_scene(scene_id)
        records = bundle.views[v]
        pts = bundle.points[bundle.offsets[v]:bundle.offsets[v + 1]].astype(np.float32)
        poses = list(records[0].get("poses") or [])

        out_records: List[Dict[str, Any]] = []
        for r in records:
            out_records.append({
                "query": str(r["query"]),
                "label": int(r["label"]),
                "weight": float(r.get("weight", 1.0)),
                "gt": _gt_from_record(r),
                "poses": poses,
            })
        return {
            "scene": scene_id,
            "view": int(records[0]["view"]),
            "view_index": int(v),
            "n_points": int(len(pts)),
            "points": pts, 
            "records": out_records,
        }

    # ---- misc ----
    def summarize(self) -> Dict[str, Any]:
        n_pos = n_neg = 0
        n_rec = 0
        scenes = 0
        for sid in self.scene_ids:
            try:
                b = self._load_scene(sid)
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"{sid}: load failed: {exc}")
                continue
            scenes += 1
            for recs in b.views:
                for r in recs:
                    n_rec += 1
                    if int(r["label"]) == 1:
                        n_pos += 1
                    else:
                        n_neg += 1
        return {
            "split": self.cfg.split, "scenes": scenes, "views": len(self.index),
            "records": n_rec, "pos": n_pos, "neg": n_neg,
            "pos_neg_ratio": (round(n_pos / max(1, n_neg), 4) if n_neg else None),
            "steps_per_epoch": len(self.index),
        }


def main() -> None:
    """
    数据管道自检：`python ft_tsp3d/data_loader.py --split train --limit-views 5`。
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=LoaderConfig.data_dir)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--limit-views", type=int, default=0)
    args = ap.parse_args()
    cfg = LoaderConfig(data_dir=args.data_dir, split=args.split, scenes=args.scenes,
                       limit_views=args.limit_views)
    ds = LocalViewDataset(cfg)
    if args.scenes or args.limit_views:
        for i in list(range(min(len(ds), 3))) + ([len(ds) - 1] if len(ds) > 3 else []):
            s = ds[i]
            r0 = s["records"][0]
            print(f"[{s['scene']} v{s['view']}] pts={s['n_points']} recs={len(s['records'])} "
                  f"q0={r0['query']} label0={r0['label']}")
    else:
        print(json.dumps(ds.summarize(), ensure_ascii=False, indent=1))
    if ds.warnings:
        print(f"[warn] {len(ds.warnings)} warnings, e.g.: {ds.warnings[:3]}")

if __name__ == "__main__":
    main()
