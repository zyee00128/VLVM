#!/usr/bin/env python3
"""Stage3 微调管道 · 数据管道（§11.2 #1）。

职责
----
1. 读取主数据集 `local_view/{split}/<scene>.npz`（`points` float16 (M, 6) +
   `offsets` (V+1,) + `records_json`），**采样粒度 = view**：同一视图的正/负样本
   共享一份点云（同一 npz 仅加载一次，场景级 LRU 缓存，重复 epoch 零磁盘开销）。
2. 可选读取 GD 侧车 `gd/{split}/<scene>.npz`（与保存视图一一对应），逐 record 计算
   教师信号（质检层见 `gd_teacher.py`）。
3. 可选读取逐场景实例表 `instances/{split}/<scene>.json`（L_s1 标签面；§6 轻量补件），
   view 级惰性计算「视锥内可见实体」（K 帧任一帧可见；`ft_geometry`）。
4. 文本 prompt = record 的 `query`（规范类名，与在线 `use_vlfm_nlp=False` 的主词口径一致）。

输出（每个 item = 一个 view）
-----------------------------
```
{
  "scene": str, "view": int, "n_points": int,
  "points": float32 (N, 6),
  "records": [
      {"query": str, "label": 0|1, "weight": float,
       "gt": None | float32 (6,)            # 正样本：canonical (x, y, z, l, w, h)
       "teacher": None | {"valid", "y", "s_max", "reason"},
       "entities": None | float32 (M, 6),   # 视锥内可见实体（L_s1；view 级共享）
       "poses": [{"x","y","floor","yaw_deg"}, ...],   # 原始帧位姿（审计用）
      }, ...
  ],
}
```

步数口径：`steps_per_epoch = len(dataset)`（每步 1 个 view；§7 的 24.6 k/epoch）。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ft_geometry import FrameGeometry, visible_entities_query_canonical  # noqa: E402
from gd_teacher import (  # noqa: E402
    GateTeacherConfig,
    GateTeacherStats,
    load_gd_sidecar,
    teacher_for_sample,
)


@dataclass
class LoaderConfig:
    data_dir: str = str(_REPO / "data" / "ft_datasets" / "local_view")
    split: str = "train"
    inst_dir: Optional[str] = None          # 默认 <data_dir>/instances/<split>
    use_sidecar: bool = True                # gd/{split}/<scene>.npz
    cache_scenes: int = 3                   # 场景级 LRU（每场景 ~80-200 MB float16）
    scenes: Optional[str] = None            # 逗号分隔 scene_id 过滤（冒烟）
    limit_views: int = 0                    # 每场景最多 N 个 view（冒烟）
    # 结构类实例（白墙 / 地面 / 天花板 / 未知）不是「物理实体」——
    # 生成器把「空白视锥（走廊、白墙）」当作负样本，L_s1 的实体集合必须排除它们，
    # 否则会教提议头在墙面处输出实体。
    struct_blacklist: Tuple[str, ...] = ("wall", "floor", "ceiling", "carpet", "unknown")
    # 可见实体（L_s1 标签面）几何参数
    ent_near_radius: float = 8.0
    ent_z_band: Tuple[float, float] = (-0.5, 3.0)   # 相对查询 floor 的高度带（去多层串扰）
    ent_grid: float = 0.20
    ent_max_pts: int = 200
    ent_min_frac: float = 0.10
    # 教师
    teacher: Optional[GateTeacherConfig] = None
    geo: Optional[FrameGeometry] = None


class _SceneBundle:
    """单场景缓存：点云 + 按 view 聚合的 records +（可选）侧车/实例表。"""

    __slots__ = ("scene_id", "points", "offsets", "views", "gd_views", "instances", "ent_cache")

    def __init__(self, scene_id: str, points: np.ndarray, offsets: np.ndarray,
                 views: List[List[dict]], gd_views: Optional[List[dict]],
                 instances: Optional[List[dict]]) -> None:
        self.scene_id = scene_id
        self.points = points
        self.offsets = offsets
        self.views = views
        self.gd_views = gd_views
        self.instances = instances
        self.ent_cache: Dict[int, np.ndarray] = {}


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


def load_instance_table(path: Path) -> Optional[List[dict]]:
    """实例表读取（schema 见 `export_instances.py`）；缺文件返回 None。"""
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    insts = data.get("instances") or []
    return list(insts)


def _is_struct_instance(inst: dict, blacklist: Sequence[str]) -> bool:
    """结构类实例（wall / floor / ceiling / unknown …）不计入 L_s1 的「物理实体」。"""
    raw = str(inst.get("raw", "")).strip().lower()
    cat = str(inst.get("cat", "") or "").strip().lower()
    for w in blacklist:
        wl = str(w).strip().lower()
        if not wl:
            continue
        if raw == wl or raw.startswith(wl) or cat == wl:
            return True
    return False


class LocalViewDataset(Dataset):
    """view 粒度数据集（每 item = 一个 view 的全部 records）。"""

    def __init__(self, cfg: LoaderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.split_dir = self.data_dir / cfg.split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"split dir not found: {self.split_dir}")
        self.inst_dir = Path(cfg.inst_dir) if cfg.inst_dir else self.data_dir / "instances" / cfg.split
        self.gd_dir = self.data_dir / "gd" / cfg.split
        self.teacher_cfg = cfg.teacher or GateTeacherConfig()
        self.geo = cfg.geo or FrameGeometry()

        scenes = self._scene_list()
        self.scene_ids: List[str] = scenes
        self._cache: "Dict[str, _SceneBundle]" = {}
        self._cache_order: List[str] = []
        self.teacher_stats = GateTeacherStats()
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

        gd_views: Optional[List[dict]] = None
        if self.cfg.use_sidecar:
            gd_path = self.gd_dir / f"{scene_id}.npz"
            if gd_path.is_file():
                gd_views = load_gd_sidecar(gd_path)
                if len(gd_views) != len(views):
                    self.warnings.append(
                        f"{scene_id}: sidecar views({len(gd_views)}) != main views({len(views)}) -> 忽略侧车")
                    gd_views = None
            else:
                self.warnings.append(f"{scene_id}: 无侧车（蒸馏项将全部弃权）")

        instances = load_instance_table(self.inst_dir / f"{scene_id}.json")
        if instances is None:
            self.warnings.append(f"{scene_id}: 无实例表（L_s1 将跳过）")
        else:
            instances = [i for i in instances if not _is_struct_instance(i, self.cfg.struct_blacklist)]

        bundle = _SceneBundle(scene_id, points, offsets, views, gd_views, instances)
        self._cache[scene_id] = bundle
        self._cache_order.append(scene_id)
        while len(self._cache_order) > max(1, int(self.cfg.cache_scenes)):
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return bundle

    # ---- entities（view 级惰性；K 帧任一帧可见） ----
    def _entities_for(self, bundle: _SceneBundle, v: int, poses: List[dict]) -> Optional[np.ndarray]:
        if bundle.instances is None:
            return None
        got = bundle.ent_cache.get(v)
        if got is None:
            q_pose = poses[-1] if poses else None
            if q_pose is None:
                got = np.zeros((0, 6), dtype=np.float32)
            else:
                ent = visible_entities_query_canonical(
                    bundle.instances, poses, q_pose, self.geo,
                    near_radius=self.cfg.ent_near_radius,
                    z_band=self.cfg.ent_z_band,
                    surface_grid=self.cfg.ent_grid,
                    max_pts=self.cfg.ent_max_pts,
                    min_frac=self.cfg.ent_min_frac,
                )
                got = (ent[:, :6].astype(np.float32) if len(ent)
                       else np.zeros((0, 6), dtype=np.float32))
            bundle.ent_cache[v] = got
        return got

    # ---- Dataset API ----
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        scene_id, v = self.index[i]
        bundle = self._load_scene(scene_id)
        records = bundle.views[v]
        pts = bundle.points[bundle.offsets[v]:bundle.offsets[v + 1]].astype(np.float32)
        poses = list(records[0].get("poses") or [])
        entities = self._entities_for(bundle, v, poses)
        gd_frames = None
        if bundle.gd_views is not None:
            gd_frames = list(bundle.gd_views[v].get("frames") or [])

        out_records: List[Dict[str, Any]] = []
        for r in records:
            gt = _gt_from_record(r)
            teacher = None
            if gd_frames is not None:
                teacher = teacher_for_sample(
                    gd_frames, str(r["query"]), int(r["label"]), gt,
                    poses[-1] if poses else {"x": 0.0, "y": 0.0, "floor": 0.0, "yaw_deg": 0.0},
                    poses, self.teacher_cfg, self.teacher_stats, self.geo,
                )
            out_records.append({
                "query": str(r["query"]),
                "label": int(r["label"]),
                "weight": float(r.get("weight", 1.0)),
                "gt": gt,
                "teacher": teacher,
                "entities": entities,
                "poses": poses,
            })
        return {
            "scene": scene_id, "view": int(records[0]["view"]), "n_points": int(len(pts)),
            "points": pts, "records": out_records,
        }

    # ---- misc ----
    def reset_teacher_stats(self) -> None:
        self.teacher_stats = GateTeacherStats()

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


def collate_view(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """bs=1 的 collate（返回单 view；bs>1 时返回 list）。"""
    return batch[0] if len(batch) == 1 else list(batch)


def main() -> None:
    """数据管道自检：`python ft_tsp3d/data_loader.py --split train --limit-views 5`。"""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=LoaderConfig.data_dir)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--limit-views", type=int, default=0)
    ap.add_argument("--no-sidecar", action="store_true")
    args = ap.parse_args()
    cfg = LoaderConfig(data_dir=args.data_dir, split=args.split, scenes=args.scenes,
                       limit_views=args.limit_views, use_sidecar=not args.no_sidecar)
    ds = LocalViewDataset(cfg)
    if args.scenes or args.limit_views:
        for i in list(range(min(len(ds), 3))) + ([len(ds) - 1] if len(ds) > 3 else []):
            s = ds[i]
            r0 = s["records"][0]
            print(f"[{s['scene']} v{s['view']}] pts={s['n_points']} recs={len(s['records'])} "
                  f"q0={r0['query']} label0={r0['label']} "
                  f"teacher={None if r0['teacher'] is None else r0['teacher']['reason']} "
                  f"entities={None if r0['entities'] is None else len(r0['entities'])}")
        print("teacher stats:", json.dumps(ds.teacher_stats.summary(), ensure_ascii=False))
    else:
        print(json.dumps(ds.summarize(), ensure_ascii=False, indent=1))
    if ds.warnings:
        print(f"[warn] {len(ds.warnings)} warnings, e.g.: {ds.warnings[:3]}")


if __name__ == "__main__":
    main()
