#!/usr/bin/env python3
"""逐场景实例 AABB 表导出（L_s1 标签面；S3 §6 轻量补件：分钟级、非重跑）。

产物：`<out-dir>/instances/<split>/<scene>.json`
```
{"scene": "00006-...", "coord": "episodic",
 "instances": [{"id": int, "raw": str, "cat": str|null,
                "center": [x, y, z], "size": [sx, sy, sz]}, ...]}
```
坐标口径（与 `data_preprocess.SceneRenderer` 完全一致，供 `ft_geometry` 使用）：
  * `center`：episodic 平面 x/y + **绝对高度** z（habitat y 经 `to_episodic` 变换）；
  * `size`：episodic 轴序 (sx, sz, sy)；
  * `cat` = 经 `grounding_dino/vocab.py` alias 归一后的 6 类规范名（未映射为 null）。

用法（需 `vlfm` 环境，含 habitat_sim）：
```
/root/miniconda3/envs/vlfm/bin/python ft_tsp3d/export_instances.py            # train+val 全量
... --splits train --limit 2                                                 # 冒烟
```
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def export_scene(scene_id: str, cfg, out_path: Path) -> int:
    from data_preprocess import SceneRenderer

    scene_dir = Path(cfg.hm3d_root) / cfg.split_dir / scene_id
    rr = SceneRenderer(scene_dir, cfg)
    try:
        insts = []
        for sid in sorted(rr.inst_center.keys()):
            insts.append({
                "id": int(sid),
                "raw": str(rr.inst_raw.get(sid, "")),
                "cat": rr.inst_class.get(sid),
                "center": rr.inst_center[sid].round(4).tolist(),
                "size": rr.inst_sizes[sid].round(4).tolist(),
            })
    finally:
        rr.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"scene": scene_id, "coord": "episodic", "instances": insts},
        ensure_ascii=False,
    ))
    return len(insts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(_REPO / "data" / "ft_datasets" / "local_view"),
                    help="主数据集根（读 splits.json 取场景清单；实例表默认写 <data-dir>/instances）")
    ap.add_argument("--out-dir", default=None, help="默认 <data-dir>/instances")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--scenes", default=None, help="逗号分隔 scene_id 覆盖")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hm3d-root", default=None, help="默认复用 data_preprocess.Config")
    ap.add_argument("--split-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from data_preprocess import Config as DPConfig
    from data_preprocess import discover_scenes

    cfg = DPConfig()
    if args.hm3d_root:
        cfg.hm3d_root = args.hm3d_root
    if args.split_dir:
        cfg.split_dir = args.split_dir

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "instances"
    splits_json = data_dir / "splits.json"
    plan = json.loads(splits_json.read_text()) if splits_json.is_file() else {}
    available = set(discover_scenes(cfg))

    t0 = time.time()
    done = skipped = failed = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        scene_ids = [str(s) for s in plan.get(split, [])] or sorted(available)
        if args.scenes:
            want = {s.strip() for s in args.scenes.split(",") if s.strip()}
            scene_ids = [s for s in scene_ids if s in want]
        scene_ids = [s for s in scene_ids if s in available]
        if args.limit > 0:
            scene_ids = scene_ids[: args.limit]
        for i, sid in enumerate(scene_ids, 1):
            dst = out_dir / split / f"{sid}.json"
            if dst.is_file() and not args.overwrite:
                skipped += 1
                continue
            try:
                n = export_scene(sid, cfg, dst)
                print(f"[{split}] {i}/{len(scene_ids)} {sid}: {n} instances ({time.time() - t0:.0f}s)")
            except Exception as exc:  # noqa: BLE001 —— 单场景失败不拖垮批次
                failed += 1
                print(f"[{split}] {i}/{len(scene_ids)} {sid}: FAILED {type(exc).__name__}: {exc}")
            done += 1
    print(f"[export_instances] done={done} skipped={skipped} failed={failed} -> {out_dir}")


if __name__ == "__main__":
    main()
