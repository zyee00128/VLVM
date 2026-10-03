#!/usr/bin/env python3
"""Stage3 · 蒸馏侧车（GD Teacher Sidecar）

把「位姿 → RGB/Depth → GD 证据」做成一条可复用链路，产出与训练侧约定的紧凑侧车。
支持两种产出模式（产物 **完全同构**，二选一或兼用）：

1) 后续补充（replay，本文件 CLI，默认）：
   对已生成的 `local_view/{split}/<scene>.npz` 逐视图回放 `records["poses"]`
   （与 data_preprocess.py 同源 sensor 配置重渲），跑 GD，写 `gd/{split}/<scene>.npz`。
   主数据集 / 主流程零改动，可分片、可断点续跑。

2) 一步生成（inline）：`python data_preprocess.py --gd-sidecar ...`
   在生成主 npz 的同一流程里，用本模块的 `FrameGdRunner` 对刚渲染的帧出证据。

侧车 schema（每场景一个 npz；`views` 顺序与主 npz `offsets` 分块一致）：
  meta_json : sensor / 词表 / 阈值 / GD 权重指纹（防约定漂移）
  views_json: [ {"view": <attempt id>, "k": K,
                 "frames": [ {"ok": bool,
                              "dets": [{"c": 规范类|None, "b": [x1,y1,x2,y2] 归一化,
                                        "s": score, "n": 深度支撑点数, "z": 中位深度(米)|None}]},
                             ... ]}, ... ]

硬规则（继承在线 `grounding_dino/client.py`）：**GD 失败 = 无信息，绝非负证据** ⇒ 每帧带 `ok` 标志；训练侧只吃 `ok=true` 帧。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# GD 权重与配置（repo 相对路径，与在线服务器同源）
GD_CONFIG_REL = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_WEIGHTS_REL = "data/groundingdino_swint_ogc.pth"


class FrameGdRunner:
    """一帧 (RGB, Depth) → 侧车帧条目；管理 GD 模型与词表映射（单进程单例）。"""

    def __init__(
        self,
        *,
        box_thr: float = 0.2,
        text_thr: float = 0.25,
        min_depth: float = 0.5,
        max_depth: float = 5.0,
        device: str = "cuda",
        quiet: bool = True,
        vocab: Optional[Sequence[str]] = None,
    ) -> None:
        import torch
        from depth_camera_filtering import filter_depth

        from vlfm.tsp3d_models.grounding_dino.vocab import (
            DEFAULT_VOCAB,
            build_alias_map,
            build_caption,
            canonicalize,
        )
        from vlfm.vlm.grounding_dino import GroundingDINO

        self._filter_depth = filter_depth
        self._canonicalize = canonicalize
        self.vocab = list(vocab) if vocab else list(DEFAULT_VOCAB)
        self.alias = build_alias_map(self.vocab)
        # 6 类合并 caption（每帧一次前向；per-record 由训练侧按 query 过滤）
        self.caption = build_caption(self.vocab, style="vocab", vocab=self.vocab)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.quiet = bool(quiet)
        # 计数（供汇总日志）
        self.n_frames = 0
        self.n_fail = 0
        self.n_dets = 0

        cfg_path = _REPO / GD_CONFIG_REL
        w_path = _REPO / GD_WEIGHTS_REL
        self.model = GroundingDINO(
            config_path=str(cfg_path),
            weights_path=str(w_path),
            caption=self.caption,
            box_threshold=float(box_thr),
            text_threshold=float(text_thr),
            device=torch.device(device),
        )

    # ---- depth support（norm → filter_depth → (0,1)）----
    def _valid_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float32)
        norm = np.zeros_like(d)
        hit = d > 0.0
        norm[hit] = (d[hit] - self.min_depth) / (self.max_depth - self.min_depth)
        np.clip(norm, 0.0, 1.0, out=norm)
        norm = self._filter_depth(norm, blur_type=None)
        return (norm > 0.0) & (norm < 1.0)

    def frame_entry(self, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, Any]:
        """返回 {"ok": bool, "dets": [{"c","b","s","n","z"}, ...]}。

        rgb: (H, W, 3/4) uint8；depth: (H, W) 度量平面 z 深度（米）。
        """
        self.n_frames += 1
        img = np.asarray(rgb)[:, :, :3]
        try:
            ctx = contextlib.redirect_stdout(io.StringIO()) if self.quiet else contextlib.nullcontext()
            with ctx:
                det = self.model.predict(img, caption=self.caption)
            boxes = det.boxes.detach().cpu().numpy().astype(np.float64).reshape(-1, 4)  # 归一化 xyxy
            scores = det.logits.detach().cpu().numpy().astype(np.float64).reshape(-1)
            phrases = [str(p) for p in det.phrases]
        except BaseException:  # noqa: BLE001 —— GD 失败 = 无信息，绝非负证据
            self.n_fail += 1
            return {"ok": False, "dets": []}

        h, w = img.shape[:2]
        valid = self._valid_depth_mask(depth)
        depth_m = np.asarray(depth, dtype=np.float32)
        dets: List[Dict[str, Any]] = []
        for b, s, ph in zip(boxes, scores, phrases):
            x1 = int(np.clip(np.floor(b[0] * w), 0, w - 1))
            x2 = int(np.clip(np.ceil(b[2] * w), x1 + 1, w))
            y1 = int(np.clip(np.floor(b[1] * h), 0, h - 1))
            y2 = int(np.clip(np.ceil(b[3] * h), y1 + 1, h))
            reg = valid[y1:y2, x1:x2]
            n = int(reg.sum())
            z = float(np.median(depth_m[y1:y2, x1:x2][reg])) if n > 0 else None
            dets.append({
                "c": self._canonicalize(ph, self.alias),   # 规范类（None = 未映射）
                "b": [round(float(v), 4) for v in b],      # 归一化 xyxy
                "s": round(float(s), 4),
                "n": n,                                    # 深度支撑点数（[0.5,5.0] m 内）
                "z": (round(z, 3) if z is not None else None),
            })
        self.n_dets += len(dets)
        return {"ok": True, "dets": dets}

    def stats(self) -> Dict[str, int]:
        return {"frames": self.n_frames, "gd_fail": self.n_fail, "dets": self.n_dets}


def sidecar_meta(runner: FrameGdRunner, sensor: Dict[str, Any]) -> Dict[str, Any]:
    """侧车 meta：防约定漂移的指纹（sensor / 词表 / 阈值）。"""
    return {
        "format": "gd_sidecar_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab": runner.vocab,
        "caption": runner.caption,
        "min_depth": runner.min_depth,
        "max_depth": runner.max_depth,
        "sensor": dict(sensor),
    }


def load_npz_views(npz_path: Path) -> List[Dict[str, Any]]:
    """按主 npz `offsets` 分块顺序还原「保存视图」序列（与 points 分块一一对应）。"""
    with np.load(npz_path, allow_pickle=False) as z:
        records = json.loads(str(z["records_json"]))
        n_views = int(np.asarray(z["offsets"]).shape[0]) - 1
    views: List[Dict[str, Any]] = []
    for r in records:
        if not views or views[-1]["view"] != int(r["view"]):
            views.append({"view": int(r["view"]), "k": int(r["k"]), "poses": r["poses"]})
    if len(views) != n_views:
        raise ValueError(
            f"{npz_path.name}: view groups({len(views)}) != offsets-1({n_views}) —— 数据集不一致"
        )
    return views


def build_scene_sidecar(
    npz_path: Path,
    dst_path: Path,
    runner: FrameGdRunner,
    cfg,
    *,
    meta: Dict[str, Any],
) -> Dict[str, int]:
    """replay 模式：回放主 npz 的逐帧位姿 → 重渲 → GD → 写侧车。"""
    from data_preprocess import SceneRenderer  # 同目录导入（复用同源 sensor 约定）

    scene_id = npz_path.stem
    scene_dir = Path(cfg.hm3d_root) / cfg.split_dir / scene_id
    views = load_npz_views(npz_path)
    rr = SceneRenderer(scene_dir, cfg)
    out_views: List[Dict[str, Any]] = []
    try:
        for v in views:
            frames_out: List[Dict[str, Any]] = []
            for p in v["poses"]:
                pos_h = np.array([float(p["x"]), float(p["floor"]), -float(p["y"])], dtype=np.float64)
                yaw_e = math.radians(float(p["yaw_deg"]))
                try:
                    obs, _sp, _R = rr.render(pos_h, yaw_e)
                    frames_out.append(
                        runner.frame_entry(obs["rgb"], np.asarray(obs["depth"], dtype=np.float32))
                    )
                except BaseException:  # noqa: BLE001 —— 渲染失败同样按"无信息"处理
                    frames_out.append({"ok": False, "dets": []})
            out_views.append({"view": v["view"], "k": len(frames_out), "frames": frames_out})
    finally:
        rr.close()

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst_path,
        meta_json=json.dumps(meta, ensure_ascii=False),
        views_json=json.dumps(out_views, ensure_ascii=False),
    )
    return {"views": len(out_views), "frames": sum(v["k"] for v in out_views)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=str(_REPO / "data" / "ft_datasets" / "local_view"),
                    help="主数据集根目录（含 {train,val}/<scene>.npz）")
    ap.add_argument("--gd-out-dir", default=None, help="侧车输出根目录（默认 <dataset-dir>/gd）")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--scenes", default=None, help="逗号分隔 scene_id 覆盖")
    ap.add_argument("--limit", type=int, default=0, help="每个 split 最多处理 N 个场景（冒烟用）")
    ap.add_argument("--shard", type=int, default=0, help="分片编号（多进程并行时用，0-based）")
    ap.add_argument("--shards", type=int, default=1, help="分片总数")
    ap.add_argument("--overwrite", action="store_true", help="已存在侧车时强制重算（默认跳过）")
    ap.add_argument("--hm3d-root", default=None, help="默认复用 data_preprocess.Config")
    ap.add_argument("--split-dir", default=None, help="主数据集的 hm3d split_dir（默认 train）")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gd-box-thr", type=float, default=0.2,
                    help="侧车存储用 GD 框阈值（低于在线 0.35，保留更多教师信息；训练侧再过滤）")
    ap.add_argument("--gd-text-thr", type=float, default=0.25)
    ap.add_argument("--no-quiet", action="store_true", help="不吞 GD 的逐帧 print")
    ap.add_argument("--dry-run", action="store_true", help="只打印处理计划，不加载 GD/仿真")
    args = ap.parse_args()

    ds_dir = Path(args.dataset_dir)
    gd_dir = Path(args.gd_out_dir) if args.gd_out_dir else ds_dir / "gd"
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    plans: List[tuple] = []
    for split in splits:
        split_dir = ds_dir / split
        if not split_dir.is_dir():
            continue
        npzs = sorted(split_dir.glob("*.npz"))
        if args.scenes:
            want = {s.strip() for s in args.scenes.split(",") if s.strip()}
            npzs = [p for p in npzs if p.stem in want]
        npzs = npzs[args.shard :: args.shards]
        if args.limit > 0:
            npzs = npzs[: args.limit]
        if npzs:
            plans.append((split, npzs))
    todo = [(s, p) for s, ps in plans for p in ps]
    print(f"[gd_sidecar] scenes to process: {len(todo)} (shard {args.shard}/{args.shards}) -> {gd_dir}")
    if args.dry_run or not todo:
        for s, ps in plans:
            print(f"  [{s}] {len(ps)} scenes, e.g. {[p.stem for p in ps[:3]]}")
        return

    from data_preprocess import Config as DPConfig

    cfg = DPConfig()
    if args.hm3d_root:
        cfg.hm3d_root = args.hm3d_root
    if args.split_dir:
        cfg.split_dir = args.split_dir
    cfg.gpu_id = args.gpu_id

    runner = FrameGdRunner(
        box_thr=args.gd_box_thr, text_thr=args.gd_text_thr,
        min_depth=cfg.min_depth, max_depth=cfg.max_depth,
        device=args.device, quiet=not args.no_quiet,
    )
    meta = sidecar_meta(runner, {
        "image_h": cfg.image_h, "image_w": cfg.image_w, "hfov_deg": cfg.hfov_deg,
        "camera_height": cfg.camera_height, "min_depth": cfg.min_depth, "max_depth": cfg.max_depth,
        "fx": round((cfg.image_w / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0), 3),
    })

    t0 = time.time()
    done = skipped = failed = 0
    for split, npz in todo:
        dst = gd_dir / split / npz.name
        if dst.is_file() and not args.overwrite:
            skipped += 1
            continue
        t = time.time()
        try:
            info = build_scene_sidecar(npz, dst, runner, cfg, meta=meta)
            print(f"[gd_sidecar] {split}/{npz.stem}: views={info['views']} frames={info['frames']} "
                  f"({time.time() - t:.1f}s)")
        except Exception as exc:  # noqa: BLE001 —— 单场景失败不拖垮整个分片
            failed += 1
            print(f"[gd_sidecar] {split}/{npz.stem}: FAILED {type(exc).__name__}: {exc}")
        done += 1
    print(f"[gd_sidecar] done={done} skipped={skipped} failed={failed} "
          f"runner={json.dumps(runner.stats(), ensure_ascii=False)} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
