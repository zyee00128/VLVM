#!/usr/bin/env python3
"""Stage3 微调管道 · GD 教师读取与质检层（S3 §4（1）（2）；§8 · 5 的输入侧）。

职责（与 `gd_sidecar.py` 产物对接，产物 schema 见其 docstring）：
  1. **侧车读取**：`gd/{split}/<scene>.npz`（`views_json` 与主 npz 的保存视图一一对应）。
  2. **教师信号** `S_max`：该视图**全部帧**中、目标类（record 的 query）检出的最高置信度；
     仅 `ok=true` 帧参与；逐检测须过**深度支撑**（`n >= n_min`、`z ∈ [z_lo, z_hi]`）才算
     「有效看见」，不满足 ⇒ 不计入（对 S_max 无贡献）。无检出 ⇒ S_max = 0。
  3. **质检层（统一前置过滤；仅作用于教师信号）**：
     * success —— 视图内无任何 `ok` 帧 ⇒ 弃权（GD 失败 = 无信息，绝非负证据）；
     * GT 投影验证 —— 正样本（label=1）中 S_max = 0 且 GT 在该视图至少一帧可见
       （`gt_visible`，由 `ft_geometry.box_visibility` 用 poses 逐帧几何投影给出）⇒
       判为 GD 漏检，该视图教师项**失效弃权**；负样本的 GD 误检（教师 > 0）不剔除
       （有意的召回侧偏置：门控 FN 不可恢复，统计并记录）。
  4. **软目标**：Ŷ = σ(logit(clamp(S, ε, 1-ε)) / T)，T = 1.5；`logit(0)` 不可直接计算，
     先 clamp（S = 0 时 Ŷ → 0）。
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class GateTeacherConfig:
    """质检层 + 软目标的超参（§4（2）/§8 · 5）。"""

    T: float = 1.5                    # 温度平滑（σ(logit(S)/T)）
    n_min: int = 40                   # 深度支撑：框内有效深度点数下限
    z_lo: float = 0.5                 # 深度支撑：中位深度下界（m）
    z_hi: float = 5.0                 # 深度支撑：中位深度上界（m）
    eps: float = 1e-6


@dataclass
class GateTeacherStats:
    """教师使用统计（每 epoch / 每 split 汇总；供训练日志）。"""

    views_total: int = 0
    valid: int = 0
    skip_no_ok_frame: int = 0         # 全帧 GD 失败 ⇒ 弃权
    skip_gd_miss_gt: int = 0          # 正样本 GD 漏检（教师 0 且 GT 可见）⇒ 弃权
    s_max_pos: List[float] = field(default_factory=list)  # 正样本 S_max（统计用）
    s_max_neg: List[float] = field(default_factory=list)
    n_dets_target: int = 0            # 目标类检出计数（过深度支撑）

    def summary(self) -> Dict[str, Any]:
        def _stat(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"n": 0}
            a = np.asarray(vals, dtype=np.float64)
            return {"n": int(a.size), "mean": round(float(a.mean()), 4),
                    "p50": round(float(np.median(a)), 4), "p90": round(float(np.percentile(a, 90)), 4)}

        return {
            "views": self.views_total,
            "valid": self.valid,
            "skip_no_ok_frame": self.skip_no_ok_frame,
            "skip_gd_miss_gt": self.skip_gd_miss_gt,
            "n_dets_target": self.n_dets_target,
            "s_max_pos": _stat(self.s_max_pos),
            "s_max_neg": _stat(self.s_max_neg),
        }


def load_gd_sidecar(npz_path: Path) -> List[Dict[str, Any]]:
    """读取侧车 npz -> `views` 列表（与主 npz 保存视图一一对应）。"""
    with np.load(npz_path, allow_pickle=False) as z:
        views = json.loads(str(z["views_json"]))
    return list(views)


def view_smax(
    frames: List[Dict[str, Any]],
    query: str,
    cfg: Optional[GateTeacherConfig] = None,
) -> Dict[str, Any]:
    """该视图对 `query` 类的教师信号（全部帧 max，深度支撑过滤，仅 ok 帧）。"""
    c = cfg or GateTeacherConfig()
    s_max = 0.0
    any_ok = False
    n_dets = 0
    for fr in frames:
        if not bool(fr.get("ok", False)):
            continue
        any_ok = True
        for d in fr.get("dets", []) or []:
            if str(d.get("c")) != str(query):
                continue
            n = d.get("n")
            z = d.get("z")
            if n is None or int(n) < c.n_min:
                continue
            if z is None or not (c.z_lo <= float(z) <= c.z_hi):
                continue
            n_dets += 1
            s_max = max(s_max, float(d.get("s", 0.0)))
    return {"s_max": s_max, "any_ok": any_ok, "n_dets": n_dets}


def soft_target(s_max: float, cfg: Optional[GateTeacherConfig] = None) -> float:
    """Ŷ = σ(logit(clamp(S, ε, 1-ε)) / T)；S = 0 ⇒ Ŷ → 0。"""
    c = cfg or GateTeacherConfig()
    s = min(max(float(s_max), c.eps), 1.0 - c.eps)
    logit = math.log(s / (1.0 - s))
    return 1.0 / (1.0 + math.exp(-logit / c.T))


def teacher_from_sidecar(
    frames: List[Dict[str, Any]],
    query: str,
    label: int,
    gt_visible: bool,
    cfg: Optional[GateTeacherConfig] = None,
    stats: Optional[GateTeacherStats] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """质检层判决 -> {"valid", "y", "s_max", "reason"}。

    弃权（valid=False）：无 ok 帧（no_ok_frame）/ 正样本 GD 漏检（gd_miss_gt）。
    `info` 可传入已算好的 `view_smax` 结果（避免重复统计）。
    """
    c = cfg or GateTeacherConfig()
    info = info if info is not None else view_smax(frames, query, c)
    if stats is not None:
        stats.views_total += 1
        stats.n_dets_target += int(info["n_dets"])

    if not info["any_ok"]:
        if stats is not None:
            stats.skip_no_ok_frame += 1
        return {"valid": False, "y": None, "s_max": None, "reason": "no_ok_frame"}

    s_max = float(info["s_max"])
    if int(label) == 1 and s_max <= 0.0 and bool(gt_visible):
        # 正样本、GT 可见、GD 全帧无有效检出 ⇒ 判 GD 漏检（用真值验证教师）
        if stats is not None:
            stats.skip_gd_miss_gt += 1
        return {"valid": False, "y": None, "s_max": s_max, "reason": "gd_miss_gt"}

    if stats is not None:
        stats.valid += 1
        (stats.s_max_pos if int(label) == 1 else stats.s_max_neg).append(s_max)
    return {"valid": True, "y": soft_target(s_max, c), "s_max": s_max, "reason": "ok"}


def teacher_for_sample(
    frames: List[Dict[str, Any]],
    query: str,
    label: int,
    gt_box_q: Optional[np.ndarray],
    q_pose: Dict[str, float],
    poses: List[Dict[str, float]],
    cfg: Optional[GateTeacherConfig] = None,
    stats: Optional[GateTeacherStats] = None,
    geo=None,
) -> Dict[str, Any]:
    """组合入口（data_loader 用）：先算 S_max；仅当「正样本 ∧ S_max=0」时才做
    GT 投影验证（惰性，逐帧几何投影由 `ft_geometry.box_visibility` 给出）。"""
    from ft_geometry import box_visibility  # 同目录（共享几何工具）

    c = cfg or GateTeacherConfig()
    info = view_smax(frames, query, c)
    gt_visible = False
    if int(label) == 1 and info["any_ok"] and float(info["s_max"]) <= 0.0 and gt_box_q is not None:
        gt_visible = bool(
            box_visibility(np.asarray(gt_box_q, dtype=np.float64).reshape(6), q_pose, poses, geo)["any_visible"]
        )
    return teacher_from_sidecar(frames, query, label, gt_visible, c, stats, info=info)


def scan_sidecar_stats(
    gd_dir: Path,
    split: str,
    max_scenes: int = 0,
) -> Dict[str, Any]:
    """离线统计（§4（5）备料）：逐场景汇总 ok 帧率 / 检出数 / 目标类覆盖。

    遍历 `gd/<split>/*.npz`（不依赖主 npz；query 维度无从对齐，只出帧级统计）。
    """
    out: Dict[str, Any] = {"scenes": 0, "views": 0, "frames": 0, "ok_frames": 0,
                           "dets": 0, "per_class": {}}
    files = sorted((Path(gd_dir) / split).glob("*.npz"))
    if max_scenes > 0:
        files = files[:max_scenes]
    for f in files:
        views = load_gd_sidecar(f)
        out["scenes"] += 1
        for v in views:
            out["views"] += 1
            for fr in v.get("frames", []):
                out["frames"] += 1
                if bool(fr.get("ok", False)):
                    out["ok_frames"] += 1
                for d in fr.get("dets", []) or []:
                    out["dets"] += 1
                    cls = str(d.get("c"))
                    out["per_class"][cls] = out["per_class"].get(cls, 0) + 1
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="GD 侧车统计/教师质检自检（§4）")
    ap.add_argument("--gd-dir", default=str(_REPO / "data" / "ft_datasets" / "local_view" / "gd"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-scenes", type=int, default=0)
    args = ap.parse_args()
    stats = scan_sidecar_stats(Path(args.gd_dir), args.split, args.max_scenes)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
