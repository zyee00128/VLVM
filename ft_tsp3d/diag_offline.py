#!/usr/bin/env python3
"""E3 离线判读：微调权重 vs 原版权重，在 val 视图上的「在线同源推理」输出对比（2026-09-30）。

判读目标（首轮评测 base 档崩坏的成因分离）
----------------------------------------
A = **分数分布下移被 sigma_tar 截断**：两权重「框召回」接近，但 base 的分数普遍更低
    （大量跌到在线准入线 0.70 之下 -> 在线看到的检测数塌缩）；
B = **输出消失 / 召回崩**：base 的框（任意分数）明显减少、真目标命中率显著低于 ref。

口径（与在线完全同源）
----------------------
* 输入 = `local_view/{split}/<scene>.npz` 的融合点云（与在线发送侧同源预处理产物，
  float16 存储 -> float32 前向）；query = record 的规范类名（`use_vlfm_nlp=False`）；
* 前向 = `TSP3D.predict`（在线服务同一条 legacy 推理路径：剪枝 + CBA + NMS 单候选），
  `sigma_sce=0.15`（近场动态放宽复刻在线）、`tau=0.15`；
* **`sigma_tar=0.0`：取全量输出（含 <0.70 的低分框）**——这是与线上唯一（且必要）
  的不同点，用于看见被在线准入线截断的部分；
* 种子：predict 内部按 `(TSP3D_SEED + 调用计数)` 固定；两次运行（A/B 权重）遍历顺序
  一致 => 相同调用序上 seed 相同，对比公平。

统计
----
* pos 样本（label=1，含 GT）：是否有框 / 最高分框与 GT AABB 的 3D IoU / 命中率
  （IoU>=0.25 与 >=0.50）/ 被剪光（pruned_all）计数 / 分数分布（全量 + 命中）；
* neg 样本（label=0）：输出框的有无与分数分布（fp 输出倾向）；
* 逐场景 / 逐类聚合；per-record 明细落 jsonl 供深挖。

用法
----
  python -u ft_tsp3d/diag_offline.py --ckpt <权重> --tag <标签> --out-dir ft_tsp3d/ft_log/e3 \
      [--neg-per-view 1] [--views-cap N] [--scenes s1,s2]
  python -u ft_tsp3d/diag_offline.py --compare <A_summary.json> <B_summary.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_loader import LoaderConfig, LocalViewDataset  # noqa: E402

# 在线同源推理参数（vlfm/policy/tsp3d_objectnav_policy.py::_query_tsp3d_client）
SIGMA_SCE = 0.15
TAU = 0.15
NEAR_FIELD_DIST = 1.0
NEAR_FIELD_SIGMA_SCALE = 0.8
CAMERA_HEIGHT = 0.88
IOU_HIT = 0.25          # 「命中真目标」的主判据（宽松）
IOU_STRONG = 0.50       # 强命中
SCORE_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-6]


# ----------------------------------------------------------------------
# 几何
# ----------------------------------------------------------------------
def aabb_from_corners(corners: np.ndarray) -> np.ndarray:
    """(8,3) corners -> (min_x, min_y, min_z, max_x, max_y, max_z)。"""
    return np.array([
        corners[:, 0].min(), corners[:, 1].min(), corners[:, 2].min(),
        corners[:, 0].max(), corners[:, 1].max(), corners[:, 2].max(),
    ], dtype=np.float64)


def aabb_from_gt(gt: np.ndarray) -> np.ndarray:
    """GT canonical (x, y, z, l, w, h)（中心+尺寸）-> min/max AABB。"""
    cx, cy, cz, l, w, h = [float(v) for v in gt[:6]]
    return np.array([cx - l / 2, cy - w / 2, cz - h / 2,
                     cx + l / 2, cy + w / 2, cz + h / 2], dtype=np.float64)


def iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    """轴对齐 3D IoU（min/max 六元组）。"""
    lo = np.maximum(a[:3], b[:3])
    hi = np.minimum(a[3:], b[3:])
    inter = float(np.prod(np.clip(hi - lo, 0.0, None)))
    va = float(np.prod(a[3:] - a[:3]))
    vb = float(np.prod(b[3:] - b[:3]))
    union = va + vb - inter
    return inter / union if union > 1e-9 else 0.0


def dynamic_sigma_sce(pts: np.ndarray) -> float:
    """复刻在线近场放宽：min_dist < 1.0 m 时 sigma_sce 按距离比例收缩。"""
    if len(pts) == 0:
        return SIGMA_SCE
    cam = np.array([0.0, 0.0, CAMERA_HEIGHT])
    d = float(np.linalg.norm(pts[:, :3] - cam, axis=1).min())
    if d < NEAR_FIELD_DIST:
        return SIGMA_SCE * max(NEAR_FIELD_SIGMA_SCALE, d / NEAR_FIELD_DIST)
    return SIGMA_SCE


# ----------------------------------------------------------------------
# 统计容器
# ----------------------------------------------------------------------
def _hist(scores: List[float]) -> Dict[str, int]:
    arr = np.asarray(scores, dtype=np.float64)
    out: Dict[str, int] = {}
    for i in range(len(SCORE_BINS) - 1):
        lo, hi = SCORE_BINS[i], SCORE_BINS[i + 1]
        out[f"{lo:.1f}-{hi:.1f}"] = int(((arr >= lo) & (arr < hi)).sum())
    return out


def _score_stats(scores: List[float]) -> Dict[str, Any]:
    if not scores:
        return {"n": 0}
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "ge070_pct": round(float((arr >= 0.70).mean() * 100.0), 2),
        "hist": _hist(scores),
        "scores": [round(float(v), 5) for v in scores],
    }


def _finalize_pos(d: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, d["n"])
    return {
        "n": d["n"],
        "has_box_pct": round(d["has_box"] / n * 100.0, 2),
        "no_box_pct": round(d["no_box"] / n * 100.0, 2),
        "pruned_all": d["pruned_all"],
        "hit025_pct": round(d["hit25"] / n * 100.0, 2),
        "hit050_pct": round(d["hit50"] / n * 100.0, 2),
        "mean_best_iou": round(d["iou_sum"] / n, 4),
        "score_all": _score_stats(d["scores"]),
        "score_hit": _score_stats(d["scores_hit"]),
    }


def _finalize_neg(d: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, d["n"])
    return {
        "n": d["n"],
        "has_box_pct": round(d["has_box"] / n * 100.0, 2),
        "score": _score_stats(d["scores"]),
    }


def _new_pos() -> Dict[str, Any]:
    return {"n": 0, "has_box": 0, "no_box": 0, "pruned_all": 0,
            "hit25": 0, "hit50": 0, "iou_sum": 0.0, "scores": [], "scores_hit": []}


def _new_neg() -> Dict[str, Any]:
    return {"n": 0, "has_box": 0, "scores": []}


# ----------------------------------------------------------------------
# run：单权重全量遍历
# ----------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    import torch
    from vlfm.vlm.tsp3d import TSP3D

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(Path(args.ckpt).resolve())
    if not Path(ckpt).is_file():
        print(f"[E3][error] ckpt 不存在: {ckpt}")
        return 1

    cfg = LoaderConfig(
        split=args.split,
        use_sidecar=False,                  # 判读不需要教师信号
        inst_dir="/nonexistent__e3_skip",   # 绕过可见实体计算（E3 不消费）
        cache_scenes=3,
        limit_views=int(args.views_cap),
        scenes=args.scenes,
    )
    ds = LocalViewDataset(cfg)
    print(f"[E3] dataset: split={args.split} views={len(ds)} scenes={len(ds.scene_ids)} "
          f"neg_per_view={args.neg_per_view} views_cap={args.views_cap}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[E3] loading ckpt: {ckpt}")
    model = TSP3D(weights_path=ckpt, device=device)

    pos, neg = _new_pos(), _new_neg()
    per_class: Dict[str, Dict[str, Any]] = {}
    jsonl_path = out_dir / f"{args.tag}_per_record.jsonl"
    jsonl = jsonl_path.open("w")
    t0 = time.time()
    call_idx = 0
    failures = 0
    views_done = 0

    try:
        for vi in range(len(ds)):
            item = ds[vi]
            pts = item["points"]
            recs = item["records"]
            picks = [r for r in recs if int(r["label"]) == 1]
            negs = [r for r in recs if int(r["label"]) == 0][: max(0, int(args.neg_per_view))]
            for r in picks + negs:
                try:
                    dets, diag = model.predict(
                        pcd=pts,
                        text=str(r["query"]),
                        sigma_sce=dynamic_sigma_sce(pts),
                        sigma_tar=0.0,          # 全量输出（判读核心）
                        tau=TAU,
                        use_vlfm_nlp=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    if failures <= 5:
                        print(f"[E3][warn] predict failed: {exc}")
                    continue

                pruned_all = bool(diag.get("pruned_all_layer1") or diag.get("pruned_all_layer0"))
                has_box = len(dets) > 0
                score: Optional[float] = float(dets[0]["confidence"]) if has_box else None
                best_iou = 0.0
                hit25 = hit50 = False
                rec = {
                    "i": call_idx, "scene": item["scene"], "view": item["view"],
                    "query": str(r["query"]), "label": int(r["label"]),
                    "has_box": has_box, "pruned_all": pruned_all,
                    "score": (round(score, 5) if score is not None else None),
                }
                if int(r["label"]) == 1 and r["gt"] is not None:
                    if has_box:
                        a = aabb_from_corners(np.asarray(dets[0]["box_3d"], dtype=np.float64))
                        best_iou = iou_3d(a, aabb_from_gt(np.asarray(r["gt"])))
                        hit25 = best_iou >= IOU_HIT
                        hit50 = best_iou >= IOU_STRONG
                    pos["n"] += 1
                    pos["has_box"] += int(has_box)
                    pos["no_box"] += int(not has_box)
                    pos["pruned_all"] += int(pruned_all)
                    pos["hit25"] += int(hit25)
                    pos["hit50"] += int(hit50)
                    pos["iou_sum"] += best_iou
                    if score is not None:
                        pos["scores"].append(score)
                        if hit25:
                            pos["scores_hit"].append(score)
                    cls = str(r["query"])
                    pc = per_class.setdefault(
                        cls, {"pos": _new_pos(), "neg": _new_neg()})
                    pc["pos"]["n"] += 1
                    pc["pos"]["has_box"] += int(has_box)
                    pc["pos"]["hit25"] += int(hit25)
                    pc["pos"]["iou_sum"] += best_iou
                    if score is not None:
                        pc["pos"]["scores"].append(score)
                    rec["best_iou"] = round(best_iou, 4)
                    rec["hit25"] = hit25
                else:
                    neg["n"] += 1
                    neg["has_box"] += int(has_box)
                    if score is not None:
                        neg["scores"].append(score)
                    cls = str(r["query"])
                    pc = per_class.setdefault(
                        cls, {"pos": _new_pos(), "neg": _new_neg()})
                    pc["neg"]["n"] += 1
                    pc["neg"]["has_box"] += int(has_box)
                    if score is not None:
                        pc["neg"]["scores"].append(score)
                jsonl.write(json.dumps(rec) + "\n")
                call_idx += 1
            views_done += 1
            if views_done % 100 == 0 or views_done == len(ds):
                el = time.time() - t0
                rate = el / max(1, call_idx)
                eta = rate * (len(ds) - views_done) * (1 + max(0, int(args.neg_per_view))) * 0.9
                print(f"[E3] {views_done}/{len(ds)} views | calls={call_idx} | "
                      f"{rate:.3f} s/call | elapsed={el / 60:.1f} min | ETA≈{eta / 60:.0f} min")
    finally:
        jsonl.close()

    wall = time.time() - t0
    summary = {
        "tag": args.tag, "ckpt": ckpt, "split": args.split,
        "sigma_tar": 0.0, "sigma_sce": SIGMA_SCE, "tau": TAU,
        "neg_per_view": int(args.neg_per_view), "views": views_done,
        "n_pos": pos["n"], "n_neg": neg["n"], "n_calls": call_idx,
        "failures": failures, "wall_sec": round(wall, 1),
        "pos": _finalize_pos(pos),
        "neg": _finalize_neg(neg),
        "per_class": {
            cls: {"pos": _finalize_pos(v["pos"]), "neg": _finalize_neg(v["neg"])}
            for cls, v in sorted(per_class.items())
        },
        "jsonl": str(jsonl_path),
    }
    sum_path = out_dir / f"{args.tag}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    p = summary["pos"]
    n = summary["neg"]
    print("=" * 72)
    print(f"[E3-summary] tag={args.tag}  ckpt={ckpt}")
    print(f"  views={views_done}  pos={p['n']}  neg={n['n']}  calls={call_idx}  "
          f"failures={failures}  wall={wall / 60:.1f} min ({wall / max(1, call_idx):.3f} s/call)")
    print(f"  POS  has_box={p['has_box_pct']}%  no_box={p['no_box_pct']}%  "
          f"pruned_all={p['pruned_all']}  hit@0.25={p['hit025_pct']}%  hit@0.50={p['hit050_pct']}%  "
          f"mean_best_iou={p['mean_best_iou']}")
    if p["score_all"].get("n"):
        s = p["score_all"]
        print(f"  POS-score(all)     mean={s['mean']}  median={s['median']}  "
              f"p10={s['p10']}  p90={s['p90']}  ge0.70={s['ge070_pct']}%")
    if p["score_hit"].get("n"):
        s = p["score_hit"]
        print(f"  POS-score(hit)     mean={s['mean']}  median={s['median']}  ge0.70={s['ge070_pct']}%")
    if n["score"].get("n"):
        s = n["score"]
        print(f"  NEG  has_box={n['has_box_pct']}%  score mean={s['mean']}  "
              f"median={s['median']}  ge0.70={s['ge070_pct']}%")
    print("  per-class(pos): " + "  ".join(
        f"{cls}[n={v['pos']['n']} hit={v['pos']['hit025_pct']}%]"
        for cls, v in summary["per_class"].items() if v["pos"]["n"] > 0))
    print(f"  summary -> {sum_path}")
    print(f"  per-record -> {jsonl_path}")
    print("=" * 72)
    return 0


# ----------------------------------------------------------------------
# compare：两份 summary 对比
# ----------------------------------------------------------------------
def _fmt(v: Optional[float], suffix: str = "%") -> str:
    return f"{v}{suffix}" if v is not None else "-"


def compare(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    pa, pb = a["pos"], b["pos"]
    na, nb = a["neg"], b["neg"]

    def d(x: float, y: float) -> str:
        return f"{x - y:+.2f}pp"

    print("=" * 78)
    print(f"[E3-compare]  A = {a['tag']} ({Path(a['ckpt']).name})  vs  B = {b['tag']} ({Path(b['ckpt']).name})")
    print(f"  A: pos={pa['n']} neg={na['n']} | B: pos={pb['n']} neg={nb['n']}")
    print("-" * 78)
    print(f"{'metric':<34}{'A':>12}{'B':>12}{'delta(A-B)':>14}")
    print(f"{'pos has_box %':<34}{_fmt(pa['has_box_pct']):>12}{_fmt(pb['has_box_pct']):>12}"
          f"{d(pa['has_box_pct'], pb['has_box_pct']):>14}")
    print(f"{'pos no_box %':<34}{_fmt(pa['no_box_pct']):>12}{_fmt(pb['no_box_pct']):>12}"
          f"{d(pa['no_box_pct'], pb['no_box_pct']):>14}")
    print(f"{'pos hit@0.25 %':<34}{_fmt(pa['hit025_pct']):>12}{_fmt(pb['hit025_pct']):>12}"
          f"{d(pa['hit025_pct'], pb['hit025_pct']):>14}")
    print(f"{'pos hit@0.50 %':<34}{_fmt(pa['hit050_pct']):>12}{_fmt(pb['hit050_pct']):>12}"
          f"{d(pa['hit050_pct'], pb['hit050_pct']):>14}")
    print(f"{'pos mean_best_iou':<34}{pa['mean_best_iou']:>12}{pb['mean_best_iou']:>12}"
          f"{pa['mean_best_iou'] - pb['mean_best_iou']:>+14.4f}")
    sa, sb = pa["score_all"], pb["score_all"]
    print(f"{'pos score(any) mean':<34}{_fmt(sa.get('mean'), ''):>12}{_fmt(sb.get('mean'), ''):>12}"
          f"{_num_delta(sa.get('mean'), sb.get('mean')):>14}")
    print(f"{'pos score(any) ge0.70 %':<34}{_fmt(sa.get('ge070_pct')):>12}{_fmt(sb.get('ge070_pct')):>12}"
          f"{d(sa.get('ge070_pct', 0.0), sb.get('ge070_pct', 0.0)):>14}")
    ha, hb = pa["score_hit"], pb["score_hit"]
    print(f"{'pos score(hit) mean':<34}{_fmt(ha.get('mean'), ''):>12}{_fmt(hb.get('mean'), ''):>12}"
          f"{_num_delta(ha.get('mean'), hb.get('mean')):>14}")
    print(f"{'pos score(hit) ge0.70 %':<34}{_fmt(ha.get('ge070_pct')):>12}{_fmt(hb.get('ge070_pct')):>12}"
          f"{d(ha.get('ge070_pct', 0.0), hb.get('ge070_pct', 0.0)):>14}")
    print(f"{'neg score mean':<34}{_fmt(na['score'].get('mean'), ''):>12}{_fmt(nb['score'].get('mean'), ''):>12}"
          f"{_num_delta(na['score'].get('mean'), nb['score'].get('mean')):>14}")
    print(f"{'neg score ge0.70 %':<34}{_fmt(na['score'].get('ge070_pct')):>12}{_fmt(nb['score'].get('ge070_pct')):>12}"
          f"{d(na['score'].get('ge070_pct', 0.0), nb['score'].get('ge070_pct', 0.0)):>14}")
    print("-" * 78)
    print("per-class pos (hit@0.25 % | n):")
    classes = sorted(set(a["per_class"]) | set(b["per_class"]))
    for cls in classes:
        va = a["per_class"].get(cls, {}).get("pos", {})
        vb = b["per_class"].get(cls, {}).get("pos", {})
        if not va and not vb:
            continue
        print(f"  {cls:<16} A: {_fmt(va.get('hit025_pct'))} (n={va.get('n', 0)})   "
              f"B: {_fmt(vb.get('hit025_pct'))} (n={vb.get('n', 0)})")
    print("-" * 78)
    # 简易判读提示（参考，不代替人工判断）
    hit_gap = pa["hit025_pct"] - pb["hit025_pct"]
    score_gap = sa.get("ge070_pct", 0.0) - sb.get("ge070_pct", 0.0)
    if hit_gap >= -5.0 and score_gap <= -5.0:
        verdict = "倾向 A：框召回接近、分数明显下移（被 0.70 线截断）"
    elif hit_gap <= -10.0:
        verdict = "倾向 B：框召回显著下降（输出消失/分布外崩）"
    else:
        verdict = "混合：召回与分数均有变化，需要看分桶与 per-class 细读"
    print(f"[hint] hit@0.25 gap(A-B)={hit_gap:+.2f}pp  ge0.70 gap={score_gap:+.2f}pp  =>  {verdict}")
    print("=" * 78)
    return 0


def _num_delta(x: Optional[float], y: Optional[float]) -> str:
    if x is None or y is None:
        return "-"
    return f"{x - y:+.4f}"


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="E3 离线判读（在线同源推理，全量输出）")
    ap.add_argument("--ckpt", type=str, default="", help="权重路径（run 模式）")
    ap.add_argument("--tag", type=str, default="", help="输出标签（run 模式）")
    ap.add_argument("--out-dir", type=str, default=str(_HERE / "ft_log" / "e3"))
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--neg-per-view", type=int, default=1)
    ap.add_argument("--views-cap", type=int, default=0)
    ap.add_argument("--scenes", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A.json", "B.json"),
                    help="对比两份 summary（不跑模型）")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])
    if not args.ckpt or not args.tag:
        ap.error("run 模式需要 --ckpt 与 --tag（或使用 --compare A B）")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
