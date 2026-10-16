#!/usr/bin/env python3
"""E3 离线判读：两个权重在 val 视图上的「在线同源推理」输出对比

口径
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
* 区分度：AUROC / AUPRC、FPR@TPR90/95、Precision@R90/95、同视图 pair accuracy；
* 分数形状与校准：正确/错误均值、gap、Cohen's d、ECE-15、Brier、NLL；
* Gate checkpoint 额外报告 presence AUROC/AUPRC、实际门槛下 recall_open / intercept；
* 逐场景 / 逐类聚合；per-record 明细落 jsonl 供深挖。

用法
----
  python -u ft_tsp3d/eval_tools/diag_offline.py --ckpt <权重> --tag <标签> --out-dir ft_tsp3d/ft_log/e3 \
      [--neg-per-view 1] [--views-cap N] [--scenes s1,s2] [--gate-threshold TAU]
  python -u ft_tsp3d/eval_tools/diag_offline.py --compare <A_summary.json> <B_summary.json>
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



# 几何
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


# 统计容器
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


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 1-based."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(scores)
    value = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(value)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Non-interpolated AP with equal-score samples grouped as one threshold."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = (labels[order] == 1).astype(np.float64)
    tp = np.cumsum(ranked_labels)
    end = np.r_[np.flatnonzero(ranked_scores[1:] != ranked_scores[:-1]), ranked_scores.size - 1]
    recall = tp[end] / n_pos
    precision = tp[end] / (end + 1)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _compact_score_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    clipped = np.clip(values.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "sd": round(float(values.std()), 6),
        "p10": round(float(np.percentile(values, 10)), 6),
        "p90": round(float(np.percentile(values, 90)), 6),
        "logit_mean": round(float(logits.mean()), 6),
        "logit_sd": round(float(logits.std()), 6),
    }


def _operating_point(scores: np.ndarray, labels: np.ndarray, target_recall: float) -> Dict[str, Any]:
    """Largest discrete threshold satisfying recall >= target_recall under score >= threshold."""
    pos = np.sort(scores[labels == 1])
    n_pos = int(pos.size)
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"target_recall": float(target_recall), "threshold": None}
    allowed_misses = int(np.floor((1.0 - float(target_recall)) * n_pos + 1e-12))
    threshold = float(pos[min(allowed_misses, n_pos - 1)])
    predicted = scores >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = n_pos - tp
    tn = n_neg - fp
    return {
        "target_recall": float(target_recall),
        "threshold": threshold,
        "recall": float(tp / n_pos),
        "fpr": float(fp / n_neg),
        "precision": float(tp / max(tp + fp, 1)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _calibration(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> Dict[str, Any]:
    if scores.size == 0:
        return {"n": 0}
    labels_f = labels.astype(np.float64)
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((scores - labels_f) ** 2))
    nll = float(-np.mean(labels_f * np.log(clipped) + (1.0 - labels_f) * np.log(1.0 - clipped)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    table: List[Dict[str, Any]] = []
    for i in range(bins):
        mask = ((scores >= edges[i]) &
                ((scores < edges[i + 1]) if i < bins - 1 else (scores <= edges[i + 1])))
        if not bool(mask.any()):
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels_f[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
        table.append({
            "lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(mask.sum()),
            "confidence": confidence, "accuracy": accuracy,
        })
    return {"n": int(scores.size), "bins": bins, "ece": ece,
            "brier": brier, "nll": nll, "table": table}


def _pair_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["scene"], record["view"]), []).append(record)
    pair_values: List[float] = []
    violation_gaps: List[float] = []
    n_candidate_views = 0
    for view_records in grouped.values():
        correct = [float(r.get("score") or 0.0) for r in view_records
                   if int(r["label"]) == 1 and bool(r.get("hit25", False))]
        target_absent = [float(r.get("score") or 0.0) for r in view_records
                         if int(r["label"]) == 0]
        errors = [float(r.get("score") or 0.0) for r in view_records
                  if not (int(r["label"]) == 1 and bool(r.get("hit25", False)))]
        for positive in correct:
            for negative in target_absent:
                pair_values.append(1.0 if positive > negative else 0.5 if positive == negative else 0.0)
        if correct and errors:
            n_candidate_views += 1
            gap = max(errors) - max(correct)
            if gap > 0.0:
                violation_gaps.append(gap)
    pair_arr = np.asarray(pair_values, dtype=np.float64)
    violation_arr = np.asarray(violation_gaps, dtype=np.float64)
    return {
        "n_pairs": int(pair_arr.size),
        "accuracy": (float(pair_arr.mean()) if pair_arr.size else None),
        "n_candidate_views": n_candidate_views,
        "n_violation_views": int(violation_arr.size),
        "violation_rate": (float(violation_arr.size / n_candidate_views)
                           if n_candidate_views else None),
        "violation_gap_mean": (float(violation_arr.mean()) if violation_arr.size else None),
        "violation_gap_median": (float(np.median(violation_arr)) if violation_arr.size else None),
    }


def _discrimination_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = np.asarray([float(record.get("score") or 0.0) for record in records], dtype=np.float64)
    labels = np.asarray([
        int(int(record["label"]) == 1 and bool(record.get("hit25", False)))
        for record in records
    ], dtype=np.int8)
    correct = scores[labels == 1]
    errors = scores[labels == 0]
    correct_stats = _compact_score_stats(correct)
    error_stats = _compact_score_stats(errors)
    prob_sd = np.sqrt((correct.var() + errors.var()) / 2.0) if correct.size and errors.size else 0.0
    correct_logits = np.log(np.clip(correct, 1e-7, 1 - 1e-7) /
                            (1 - np.clip(correct, 1e-7, 1 - 1e-7)))
    error_logits = np.log(np.clip(errors, 1e-7, 1 - 1e-7) /
                          (1 - np.clip(errors, 1e-7, 1 - 1e-7)))
    logit_sd = (np.sqrt((correct_logits.var() + error_logits.var()) / 2.0)
                if correct.size and errors.size else 0.0)
    probability_gap = (float(correct.mean() - errors.mean()) if correct.size and errors.size else None)
    logit_gap = (float(correct_logits.mean() - error_logits.mean())
                 if correct.size and errors.size else None)
    return {
        "definition": "correct := label=1 and Top-1 IoU>=0.25; all other records := error",
        "n": int(scores.size), "n_correct": int(correct.size), "n_error": int(errors.size),
        "correct_rate": float(correct.size / max(scores.size, 1)),
        "auroc": _binary_auc(scores, labels),
        "auprc": _average_precision(scores, labels),
        "at_tpr90": _operating_point(scores, labels, 0.90),
        "at_tpr95": _operating_point(scores, labels, 0.95),
        "correct_score": correct_stats,
        "error_score": error_stats,
        "score_gap": probability_gap,
        "logit_gap": logit_gap,
        "cohen_d": (float(probability_gap / prob_sd) if probability_gap is not None and prob_sd > 0 else None),
        "logit_cohen_d": (float(logit_gap / logit_sd) if logit_gap is not None and logit_sd > 0 else None),
        "pair": _pair_metrics(records),
        "calibration": _calibration(scores, labels, bins=15),
    }


def _presence_gate_metrics(records: List[Dict[str, Any]], threshold: Optional[float]) -> Optional[Dict[str, Any]]:
    selected = [record for record in records if record.get("presence_prob") is not None]
    if not selected:
        return None
    scores = np.asarray([float(record["presence_prob"]) for record in selected], dtype=np.float64)
    labels = np.asarray([int(record["label"]) for record in selected], dtype=np.int8)
    out: Dict[str, Any] = {
        "definition": "positive := target class is present in the view",
        "n": int(scores.size), "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
        "auroc": _binary_auc(scores, labels), "auprc": _average_precision(scores, labels),
        "positive_score": _compact_score_stats(scores[labels == 1]),
        "negative_score": _compact_score_stats(scores[labels == 0]),
        "threshold": threshold,
    }
    if threshold is not None:
        closed = scores < float(threshold)
        pos = labels == 1
        neg = labels == 0
        out["at_threshold"] = {
            "recall_open": float((~closed[pos]).mean()) if bool(pos.any()) else None,
            "false_close": float(closed[pos].mean()) if bool(pos.any()) else None,
            "intercept": float(closed[neg].mean()) if bool(neg.any()) else None,
            "target_absent_fpr": float((~closed[neg]).mean()) if bool(neg.any()) else None,
            "n_false_close": int((closed & pos).sum()),
            "n_intercept": int((closed & neg).sum()),
        }
    return out


def _f4(value: Optional[float]) -> str:
    return "-" if value is None else f"{float(value):.4f}"


# run：单权重全量遍历
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
    gate_enabled = bool(model.ft_config.get("enable_gate", False))
    effective_gate_threshold = args.gate_threshold
    if gate_enabled and effective_gate_threshold is None:
        effective_gate_threshold = float(model.ft_config.get("gate_threshold", 0.30))
    print(f"[E3] gate: enabled={gate_enabled} threshold={effective_gate_threshold}")

    pos, neg = _new_pos(), _new_neg()
    per_class: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
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
                        gate_threshold=args.gate_threshold,
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
                presence_prob = diag.get("presence_prob")
                if presence_prob is not None:
                    rec["presence_prob"] = round(float(presence_prob), 6)
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
                records.append(rec)
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
    discrimination = _discrimination_metrics(records)
    presence_gate = _presence_gate_metrics(records, effective_gate_threshold)
    summary = {
        "tag": args.tag, "ckpt": ckpt, "split": args.split,
        "sigma_tar": 0.0, "sigma_sce": SIGMA_SCE, "tau": TAU,
        "gate_enabled": gate_enabled, "gate_threshold": effective_gate_threshold,
        "neg_per_view": int(args.neg_per_view), "views": views_done,
        "n_pos": pos["n"], "n_neg": neg["n"], "n_calls": call_idx,
        "failures": failures, "wall_sec": round(wall, 1),
        "pos": _finalize_pos(pos),
        "neg": _finalize_neg(neg),
        "per_class": {
            cls: {"pos": _finalize_pos(v["pos"]), "neg": _finalize_neg(v["neg"])}
            for cls, v in sorted(per_class.items())
        },
        "discrimination": discrimination,
        "presence_gate": presence_gate,
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
    disc = summary["discrimination"]
    r90, r95 = disc["at_tpr90"], disc["at_tpr95"]
    print(f"  DISC auroc={_f4(disc['auroc'])}  auprc={_f4(disc['auprc'])}  "
          f"pair_acc={_f4(disc['pair']['accuracy'])}  gap={_f4(disc['score_gap'])}")
    print(f"  SHAPE cohen_d={_f4(disc['cohen_d'])}  logit_gap={_f4(disc['logit_gap'])}  "
          f"logit_cohen_d={_f4(disc['logit_cohen_d'])}  "
          f"view_violation={_f4(disc['pair']['violation_rate'])}  "
          f"violation_gap={_f4(disc['pair']['violation_gap_mean'])}")
    print(f"  HIGH-RECALL fpr@tpr90={_f4(r90.get('fpr'))}  precision@r90={_f4(r90.get('precision'))}  "
          f"fpr@tpr95={_f4(r95.get('fpr'))}  precision@r95={_f4(r95.get('precision'))}")
    calibration = disc["calibration"]
    print(f"  CAL ece15={_f4(calibration.get('ece'))}  brier={_f4(calibration.get('brier'))}  "
          f"nll={_f4(calibration.get('nll'))}")
    gate_metrics = summary.get("presence_gate")
    if gate_metrics is not None:
        at_gate = gate_metrics.get("at_threshold", {})
        print(f"  GATE auroc={_f4(gate_metrics['auroc'])}  auprc={_f4(gate_metrics['auprc'])}  "
              f"threshold={gate_metrics.get('threshold')}  "
              f"recall_open={at_gate.get('recall_open')}  intercept={at_gate.get('intercept')}")
    print("  per-class(pos): " + "  ".join(
        f"{cls}[n={v['pos']['n']} hit={v['pos']['hit025_pct']}%]"
        for cls, v in summary["per_class"].items() if v["pos"]["n"] > 0))
    print(f"  summary -> {sum_path}")
    print(f"  per-record -> {jsonl_path}")
    print("=" * 72)
    return 0


# compare：两份 summary 对比
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
    da, db = a.get("discrimination"), b.get("discrimination")
    if da and db:
        metric_rows = [
            ("AUROC", da.get("auroc"), db.get("auroc")),
            ("AUPRC", da.get("auprc"), db.get("auprc")),
            ("FPR@TPR90", da.get("at_tpr90", {}).get("fpr"), db.get("at_tpr90", {}).get("fpr")),
            ("FPR@TPR95", da.get("at_tpr95", {}).get("fpr"), db.get("at_tpr95", {}).get("fpr")),
            ("Precision@R90", da.get("at_tpr90", {}).get("precision"), db.get("at_tpr90", {}).get("precision")),
            ("Precision@R95", da.get("at_tpr95", {}).get("precision"), db.get("at_tpr95", {}).get("precision")),
            ("pair accuracy", da.get("pair", {}).get("accuracy"), db.get("pair", {}).get("accuracy")),
            ("view violation rate", da.get("pair", {}).get("violation_rate"),
             db.get("pair", {}).get("violation_rate")),
            ("violation gap mean", da.get("pair", {}).get("violation_gap_mean"),
             db.get("pair", {}).get("violation_gap_mean")),
            ("score gap", da.get("score_gap"), db.get("score_gap")),
            ("logit gap", da.get("logit_gap"), db.get("logit_gap")),
            ("Cohen's d", da.get("cohen_d"), db.get("cohen_d")),
            ("logit Cohen's d", da.get("logit_cohen_d"), db.get("logit_cohen_d")),
            ("ECE-15", da.get("calibration", {}).get("ece"), db.get("calibration", {}).get("ece")),
            ("Brier", da.get("calibration", {}).get("brier"), db.get("calibration", {}).get("brier")),
            ("NLL", da.get("calibration", {}).get("nll"), db.get("calibration", {}).get("nll")),
        ]
        print("-" * 78)
        for name, value_a, value_b in metric_rows:
            print(f"{name:<34}{_f4(value_a):>12}{_f4(value_b):>12}"
                  f"{_num_delta(value_a, value_b):>14}")
    ga, gb = a.get("presence_gate"), b.get("presence_gate")
    if ga or gb:
        print("-" * 78)
        gate_rows = [
            ("Gate AUROC", (ga or {}).get("auroc"), (gb or {}).get("auroc")),
            ("Gate AUPRC", (ga or {}).get("auprc"), (gb or {}).get("auprc")),
            ("Gate recall_open", (ga or {}).get("at_threshold", {}).get("recall_open"),
             (gb or {}).get("at_threshold", {}).get("recall_open")),
            ("Gate intercept", (ga or {}).get("at_threshold", {}).get("intercept"),
             (gb or {}).get("at_threshold", {}).get("intercept")),
        ]
        for name, value_a, value_b in gate_rows:
            print(f"{name:<34}{_f4(value_a):>12}{_f4(value_b):>12}"
                  f"{_num_delta(value_a, value_b):>14}")
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
    ap.add_argument("--gate-threshold", type=float, default=None,
                    help="显式 Presence Gate 门槛；省略时使用 checkpoint/default 配置")
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
