#!/usr/bin/env python3
"""SP1 离线统计：S-penalty 逐检测记录的证据级判读

口径
----
* 输入 = 导航评测落盘的 per-detection jsonl（`<det_log_dir>/s4_sp1_detections.jsonl`，
  由 `vlfm/utils/episode_stats_logger.dump_detection_logs` 写出）。每条
  `{"kind": "det"}` 记录携带 raw S-penalty 输入（`conf` / `s` / `w_s` / `branch` /
  `dist` / `admitted`）与该检测的 TP 标签（`tp`：检测质心落在 GT 目标掩码
  膨胀 10 px 内，与 `aggregate_detect_stats` 同一判据）。
* 样本 = 每一步写入前的全部 TSP3D 候选（含被拒检测）；`admitted=True` 是入记忆子集。
* **未落盘层**：最终被选导航候选（nav goal）当前不进 jsonl；SP1 只报 candidate 层与
  admitted 子集两层。
* `covered` = 落盘时 `s` 有值且 `>0`；`uncovered` = `s is None` 或 `<=0`，单独成一路。
* `branch` 依赖落盘时的 `t_S`。本工具用 `--t-s` 按固定 `s_min=0.1` 对 raw `s`
  重算准入；S 查询半径固定为 0.5m，不参与搜索。

统计输出（只输出统计量，SP2 的搜索范围由人工据此确认）
----
* 覆盖率、TP/FP 构成；三路（covered-high / covered-low / uncovered）precision 与 TP recall；
* raw `c`、融合 `c'`、S-only 的 AUROC/AUPRC，及 S-only 相对 raw `c` 的 AUPRC/AUROC 增量；
* 高召回工作点 FPR@TPR90/95、Precision@R90/95（raw `c` 与融合 `c'`）；
* TP/FP 的 S 分位数（p10/p25/p50/p75/p90）与 covered 比例——供 SP2 收缩 `t_S`；
* 按类别 / 场景 / 距离档分层；
* `--grid-t-s`：在给定网格上重算 `c'`，输出每组 precision / TP recall。

用法
----
  python -u ft_tsp3d/eval_tools/s_evidence.py \
      --jsonl <输出目录 或 jsonl 文件> [--sigma-tar 0.60] [--uncovered-w 1.0] \
      [--grid-t-s 0.110,0.1125,0.115,0.1175,0.120] [--out sp1.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlfm.s_penalty_constants import S_MIN, S_QUERY_RADIUS_M

DET_FILE = "s4_sp1_detections.jsonl"
DIST_BANDS: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 1e9))
PROPOSAL_THRESHOLDS: Tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank-based AUROC with average ranks for ties; None if one class is missing."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    s_sorted = s[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Tie-aware average precision (step-wise recall increments); None if no positives."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool).astype(float)
    n_pos = float(y.sum())
    if n_pos <= 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    tp = np.cumsum(y[order])
    fp = np.cumsum(1.0 - y[order])
    keep = np.r_[np.where(np.diff(s_sorted) != 0)[0], s_sorted.size - 1]
    prec = tp[keep] / np.maximum(tp[keep] + fp[keep], 1e-12)
    rec = tp[keep] / n_pos
    ap = 0.0
    prev = 0.0
    for p, r in zip(prec, rec):
        ap += float(p) * float(r - prev)
        prev = float(r)
    return float(ap)


def working_point(
    scores: Sequence[float], labels: Sequence[bool], target_recall: float
) -> Optional[Dict[str, float]]:
    """Threshold achieving >= target_recall with the highest possible cut (min FPR)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    ok = np.where(tp >= target_recall * n_pos)[0]
    if ok.size == 0:
        return None
    cut = int(ok[0])
    n_neg = int((~y).sum())
    return {
        "threshold": float(s_sorted[cut]),
        "recall": float(tp[cut] / n_pos),
        "fpr": float(fp[cut] / n_neg) if n_neg else 0.0,
        "precision": float(tp[cut] / max(tp[cut] + fp[cut], 1)),
    }


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    qs = (10, 25, 50, 75, 90)
    out = {f"p{q}": float(np.percentile(a, q)) for q in qs}
    out["n"] = int(a.size)
    out["mean"] = float(a.mean())
    return out


def _dist_bin(dist: Optional[float]) -> str:
    if dist is None or not np.isfinite(dist):
        return "unknown"
    for lo, hi in DIST_BANDS:
        if lo <= dist < hi:
            return f"{lo:g}-{hi:g}m" if hi < 1e8 else f">={lo:g}m"
    return "unknown"


# ---------------------------------------------------------------------------
# loading / reweighting
# ---------------------------------------------------------------------------
def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def load_detections(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dets: List[Dict[str, Any]] = []
    eps: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = rec.get("kind")
                if kind == "det":
                    dets.append(rec)
                elif kind == "ep":
                    eps.append(rec)
    return dets, eps


def resolve_paths(target: str) -> List[Path]:
    p = Path(target)
    if p.is_dir():
        found = sorted(p.glob("*.jsonl"))
        if not found:
            raise SystemExit(f"目录内没有 jsonl: {p}")
        return found
    if not p.is_file():
        raise SystemExit(f"输入不存在: {p}")
    return [p]


def reweight(s: Optional[float], t_s: float, uncovered_w: float) -> float:
    """w_S for logged raw S with fixed ``s_min=0.1`` and variable ``t_S``."""
    if s is None or s <= 0.0:
        return float(uncovered_w)
    return float(min(1.0, max(float(s), S_MIN) / max(float(t_s), 1e-9)))


def _covered(det: Dict[str, Any]) -> bool:
    s = _f(det.get("s"))
    return s is not None and s > 0.0


def _branch_now(det: Dict[str, Any], t_s: float) -> str:
    s = _f(det.get("s"))
    if s is None or s <= 0.0:
        return "uncovered"
    return "covered-high" if s >= t_s else "covered-low"


def _admit_stats(
    dets: Sequence[Dict[str, Any]], t_s: float, uncovered_w: float, sigma_tar: float
) -> Dict[str, Any]:
    n_tp = sum(1 for d in dets if d.get("tp"))
    tp_kept = fp_kept = n_kept = 0
    for det in dets:
        conf = _f(det.get("conf"))
        if conf is None:
            continue
        w = reweight(_f(det.get("s")), t_s, uncovered_w)
        if conf * w < sigma_tar:
            continue
        n_kept += 1
        if det.get("tp"):
            tp_kept += 1
        else:
            fp_kept += 1
    return {
        "t_S": float(t_s),
        "s_min": S_MIN,
        "derived_min_w": float(min(1.0, S_MIN / max(float(t_s), 1e-9))),
        "radius_m": S_QUERY_RADIUS_M,
        "uncovered_w": float(uncovered_w),
        "admitted": n_kept,
        "tp_kept": tp_kept,
        "fp_kept": fp_kept,
        "precision": float(tp_kept / max(n_kept, 1)),
        "tp_recall": float(tp_kept / max(n_tp, 1)),
    }


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarize(
    dets: Sequence[Dict[str, Any]],
    eps: Sequence[Dict[str, Any]],
    sigma_tar: float,
    t_s: float,
    uncovered_w: float,
    sigma_prop_floor: Optional[float] = None,
) -> Dict[str, Any]:
    n = len(dets)
    if n == 0:
        return {"n_det": 0, "n_ep": len(eps)}

    tp_flags = [bool(d.get("tp")) for d in dets]
    conf = np.array([_f(d.get("conf")) or 0.0 for d in dets], dtype=float)
    weights = np.array(
        [reweight(_f(d.get("s")), t_s, uncovered_w) for d in dets], dtype=float
    )
    conf_amp = conf * weights
    s_raw = np.array([_f(d.get("s")) if _f(d.get("s")) is not None else 0.0 for d in dets])
    covered = np.array([_covered(d) for d in dets])

    branches: Dict[str, Dict[str, Any]] = {}
    for name in ("covered-high", "covered-low", "uncovered"):
        idx = [i for i, d in enumerate(dets) if _branch_now(d, t_s) == name]
        tp = sum(1 for i in idx if tp_flags[i])
        admitted = sum(1 for i in idx if conf_amp[i] >= sigma_tar)
        branches[name] = {
            "n": len(idx),
            "tp": tp,
            "fp": len(idx) - tp,
            "precision": float(tp / max(len(idx), 1)),
            "tp_recall": float(tp / max(sum(tp_flags), 1)),
            "admitted": admitted,
        }

    s_cov = [float(v) for v, c in zip(s_raw, covered) if c]
    s_tp = [float(v) for v, c, f in zip(s_raw, covered, tp_flags) if c and f]
    s_fp = [float(v) for v, c, f in zip(s_raw, covered, tp_flags) if c and not f]

    proposal_shadow: Dict[str, Dict[str, Any]] = {}
    n_tp_all = int(sum(tp_flags))
    for threshold in PROPOSAL_THRESHOLDS:
        field = f"pass_prop_{int(round(threshold * 100)):03d}"
        # New logs carry explicit flags; raw confidence is a compatible
        # fallback for SP1 files written before sigma_prop was split out.
        selected = [
            d for d in dets
            if bool(d.get(field, (_f(d.get("conf")) or 0.0) >= threshold))
        ]
        tp_selected = sum(1 for d in selected if d.get("tp"))
        proposal_shadow[f"{threshold:.2f}"] = {
            "n": len(selected),
            "tp": tp_selected,
            "fp": len(selected) - tp_selected,
            "precision": float(tp_selected / max(len(selected), 1)),
            "tp_recall": float(tp_selected / max(n_tp_all, 1)),
            "observable": bool(
                sigma_prop_floor is not None
                and float(sigma_prop_floor) <= threshold + 1e-12
            ),
        }

    logged_floors = sorted(
        {
            float(value)
            for d in dets
            if (value := _f(d.get("sigma_prop_floor"))) is not None
        }
    )
    min_conf = min((float(v) for v in conf), default=None)
    shadow_counts = [proposal_shadow[f"{value:.2f}"]["n"] for value in PROPOSAL_THRESHOLDS]
    proposal_shadow_contract = {
        "requested_floor": sigma_prop_floor,
        "logged_floors": logged_floors,
        "thresholds": list(PROPOSAL_THRESHOLDS),
        "all_thresholds_observable": bool(
            sigma_prop_floor is not None
            and float(sigma_prop_floor) <= min(PROPOSAL_THRESHOLDS) + 1e-12
        ),
        "counts_monotone": all(
            left >= right for left, right in zip(shadow_counts, shadow_counts[1:])
        ),
        "min_logged_conf": min_conf,
        "n_below_admission": int(sum(float(value) < sigma_tar for value in conf)),
    }

    def _disc(scores: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if mask is None:
            sc, lb = scores, tp_flags
        else:
            sc = [float(v) for v, m in zip(scores, mask) if m]
            lb = [f for f, m in zip(tp_flags, mask) if m]
        return {
            "n": int(len(sc)),
            "n_pos": int(sum(1 for v in lb if v)),
            "auroc": binary_auc(sc, lb),
            "auprc": average_precision(sc, lb),
            "wp90": working_point(sc, lb, 0.90),
            "wp95": working_point(sc, lb, 0.95),
        }

    out: Dict[str, Any] = {
        "n_det": n,
        "n_ep": len(eps),
        "n_tp": int(sum(tp_flags)),
        "precision_raw": float(sum(tp_flags) / n),
        "coverage": float(covered.sum() / n),
        "proposal_shadow": proposal_shadow,
        "proposal_shadow_contract": proposal_shadow_contract,
        "branches": branches,
        "discrimination": {
            "raw_c": _disc(conf),
            "fused_c_amp": _disc(conf_amp),
            "s_only_covered": _disc(s_raw, covered),
            "s_with_uncovered_zero": _disc(s_raw),
        },
        "s_quantiles": {
            "covered": _quantiles(s_cov),
            "covered_tp": _quantiles(s_tp),
            "covered_fp": _quantiles(s_fp),
        },
        "by_class": {},
        "by_scene": {},
        "by_dist": {},
        "admitted_under_logged_cfg": _admit_stats(dets, t_s, uncovered_w, sigma_tar),
    }

    keys = sorted({str(d.get("target_object") or "?") for d in dets})
    for key in keys:
        sub = [d for d in dets if str(d.get("target_object") or "?") == key]
        f = [bool(d.get("tp")) for d in sub]
        out["by_class"][key] = {
            "n": len(sub),
            "tp": int(sum(f)),
            "precision": float(sum(f) / max(len(sub), 1)),
            "coverage": float(sum(1 for d in sub if _covered(d)) / max(len(sub), 1)),
            "s_covered": _quantiles([_f(d.get("s")) or 0.0 for d in sub if _covered(d)]),
        }
    for key in sorted({str(d.get("scene") or "?") for d in dets}):
        sub = [d for d in dets if str(d.get("scene") or "?") == key]
        f = [bool(d.get("tp")) for d in sub]
        out["by_scene"][key] = {
            "n": len(sub),
            "tp": int(sum(f)),
            "precision": float(sum(f) / max(len(sub), 1)),
        }
    for band in [f"{lo:g}-{hi:g}m" if hi < 1e8 else f">={lo:g}m" for lo, hi in DIST_BANDS] + [
        "unknown"
    ]:
        sub = [d for d in dets if _dist_bin(_f(d.get("dist"))) == band]
        if not sub:
            continue
        f = [bool(d.get("tp")) for d in sub]
        out["by_dist"][band] = {
            "n": len(sub),
            "tp": int(sum(f)),
            "precision": float(sum(f) / max(len(sub), 1)),
            "coverage": float(sum(1 for d in sub if _covered(d)) / max(len(sub), 1)),
        }
    return out


def grid_table(
    dets: Sequence[Dict[str, Any]],
    t_s_list: Sequence[float],
    uncovered_w: float,
    sigma_tar: float,
) -> List[Dict[str, Any]]:
    return [
        _admit_stats(dets, t_s, uncovered_w, sigma_tar)
        for t_s in t_s_list
    ]


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def print_report(summary: Dict[str, Any], grid: Sequence[Dict[str, Any]]) -> None:
    if summary.get("n_det", 0) == 0:
        print("无检测记录（n_det=0）")
        return
    print("=" * 72)
    print(f"SP1 S-evidence | det={summary['n_det']} ep={summary['n_ep']} "
          f"tp={summary['n_tp']} ({summary['precision_raw'] * 100:.2f}%) "
          f"coverage={summary['coverage'] * 100:.2f}%")
    print("-- branch split (recomputed at logged t_S) --")
    print(f"  {'branch':<14}{'n':>8}{'tp':>7}{'precision':>11}{'tp_recall':>11}{'admitted':>10}")
    for name, b in summary["branches"].items():
        print(f"  {name:<14}{b['n']:>8}{b['tp']:>7}{b['precision']:>11.4f}"
              f"{b['tp_recall']:>11.4f}{b['admitted']:>10}")
    print("-- sigma_prop shadow (same online trajectory; not closed-loop) --")
    contract = summary.get("proposal_shadow_contract", {})
    print(
        "  contract: "
        f"floor={contract.get('requested_floor')} "
        f"logged={contract.get('logged_floors')} "
        f"all_observable={contract.get('all_thresholds_observable')} "
        f"min_conf={contract.get('min_logged_conf')} "
        f"n_below_admission={contract.get('n_below_admission')}"
    )
    print(f"  {'threshold':>9}{'n':>8}{'tp':>7}{'fp':>7}{'precision':>11}{'tp_recall':>11}")
    for threshold, d in summary.get("proposal_shadow", {}).items():
        print(
            f"  {threshold:>9}{d['n']:>8}{d['tp']:>7}{d['fp']:>7}"
            f"{d['precision']:>11.4f}{d['tp_recall']:>11.4f}"
        )
    print("-- discrimination --")
    print(f"  {'score':<24}{'n':>8}{'n_pos':>7}{'auroc':>9}{'auprc':>9}"
          f"{'FPR@R90':>10}{'P@R90':>9}{'FPR@R95':>10}{'P@R95':>9}")
    for name, d in summary["discrimination"].items():
        w90, w95 = d.get("wp90"), d.get("wp95")
        print(
            f"  {name:<24}{d['n']:>8}{d['n_pos']:>7}{_fmt(d['auroc']):>9}{_fmt(d['auprc']):>9}"
            f"{_fmt((w90 or {}).get('fpr'), 3):>10}{_fmt((w90 or {}).get('precision'), 3):>9}"
            f"{_fmt((w95 or {}).get('fpr'), 3):>10}{_fmt((w95 or {}).get('precision'), 3):>9}"
        )
    print("-- S quantiles (covered) --")
    for name, q in summary["s_quantiles"].items():
        if not q:
            print(f"  {name:<14} (empty)")
            continue
        print(f"  {name:<14} n={q['n']:>6} mean={q['mean']:.4f} "
              + " ".join(f"{k}={q[k]:.4f}" for k in ("p10", "p25", "p50", "p75", "p90")))
    print("-- by class --")
    for cls, d in sorted(summary["by_class"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {cls:<20} n={d['n']:>6} tp={d['tp']:>5} prec={d['precision']:.4f} "
              f"cov={d['coverage']:.4f}")
    print("-- by distance --")
    for band, d in summary["by_dist"].items():
        print(f"  {band:<12} n={d['n']:>6} tp={d['tp']:>5} prec={d['precision']:.4f} "
              f"cov={d['coverage']:.4f}")
    if grid:
        print("-- t_S grid replay (s_min=0.1, radius=0.5m fixed) --")
        print(f"  {'t_S':>7}{'min_w':>9}{'admitted':>10}{'tp_kept':>9}{'fp_kept':>9}"
              f"{'precision':>11}{'tp_recall':>11}")
        for row in grid:
            print(f"  {row['t_S']:>7.4f}{row['derived_min_w']:>9.4f}{row['admitted']:>10}"
                  f"{row['tp_kept']:>9}{row['fp_kept']:>9}{row['precision']:>11.4f}"
                  f"{row['tp_recall']:>11.4f}")


def _csv_floats(text: Optional[str], default: Sequence[float]) -> List[float]:
    if not text:
        return list(default)
    return [float(v) for v in text.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="SP1 S-penalty per-detection evidence summary")
    ap.add_argument("--jsonl", required=True,
                    help=f"输出目录（扫 *.jsonl）或单个 jsonl；默认文件名 {DET_FILE}")
    ap.add_argument("--sigma-tar", type=float, default=0.60, help="准入原始门限（默认 0.60）")
    ap.add_argument("--t-s", type=float, default=None,
                    help="三路判定用的 t_S；缺省取 ep 头行记录的 t_S")
    ap.add_argument("--uncovered-w", type=float, default=1.0, help="uncovered 权重（默认 1.0）")
    ap.add_argument(
        "--require-proposal-shadow",
        action="store_true",
        help="要求服务候选下限覆盖 0.45/0.50/0.55/0.60 四档，并校验落盘 provenance",
    )
    ap.add_argument("--grid-t-s", default=None,
                    help="逗号分隔，例如 0.110,0.1125,0.115,0.1175,0.120")
    ap.add_argument("--out", default=None, help="可选：把完整 summary 写成 json")
    args = ap.parse_args()

    paths = resolve_paths(args.jsonl)
    dets, eps = load_detections(paths)
    print(f"loaded {len(dets)} detections / {len(eps)} episodes from "
          f"{', '.join(str(p) for p in paths)}")

    cfg = {}
    for ep in eps:
        cfg.update(ep.get("cfg") or {})
    # Prefer the Phase-G schema while retaining compatibility with earlier SP1 logs.
    t_s = args.t_s if args.t_s is not None else float(
        cfg.get("t_S", cfg.get("s_penalty_thresh", 0.15))
    )
    sigma_prop_floor_raw = cfg.get("sigma_prop")
    sigma_prop_floor = (
        None if sigma_prop_floor_raw is None else float(sigma_prop_floor_raw)
    )
    summary = summarize(
        dets,
        eps,
        args.sigma_tar,
        t_s,
        args.uncovered_w,
        sigma_prop_floor,
    )
    if args.require_proposal_shadow:
        contract = summary["proposal_shadow_contract"]
        if not contract["all_thresholds_observable"]:
            raise SystemExit(
                "proposal shadow is truncated: sigma_prop must be <= 0.45"
            )
        if contract["logged_floors"] != [sigma_prop_floor]:
            raise SystemExit(
                "proposal shadow provenance mismatch: "
                f"cfg={sigma_prop_floor}, logged={contract['logged_floors']}"
            )
        if not contract["counts_monotone"]:
            raise SystemExit("proposal shadow counts are not monotone")
    summary["_inputs"] = {"files": [str(p) for p in paths], "t_S": t_s,
                          "s_min": S_MIN, "radius_m": S_QUERY_RADIUS_M,
                          "sigma_tar": args.sigma_tar, "uncovered_w": args.uncovered_w,
                          "sigma_prop_floor": sigma_prop_floor,
                          "cfg_from_ep_header": cfg}
    grid = grid_table(
        dets,
        _csv_floats(args.grid_t_s, [0.110, 0.1125, 0.115, 0.1175, 0.120]),
        args.uncovered_w,
        args.sigma_tar,
    ) if args.grid_t_s else []

    print_report(summary, grid)

    if args.out:
        payload = {"summary": summary, "grid": grid}
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
