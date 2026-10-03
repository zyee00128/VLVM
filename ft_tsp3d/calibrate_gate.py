#!/usr/bin/env python3
"""τ_gate 训练后校准（§8 · 2 / §11.3 #2）。

输入：`pipeline.py --dump-gate` 的产物 npz（`p_exist` / `y` / `weight` / `scene` / `view`；
`y` = record 的 `label`，即「查询类在本视图任一实例像素 ≥ min_vis_px(40)」）。

方法（保召回约束，§9 门控=召回守卫）：
  * 门控短路条件 = `P_exist ≤ τ` ⇒ 开闸 = `P_exist > τ`；
  * 在「召回 ≥ r_target」约束下取 **最大 τ**（实质 = 正样本分布的低分位，取
    `τ* = quantile(P | y=1, 1 − r_target)` 并回校 FN 数）；
  * 输出 τ 扫描表（拦截率 / 召回 / 误关率）+ 建议 τ + AUC，供人工裁决与在线部署。

判定口径提醒：门控 FN（真目标被门掉）不可恢复，优先保召回；本阶段判定指标 =
假阳性早停拦截率 / 门控召回（GT 存在视图开闸率）/ 误关率 + SR（非 Gap）。

用法：
```
python ft_tsp3d/calibrate_gate.py --dump ft_runs/m4/gate_val.npz
python ft_tsp3d/calibrate_gate.py --dump ... --tau 0.30        # 只报该 τ 的指标
```
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auc_score(p: np.ndarray, y: np.ndarray) -> Optional[float]:
    """P(P_exist_pos > P_exist_neg)（并列取 0.5），无正/负样本时 None。"""
    pos = p[y == 1]
    neg = p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    ranks = _rankdata(p)
    n_pos, n_neg = int(pos.size), int(neg.size)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gate_metrics(p: np.ndarray, y: np.ndarray, tau: float, weight: Optional[np.ndarray] = None) -> Dict:
    """给定 τ 的门控三指标：拦截率（y=0 被门掉）/ 召回（y=1 开闸）/ 误关率（y=1 被门掉）。"""
    short = p <= float(tau)
    out: Dict[str, object] = {"tau": float(tau), "n": int(p.size)}
    if (y == 1).any():
        out["recall_open"] = float((~short[y == 1]).mean())        # 门控召回（开闸率）
        out["false_close"] = float(short[y == 1].mean())           # 误关率
        out["n_fn"] = int((short & (y == 1)).sum())
    if (y == 0).any():
        out["intercept"] = float(short[y == 0].mean())             # 假阳性早停拦截率（离线口径）
        out["n_pass"] = int((~short & (y == 0)).sum())
    if weight is not None and weight.size == p.size and (y == 0).any():
        w = weight.astype(np.float64)
        out["intercept_w"] = float((w[short & (y == 0)].sum()) / max(w[y == 0].sum(), 1e-9))
    return out


def recommend_tau(p: np.ndarray, y: np.ndarray, target_recall: float = 0.995) -> Dict:
    """保召回约束下的最大 τ（正样本低分位 + FN 回校）。"""
    pos = np.sort(p[y == 1])
    if pos.size == 0:
        return {"tau": None, "reason": "no_positive_samples"}
    k = int(math.floor((1.0 - float(target_recall)) * pos.size))  # 允许的 FN 数
    tau = float(pos[min(k, pos.size - 1)])
    # 回校：开闸数（pos > tau）须 ≥ target_recall·n（把 τ 往下压到满足）
    while tau > 0.0 and float((pos > tau).mean()) < target_recall:
        tau = float(np.nextafter(tau, 0.0))
        if tau <= 0.0:
            break
    return {"tau": tau, "target_recall": float(target_recall),
            "recall_open": float((pos > tau).mean()), "n_fn_allowed": k}


def scan_table(p: np.ndarray, y: np.ndarray, n: int = 25) -> List[Dict]:
    taus = np.linspace(0.02, 0.98, int(n))
    return [gate_metrics(p, y, float(t)) for t in taus]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", required=True, help="pipeline --dump-gate 产物 npz")
    ap.add_argument("--tau", type=float, default=None, help="给定 τ 时只报该值指标")
    ap.add_argument("--target-recall", type=float, default=0.995)
    ap.add_argument("--table", type=int, default=25, help="扫描表点数（0 = 不打印）")
    ap.add_argument("--out", default=None, help="报告落盘 json")
    args = ap.parse_args()

    z = np.load(args.dump, allow_pickle=False)
    p = np.asarray(z["p_exist"], dtype=np.float64)
    y = np.asarray(z["y"], dtype=np.int64)
    w = np.asarray(z["weight"], dtype=np.float64) if "weight" in z else None
    if p.size == 0:
        raise SystemExit("dump 为空（val 未收集到 P_exist）")

    report: Dict = {
        "dump": str(args.dump), "n": int(p.size),
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "auc": auc_score(p, y),
        "pos_stats": {"p50": float(np.median(p[y == 1])) if (y == 1).any() else None,
                      "p05": float(np.percentile(p[y == 1], 5)) if (y == 1).any() else None,
                      "p95": float(np.percentile(p[y == 1], 95)) if (y == 1).any() else None},
        "neg_stats": {"p50": float(np.median(p[y == 0])) if (y == 0).any() else None,
                      "p95": float(np.percentile(p[y == 0], 95)) if (y == 0).any() else None},
    }
    if args.tau is not None:
        report["at_tau"] = gate_metrics(p, y, args.tau, w)
    else:
        report["recommended"] = recommend_tau(p, y, args.target_recall)
        rec = report["recommended"]
        if rec.get("tau") is not None:
            report["at_recommended"] = gate_metrics(p, y, float(rec["tau"]), w)
        if args.table > 0:
            report["scan"] = scan_table(p, y, args.table)

    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.table > 0 and args.tau is None and "scan" in report:
        print("\n   tau   intercept   recall_open   false_close")
        for row in report["scan"]:
            print(f"  {row['tau']:.2f}   {row.get('intercept', float('nan')):.4f}      "
                  f"{row.get('recall_open', float('nan')):.4f}       {row.get('false_close', float('nan')):.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print("dumped:", args.out)


if __name__ == "__main__":
    main()
