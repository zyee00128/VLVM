#!/usr/bin/env python3
"""Stage3 微调 · 指标统计脚本（§11.3 #3）。

两个子命令：
  * `gate`   —— 离线门控指标（读 `pipeline --dump-gate` 产物）：假阳性早停拦截率 /
    门控召回（GT 存在视图开闸率）/ 误关率 + AUC + 分场景细分（判读口径 = §4（5）：
    拦截率 / 召回 / 误关率 + SR；**非 Gap**）。
  * `online` —— 从评测 `run.log` 提取：SR / Oracle / spl / soft_spl / 检测级
    precision 等汇总键值 + 失败归因行（fp / fn / nv）。格式随 trainer 输出演进，
    必要时在此更新锚点正则。

用法：
```
python ft_tsp3d/metrics.py gate   --dump ft_runs/m4/gate_val.npz --tau 0.30
python ft_tsp3d/metrics.py online --log outputs/eval_hm3d_2026-xx-xx_xxxx_VLVM.log
```
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np

from calibrate_gate import auc_score, gate_metrics


# ---------------------------------------------------------------------------
# gate 子命令
# ---------------------------------------------------------------------------


def cmd_gate(args: argparse.Namespace) -> None:
    z = np.load(args.dump, allow_pickle=False)
    p = np.asarray(z["p_exist"], dtype=np.float64)
    y = np.asarray(z["y"], dtype=np.int64)
    w = np.asarray(z["weight"], dtype=np.float64) if "weight" in z else None
    tau = float(args.tau)

    report: Dict[str, Any] = {
        "dump": str(args.dump), "tau": tau, "n": int(p.size),
        "overall": gate_metrics(p, y, tau, w),
        "auc": auc_score(p, y),
    }
    if "scene" in z:
        scenes = np.asarray(z["scene"])
        by_scene = {}
        for s in sorted(set(scenes.tolist())):
            m = scenes == s
            by_scene[s] = gate_metrics(p[m], y[m], tau, w[m] if w is not None else None)
        report["by_scene"] = by_scene
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print("dumped:", args.out)


# ---------------------------------------------------------------------------
# online 子命令（评测日志提取；宽容匹配 + 最后值优先）
# ---------------------------------------------------------------------------

_PATTERNS: Dict[str, str] = {
    # 注意排除 "Oracle Success Rate" 行（否则子串会污染 SR 提取）
    "success_rate": r"(?<!Oracle )[Ss]uccess\s*[Rr]ate\s*[:：]\s*([0-9.]+)",
    "oracle": r"[Oo]racle\s*[Ss]uccess\s*[Rr]ate\s*[:：]\s*([0-9.]+)",
    "spl": r"\bspl\b\s*[:：]\s*([0-9.]+)",
    "soft_spl": r"\bsoft[_ ]spl\b\s*[:：]\s*([0-9.]+)",
    "avg_steps": r"\bavg[_ ]steps\b\s*[:：]\s*([0-9.]+)",
    "fps": r"\bfps\b\s*[:：]\s*([0-9.]+)",
    "precision": r"[Pp]recision\s*[:：]?\s*([0-9.]+)\s*%?",
}

_CAUSE_HINTS = ("fp", "fn", "nv", "bad_stop", "timeout")


def cmd_online(args: argparse.Namespace) -> None:
    log = Path(args.log)
    if not log.is_file():
        raise SystemExit(f"log not found: {log}")
    text = log.read_text(errors="ignore")
    lines = text.splitlines()

    summary: Dict[str, Any] = {"log": str(log), "lines": len(lines), "metrics": {}}
    for key, pat in _PATTERNS.items():
        hits = re.findall(pat, text)
        if hits:
            vals = [float(v) for v in hits]
            summary["metrics"][key] = {"last": vals[-1], "n_hits": len(vals),
                                       "min": min(vals), "max": max(vals)}
    # 失败归因行：包含 fp/fn/nv 关键字的汇总行（保留原文，人工核读）
    cause_lines = [ln.strip() for ln in lines
                   if any(h in ln for h in _CAUSE_HINTS) and len(ln) < 220]
    summary["cause_lines"] = cause_lines[-args.cause_tail:] if args.cause_tail > 0 else []

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print("dumped:", args.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="离线门控三指标（τ 给定）")
    g.add_argument("--dump", required=True)
    g.add_argument("--tau", type=float, required=True)
    g.add_argument("--out", default=None)
    g.set_defaults(fn=cmd_gate)

    o = sub.add_parser("online", help="评测日志汇总提取")
    o.add_argument("--log", required=True)
    o.add_argument("--cause-tail", type=int, default=12, help="输出最后 N 条失败归因相关行")
    o.add_argument("--out", default=None)
    o.set_defaults(fn=cmd_online)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
