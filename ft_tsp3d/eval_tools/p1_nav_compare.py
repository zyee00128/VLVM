#!/usr/bin/env python3
"""Build the Phase-G P1 navigation comparison after the t_S grid finishes."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METRICS = ("success", "oracle", "spl", "soft_spl", "steps")


def load_episodes(path: Path, scenes: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = set(scenes)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "episode" and row.get("scene") in wanted:
                rows.append(row)
    return rows


def aggregate_episodes(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    if not n:
        raise ValueError("empty episode set")
    return {
        "n": n,
        "success": sum(float(row.get("success", 0.0)) for row in rows) / n,
        "oracle": sum(float(row.get("oracle_success", 0.0)) for row in rows) / n,
        "spl": sum(float(row.get("spl", 0.0)) for row in rows) / n,
        "soft_spl": sum(float(row.get("soft_spl", 0.0)) for row in rows) / n,
        "steps": sum(float(row.get("steps_count", 0.0)) for row in rows) / n,
    }


def parse_last_scene_table(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse the last VLFM/VLVM ``Per-scene metrics`` table in a text log."""
    row_re = re.compile(
        r"^(\S+)\s+(\d+)\s+([0-9.]+)%\s+([0-9.]+)%\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
    )
    current: Optional[Dict[str, Dict[str, float]]] = None
    last: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "=== Per-scene metrics (multi-scene run) ===":
                current = {}
                continue
            if current is None:
                continue
            match = row_re.match(line)
            if match:
                scene, n, success, oracle, spl, soft_spl, steps = match.groups()
                current[scene] = {
                    "n": int(n),
                    "success": float(success) / 100.0,
                    "oracle": float(oracle) / 100.0,
                    "spl": float(spl),
                    "soft_spl": float(soft_spl),
                    "steps": float(steps),
                }
            elif line.startswith("===") and current:
                last = current
                current = None
    if current:
        last = current
    if not last:
        raise ValueError(f"no per-scene table found in {path}")
    return last


def aggregate_scene_table(
    table: Mapping[str, Mapping[str, float]], scenes: Sequence[str]
) -> Dict[str, float]:
    missing = [scene for scene in scenes if scene not in table]
    if missing:
        raise ValueError(f"missing scenes: {missing}")
    total = sum(int(table[scene]["n"]) for scene in scenes)
    out: Dict[str, float] = {"n": total}
    for metric in METRICS:
        if metric in ("success", "oracle"):
            # The historical text tables print rates to one decimal place.
            # Recover integer episode counts before pooling so 297-episode SR
            # agrees with the original count-based result.
            numerator = sum(
                round(float(table[scene][metric]) * int(table[scene]["n"]))
                for scene in scenes
            )
        else:
            numerator = sum(
                float(table[scene][metric]) * int(table[scene]["n"])
                for scene in scenes
            )
        out[metric] = numerator / total
    return out


def episode_scene_table(
    rows: Sequence[Mapping[str, Any]], scenes: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    return {
        scene: aggregate_episodes([row for row in rows if row.get("scene") == scene])
        for scene in scenes
    }


def paired_delta(
    new_rows: Sequence[Mapping[str, Any]], legacy_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    def index(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
        return {
            (str(row.get("scene")), str(row.get("episode_id"))): row
            for row in rows
        }

    new_idx, old_idx = index(new_rows), index(legacy_rows)
    keys = sorted(set(new_idx) & set(old_idx))
    gained = sum(
        int(new_idx[key].get("success", 0)) == 1
        and int(old_idx[key].get("success", 0)) == 0
        for key in keys
    )
    lost = sum(
        int(new_idx[key].get("success", 0)) == 0
        and int(old_idx[key].get("success", 0)) == 1
        for key in keys
    )
    return {"matched": len(keys), "success_gained": gained, "success_lost": lost}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def metric_row(name: str, t_s: str, stats: Mapping[str, float]) -> str:
    return (
        f"| {name} | {t_s} | {int(stats['n'])} | {pct(stats['success'])} | "
        f"{pct(stats['oracle'])} | {stats['spl']:.4f} | {stats['soft_spl']:.4f} | "
        f"{stats['steps']:.4f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--legacy-episodes",
        default="outputs/phase_g_p1_2026-10-17_220022/p1_c_r05/detections/episodes.jsonl",
    )
    parser.add_argument(
        "--p0-log", default="outputs/phase_g_sp1_2026-10-16_213713/navigation.log"
    )
    parser.add_argument(
        "--vlfm-log", default="outputs/eval_hm3d_2026-09-25_115044_vlfm_hm3d_full.log"
    )
    parser.add_argument(
        "--scenes", default="5cdEh9F2hJL,mL8ThkuaVTM,QaLdnwvtxbs"
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    configs: List[Dict[str, Any]] = []
    for path in sorted(run_root.glob("p1_ts*/detections/episodes.jsonl")):
        rows = load_episodes(path, scenes)
        cfg = (rows[0].get("cfg") or {}) if rows else {}
        t_s = float(cfg.get("t_S"))
        configs.append(
            {
                "name": path.parents[1].name,
                "t_S": t_s,
                "episodes": rows,
                "overall": aggregate_episodes(rows),
                "by_scene": episode_scene_table(rows, scenes),
            }
        )
    if not configs:
        raise SystemExit(f"no completed p1_ts episode logs under {run_root}")

    legacy_rows = load_episodes(Path(args.legacy_episodes), scenes)
    legacy = {
        "name": "legacy P1-C-R05",
        "t_S": 0.115,
        "overall": aggregate_episodes(legacy_rows),
        "by_scene": episode_scene_table(legacy_rows, scenes),
    }
    p0_table = parse_last_scene_table(Path(args.p0_log))
    vlfm_table = parse_last_scene_table(Path(args.vlfm_log))
    p0 = {
        "name": "P0/SP1",
        "t_S": 0.150,
        "overall": aggregate_scene_table(p0_table, scenes),
        "by_scene": {scene: p0_table[scene] for scene in scenes},
    }
    vlfm = {
        "name": "VLFM",
        "t_S": None,
        "overall": aggregate_scene_table(vlfm_table, scenes),
        "by_scene": {scene: vlfm_table[scene] for scene in scenes},
    }

    # Primary metric = SR. Differences smaller than one episode in this 297-episode
    # panel are treated as a tie, then resolved by SPL, SoftSPL and fewer steps.
    episode_resolution = 1.0 / int(configs[0]["overall"]["n"])
    max_sr = max(item["overall"]["success"] for item in configs)
    sr_tied = [
        item for item in configs
        if max_sr - item["overall"]["success"] < episode_resolution - 1e-12
    ]
    best = max(
        sr_tied,
        key=lambda item: (
            item["overall"]["spl"],
            item["overall"]["soft_spl"],
            -item["overall"]["steps"],
        ),
    )
    paired = {
        item["name"]: paired_delta(item["episodes"], legacy_rows) for item in configs
    }

    lines = [
        "# Phase G P1 navigation comparison",
        "",
        "同一组三场景共 297 集。新 P1 内部以 SR 为主指标；差异不足 1/297（0.34 pp）按并列处理，依次用 SPL、SoftSPL、较少 steps 破同分。",
        "",
        "| 配置 | t_S | episodes | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in configs:
        lines.append(metric_row(item["name"], f"{item['t_S']:.4f}", item["overall"]))
    for item in (legacy, p0, vlfm):
        t_s = "-" if item["t_S"] is None else f"{item['t_S']:.4f}"
        lines.append(metric_row(item["name"], t_s, item["overall"]))

    lines.extend(
        [
            "",
            f"**新 P1 最佳档：{best['name']}（t_S={best['t_S']:.4f}）。**",
            "",
            "## 相对 legacy P1-C-R05 的逐集变化",
            "",
            "| 新配置 | matched | success gained | success lost | net |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in configs:
        row = paired[item["name"]]
        lines.append(
            f"| {item['name']} | {row['matched']} | {row['success_gained']} | "
            f"{row['success_lost']} | {row['success_gained'] - row['success_lost']:+d} |"
        )

    lines.extend(["", "## 逐场景", ""])
    for scene in scenes:
        lines.extend(
            [
                f"### {scene}",
                "",
                "| 配置 | t_S | episodes | SR ↑ | Oracle ↑ | SPL ↑ | SoftSPL ↑ | steps ↓ |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in configs:
            lines.append(metric_row(item["name"], f"{item['t_S']:.4f}", item["by_scene"][scene]))
        for item in (legacy, p0, vlfm):
            t_s = "-" if item["t_S"] is None else f"{item['t_S']:.4f}"
            lines.append(metric_row(item["name"], t_s, item["by_scene"][scene]))
        lines.append("")

    lines.extend(
        [
            "## 公式影响的解释边界",
            "",
            "- 与 legacy P1-C-R05 的逐集变化可以判断新方案是否改变闭环结果，但新网格没有相同的 `t_S=0.115`，因此不是严格的公式单变量因果对照。",
            "- `t_S=0.1500` 与 P0/SP1 具有相同 t_S 和 radius，但 P0 使用旧 penalty 公式且日志契约更早，差异仍不能全部归因于单一参数。",
            "- VLFM 是跨框架导航参照，只用于判断最终导航水平，不用于证明 S-penalty 公式的因果作用。",
        ]
    )
    markdown = "\n".join(lines) + "\n"

    payload = {
        "scenes": scenes,
        "selection_rule": "SR; within 1 episode tie -> SPL -> SoftSPL -> fewer steps",
        "best_new": {"name": best["name"], "t_S": best["t_S"]},
        "new": [
            {k: value for k, value in item.items() if k != "episodes"}
            for item in configs
        ],
        "references": [legacy, p0, vlfm],
        "paired_vs_legacy": paired,
        "formula_effect_isolated": False,
    }
    out_md = Path(args.out_md) if args.out_md else run_root / "p1_navigation_comparison.md"
    out_json = Path(args.out_json) if args.out_json else run_root / "p1_navigation_comparison.json"
    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(markdown)
    print(f"saved -> {out_md}")
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
