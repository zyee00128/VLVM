"""S5 offline calibration on logged detection runs (no simulation).

Consumes a run's ``detections/`` logs and reports:
  * stop-rule replay at the final nav-goal decision point
  * track-level upgrade rules (num_obs / max sigma_c / close re-detection)
  * delete-threshold proxy ("frame had detections but none matched the track")
  * admission band expansion cost

Usage:
  python -m vlfm.eval_tools.s5_offline_cal --run-dir outputs/s2_2026-10-23_012020
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(row: Dict[str, Any]) -> Tuple[str, int]:
    return str(row.get("scene")), int(row.get("episode_id"))


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    data = sorted(float(v) for v in values)
    if not data:
        return {}
    def pick(frac: float) -> float:
        return data[min(len(data) - 1, int(frac * len(data)))]
    return {
        "n": len(data),
        "p10": pick(0.10), "p25": pick(0.25), "p50": pick(0.50),
        "p75": pick(0.75), "p90": pick(0.90), "max": data[-1],
    }


def build_tracks(
    memory: List[Dict[str, Any]],
    cand_dist: Dict[Tuple[str, int, str], float],
) -> Dict[str, Dict[str, Any]]:
    tracks: Dict[str, Dict[str, Any]] = {}
    for row in memory:
        if row.get("event") not in ("new", "merge", "upgrade"):
            continue
        uid = row.get("memory_uid")
        if uid is None:
            continue
        track = tracks.setdefault(uid, {
            "tp": bool(row.get("tp")),
            "num_obs": 0,
            "max_sigma": 0.0,
            "min_dist": None,
            "steps": [],
            "state": "confirmed",
            "initial_state": str(row.get("lifecycle_state") or "confirmed"),
            "ever_provisional": str(row.get("lifecycle_state")) == "provisional",
            "deleted": False,
            "episode": _key(row),
        })
        track["tp"] = bool(row.get("tp"))
        track["num_obs"] = max(int(track["num_obs"]), int(row.get("num_obs") or 1))
        track["max_sigma"] = max(track["max_sigma"], float(row.get("sigma_c_max") or 0.0))
        if row.get("event") == "new":
            track["initial_state"] = str(row.get("lifecycle_state") or "confirmed")
            track["ever_provisional"] = track["initial_state"] == "provisional"
        dist = row.get("detect_dist")
        if dist is None:
            dist = cand_dist.get((_key(row)[0], _key(row)[1], str(row.get("candidate_id"))))
        if dist is not None:
            dist = float(dist)
            track["min_dist"] = dist if track["min_dist"] is None else min(track["min_dist"], dist)
        track["steps"].append(int(row.get("step") or 0))
        track["state"] = str(row.get("lifecycle_state") or track["state"])
    for row in memory:
        if row.get("event") == "delete" and row.get("memory_uid") in tracks:
            tracks[row["memory_uid"]]["deleted"] = True
    return tracks


def stop_rule_table(
    episodes: List[Dict[str, Any]],
    nav_goals: List[Dict[str, Any]],
    tracks: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_ep: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in nav_goals:
        by_ep[_key(row)].append(row)

    finals = []
    for episode in episodes:
        rows = by_ep.get(_key(episode), [])
        if not rows:
            continue
        goal = max(rows, key=lambda r: int(r.get("step") or 0))
        track = tracks.get(goal.get("memory_uid"))
        if track is None:
            continue
        final_step = int(goal.get("step") or 0)
        recent = [s for s in track["steps"] if final_step - 3 <= s <= final_step]
        finals.append({
            "tp": bool(goal.get("tp")),
            "num_obs": int(track["num_obs"]),
            "max_sigma": float(track["max_sigma"]),
            "min_dist": track["min_dist"],
            "state": track["state"],
            "recent_redetect_3": bool(recent),
        })

    def rule(name: str, fn) -> Dict[str, Any]:
        allowed = [f for f in finals if fn(f)]
        tp_kept = sum(1 for f in allowed if f["tp"])
        fp_kept = sum(1 for f in allowed if not f["tp"])
        tp_total = sum(1 for f in finals if f["tp"])
        fp_total = sum(1 for f in finals if not f["tp"])
        return {
            "rule": name,
            "allowed": len(allowed),
            "tp_kept": tp_kept,
            "tp_lost": tp_total - tp_kept,
            "fp_kept": fp_kept,
            "fp_avoided": fp_total - fp_kept,
        }

    rules = [
        rule("baseline(现状)", lambda f: True),
        rule("num_obs>=2", lambda f: f["num_obs"] >= 2),
        rule("num_obs>=3", lambda f: f["num_obs"] >= 3),
        rule("state=confirmed", lambda f: f["state"] == "confirmed"),
        rule("最近3步内重检", lambda f: f["recent_redetect_3"]),
        rule("num_obs>=2 且 最近3步重检",
             lambda f: f["num_obs"] >= 2 and f["recent_redetect_3"]),
        rule("num_obs>=2 且 max_sigma>=0.60",
             lambda f: f["num_obs"] >= 2 and f["max_sigma"] >= 0.60),
        rule("num_obs>=2 且 最近距离<=2.5",
             lambda f: f["num_obs"] >= 2 and f["min_dist"] is not None and f["min_dist"] <= 2.5),
    ]
    return rules


def upgrade_table(tracks: Dict[str, Dict[str, Any]], provisional_only: bool) -> List[Dict[str, Any]]:
    pool = [t for t in tracks.values() if (t["ever_provisional"] or not provisional_only)]
    tp_total = sum(1 for t in pool if t["tp"])
    rules = [
        ("全部", lambda t: True),
        ("num_obs>=2", lambda t: t["num_obs"] >= 2),
        ("num_obs>=3", lambda t: t["num_obs"] >= 3),
        ("max_sigma>=0.60", lambda t: t["max_sigma"] >= 0.60),
        ("num_obs>=2 且 max_sigma>=0.60",
         lambda t: t["num_obs"] >= 2 and t["max_sigma"] >= 0.60),
        ("num_obs>=2 且 最近距离<=2.5",
         lambda t: t["num_obs"] >= 2 and t["min_dist"] is not None and t["min_dist"] <= 2.5),
    ]
    rows = []
    for name, fn in rules:
        sel = [t for t in pool if fn(t)]
        tp = sum(1 for t in sel if t["tp"])
        rows.append({
            "rule": name,
            "n": len(sel),
            "tp": tp,
            "fp": len(sel) - tp,
            "precision": (tp / len(sel)) if sel else 0.0,
            "recall": (tp / tp_total) if tp_total else 0.0,
        })
    return rows


def delete_proxy(
    candidates: List[Dict[str, Any]],
    tracks: Dict[str, Dict[str, Any]],
    merge_dist: float = 0.5,
) -> Dict[str, Any]:
    frames: Dict[Tuple[str, int], Dict[int, List[List[float]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in candidates:
        centroid = row.get("centroid")
        if centroid is None:
            continue
        frames[_key(row)][int(row.get("step") or 0)].append([float(centroid[0]), float(centroid[1])])

    runs_tp: List[int] = []
    runs_fp: List[int] = []
    for track in tracks.values():
        steps = frames.get(track["episode"])
        if not steps or not track["steps"]:
            continue
        start, end = min(track["steps"]), max(track["steps"])
        run = 0
        best = 0
        for step in sorted(steps):
            if step < start or step > end:
                continue
            matched = any(
                (xyz[0] ** 2 + xyz[1] ** 2) ** 0.5 <= merge_dist
                for xyz in steps[step]
            )
            run = 0 if matched else run + 1
            best = max(best, run)
        (runs_tp if track["tp"] else runs_fp).append(best)

    def share(values: List[int], thr: int) -> float:
        return (sum(1 for v in values if v >= thr) / len(values)) if values else 0.0

    return {
        "tracks_tp": len(runs_tp),
        "tracks_fp": len(runs_fp),
        "tp_ge2_share": share(runs_tp, 2),
        "fp_ge2_share": share(runs_fp, 2),
        "tp_ge3_share": share(runs_tp, 3),
        "fp_ge3_share": share(runs_fp, 3),
        "tp_run": _quantiles(runs_tp),
        "fp_run": _quantiles(runs_fp),
    }


def admission_table(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    occ = [c for c in candidates if c.get("occ_passed")]
    rules = [
        ("现状 0.55 / 1.5-4.0", 0.55, 1.5, 4.0),
        ("0.50 / 1.5-4.0", 0.50, 1.5, 4.0),
        ("0.55 / 1.0-4.0", 0.55, 1.0, 4.0),
        ("0.50 / 1.0-4.0", 0.50, 1.0, 4.0),
        ("0.45 / 1.0-4.0", 0.45, 1.0, 4.0),
    ]
    rows = []
    for name, sigma, dlo, dhi in rules:
        keep = [
            c for c in occ
            if float(c.get("sigma_c") or 0.0) >= 0.60
            or (
                float(c.get("sigma_c") or 0.0) >= sigma
                and dlo <= float(c.get("distance") or 0.0) <= dhi
            )
        ]
        tp = sum(1 for c in keep if c.get("tp"))
        episodes = {_key(c) for c in keep if c.get("tp")}
        rows.append({
            "rule": name, "n": len(keep), "tp": tp,
            "precision": (tp / len(keep)) if keep else 0.0,
            "episodes_with_tp": len(episodes),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    det = os.path.join(args.run_dir, "detections")
    episodes = _load(os.path.join(det, "episodes.jsonl"))
    candidates = _load(os.path.join(det, "candidates.jsonl"))
    memory = _load(os.path.join(det, "memory_events.jsonl"))
    nav_goals = _load(os.path.join(det, "nav_goals.jsonl"))

    tracks = build_tracks(
        memory,
        {
            (_key(c)[0], _key(c)[1], str(c.get("candidate_id"))): float(c.get("distance"))
            for c in candidates if c.get("distance") is not None
        },
    )
    report = {
        "run_dir": args.run_dir,
        "episodes": len(episodes),
        "success": sum(int(e.get("success") or 0) for e in episodes),
        "tracks": len(tracks),
        "stop_rules": stop_rule_table(episodes, nav_goals, tracks),
        "upgrade_all": upgrade_table(tracks, provisional_only=False),
        "upgrade_provisional": upgrade_table(tracks, provisional_only=True),
        "delete_proxy": delete_proxy(candidates, tracks),
        "admission": admission_table(candidates),
    }

    print(f"== 离线标定 {args.run_dir}")
    print(f"episodes={report['episodes']} success={report['success']} tracks={report['tracks']}")

    print("\n[停止规则 @ 最后决策点]")
    print(f"{'规则':28s} {'允许':>5s} {'TP保留':>7s} {'TP丢失':>7s} {'FP保留':>7s} {'FP规避':>7s}")
    for row in report["stop_rules"]:
        print(f"{row['rule']:28s} {row['allowed']:5d} {row['tp_kept']:7d} "
              f"{row['tp_lost']:7d} {row['fp_kept']:7d} {row['fp_avoided']:7d}")

    for key, label in (("upgrade_all", "升级规则（全部轨道）"),
                       ("upgrade_provisional", "升级规则（provisional 子集）")):
        print(f"\n[{label}]")
        print(f"{'规则':30s} {'n':>4s} {'TP':>4s} {'prec':>6s} {'recall':>7s}")
        for row in report[key]:
            print(f"{row['rule']:30s} {row['n']:4d} {row['tp']:4d} "
                  f"{row['precision']:.3f} {row['recall']:.3f}")

    proxy = report["delete_proxy"]
    print("\n[删除阈值代理：连续未匹配帧数]")
    print(f"TP tracks={proxy['tracks_tp']} FP tracks={proxy['tracks_fp']}")
    print(f"  ≥2 帧比例: TP {proxy['tp_ge2_share']:.3f} / FP {proxy['fp_ge2_share']:.3f}")
    print(f"  ≥3 帧比例: TP {proxy['tp_ge3_share']:.3f} / FP {proxy['fp_ge3_share']:.3f}")

    print("\n[准入扩张]")
    print(f"{'口径':24s} {'n':>5s} {'TP':>5s} {'prec':>6s} {'有TP集':>7s}")
    for row in report["admission"]:
        print(f"{row['rule']:24s} {row['n']:5d} {row['tp']:5d} "
              f"{row['precision']:.3f} {row['episodes_with_tp']:7d}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
