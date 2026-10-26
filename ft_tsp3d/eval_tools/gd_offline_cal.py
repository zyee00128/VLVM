"""GD offline calibration from a log-only navigation run.

Consumes the files written by `GdCalibrationLogger` through the episode logger:

  detections/gd_frames.jsonl
  detections/gd_proposals.jsonl
  detections/episodes.jsonl
  detections/candidates.jsonl
  detections/memory_events.jsonl
  detections/track_frames.jsonl
  detections/nav_goals.jsonl

Outputs one JSON bundle plus a compact markdown summary.  It never starts a
simulator or a model server.

Usage:
  python -m ft_tsp3d.eval_tools.gd_offline_cal --run-dir outputs/<run>
  python -m ft_tsp3d.eval_tools.gd_offline_cal --detections-dir outputs/<run>/detections
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------
def _load(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(row: Dict[str, Any]) -> Tuple[str, int]:
    return str(row.get("scene")), int(row.get("episode_id", -1))


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    data = sorted(float(v) for v in values if np.isfinite(float(v)))
    if not data:
        return {}

    def pick(frac: float) -> float:
        return data[min(len(data) - 1, int(frac * len(data)))]

    return {
        "n": len(data),
        "p10": pick(0.10),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "max": data[-1],
    }


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    """Rank-based AUC; None when one class is missing."""
    pairs = [(float(s), bool(y)) for s, y in zip(scores, labels) if np.isfinite(float(s))]
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # average ranks (1-based), ties share the mean rank
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = np.zeros(len(pairs), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    pos_rank_sum = float(sum(ranks[i] for i, (_, y) in enumerate(pairs) if y))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _rate(n: int, tp: int) -> float:
    return (float(tp) / float(n)) if n else 0.0


def _bin(value: Optional[float], edges: Sequence[float]) -> str:
    if value is None or not np.isfinite(float(value)):
        return "none"
    v = float(value)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= v < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f">={edges[-1]:.2f}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _proposal_table(rows: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any]:
    tp = sum(1 for r in rows if bool(r.get("tp")))
    return {
        "group": name,
        "n": len(rows),
        "tp": tp,
        "fp": len(rows) - tp,
        "precision": _rate(len(rows), tp),
    }


def proposal_tables(
    proposals: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def group_by(fn: Callable[[Dict[str, Any]], str]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for row in proposals:
            buckets[fn(row)].append(row)
        return [
            _proposal_table(rows, key)
            for key, rows in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        ]

    return {
        "all": _proposal_table(proposals, "all"),
        "by_class": group_by(lambda r: str(r.get("canon") or "none")),
        "by_conf": group_by(lambda r: _bin(r.get("conf"), [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])),
        "by_distance": group_by(lambda r: _bin(r.get("dist_robot"), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])),
        "by_suspect": group_by(lambda r: "suspect" if r.get("suspect") else "clean"),
        "by_in_fov": group_by(lambda r: "in_fov" if r.get("in_fov") else "out_fov"),
        "by_depth_ok": group_by(lambda r: "depth_ok" if r.get("depth_ok") else "depth_far"),
        "by_gd_pos": group_by(lambda r: "gd_pos" if r.get("gd_pos") else "gd_neg"),
        "by_candidate_pos": group_by(
            lambda r: "candidate_pos" if r.get("gd_candidate_pos") else "candidate_neg"
        ),
        "by_new_info": group_by(
            lambda r: "new_info" if r.get("new_info") else "repeat_pose"
        ),
        "by_gd_hits": group_by(
            lambda r: f"hits={int(r.get('gd_hits_after') or 0)}"
        ),
    }


def coverage_tables(
    episodes: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    proposals: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    occ_tp: set = set()
    admitted_tp: set = set()
    for row in candidates:
        if not bool(row.get("tp")):
            continue
        k = _key(row)
        if bool(row.get("occ_passed")):
            occ_tp.add(k)
        if bool(row.get("admitted")):
            admitted_tp.add(k)

    by_ep: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in proposals:
        by_ep[_key(row)].append(row)

    def summarize(pool: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
        rows = []
        for episode in pool:
            key = _key(episode)
            props = by_ep.get(key, [])
            hit = any(bool(p.get("tp")) for p in props)
            rows.append(
                {
                    "key": key,
                    "target": episode.get("target_object"),
                    "success": int(episode.get("success") or 0),
                    "failure": episode.get("failure_cause"),
                    "hit": hit,
                    "n_proposals": len(props),
                    "n_tp_proposals": sum(1 for p in props if bool(p.get("tp"))),
                }
            )
        by_class: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            by_class[str(row["target"])].append(row)

        def class_row(name: str, group: List[Dict[str, Any]]) -> Dict[str, Any]:
            hit = sum(1 for r in group if r["hit"])
            return {
                "target": name,
                "n_episodes": len(group),
                "n_hit": hit,
                "hit_rate": _rate(len(group), hit),
                "success_current": sum(r["success"] for r in group),
                "n_hit": hit,
            }

        return {
            "label": label,
            "n_episodes": len(rows),
            "n_hit": sum(1 for r in rows if r["hit"]),
            "hit_rate": _rate(len(rows), sum(1 for r in rows if r["hit"])),
            "by_class": [class_row(k, v) for k, v in sorted(by_class.items())],
            "episodes": rows,
        }

    no_occ_tp = [e for e in episodes if _key(e) not in occ_tp]
    no_admitted_tp = [e for e in episodes if _key(e) not in admitted_tp]
    return {
        "episodes": len(episodes),
        "episodes_occ_tp": len(episodes) - len(no_occ_tp),
        "episodes_admitted_tp": len(episodes) - len(no_admitted_tp),
        "no_occ_tp": summarize(no_occ_tp, "no post-occ TP"),
        "no_admitted_tp": summarize(no_admitted_tp, "no admitted TP"),
    }


def track_tables(
    proposals: Sequence[Dict[str, Any]],
    track_frames: Sequence[Dict[str, Any]],
    memory_events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    track_tp: Dict[str, bool] = {}
    for row in track_frames:
        uid = row.get("memory_uid")
        if uid is None:
            continue
        track_tp[str(uid)] = bool(track_tp.get(str(uid), False) or row.get("tp"))
    for row in memory_events:
        uid = row.get("memory_uid")
        if uid is None or row.get("event") not in ("new", "merge", "upgrade"):
            continue
        track_tp[str(uid)] = bool(track_tp.get(str(uid), False) or row.get("tp"))

    per_track: Dict[str, Dict[str, Any]] = {}
    for row in proposals:
        uid = row.get("memory_uid")
        if uid is None:
            continue
        entry = per_track.setdefault(
            str(uid),
            {
                "memory_uid": str(uid),
                "target": row.get("target_object"),
                "n": 0,
                "gd_pos": False,
                "gd_hits_max": 0,
                "gd_conf_max": 0.0,
                "gd_conf_last": 0.0,
                "suspect": False,
                "states": set(),
                "num_obs_max": 0,
                "sigma_c_max": 0.0,
            },
        )
        entry["n"] += 1
        entry["gd_pos"] = bool(entry["gd_pos"] or row.get("gd_pos"))
        entry["gd_hits_max"] = max(int(entry["gd_hits_max"]), int(row.get("gd_hits_after") or 0))
        entry["gd_conf_max"] = max(float(entry["gd_conf_max"]), float(row.get("conf") or 0.0))
        entry["suspect"] = bool(entry["suspect"] or row.get("suspect"))
        match = row.get("match_track") or {}
        if match.get("lifecycle_state"):
            entry["states"].add(str(match.get("lifecycle_state")))
        entry["num_obs_max"] = max(int(entry["num_obs_max"]), int(match.get("num_obs") or 0))
        entry["sigma_c_max"] = max(float(entry["sigma_c_max"]), float(match.get("sigma_c_max") or 0.0))

    tracks: List[Dict[str, Any]] = []
    for uid, entry in per_track.items():
        if uid not in track_tp:
            continue
        row = dict(entry)
        row["states"] = sorted(entry["states"])
        row["tp"] = bool(track_tp[uid])
        tracks.append(row)

    def rule(name: str, fn: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
        selected = [t for t in tracks if fn(t)]
        tp = sum(1 for t in selected if t["tp"])
        base_tp = sum(1 for t in tracks if t["tp"])
        return {
            "rule": name,
            "n": len(selected),
            "tp": tp,
            "fp": len(selected) - tp,
            "precision": _rate(len(selected), tp),
            "recall": _rate(base_tp, tp),
        }

    rules = [
        rule("all", lambda t: True),
        rule("gd_pos", lambda t: bool(t["gd_pos"])),
        rule("gd_hits>=1", lambda t: int(t["gd_hits_max"]) >= 1),
        rule("gd_hits>=2", lambda t: int(t["gd_hits_max"]) >= 2),
        rule("gd_hits>=3", lambda t: int(t["gd_hits_max"]) >= 3),
        rule("gd_conf>=0.3", lambda t: float(t["gd_conf_max"]) >= 0.3),
        rule("gd_conf>=0.4", lambda t: float(t["gd_conf_max"]) >= 0.4),
        rule("gd_conf>=0.5", lambda t: float(t["gd_conf_max"]) >= 0.5),
        rule("gd_pos & gd_hits>=2", lambda t: bool(t["gd_pos"]) and int(t["gd_hits_max"]) >= 2),
        rule("no gd_pos", lambda t: not bool(t["gd_pos"])),
    ]

    labels = [bool(t["tp"]) for t in tracks]
    return {
        "n_tracks": len(tracks),
        "n_tp": sum(1 for t in tracks if t["tp"]),
        "n_fp": sum(1 for t in tracks if not t["tp"]),
        "rules": rules,
        "auc": {
            "gd_conf_max": _auc(labels, [float(t["gd_conf_max"]) for t in tracks]),
            "gd_hits_max": _auc(labels, [float(t["gd_hits_max"]) for t in tracks]),
            "gd_pos": _auc(labels, [1.0 if t["gd_pos"] else 0.0 for t in tracks]),
            "n_proposals": _auc(labels, [float(t["n"]) for t in tracks]),
        },
        "tracks": tracks,
    }


def stop_tables(
    nav_goals: Sequence[Dict[str, Any]],
    proposals: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    final: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in nav_goals:
        key = _key(row)
        prev = final.get(key)
        if prev is None or int(row.get("step") or 0) >= int(prev.get("step") or 0):
            final[key] = row

    by_track: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in proposals:
        uid = row.get("memory_uid")
        if uid is not None:
            by_track[str(uid)].append(row)

    rows: List[Dict[str, Any]] = []
    for goal in final.values():
        uid = goal.get("memory_uid")
        props = by_track.get(str(uid), []) if uid is not None else []
        rows.append(
            {
                "tp": bool(goal.get("tp")),
                "state": goal.get("lifecycle_state"),
                "num_obs": goal.get("num_obs"),
                "gd_pos": any(bool(p.get("gd_pos")) for p in props),
                "gd_hits": max([int(p.get("gd_hits_after") or 0) for p in props] or [0]),
                "gd_conf": max([float(p.get("conf") or 0.0) for p in props] or [0.0]),
                "n_gd": len(props),
            }
        )

    def rule(name: str, fn: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
        selected = [r for r in rows if fn(r)]
        tp = sum(1 for r in selected if r["tp"])
        base_tp = sum(1 for r in rows if r["tp"])
        base_fp = sum(1 for r in rows if not r["tp"])
        kept_fp = len(selected) - tp
        return {
            "rule": name,
            "allowed": len(selected),
            "tp_kept": tp,
            "tp_lost": base_tp - tp,
            "fp_kept": kept_fp,
            "fp_avoided": base_fp - kept_fp,
            "precision": _rate(len(selected), tp),
        }

    return {
        "n_goals": len(rows),
        "tp": sum(1 for r in rows if r["tp"]),
        "fp": sum(1 for r in rows if not r["tp"]),
        "rules": [
            rule("baseline", lambda r: True),
            rule("state=confirmed", lambda r: r["state"] == "confirmed"),
            rule("confirmed & gd_pos", lambda r: r["state"] == "confirmed" and r["gd_pos"]),
            rule("confirmed & gd_hits>=1", lambda r: r["state"] == "confirmed" and r["gd_hits"] >= 1),
            rule("confirmed & gd_hits>=2", lambda r: r["state"] == "confirmed" and r["gd_hits"] >= 2),
            rule("confirmed & gd_conf>=0.3",
                 lambda r: r["state"] == "confirmed" and r["gd_conf"] >= 0.3),
            rule("confirmed & gd_conf>=0.4",
                 lambda r: r["state"] == "confirmed" and r["gd_conf"] >= 0.4),
        ],
        "goals": rows,
    }


def cost_tables(frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gd_frames = [f for f in frames if bool((f.get("gd") or {}).get("ran"))]
    latencies = [
        float((f.get("gd") or {}).get("latency_ms"))
        for f in gd_frames
        if (f.get("gd") or {}).get("latency_ms") is not None
    ]
    proposals = [int((f.get("gd") or {}).get("n_proposals") or 0) for f in gd_frames]
    decisions: collections.Counter = collections.Counter()
    reasons: collections.Counter = collections.Counter()
    for frame in frames:
        gd = frame.get("gd") or {}
        trace = gd.get("trace") or {}
        reason = trace.get("reason")
        if reason:
            reasons[str(reason)] += 1
        for decision in trace.get("decisions") or []:
            decisions[str(decision.get("reason") or "unknown")] += 1
    return {
        "frames": len(frames),
        "gd_frames": len(gd_frames),
        "gd_call_rate": _rate(len(frames), len(gd_frames)),
        "latency_ms": _quantiles(latencies),
        "proposals": _quantiles(proposals),
        "proposals_total": sum(proposals),
        "reject_decisions": dict(decisions),
        "gd_reasons": dict(reasons),
        "max_buffer_size": max([int((f.get("stats") or {}).get("buffer_size") or 0) for f in frames] or [0]),
        "max_tracked_keys": max([int((f.get("stats") or {}).get("tracked_keys") or 0) for f in frames] or [0]),
    }


def selection_context(
    nav_goals: Sequence[Dict[str, Any]],
    frames: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Count nav-goal decisions where another track already had a GD positive.

    This is a *context* count, not a replay: the logger snapshots the tracks at
    that step, so we can see whether a GD-positive alternative existed without
    claiming that the selector would have chosen it.
    """
    frame_index: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for frame in frames:
        frame_index[(*_key(frame), int(frame.get("step") or 0))] = frame

    def band(sigma: Optional[float]) -> int:
        if sigma is None:
            return 9
        value = float(sigma)
        if 0.65 <= value < 0.70:
            return 0
        if 0.60 <= value < 0.65:
            return 1
        if 0.55 <= value < 0.60:
            return 2
        if value >= 0.70:
            return 3
        return 4

    n = 0
    with_alt = 0
    for goal in nav_goals:
        frame = frame_index.get((*_key(goal), int(goal.get("step") or 0)))
        if frame is None:
            continue
        n += 1
        chosen_id = goal.get("memory_id")
        chosen_band = int(goal.get("sigma_band") if goal.get("sigma_band") is not None else 9)
        for track in frame.get("tracks") or []:
            if track.get("memory_id") == chosen_id:
                continue
            if int(track.get("gd_hits") or 0) <= 0:
                continue
            if band(track.get("sigma_c_latest")) <= chosen_band:
                with_alt += 1
                break
    return {
        "n_nav_goals": n,
        "n_with_gd_positive_alternative": with_alt,
        "share_with_gd_positive_alternative": _rate(n, with_alt),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=None, help="outputs/<run> (uses <run>/detections)")
    parser.add_argument("--detections-dir", default=None, help="directory containing the jsonl files")
    parser.add_argument("--out", default=None, help="output JSON (default: <detections>/gd_offline_cal.json)")
    args = parser.parse_args()

    if args.detections_dir:
        det_dir = args.detections_dir
    elif args.run_dir:
        det_dir = os.path.join(args.run_dir, "detections")
    else:
        raise SystemExit("provide --run-dir or --detections-dir")
    det_dir = os.path.abspath(det_dir)
    out_path = args.out or os.path.join(det_dir, "gd_offline_cal.json")

    episodes = _load(os.path.join(det_dir, "episodes.jsonl"))
    candidates = _load(os.path.join(det_dir, "candidates.jsonl"))
    proposals = _load(os.path.join(det_dir, "gd_proposals.jsonl"))
    frames = _load(os.path.join(det_dir, "gd_frames.jsonl"))
    track_frames = _load(os.path.join(det_dir, "track_frames.jsonl"))
    memory_events = _load(os.path.join(det_dir, "memory_events.jsonl"))
    nav_goals = _load(os.path.join(det_dir, "nav_goals.jsonl"))

    result = {
        "meta": {
            "detections_dir": det_dir,
            "episodes": len(episodes),
            "gd_frames": len(frames),
            "gd_proposals": len(proposals),
            "track_frames": len(track_frames),
            "memory_events": len(memory_events),
            "nav_goals": len(nav_goals),
            "note": (
                "GD is log-only.  Missing gd_*.jsonl means the run did not enable "
                "GD_CALIB_ENABLE=1 or the logger was disabled after an error."
            ),
        },
        "proposal_precision": proposal_tables(proposals),
        "coverage": coverage_tables(episodes, candidates, proposals),
        "track_arbitration": track_tables(proposals, track_frames, memory_events),
        "stop": stop_tables(nav_goals, proposals),
        "selection": selection_context(nav_goals, frames),
        "cost": cost_tables(frames),
    }

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)

    print(f"[gd_offline_cal] wrote {out_path}")
    _print_summary(result)


def _print_summary(result: Dict[str, Any]) -> None:
    meta = result["meta"]
    print("\n## GD offline calibration\n")
    print(f"- episodes: {meta['episodes']} | gd_frames: {meta['gd_frames']} | gd_proposals: {meta['gd_proposals']}")
    prop = result["proposal_precision"]["all"]
    print(f"- proposal precision: {prop['tp']}/{prop['n']} = {prop['precision']:.2%}")
    cov = result["coverage"]
    print(
        f"- no post-occ TP episodes: {cov['no_occ_tp']['n_episodes']} | "
        f"GD hit: {cov['no_occ_tp']['n_hit']} ({cov['no_occ_tp']['hit_rate']:.2%})"
    )
    track = result["track_arbitration"]
    print(f"- tracks with GD association: {track['n_tracks']} | TP {track['n_tp']} / FP {track['n_fp']}")
    print("")
    print("| rule | n | TP | FP | precision | recall |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for row in track["rules"]:
        print(
            f"| {row['rule']} | {row['n']} | {row['tp']} | {row['fp']} | "
            f"{row['precision']:.2%} | {row['recall']:.2%} |"
        )
    print("")
    print("| stop rule | allowed | TP kept | TP lost | FP kept | FP avoided | precision |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in result["stop"]["rules"]:
        print(
            f"| {row['rule']} | {row['allowed']} | {row['tp_kept']} | {row['tp_lost']} | "
            f"{row['fp_kept']} | {row['fp_avoided']} | {row['precision']:.2%} |"
        )
    cost = result["cost"]
    print("")
    print(
        f"- GD cost: calls {cost['gd_frames']}/{cost['frames']} frames, "
        f"latency p50={cost['latency_ms'].get('p50', 0):.2f} ms, "
        f"proposals/frame p50={cost['proposals'].get('p50', 0):.1f}"
    )
    print(f"- selection context: {result['selection']}")


if __name__ == "__main__":
    main()
