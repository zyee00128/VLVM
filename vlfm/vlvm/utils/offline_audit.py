"""S9 offline episode-level audit for VLVM navigation runs (read-only, no simulation).

Turns one run's ``detections/*.jsonl`` into an actionable failure ledger so that the
next mechanism is chosen by a measured failure split instead of by guesswork.

Sections
--------
1. ``buckets``     : ``failure_cause`` x ``oracle_success`` x steps, plus per-scene SR.
2. ``attribution`` : per-episode cross-tab of
                     (has TP candidate) x (TP candidate ever admitted) x (TP nav goal).
                     This separates "never detected" / "detected but never admitted" /
                     "admitted but not used" instead of lumping them into one bucket.
3. ``fp``          : ``false_positive`` split by the same ladder + lock metadata.
4. ``never_saw``   : stairs-required vs feasible, candidate volume, trajectory coverage.
5. ``thresholds``  : per-class ``sigma_c`` calibration on candidate-level TP/FP labels.
6. ``fusion``      : recompute the S8 section-3 AUC claim (``c_raw`` vs ``c_fused_*``).

Data contract (see ``vlfm/utils/episode_stats_logger.py``)
----------------------------------------------------------
* ``episodes.jsonl``     one row per episode, carries ``failure_cause`` / ``oracle_success``
                        / ``first_wrong_lock_step`` / ``wrong_stop_before_target_coverage``.
* ``candidates.jsonl``   one row per TSP3D candidate; ``tp`` = centroid inside the dilated
                        GT target bbox; ``sigma_c`` = ``c_raw * w_S``; ``admitted`` = final
                        admission flag after the provisional gate.
* ``memory_events.jsonl`` ``memory_uid`` -> ``tp`` (memory-level identity label).
* ``nav_goals.jsonl``    selected goal per step; ``tp`` labels the selected centroid.
* ``stop_shadow.jsonl``  per-step STOP-rule re-evaluation; ``stop_eligible_current``.
* ``gd_frames.jsonl``    per-frame robot pose (streamed; only xy/z kept).
* ``fusion_field.jsonl`` ``fusion_candidate`` rows are **row-aligned** with the
                        ``candidates.jsonl`` rows that carry ``c_raw`` (same order).

Usage
-----
    PYTHONPATH=$PWD python -m vlfm.vlvm.utils.offline_audit \
        --run-dir outputs/s8_fusion_shadow_3scene_2026-11-04_215549 \
        --json-out outputs/s8_fusion_shadow_3scene_2026-11-04_215549/offline_audit.json \
        --md-out /tmp/offline_audit.md
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EpKey = Tuple[str, int]

# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ep_key(row: Dict[str, Any]) -> EpKey:
    return str(row.get("scene")), int(row.get("episode_id"))


def group_by_episode(rows: Iterable[Dict[str, Any]]) -> Dict[EpKey, List[Dict[str, Any]]]:
    out: Dict[EpKey, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        out[ep_key(row)].append(row)
    return out


def quantiles(values: Sequence[float]) -> Dict[str, float]:
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return {}

    def pick(frac: float) -> float:
        return data[min(len(data) - 1, int(frac * len(data)))]

    return {
        "n": len(data),
        "p10": pick(0.10),
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "max": data[-1],
    }


def rate(num: int, den: int) -> Optional[float]:
    return None if den <= 0 else num / den


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUC with tie-averaged ranks. ``None`` if one side is empty."""
    pos = [float(v) for v in pos if v is not None]
    neg = [float(v) for v in neg if v is not None]
    if not pos or not neg:
        return None
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(r for r, (_, label) in zip(ranks, merged) if label == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def sweep_threshold(values: Sequence[float], labels: Sequence[bool], grid: Sequence[float]) -> List[Dict[str, Any]]:
    """Admission sweep: keep ``value >= t``."""
    rows = []
    pairs = [(float(v), bool(l)) for v, l in zip(values, labels) if v is not None]
    n_tp = sum(1 for _, l in pairs if l)
    n_fp = len(pairs) - n_tp
    for t in grid:
        kept = [(v, l) for v, l in pairs if v >= t]
        tp = sum(1 for _, l in kept if l)
        fp = len(kept) - tp
        rows.append(
            {
                "t": round(float(t), 3),
                "n_kept": len(kept),
                "tp": tp,
                "fp": fp,
                "precision": rate(tp, tp + fp),
                "recall": rate(tp, n_tp),
                "fp_keep_rate": rate(fp, n_fp),
            }
        )
    return rows


def best_threshold(values: Sequence[float], labels: Sequence[bool]) -> Optional[Dict[str, Any]]:
    """Threshold maximising Youden J = TPR - FPR (ties broken by more TP kept)."""
    pairs = sorted(((float(v), bool(l)) for v, l in zip(values, labels) if v is not None), reverse=True)
    n_tp = sum(1 for _, l in pairs if l)
    n_fp = len(pairs) - n_tp
    if not n_tp or not n_fp:
        return None
    tp = fp = 0
    best: Optional[Dict[str, Any]] = None
    idx = 0
    while idx < len(pairs):
        t = pairs[idx][0]
        while idx < len(pairs) and pairs[idx][0] >= t:
            if pairs[idx][1]:
                tp += 1
            else:
                fp += 1
            idx += 1
        j = tp / n_tp - fp / n_fp
        cand = {"t": t, "j": j, "tp": tp, "fp": fp}
        if best is None or (cand["j"], cand["tp"]) > (best["j"], best["tp"]):
            best = cand
    return best


# ---------------------------------------------------------------------------
# section 1/2: episode buckets and attribution ladder
# ---------------------------------------------------------------------------


def build_episode_index(
    episodes: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    nav_goals: List[Dict[str, Any]],
    memory_events: List[Dict[str, Any]],
    stop_shadow: List[Dict[str, Any]],
) -> Dict[EpKey, Dict[str, Any]]:
    cand_by = group_by_episode(candidates)
    nav_by = group_by_episode(nav_goals)
    mem_by = group_by_episode(memory_events)
    stop_by = group_by_episode(stop_shadow)

    index: Dict[EpKey, Dict[str, Any]] = {}
    for row in episodes:
        key = ep_key(row)
        cands = cand_by.get(key, [])
        navs = sorted(nav_by.get(key, []), key=lambda r: int(r.get("step") or 0))
        mems = mem_by.get(key, [])
        stops = stop_by.get(key, [])

        tp_cands = [c for c in cands if c.get("tp")]
        tp_admitted = [c for c in tp_cands if c.get("admitted")]
        tp_navs = [g for g in navs if g.get("tp")]
        tp_mems = {str(m.get("memory_uid")) for m in mems if m.get("tp") and m.get("memory_uid")}

        final_nav = navs[-1] if navs else None
        index[key] = {
            "episode": row,
            "n_candidates": len(cands),
            "n_tp_candidates": len(tp_cands),
            "n_tp_admitted": len(tp_admitted),
            "tp_first_step": min((int(c.get("step") or 0) for c in tp_cands), default=None),
            "tp_first_admit_step": min((int(c.get("step") or 0) for c in tp_admitted), default=None),
            "n_tp_nav_goals": len(tp_navs),
            "tp_first_nav_step": min((int(g.get("step") or 0) for g in tp_navs), default=None),
            "n_memories": len({str(m.get("memory_id")) for m in mems if m.get("memory_id")}),
            "n_tp_memories": len(tp_mems),
            "n_nav_goals": len(navs),
            "final_nav": final_nav,
            "first_nav": navs[0] if navs else None,
            "stop_eligible_ever": any(bool(s.get("stop_eligible_current")) for s in stops),
            "stop_goal_tp_ever": any(bool(s.get("goal_tp")) for s in stops),
            "n_stop_shadow": len(stops),
        }
    return index


def episode_stage(info: Dict[str, Any]) -> str:
    """Where in the detect -> admit -> use ladder this episode stopped."""
    if info["n_tp_candidates"] == 0:
        return "no_tp_detected"
    if not info["n_tp_admitted"]:
        return "tp_not_admitted"
    if info["n_tp_nav_goals"] == 0:
        return "tp_admitted_not_used"
    return "tp_used"


def attribution_ladder(index: Dict[EpKey, Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-tab of detection / admission / usage for every failure bucket."""
    ladder = {"no_tp_detected": 0, "tp_not_admitted": 0, "tp_admitted_not_used": 0, "tp_used": 0}
    table: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: dict(ladder))
    for info in index.values():
        stage = episode_stage(info)
        cause = str(info["episode"].get("failure_cause") or "unknown")
        table[cause][stage] += 1
        table[cause]["__n__"] = table[cause].get("__n__", 0) + 1

    out = {}
    for cause, counts in sorted(table.items(), key=lambda kv: -kv[1].get("__n__", 0)):
        n = counts.get("__n__", 0)
        out[cause] = {
            "n": n,
            **{stage: counts[stage] for stage in ladder},
            "share": {stage: rate(counts[stage], n) for stage in ladder},
        }
    return out


def episode_buckets(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cause: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in episodes:
        by_cause[str(row.get("failure_cause") or "unknown")].append(row)

    rows = []
    for cause, items in sorted(by_cause.items(), key=lambda kv: -len(kv[1])):
        steps = [float(r.get("steps_count") or 0) for r in items]
        rows.append(
            {
                "failure_cause": cause,
                "n": len(items),
                "share": rate(len(items), len(episodes)),
                "oracle1": sum(1 for r in items if int(r.get("oracle_success") or 0) == 1),
                "steps_p50": statistics.median(steps) if steps else None,
                "steps_p90": quantiles(steps).get("p90"),
                "hit_500": sum(1 for s in steps if s >= 500),
            }
        )

    by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in episodes:
        by_scene[ep_key(row)[0]].append(row)
    scenes = []
    for scene, items in sorted(by_scene.items()):
        steps = [float(r.get("steps_count") or 0) for r in items]
        scenes.append(
            {
                "scene": scene,
                "n": len(items),
                "success": sum(1 for r in items if int(r.get("success") or 0) == 1),
                "sr": rate(sum(1 for r in items if int(r.get("success") or 0) == 1), len(items)),
                "steps_p50": statistics.median(steps) if steps else None,
            }
        )

    n_succ = sum(1 for r in episodes if int(r.get("success") or 0) == 1)
    oracle_conflict = sum(
        1
        for r in episodes
        if int(r.get("success") or 0) == 1 and int(r.get("oracle_success") or 0) == 0
    )
    scene_cause: Dict[str, Dict[str, int]] = {}
    for row in episodes:
        scene = ep_key(row)[0]
        cause = str(row.get("failure_cause") or "unknown")
        scene_cause.setdefault(scene, {})
        scene_cause[scene][cause] = scene_cause[scene].get(cause, 0) + 1
    return {
        "n": len(episodes),
        "success": n_succ,
        "sr": rate(n_succ, len(episodes)),
        "oracle1": sum(1 for r in episodes if int(r.get("oracle_success") or 0) == 1),
        "oracle_success_conflict": oracle_conflict,
        "by_cause": rows,
        "by_scene": scenes,
        "scene_cause": scene_cause,
    }


# ---------------------------------------------------------------------------
# section 3: false-positive autopsy
# ---------------------------------------------------------------------------


def fp_autopsy(
    index: Dict[EpKey, Dict[str, Any]],
    memory_events: List[Dict[str, Any]],
    nav_goals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    memory_events_by_ep = group_by_episode(memory_events)
    nav_goals_by_ep = group_by_episode(nav_goals)
    fps = [i for i in index.values() if str(i["episode"].get("failure_cause")) == "false_positive"]
    cats = {
        "no_tp_detected": [],
        "tp_not_admitted": [],
        "tp_admitted_not_used": [],
        "tp_used": [],
    }
    for info in fps:
        if info["n_tp_candidates"] == 0:
            cats["no_tp_detected"].append(info)
        elif not info["n_tp_admitted"]:
            cats["tp_not_admitted"].append(info)
        elif info["n_tp_nav_goals"] == 0:
            cats["tp_admitted_not_used"].append(info)
        else:
            cats["tp_used"].append(info)

    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {"n": 0}
        lock_steps = [int(i["final_nav"].get("step") or 0) for i in items if i["final_nav"]]
        lock_dist = [float(i["final_nav"].get("distance") or 0) for i in items if i["final_nav"]]
        first_lock = [
            int(i["episode"].get("first_wrong_lock_step"))
            for i in items
            if i["episode"].get("first_wrong_lock_step") is not None
        ]
        return {
            "n": len(items),
            "oracle1": sum(1 for i in items if int(i["episode"].get("oracle_success") or 0) == 1),
            "oracle0": sum(1 for i in items if int(i["episode"].get("oracle_success") or 0) == 0),
            "stop_eligible_ever": sum(1 for i in items if i["stop_eligible_ever"]),
            "multi_memory": sum(1 for i in items if i["n_memories"] >= 2),
            "n_memories_p50": statistics.median([i["n_memories"] for i in items]),
            "n_candidates_p50": statistics.median([i["n_candidates"] for i in items]),
            "first_wrong_lock_p50": statistics.median(first_lock) if first_lock else None,
            "lock_step_p50": statistics.median(lock_steps) if lock_steps else None,
            "lock_dist_p50": statistics.median(lock_dist) if lock_dist else None,
            "lock_origin": dict(collections.Counter(str(i["final_nav"].get("origin")) for i in items if i["final_nav"])),
            "lock_tier": dict(
                collections.Counter(str(i["final_nav"].get("selection_tier_name")) for i in items if i["final_nav"])
            ),
            "lock_before_tp_first_step": sum(
                1
                for i in items
                if i["final_nav"]
                and i["tp_first_step"] is not None
                and int(i["final_nav"].get("step") or 0) >= i["tp_first_step"]
            ),
        }

    # TD: time-ordered diagnostic, "TP candidate was available while the wrong
    # memory was already the goal" -> a re-ranking (not a detector) fix.
    td = [
        i
        for i in fps
        if i["tp_first_step"] is not None
        and i["final_nav"] is not None
        and i["tp_first_step"] <= int(i["final_nav"].get("step") or 0)
    ]

    detail = fp_tp_used_detail(cats["tp_used"], memory_events_by_ep, nav_goals_by_ep)

    oracle_stop = {}
    for name, items in cats.items():
        o1 = [i for i in items if int(i["episode"].get("oracle_success") or 0) == 1]
        o0 = [i for i in items if int(i["episode"].get("oracle_success") or 0) == 0]
        oracle_stop[name] = {
            "o1_stop": sum(1 for i in o1 if i["stop_eligible_ever"]),
            "o1_nostop": sum(1 for i in o1 if not i["stop_eligible_ever"]),
            "o0_stop": sum(1 for i in o0 if i["stop_eligible_ever"]),
            "o0_nostop": sum(1 for i in o0 if not i["stop_eligible_ever"]),
        }

    return {
        "n": len(fps),
        "categories": {name: summarize(items) for name, items in cats.items()},
        "tp_candidate_while_wrong_goal": len(td),
        "tp_step_before_first_wrong_lock": sum(
            1
            for i in td
            if i["episode"].get("first_wrong_lock_step") is not None
            and i["tp_first_step"] < int(i["episode"]["first_wrong_lock_step"])
        ),
        "tp_used_detail": detail,
        "oracle_stop": oracle_stop,
    }


def fp_tp_used_detail(
    episodes: List[Dict[str, Any]],
    mem_by: Dict[EpKey, List[Dict[str, Any]]],
    nav_by: Dict[EpKey, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """For fp episodes where the TRUE target was once the goal, why did it leak?

    These are the only fp episodes where a ranking / STOP-qualification decision
    (not a detector miss) is responsible.
    """
    if not episodes:
        return {"n": 0}
    blacklisted = 0
    blacklist_reasons: collections.Counter = collections.Counter()
    min_dist: List[float] = []
    nav_steps: List[int] = []
    first_step: List[int] = []
    last_step: List[int] = []
    final_after: int = 0
    for info in episodes:
        key = ep_key(info["episode"])
        navs = nav_by.get(key, [])
        tp_uids = {str(g.get("memory_uid")) for g in navs if g.get("tp") and g.get("memory_uid")}
        tp_rows = [g for g in navs if g.get("tp")]
        if tp_rows:
            d = [float(g.get("distance")) for g in tp_rows if g.get("distance") is not None]
            if d:
                min_dist.append(min(d))
            nav_steps.append(len(tp_rows))
            first_step.append(min(int(g.get("step") or 0) for g in tp_rows))
            last_step.append(max(int(g.get("step") or 0) for g in tp_rows))
            final_step = int(info["final_nav"].get("step") or 0) if info["final_nav"] else 0
            if final_step > max(int(g.get("step") or 0) for g in tp_rows):
                final_after += 1
        for row in mem_by.get(key, []):
            if str(row.get("event")) == "stop_blacklist" and str(row.get("memory_uid")) in tp_uids:
                blacklisted += 1
                blacklist_reasons[str(row.get("reason") or "unknown")] += 1
                break
    return {
        "n": len(episodes),
        "tp_memory_blacklisted": blacklisted,
        "blacklist_reasons": dict(blacklist_reasons),
        "min_dist_to_tp_p50": statistics.median(min_dist) if min_dist else None,
        "min_dist_to_tp_p10": quantiles(min_dist).get("p10") if min_dist else None,
        "min_dist_to_tp_max": max(min_dist) if min_dist else None,
        "tp_goal_rows_p50": statistics.median(nav_steps) if nav_steps else None,
        "tp_goal_first_step_p50": statistics.median(first_step) if first_step else None,
        "tp_goal_last_step_p50": statistics.median(last_step) if last_step else None,
        "wrong_lock_after_tp_last_step": final_after,
    }


# ---------------------------------------------------------------------------
# section 4: never-saw autopsy
# ---------------------------------------------------------------------------


def _trajectory_stats(run_dir: str) -> Dict[EpKey, Dict[str, Any]]:
    """Stream ``gd_frames.jsonl`` and keep only pose-derived coverage numbers."""
    path = os.path.join(run_dir, "detections", "gd_frames.jsonl")
    if not os.path.exists(path):
        return {}
    cells: Dict[EpKey, set] = collections.defaultdict(set)
    zrange: Dict[EpKey, List[float]] = collections.defaultdict(lambda: [math.inf, -math.inf])
    last: Dict[EpKey, Tuple[float, float]] = {}
    length: Dict[EpKey, float] = collections.defaultdict(float)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            xy = row.get("robot_xy")
            xyz = row.get("robot_xyz")
            if not xy or len(xy) < 2:
                continue
            key = ep_key(row)
            x, y = float(xy[0]), float(xy[1])
            cells[key].add((int(math.floor(x / 0.5)), int(math.floor(y / 0.5))))
            if xyz and len(xyz) >= 3:
                z = float(xyz[2])
                lo, hi = zrange[key]
                zrange[key] = [min(lo, z), max(hi, z)]
            if key in last:
                px, py = last[key]
                length[key] += math.hypot(x - px, y - py)
            last[key] = (x, y)
    out = {}
    for key in cells:
        lo, hi = zrange.get(key, [0.0, 0.0])
        out[key] = {
            "cells_05m": len(cells[key]),
            "path_len_m": length.get(key, 0.0),
            "z_span_m": (hi - lo) if lo < math.inf else None,
        }
    return out


def never_saw_autopsy(index: Dict[EpKey, Dict[str, Any]], traj: Dict[EpKey, Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for key, info in index.items():
        cause = str(info["episode"].get("failure_cause") or "")
        if cause.startswith("never_saw_target"):
            groups[cause].append(info)
        elif cause == "did_not_fail":
            groups["success"].append(info)

    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {"n": 0}
        steps = [float(i["episode"].get("steps_count") or 0) for i in items]
        ncand = [i["n_candidates"] for i in items]
        zero_cand = sum(1 for v in ncand if v == 0)
        keys = [k for k, i in index.items() if i in items]
        cov = [traj[k]["cells_05m"] for k in keys if k in traj]
        path = [traj[k]["path_len_m"] for k in keys if k in traj]
        zsp = [traj[k]["z_span_m"] for k in keys if k in traj and traj[k]["z_span_m"] is not None]
        navc = [i["n_nav_goals"] for i in items]
        last_nav = [
            int(i["final_nav"].get("step") or 0) for i in items if i["final_nav"]
        ]
        return {
            "n": len(items),
            "steps_p50": statistics.median(steps) if steps else None,
            "hit_500": sum(1 for s in steps if s >= 500),
            "candidates_p50": statistics.median(ncand) if ncand else None,
            "zero_candidate": zero_cand,
            "n_nav_goals_p50": statistics.median(navc) if navc else None,
            "zero_nav_goal": sum(1 for v in navc if v == 0),
            "last_nav_step_p50": statistics.median(last_nav) if last_nav else None,
            "max_candidates": max(ncand) if ncand else 0,
            "n_tp_candidates_total": sum(i["n_tp_candidates"] for i in items),
            "coverage_cells_p50": statistics.median(cov) if cov else None,
            "coverage_cells_p90": quantiles(cov).get("p90") if cov else None,
            "path_len_p50": statistics.median(path) if path else None,
            "z_span_p50": statistics.median(zsp) if zsp else None,
            "z_span_max": max(zsp) if zsp else None,
        }

    out = {name: summarize(items) for name, items in sorted(groups.items())}
    out["_total_never_saw"] = sum(len(v) for k, v in groups.items() if k != "success")
    return out


# ---------------------------------------------------------------------------
# section 4b: same-floor vs cross-floor split (needs the ObjectNav content JSON)
# ---------------------------------------------------------------------------


def load_objectnav_content(content_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load per-scene episode files; returns scene -> {episodes, goals: category -> [goal]}."""
    out: Dict[str, Dict[str, Any]] = {}
    if not content_dir or not os.path.isdir(content_dir):
        return out
    for name in sorted(os.listdir(content_dir)):
        if not name.endswith(".json.gz"):
            continue
        scene = name[: -len(".json.gz")]
        with gzip.open(os.path.join(content_dir, name), "r") as handle:
            data = json.load(handle)
        episodes = data.get("episodes") or []
        cats = {str(e.get("object_category")) for e in episodes}
        goals: Dict[str, List[Dict[str, Any]]] = {}
        for key, items in (data.get("goals_by_category") or {}).items():
            for cat in cats:
                if key.endswith("_" + cat):
                    goals.setdefault(cat, []).extend(items)
                    break
        out[scene] = {"episodes": episodes, "goals": goals}
    return out


def same_floor_goal(scene_data: Dict[str, Any], ep_index: int) -> Dict[str, Any]:
    """Reproduce ``TopDownMap._is_on_same_floor``: ``y0 <= y < y0 + 2.0``.

    ``y`` is the goal AABB centre (``goal["position"]``), which is exactly what
    ``FrontierExplorationMap._draw_target_bbox_mask`` tests before drawing a bbox.
    An episode with no same-floor goal cannot produce ``tp=True`` for any candidate.
    """
    episodes = scene_data.get("episodes") or []
    if ep_index < 0 or ep_index >= len(episodes):
        return {"available": False}
    ep = episodes[ep_index]
    y0 = float(ep["start_position"][1])
    goals = (scene_data.get("goals") or {}).get(str(ep.get("object_category")), [])
    ys = [float(g["position"][1]) for g in goals if g.get("position") is not None]
    return {
        "available": True,
        "category": ep.get("object_category"),
        "start_y": y0,
        "n_goals": len(ys),
        "same_floor": any(y0 <= y < y0 + 2.0 for y in ys),
        "min_dy": min((abs(y - y0) for y in ys), default=None),
    }


def floor_split(index: Dict[EpKey, Dict[str, Any]], content: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Split every bucket by "does a goal instance exist on the agent's start floor"."""
    if not content:
        return {"available": False}

    rows = []
    missing = 0
    for key, info in index.items():
        scene, ep_idx = key
        scene_data = content.get(scene)
        if not scene_data:
            missing += 1
            continue
        f = same_floor_goal(scene_data, ep_idx)
        if not f.get("available"):
            missing += 1
            continue
        rows.append({"key": key, "info": info, **f})
    if not rows:
        return {"available": False, "missing": missing}

    by_cause: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"n": 0, "same": 0, "cross": 0}
    )
    for row in rows:
        cause = str(row["info"]["episode"].get("failure_cause") or "unknown")
        by_cause[cause]["n"] += 1
        by_cause[cause]["same" if row["same_floor"] else "cross"] += 1

    fp_stage: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"same": 0, "cross": 0})
    for row in rows:
        if str(row["info"]["episode"].get("failure_cause")) != "false_positive":
            continue
        key = "same" if row["same_floor"] else "cross"
        fp_stage[episode_stage(row["info"])][key] += 1

    by_scene: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"n": 0, "success": 0, "same_n": 0}
    )
    for row in rows:
        scene = row["key"][0]
        by_scene[scene]["n"] += 1
        if int(row["info"]["episode"].get("success") or 0) == 1:
            by_scene[scene]["success"] += 1
        if row["same_floor"]:
            by_scene[scene]["same_n"] += 1

    same_rows = [r for r in rows if r["same_floor"]]
    same_success = sum(1 for r in same_rows if int(r["info"]["episode"].get("success") or 0) == 1)

    def _features(info: Dict[str, Any]) -> Dict[str, float]:
        nav = info.get("final_nav") or {}
        return {
            "sigma": float(nav.get("quality_score") or 0.0),
            "num_obs": float(nav.get("num_obs") or 0.0),
            "gd_hits": float(nav.get("gd_hits") or 0.0),
            "dist": float(nav.get("distance") or 0.0),
            "n_selectable": float(nav.get("n_selectable") or 0.0),
            "step": float(nav.get("step") or 0.0),
            "gd_tsp": 1.0 if str(nav.get("selection_tier_name")) == "gd_tsp" else 0.0,
        }

    # Can any per-stop feature separate "stop was right" from "stop was wrong"
    # when the target does exist on the same floor?
    pos, neg = [], []
    for row in same_rows:
        info = row["info"]
        success = int(info["episode"].get("success") or 0) == 1
        is_fp = str(info["episode"].get("failure_cause")) == "false_positive"
        if not info.get("final_nav"):
            continue
        if success:
            pos.append(_features(info))
        elif is_fp and episode_stage(info) == "no_tp_detected":
            neg.append(_features(info))
    sep = {}
    if pos and neg:
        for name in pos[0]:
            sep[name] = auc([p[name] for p in pos], [q[name] for q in neg])

    return {
        "available": True,
        "missing": missing,
        "n": len(rows),
        "n_same": len(same_rows),
        "n_cross": len(rows) - len(same_rows),
        "same_success": same_success,
        "same_sr": rate(same_success, len(same_rows)),
        "by_cause": dict(by_cause),
        "fp_by_stage": dict(fp_stage),
        "by_scene": {k: {**v, "same_sr": rate(v["success"], v["same_n"])} for k, v in by_scene.items()},
        "separability_success_vs_fp_same": {
            "n_pos": len(pos),
            "n_neg": len(neg),
            "auc": sep,
        },
    }


# ---------------------------------------------------------------------------
# section 5: per-class threshold calibration
# ---------------------------------------------------------------------------


def threshold_calibration(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    grid = [round(0.30 + 0.05 * i, 2) for i in range(12)]  # 0.30 .. 0.85
    rows = [c for c in candidates if c.get("sigma_c") is not None]
    vals = [float(c["sigma_c"]) for c in rows]
    labels = [bool(c.get("tp")) for c in rows]

    overall = {
        "n": len(rows),
        "tp": sum(1 for l in labels if l),
        "auc_sigma_c": auc([v for v, l in zip(vals, labels) if l], [v for v, l in zip(vals, labels) if not l]),
        "auc_conf": auc(
            [float(c["conf"]) for c in rows if c.get("tp") and c.get("conf") is not None],
            [float(c["conf"]) for c in rows if not c.get("tp") and c.get("conf") is not None],
        ),
        "sweep": sweep_threshold(vals, labels, grid),
        "best": best_threshold(vals, labels),
        "tp_sigma": quantiles([v for v, l in zip(vals, labels) if l]),
        "fp_sigma": quantiles([v for v, l in zip(vals, labels) if not l]),
    }

    per_class: Dict[str, Any] = {}
    by_class: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for c in rows:
        by_class[str(c.get("target_class") or c.get("target_object") or "unknown")].append(c)
    for cls, items in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        v = [float(c["sigma_c"]) for c in items]
        l = [bool(c.get("tp")) for c in items]
        n_tp = sum(1 for x in l if x)
        at_global = [c for c in items if float(c["sigma_c"]) >= 0.60]
        per_class[cls] = {
            "n": len(items),
            "tp": n_tp,
            "tp_rate": rate(n_tp, len(items)),
            "auc": auc([x for x, y in zip(v, l) if y], [x for x, y in zip(v, l) if not y]),
            "at_0.60": {
                "kept": len(at_global),
                "tp": sum(1 for c in at_global if c.get("tp")),
                "fp": sum(1 for c in at_global if not c.get("tp")),
            },
            "best": best_threshold(v, l),
            "tp_sigma_p50": quantiles([x for x, y in zip(v, l) if y]).get("p50"),
            "fp_sigma_p50": quantiles([x for x, y in zip(v, l) if not y]).get("p50"),
        }
    return {"overall": overall, "per_class": per_class}


# ---------------------------------------------------------------------------
# section 6: fusion AUC recomputation
# ---------------------------------------------------------------------------


def fusion_recheck(candidates: List[Dict[str, Any]], fusion_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cand_with_raw = [c for c in candidates if c.get("c_raw") is not None]
    fusion_cands = [r for r in fusion_rows if r.get("kind") == "fusion_candidate"]
    n = min(len(cand_with_raw), len(fusion_cands))
    aligned = 0
    for a, b in zip(cand_with_raw[:n], fusion_cands[:n]):
        if (
            a.get("scene") == b.get("scene")
            and int(a.get("episode_id")) == int(b.get("episode_id"))
            and int(a.get("step") or 0) == int(b.get("step") or 0)
            and abs(float(a["c_raw"]) - float(b["c_raw"])) < 1e-9
        ):
            aligned += 1

    labels = [bool(a.get("tp")) for a in cand_with_raw[:n]]

    def score(field: str) -> Optional[float]:
        pos = [float(b[field]) for b, l in zip(fusion_cands[:n], labels) if l and b.get(field) is not None]
        neg = [float(b[field]) for b, l in zip(fusion_cands[:n], labels) if not l and b.get(field) is not None]
        return auc(pos, neg)

    neg_rows = [r for r in fusion_rows if r.get("kind") == "fusion_negative"]
    return {
        "n_candidates_with_raw": len(cand_with_raw),
        "n_fusion_candidates": len(fusion_cands),
        "row_aligned": aligned,
        "aligned_ratio": rate(aligned, n),
        "auc": {
            "c_raw": score("c_raw"),
            "c_vol_pts": score("c_vol_pts"),
            "c_vol_box": score("c_vol_box"),
            "c_active": score("c_active"),
        },
        "n_fusion_negative_rows": len(neg_rows),
        "neg_triggered_candidates": sum(1 for r in fusion_cands if r.get("neg_triggered")),
        "neg_triggered_negative_rows": sum(1 for r in neg_rows if r.get("neg_triggered")),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def audit_run(run_dir: str, content_dir: Optional[str] = None) -> Dict[str, Any]:
    det = os.path.join(run_dir, "detections")
    episodes = load_jsonl(os.path.join(det, "episodes.jsonl"))
    candidates = load_jsonl(os.path.join(det, "candidates.jsonl"))
    nav_goals = load_jsonl(os.path.join(det, "nav_goals.jsonl"))
    memory_events = load_jsonl(os.path.join(det, "memory_events.jsonl"))
    stop_shadow = load_jsonl(os.path.join(det, "stop_shadow.jsonl"))
    fusion_rows = load_jsonl(os.path.join(det, "fusion_field.jsonl"))
    traj = _trajectory_stats(run_dir)
    content = load_objectnav_content(content_dir) if content_dir else {}

    index = build_episode_index(episodes, candidates, nav_goals, memory_events, stop_shadow)
    return {
        "run_dir": os.path.abspath(run_dir),
        "rows": {
            "episodes": len(episodes),
            "candidates": len(candidates),
            "nav_goals": len(nav_goals),
            "memory_events": len(memory_events),
            "stop_shadow": len(stop_shadow),
            "fusion_field": len(fusion_rows),
            "trajectories": len(traj),
            "content_scenes": len(content),
        },
        "buckets": episode_buckets(episodes),
        "attribution": attribution_ladder(index),
        "fp": fp_autopsy(index, memory_events, nav_goals),
        "never_saw": never_saw_autopsy(index, traj),
        "floor_split": floor_split(index, content),
        "thresholds": threshold_calibration(candidates),
        "fusion": fusion_recheck(candidates, fusion_rows),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(num: Optional[int], den: Optional[int]) -> str:
    if num is None or not den:
        return "-"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


def render_markdown(report: Dict[str, Any]) -> str:
    out: List[str] = []
    add = out.append
    add(f"# Offline audit: `{os.path.basename(report['run_dir'])}`\n")
    rows = report["rows"]
    add("日志规模: " + ", ".join(f"`{k}`={v}" for k, v in rows.items()) + "\n")

    b = report["buckets"]
    add("## 1. 失败桶\n")
    add(f"- 集数 {b['n']}，成功 {b['success']}（SR {_fmt(b['sr'])}），`oracle_success=1` {b['oracle1']}")
    add(f"- `success=1 & oracle_success=0` 冲突 {b['oracle_success_conflict']} 集（fog-of-war bbox mask 偏差）\n")
    add("| failure_cause | n | share | oracle1 | steps p50 | steps p90 | steps>=500 |")
    add("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in b["by_cause"]:
        add(
            f"| `{r['failure_cause']}` | {r['n']} | {_fmt(r['share'])} | {r['oracle1']} | "
            f"{_fmt(r['steps_p50'], 1)} | {_fmt(r['steps_p90'], 1)} | {r['hit_500']} |"
        )
    add("")
    add("| scene | n | success | SR | steps p50 |")
    add("| :--- | ---: | ---: | ---: | ---: |")
    for r in b["by_scene"]:
        add(f"| `{r['scene']}` | {r['n']} | {r['success']} | {_fmt(r['sr'])} | {_fmt(r['steps_p50'], 1)} |")
    add("")
    short = {
        "did_not_fail": "SR",
        "false_positive": "fp",
        "never_saw_target_did_not_travel_stairs_likely_infeasible": "ns_stairs",
        "never_saw_target_did_not_travel_stairs_feasible": "ns_feasible",
        "never_saw_target_traveled_stairs": "ns_traveled",
        "timeout_true_positive": "timeout",
        "bad_stop_true_positive": "bad_stop",
        "false_negative": "fn",
    }
    causes = [c["failure_cause"] for c in b["by_cause"]]
    add("| scene | " + " | ".join(short.get(c, c) for c in causes) + " |")
    add("| :--- | " + " | ".join("---:" for _ in causes) + " |")
    for scene, counts in (b.get("scene_cause") or {}).items():
        add(f"| `{scene}` | " + " | ".join(str(counts.get(c, 0)) for c in causes) + " |")
    add("")

    add("## 2. 归因阶梯（检测 -> 准入 -> 使用）\n")
    add("| failure_cause | n | 无 TP 候选 | 有 TP 未准入 | 已准入未被用 | TP 已被用 |")
    add("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for cause, r in report["attribution"].items():
        add(
            f"| `{cause}` | {r['n']} | {r['no_tp_detected']} | {r['tp_not_admitted']} | "
            f"{r['tp_admitted_not_used']} | {r['tp_used']} |"
        )
    add("")

    fp = report["fp"]
    add("## 3. false_positive 分账\n")
    add("| 子类 | n | oracle1 | oracle0 | 曾 STOP 资格 | 多 memory | lock step p50 | lock dist p50 | lock tier |")
    add("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for name, s in fp["categories"].items():
        if not s.get("n"):
            continue
        add(
            f"| `{name}` | {s['n']} | {s['oracle1']} | {s['oracle0']} | {s['stop_eligible_ever']} | "
            f"{s['multi_memory']} | {_fmt(s['lock_step_p50'], 1)} | {_fmt(s['lock_dist_p50'], 2)} | "
            f"{s['lock_tier']} |"
        )
    add("")
    add("| 子类 | oracle=1 且曾获 STOP 资格 | oracle=1 从未 | oracle=0 且曾获 STOP 资格 | oracle=0 从未 |")
    add("| :--- | ---: | ---: | ---: | ---: |")
    for name, s in fp["categories"].items():
        if not s.get("n"):
            continue
        c = fp["oracle_stop"][name]
        add(f"| `{name}` | {c['o1_stop']} | {c['o1_nostop']} | {c['o0_stop']} | {c['o0_nostop']} |")
    add("")
    add(f"\n- 错目标锁定期间 TP 候选已出现的集: **{fp['tp_candidate_while_wrong_goal']}** / {fp['n']}")
    add(f"- 其中 TP 首次出现 **早于** 首次锁错步: **{fp['tp_step_before_first_wrong_lock']}**")
    d = fp.get("tp_used_detail") or {}
    if d.get("n"):
        add("")
        add("### 3.1 `tp_used` 子类：真目标当过 goal 却仍未成功\n")
        add(f"- n={d['n']}，其中真目标 memory 被 `stop_blacklist`: **{d['tp_memory_blacklisted']}**")
        add(f"- blacklist reason: `{d.get('blacklist_reasons')}`")
        add(
            f"- 真目标当 goal 时最近距离: p10 {_fmt(d['min_dist_to_tp_p10'], 2)} / "
            f"p50 {_fmt(d['min_dist_to_tp_p50'], 2)} / max {_fmt(d['min_dist_to_tp_max'], 2)} m"
        )
        add(
            f"- 真目标作 goal 的行数 p50 {_fmt(d['tp_goal_rows_p50'], 1)}，首次 {_fmt(d['tp_goal_first_step_p50'], 1)} 步，"
            f"最后一次 {_fmt(d['tp_goal_last_step_p50'], 1)} 步"
        )
        add(f"- 最终锁错 **晚于** 真目标最后一次作 goal: **{d['wrong_lock_after_tp_last_step']}** / {d['n']}")
    add("")

    ns = report["never_saw"]
    add("## 4. never_saw_target 分账\n")
    add("| 组 | n | steps p50 | steps>=500 | 候选 p50 | 零候选集 | goal 数 p50 | 零 goal 集 | 最后 goal 步 p50 | TP 候选总数 | 覆盖 0.5m 格 p50 | 路径 p50 (m) | z span p50 (m) |")
    add("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, s in ns.items():
        if name.startswith("_"):
            continue
        add(
            f"| `{name}` | {s['n']} | {_fmt(s['steps_p50'], 1)} | {s['hit_500']} | {_fmt(s['candidates_p50'], 1)} | "
            f"{s['zero_candidate']} | {_fmt(s.get('n_nav_goals_p50'), 1)} | {s.get('zero_nav_goal', 0)} | "
            f"{_fmt(s.get('last_nav_step_p50'), 1)} | {s['n_tp_candidates_total']} | "
            f"{_fmt(s['coverage_cells_p50'], 1)} | {_fmt(s['path_len_p50'], 1)} | {_fmt(s['z_span_p50'], 2)} |"
        )
    add("")

    fl = report.get("floor_split") or {}
    if fl.get("available"):
        add("## 4b. 同层 / 跨层拆分（目标是否存在于起始楼层）\n")
        add(
            f"- 覆盖 {fl['n']} 集（缺 {fl['missing']}）；同层 **{fl['n_same']}**、跨层 **{fl['n_cross']}**；"
            f"同层 SR **{fl['same_success']}/{fl['n_same']} = {_fmt(fl['same_sr'])}**"
        )
        add("")
        add("| failure_cause | n | 同层目标存在 | 跨层（需上楼） |")
        add("| :--- | ---: | ---: | ---: |")
        for cause, c in sorted(fl["by_cause"].items(), key=lambda kv: -kv[1]["n"]):
            add(f"| `{cause}` | {c['n']} | {c['same']} | {c['cross']} |")
        add("")
        add("| fp 子类 | 同层 | 跨层 |")
        add("| :--- | ---: | ---: |")
        for stage, c in fl["fp_by_stage"].items():
            add(f"| `{stage}` | {c['same']} | {c['cross']} |")
        add("")
        add("| scene | n | 同层 n | 同层 success | 同层 SR |")
        add("| :--- | ---: | ---: | ---: | ---: |")
        for scene, c in fl["by_scene"].items():
            add(f"| `{scene}` | {c['n']} | {c['same_n']} | {c['success']} | {_fmt(c['same_sr'])} |")
        add("")
        sep = fl.get("separability_success_vs_fp_same") or {}
        if sep.get("auc"):
            add(
                f"终态可分性（正样本=同层成功的终止候选 {sep['n_pos']} 个，负样本=同层 fp `no_tp_detected` "
                f"的终止候选 {sep['n_neg']} 个）：\n"
            )
            add("| 终态量 | AUC |")
            add("| :--- | ---: |")
            for name, value in sep["auc"].items():
                add(f"| `{name}` | {_fmt(value)} |")
            add("")
    else:
        add("## 4b. 同层 / 跨层拆分\n")
        add("- 未加载 ObjectNav content JSON，跳过（用 `--content-dir` 指定）\n")

    th = report["thresholds"]["overall"]
    add("## 5. candidate 级阈值标定\n")
    add(f"- n={th['n']}，TP={th['tp']}，AUC(`sigma_c`)={_fmt(th['auc_sigma_c'])}，AUC(`conf`)={_fmt(th['auc_conf'])}")
    add(f"- TP `sigma_c`: p10 {_fmt(th['tp_sigma'].get('p10'))} / p50 {_fmt(th['tp_sigma'].get('p50'))} / p90 {_fmt(th['tp_sigma'].get('p90'))}")
    add(f"- FP `sigma_c`: p10 {_fmt(th['fp_sigma'].get('p10'))} / p50 {_fmt(th['fp_sigma'].get('p50'))} / p90 {_fmt(th['fp_sigma'].get('p90'))}\n")
    add("| t | kept | TP | FP | precision | recall | FP keep rate |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in th["sweep"]:
        add(
            f"| {r['t']:.2f} | {r['n_kept']} | {r['tp']} | {r['fp']} | {_fmt(r['precision'])} | "
            f"{_fmt(r['recall'])} | {_fmt(r['fp_keep_rate'])} |"
        )
    add("")
    add("| class | n | TP | TP rate | AUC | @0.60 kept/TP/FP | best t (J) | TP p50 | FP p50 |")
    add("| :--- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: |")
    for cls, r in report["thresholds"]["per_class"].items():
        best = r.get("best") or {}
        add(
            f"| `{cls}` | {r['n']} | {r['tp']} | {_fmt(r['tp_rate'])} | {_fmt(r['auc'])} | "
            f"{r['at_0.60']['kept']}/{r['at_0.60']['tp']}/{r['at_0.60']['fp']} | "
            f"{_fmt(best.get('t'), 2)} ({_fmt(best.get('j'))}) | "
            f"{_fmt(r['tp_sigma_p50'])} | {_fmt(r['fp_sigma_p50'])} |"
        )
    add("")

    fu = report["fusion"]
    add("## 6. 复核 S8 §3 融合 AUC\n")
    add(f"- 对齐: {fu['row_aligned']}/{fu['n_fusion_candidates']}（ratio {_fmt(fu['aligned_ratio'])}）")
    for name, value in fu["auc"].items():
        add(f"- AUC(`{name}`) = {_fmt(value)}")
    add(f"\n- `fusion_negative` 行 {fu['n_fusion_negative_rows']}，其中 `neg_triggered` {fu['neg_triggered_negative_rows']}")
    add("")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline failure audit for a VLVM navigation run.")
    parser.add_argument("--run-dir", required=True, help="run directory containing detections/*.jsonl")
    parser.add_argument("--json-out", default=None, help="write the full report as JSON here")
    parser.add_argument("--md-out", default=None, help="write the markdown summary here")
    parser.add_argument(
        "--content-dir",
        default="data/datasets/objectnav/hm3d/v1/val/content",
        help="per-scene ObjectNav episode JSONs (for the same-floor split); '' disables it",
    )
    args = parser.parse_args(argv)

    report = audit_run(args.run_dir, content_dir=args.content_dir or None)
    md = render_markdown(report)
    print(md)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\n[wrote] {args.json_out}")
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as handle:
            handle.write(md)
        print(f"[wrote] {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
