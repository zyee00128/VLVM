"""Explore mode switching: semantic-value frontier vs. nearest frontier.

Pure decision logic, knows nothing about the policy.  The policy supplies the
value score and receives a mode name; see ``Results/VLVM-S9.md`` §1.

Motivation (ApexNav adaptive switching): semantic-value frontier selection is
only worth its long detours when there is *some* semantic signal to follow.
With no signal at all it just walks to the far side of the house, which is the
main reason ``never_saw_target`` episodes burn their step budget.  In that case
the cheaper policy is "expand where I already am": the nearest frontier.

The switch has a **single** trigger, the pure ValueMap score ``S`` of the best
frontier of this frame.  An earlier design also admitted a TSP3D-confidence
trigger (``sigma >= 0.50 and dist <= 4 m``), but measured on 31662 explore
frames it fired on 0.2% of them - ``sigma_prop=0.45`` already filters every
weak detection before it becomes a candidate, so the survivors sit on the
floor and the branch was dead weight.  It has been removed: the only knob that
actually moves behaviour is ``sem_value_min``, and its value can only be
decided by A/B experiments, not by offline shadow replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


MODE_SEMANTIC = "semantic"
MODE_NEAREST = "nearest"


@dataclass(frozen=True)
class ExploreModeConfig:
    """Thresholds of the explore mode switch (see ``Results/VLVM-S9.md`` §1)."""

    enable: bool = True
    # Semantic signal: the **pure ValueMap score ``S``** of the best frontier,
    # excluding ``h_lam * H1``.  Mixing passability into the gate would shift
    # it by ~+0.30 in every indoor frame, which is why an earlier 0.90 gate
    # could never fire (``S`` max is ~0.30).  ``S`` is the same quantity as the
    # S-penalty field: BLIP-2 ITM cosine fused into the map.
    # Calibrated at the p75 of the observed S distribution (297 episodes).
    sem_value_min: float = 0.14
    # nearest mode: keep pursuing the previous frontier while it is this close
    sticky_radius: float = 0.50
    # logging
    log: bool = True

    def __post_init__(self) -> None:
        if self.sem_value_min < 0.0:
            raise ValueError("sem_value_min must be non-negative")
        if self.sticky_radius < 0.0:
            raise ValueError("sticky_radius must be non-negative")


def classify(
    sem_value: float,
    cfg: ExploreModeConfig,
) -> Tuple[str, Optional[str]]:
    """Return ``(mode, reason)`` for one explore frame.

    ``sem_value`` is the pure ValueMap score ``S`` (no ``h_lam * H1``).
    ``reason`` is ``"value"`` when the semantic gate fired, else ``None``
    (no semantic signal -> the nearest frontier is used).
    """
    if float(sem_value) >= float(cfg.sem_value_min):
        return MODE_SEMANTIC, "value"
    return MODE_NEAREST, None


def nearest_index(
    points: np.ndarray,
    robot_xy: np.ndarray,
    sticky_point: Optional[np.ndarray] = None,
    sticky_radius: float = 0.5,
) -> int:
    """Index of the frontier to pursue in nearest mode (``-1`` if empty).

    ``points`` is the value-sorted frontier array; the caller keeps using this
    ordering so ``values[index]`` stays aligned.  If ``sticky_point`` is still
    present (within ``sticky_radius``) it is kept, which prevents the two
    closest frontiers from alternating frame by frame.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return -1
    pts = pts.reshape(-1, 2) if pts.ndim == 1 else pts[:, :2]
    robot = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]

    if sticky_point is not None:
        sticky = np.asarray(sticky_point, dtype=np.float64).reshape(-1)[:2]
        if sticky.size == 2 and np.all(np.isfinite(sticky)):
            gaps = np.linalg.norm(pts - sticky[None, :], axis=1)
            within = np.where(gaps <= float(sticky_radius))[0]
            if within.size:
                # Prefer the frontier closest to the one we were pursuing.
                return int(within[int(np.argmin(gaps[within]))])

    gaps = np.linalg.norm(pts - robot[None, :], axis=1)
    return int(np.argmin(gaps))


def decision_row(
    mode: str,
    reason: Optional[str],
    best_value: float,
    n_frontiers: int,
    chosen: Optional[np.ndarray],
    robot_xy: np.ndarray,
    cfg: ExploreModeConfig,
    sem_goal: Optional[np.ndarray] = None,
    near_goal: Optional[np.ndarray] = None,
    value_with_h1: Optional[float] = None,
) -> dict:
    """One ``explore_mode.jsonl`` row (no side effects).

    ``best_value`` is the pure ``S`` used by the gate; ``value_with_h1`` keeps
    the legacy ``S + h_lam*H1`` number for continuity with earlier runs.
    """
    row = {
        "mode": str(mode),
        "reason": reason,
        "sem_s": round(float(best_value), 5),
        "best_value": round(float(best_value), 5),
        "n_frontiers": int(n_frontiers),
        "sem_value_min": float(cfg.sem_value_min),
    }
    if chosen is not None:
        goal = np.asarray(chosen, dtype=np.float64).reshape(-1)[:2]
        if goal.size == 2:
            row["goal"] = [round(float(v), 3) for v in goal]
            row["goal_dist"] = round(
                float(
                    np.linalg.norm(
                        goal - np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
                    )
                ),
                3,
            )

    # The two candidate goals that the switch is choosing *between*.
    robot = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
    for key, point in (("sem_goal", sem_goal), ("near_goal", near_goal)):
        if point is None:
            continue
        goal = np.asarray(point, dtype=np.float64).reshape(-1)[:2]
        if goal.size != 2 or not np.all(np.isfinite(goal)):
            continue
        row[key] = [round(float(v), 3) for v in goal]
        row[f"{key}_dist"] = round(float(np.linalg.norm(goal - robot)), 3)
    if value_with_h1 is not None:
        row["value_with_h1"] = round(float(value_with_h1), 5)
    return row
