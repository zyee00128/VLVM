"""Three-point bookkeeping fields for the per-memory logs (S8 / M-0).

The memory manager keeps three points per target memory:

* centroid      -- EMA centre used for merge/selection/region keys
* first surface -- first written near-surface point; **navigation goal**
* latest surface-- most recent observed near-surface point; recorded only

Only the first two are consumed today.  ``surface_latest`` exists so the
staleness of the navigation goal can be measured offline (``latest - first``
displacement, whether ``latest`` leaves the target bbox, share of stale first
points).  Nothing in this module changes navigation, admission, merge, STOP or
any threshold: it only formats state fields for the logs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


NAV_POINT_SOURCE = "surface_first"


def _as_point(value: Any) -> Optional[list]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return [float(v) for v in arr[:3]]


def _shift(a: Optional[list], b: Optional[list]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a[:2]) - np.asarray(b[:2])))


def three_point_fields(
    state: Optional[Dict[str, Any]],
    centroid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Format the three points of one memory for logging.

    ``state`` is the lifecycle state dict returned by
    ``TargetMemoryManager.state_of`` (a shallow copy).  Missing fields are
    logged as ``None`` so older memories stay row-compatible.
    """
    state = state or {}
    center = _as_point(centroid if centroid is not None else state.get("centroid"))
    first = _as_point(state.get("surface_first"))
    latest = _as_point(state.get("surface_latest"))
    if first is None:
        # Fall back to the navigation surface stored by the caller.
        first = _as_point(state.get("surface"))
    return {
        "point_centroid": center,
        "point_surface_first": first,
        "point_surface_first_step": state.get("surface_first_step"),
        "point_surface_latest": latest,
        "point_surface_latest_step": state.get("surface_latest_step"),
        "point_first_latest_shift_m": _shift(first, latest),
        "nav_point_source": NAV_POINT_SOURCE,
    }


def three_point_fields_for(
    memory_manager: Any,
    centroid: Sequence[float],
    *,
    tol: float = 0.3,
) -> Dict[str, Any]:
    """Convenience wrapper: look the memory up by centroid, then format."""
    try:
        state = memory_manager.state_of(
            np.asarray(centroid, dtype=np.float64).reshape(-1), tol=tol
        )
    except Exception:  # noqa: BLE001 - bookkeeping must never break a run
        state = {}
    return three_point_fields(state, centroid=centroid)


def summarize_three_points(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Offline summary of ``point_first_latest_shift_m`` over logged rows."""
    shifts = [
        float(r["point_first_latest_shift_m"])
        for r in rows
        if r.get("point_first_latest_shift_m") is not None
    ]
    if not shifts:
        return {"n": 0}
    arr = np.asarray(shifts, dtype=np.float64)
    qs = np.quantile(arr, [0.5, 0.75, 0.9, 0.99])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(qs[0]),
        "p75": float(qs[1]),
        "p90": float(qs[2]),
        "p99": float(qs[3]),
        "max": float(arr.max()),
        "share_over_0.5m": float(np.mean(arr > 0.5)),
        "share_over_1.0m": float(np.mean(arr > 1.0)),
    }
