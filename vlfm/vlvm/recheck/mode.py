"""Recheck mode: explore frontier exhausted -> delayed STOP.

Every behaviour has exactly one job:

* trigger  - the frontier is empty, so explore would STOP; recheck defers it.
* queue    - the retained memories are waypoints that keep the robot moving
             and generate close-range observations on the way.
* order    - the target-selection comparator decides the tour priority.
* budget   - ``recheck_arrive_steps`` is the only throttle: one fixed arrival
             budget per waypoint.  A distance-adaptive budget was tried and
             removed (it only shortened the budget of close waypoints and
             caused timeouts the fixed value would not have produced).

It never evaluates, promotes or releases anything, and it does not cap the
whole queue: the tour always runs to the end and the controlled explore STOP
fires exactly where the last waypoint leaves the robot.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from torch import Tensor


class RecheckMixin:
    def _init_recheck(
        self,
        *,
        enable: bool = True,
        arrive_steps: int = 45,
    ) -> None:
        """Initialise the delayed-STOP switches (hyper-parameters)."""
        self._recheck_enabled = bool(enable)
        self._recheck_arrive_steps = max(5, int(arrive_steps))
        self._recheck_reset()

    def _recheck_reset(self) -> None:
        self._recheck_active = False
        self._recheck_queue: List[Dict[str, Any]] = []
        self._recheck_idx = 0
        self._recheck_total_steps = 0
        self._recheck_attempted: set = set()

    def _log_recheck_event(self, event: str, **fields: Any) -> None:
        """Write one recheck row into the memory-event stream."""
        row: Dict[str, Any] = {
            "event": str(event),
            "step": int(fields.pop("step", self._num_steps)),
        }
        row.update(fields)
        self._log_memory_event(row)

    def _recheck_candidates(
        self, robot_xyz: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Selection-ordered memory list for recheck (blacklist NOT filtered)."""
        robot_xy = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2]
        rows: List[Dict[str, Any]] = []
        for cls in [
            c.strip()
            for c in str(self._target_object).split("|")
            if c.strip()
        ]:
            centroids = self._target_3d_memory.get(cls) or []
            surfaces = self._target_surface_memory.get(cls) or []
            verify = self._target_verify_state.get(cls) or []
            for i, centroid in enumerate(centroids):
                c = np.asarray(centroid, dtype=np.float64).reshape(-1)
                if c.size < 2:
                    continue
                st = self._memory_manager.state_of(c)
                memory_id = st.get("memory_id")
                if memory_id is None:
                    continue
                memory_id = str(memory_id)
                if memory_id in self._recheck_attempted:
                    continue
                state_name = str(st.get("lifecycle_state") or "confirmed")
                if state_name == "stable":
                    continue
                origin = str(st.get("origin") or "")
                promoted = bool(st.get("promoted_from_gd_only", False))
                gd_tsp = origin == "gd-tsp" or promoted
                gd_only = bool(st.get("gd_only", False)) or state_name == "gd"
                if gd_tsp:
                    tier = 0
                elif gd_only:
                    tier = 1
                elif state_name == "confirmed":
                    tier = 2
                elif state_name == "provisional":
                    tier = 3
                else:
                    tier = 4
                score = (
                    float(verify[i][3])
                    if i < len(verify) and len(verify[i]) > 3
                    else 0.0
                )
                obs = (
                    int(verify[i][0])
                    if i < len(verify) and len(verify[i]) >= 1
                    else 1
                )
                last_obs = int(st.get("last_obs_step", self._num_steps))
                stale = max(
                    0,
                    max(0, int(self._memory_manager.frame) - 1) - last_obs,
                )
                goal = c[:2].copy()
                if self._goal_use_surface and i < len(surfaces):
                    goal = np.asarray(
                        surfaces[i], dtype=np.float64
                    ).reshape(-1)[:2]
                rows.append(
                    {
                        "memory_id": memory_id,
                        "target_class": str(cls),
                        "centroid": c[:3].copy(),
                        "goal_xy": goal.copy(),
                        "tier": int(tier),
                        "sigma_band": int(self._sigma_band(score)),
                        "gd_hits": int(st.get("gd_hits", 0) or 0),
                        "distance": float(np.linalg.norm(goal - robot_xy)),
                        "num_obs": int(obs),
                        "stale": int(stale),
                        "lifecycle_state": state_name,
                        "origin": origin or None,
                        "blacklisted": bool(memory_id in self._stop_blacklist),
                    }
                )
        # Ordering = the target-selection comparator, not nearest-first:
        # tier -> sigma band -> gd_hits -> 0.5 m distance bucket -> num_obs -> stale.
        def _selection_key(r: Dict[str, Any]) -> Any:
            tier = int(r["tier"])
            band = int(r["sigma_band"]) if tier in (0, 2, 3) else 0
            hits = -int(r["gd_hits"]) if tier in (0, 1) else 0
            bucket = int(round(float(r["distance"]) / 0.5))
            obs = int(r["num_obs"])
            return (
                tier,
                band,
                hits,
                bucket,
                0 if obs >= 3 else 1,
                -obs,
                int(r["stale"]),
                str(r["memory_id"]),
            )

        rows.sort(key=_selection_key)
        return rows

    def _recheck_begin(self, robot_xyz: np.ndarray) -> bool:
        """Build the recheck queue once; False when nothing can be checked."""
        if not self._recheck_enabled:
            return False
        try:
            rows = self._recheck_candidates(robot_xyz)
        except BaseException as exc:  # noqa: BLE001 - recheck must not kill nav
            print(
                f"[RECHECK] candidate build failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        if not rows:
            return False
        self._recheck_queue = rows
        self._recheck_idx = 0
        self._recheck_active = True
        self._recheck_total_steps = 0
        rows[0]["start_step"] = int(self._num_steps)
        print(
            "[RECHECK] start "
            f"n={len(rows)} step={int(self._num_steps)} "
            f"first={rows[0]['memory_id']} tier={rows[0]['tier']} "
            f"dist={rows[0]['distance']:.2f} "
            f"blacklisted={rows[0]['blacklisted']}",
            flush=True,
        )
        self._log_recheck_event(
            "recheck_start",
            n_candidates=len(rows),
            candidates=[
                {
                    "memory_id": r["memory_id"],
                    "tier": r["tier"],
                    "distance": r["distance"],
                    "blacklisted": r["blacklisted"],
                }
                for r in rows
            ],
        )
        return True

    def _recheck_act(
        self,
        observations: "TensorDict",
        robot_xyz: np.ndarray,
    ) -> Optional[Tensor]:
        """Drive the tour; STOP exactly where the last waypoint leaves us."""
        if not self._recheck_active:
            return None
        try:
            self._recheck_total_steps += 1
            robot_xy = np.asarray(
                robot_xyz, dtype=np.float64
            ).reshape(-1)[:2]
            if self._recheck_idx >= len(self._recheck_queue):
                return self._recheck_stop(robot_xy)
            item = self._recheck_queue[self._recheck_idx]
            item.setdefault("start_step", int(self._num_steps))
            rho = float(np.linalg.norm(item["goal_xy"] - robot_xy))
            if rho > float(self._pointnav_stop_radius):
                used = int(self._num_steps) - int(item["start_step"])
                if used > self._recheck_arrive_steps:
                    self._log_recheck_event(
                        "recheck_result",
                        memory_id=item["memory_id"],
                        outcome="arrive_timeout",
                        steps_used=int(used),
                        distance=rho,
                        blacklisted=item["blacklisted"],
                    )
                    print(
                        "[RECHECK] arrive timeout "
                        f"memory_id={item['memory_id']} "
                        f"steps={int(used)} dist={rho:.2f}",
                        flush=True,
                    )
                    return self._recheck_advance(robot_xy)
                return self._pointnav(item["goal_xy"], stop=False)
            # Arrival itself is the payload: the robot is now next to a
            # retained memory and this frame's TSP3D/GD queries already ran.
            self._log_recheck_event(
                "recheck_result",
                memory_id=item["memory_id"],
                outcome="arrived",
                steps_used=int(self._num_steps) - int(item["start_step"]),
                distance=rho,
                blacklisted=item["blacklisted"],
            )
            print(
                "[RECHECK] arrived "
                f"memory_id={item['memory_id']} dist={rho:.2f} "
                f"step={int(self._num_steps)}",
                flush=True,
            )
            return self._recheck_advance(robot_xy)
        except BaseException as exc:  # noqa: BLE001 - recheck must not kill nav
            print(
                f"[RECHECK] disabled after error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            self._recheck_active = False
            return None

    def _recheck_advance(self, robot_xyz: np.ndarray) -> Tensor:
        item = self._recheck_queue[self._recheck_idx]
        self._recheck_attempted.add(str(item["memory_id"]))
        self._recheck_idx += 1
        if self._recheck_idx >= len(self._recheck_queue):
            robot_xy = np.asarray(
                robot_xyz, dtype=np.float64
            ).reshape(-1)[:2]
            return self._recheck_stop(robot_xy)
        # Static comparator order: the selection key does not depend on where
        # the robot currently stands, so the queue is never re-sorted.
        nxt = self._recheck_queue[self._recheck_idx]
        nxt["start_step"] = int(self._num_steps)
        print(
            "[RECHECK] next "
            f"memory_id={nxt['memory_id']} tier={nxt['tier']} "
            f"dist={nxt['distance']:.2f} blacklisted={nxt['blacklisted']}",
            flush=True,
        )
        return self._pointnav(nxt["goal_xy"], stop=False)

    def _recheck_stop(self, robot_xyz: np.ndarray) -> Tensor:
        """Emit the controlled explore STOP exactly where the tour ended."""
        self._recheck_active = False
        self._explore_exhausted = True
        robot_xy = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)[:2]
        best = self._recheck_queue[0] if self._recheck_queue else None
        best_id = str(best["memory_id"]) if best is not None else None
        best_dist = (
            float(
                np.linalg.norm(
                    np.asarray(best["goal_xy"], dtype=np.float64).reshape(-1)[:2]
                    - robot_xy
                )
            )
            if best is not None
            else 0.0
        )
        # ``best_distance`` is log-only: it records how far the STOP location
        # ended up from the top-ranked waypoint (the anchor idea, unenforced).
        self._log_recheck_event(
            "recheck_exhausted",
            total_steps=int(self._recheck_total_steps),
            outcome="queue_exhausted",
            best_memory_id=best_id,
            best_distance=best_dist,
        )
        print(
            "[RECHECK] exhausted; STOP "
            f"step={int(self._num_steps)} "
            f"total_steps={int(self._recheck_total_steps)} "
            f"best={best_id} best_dist={best_dist:.2f}",
            flush=True,
        )
        return self._stop_action

