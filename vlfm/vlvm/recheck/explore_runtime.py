"""Explore runtime: the semantic-value / nearest-frontier switch and its fallbacks.

Moved out of ``tsp3d_objectnav_policy.py`` so the policy file keeps only the
navigation pipeline.  Everything here is *behaviour*, not just logging: the
explore mode decision changes which frontier is pursued, and ``_explore_safe``
owns the stuck / exhausted / recheck fallback ladder.

Contract with the policy (attributes it must provide):

* ``self._explore``            - subclass frontier policy (``BaseITMPolicy``).
* ``self._stop_action``        - STOP tensor.
* ``self._observations_cache`` - per-step cache with ``robot_xy`` /
                                 ``frontier_sensor``.
* ``self._num_steps``          - step counter.
* ``self._recheck_*``          - supplied by :class:`ExplRecheckMixin`.
* ``self._explore_stuck_steps`` / ``_explore_stuck_move_m`` - stuck budget.

State owned here (initialised by :meth:`._init_explore_mode`, cleared by
:meth:`._reset_explore_mode`):

``_explore_mode_cfg`` / ``_explore_mode_rows`` / ``_explore_stuck_count`` /
``_last_explore_xy`` / ``_explore_forced_fallback`` /
``_explore_frontier_blacklist`` / ``_explore_exhausted`` /
``_explore_stop_suppressed``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from torch import Tensor

from vlfm.vlvm.recheck.explore_mode import ExploreModeConfig
from vlfm.vlvm.recheck.explore_mode import decision_row as explore_mode_row

try:  # habitat is optional for offline tooling
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:  # pragma: no cover
    TensorDict = None  # type: ignore[assignment]


class ExploreRuntimeMixin:
    """Explore-mode switch, stuck detection and the recheck fallback ladder."""

    # ---- lifecycle -----------------------------------------------------
    def _init_explore_mode(
        self,
        enable: bool = True,
        sem_value_min: float = 0.14,
        sticky_radius: float = 0.50,
        stuck_steps: int = 15,
        stuck_move_m: float = 0.05,
    ) -> None:
        """Set up the semantic-vs-nearest frontier switch.

        The decision itself lives in :mod:`vlfm.vlvm.recheck.explore_mode` and
        is applied by ``BaseITMPolicy._explore``.  The only knob is
        ``sem_value_min``, the pure ValueMap score ``S`` of the best frontier;
        it can only be calibrated by A/B runs, not by offline shadow replay.
        Logging is not a separate switch: it follows ``enable``.
        """
        self._explore_mode_cfg: ExploreModeConfig = ExploreModeConfig(
            enable=bool(enable),
            sem_value_min=float(sem_value_min),
            sticky_radius=float(sticky_radius),
            log=bool(enable),
        )
        self._explore_stuck_steps = int(stuck_steps)
        self._explore_stuck_move_m = float(stuck_move_m)
        self._reset_explore_mode()

    def _reset_explore_mode(self) -> None:
        """Clear per-episode explore state (called from ``policy._reset``)."""
        self._explore_mode_rows = []
        self._explore_stuck_count = 0
        self._explore_exhausted = False
        self._explore_stop_suppressed = 0
        self._explore_fallback_turns = 0
        self._explore_forced_fallback = False
        self._last_explore_xy = None
        self._explore_frontier_blacklist = []

    # ---- logging -------------------------------------------------------
    def _explore_mode_note(
        self,
        mode: str,
        reason,
        top_value: float,
        n_frontiers: int,
        chosen,
        robot_xy: np.ndarray,
        sem_goal=None,
        near_goal=None,
        value_with_h1=None,
        n_blacklisted: int = 0,
    ) -> None:
        """Record one ``explore_mode.jsonl`` row; never changes the action.

        ``top_value`` is the pure ``S`` the gate used; ``value_with_h1`` is the
        legacy ``S + h_lam*H1`` number, logged only.
        """
        cfg = getattr(self, "_explore_mode_cfg", None)
        if cfg is None or not bool(cfg.log):
            return
        rows = getattr(self, "_explore_mode_rows", None)
        if rows is None:
            return
        row = explore_mode_row(
            mode=mode,
            reason=reason,
            best_value=top_value,
            n_frontiers=n_frontiers,
            chosen=chosen,
            robot_xy=robot_xy,
            cfg=cfg,
            sem_goal=sem_goal,
            near_goal=near_goal,
            value_with_h1=value_with_h1,
        )
        row["step"] = int(self._num_steps)
        row["n_blacklisted"] = int(n_blacklisted)
        rows.append(row)
        if str(mode) != "semantic":
            print(
                "[EXPLORE_MODE] nearest frontier "
                f"step={int(self._num_steps)} "
                f"S={float(top_value):.4f} "
                f"n_frontiers={int(n_frontiers)} "
                f"goal_dist={row.get('goal_dist')}",
                flush=True,
            )

    # ---- runtime -------------------------------------------------------
    @staticmethod
    def _is_stop_action(action: Tensor) -> bool:
        """Return True only for the discrete Habitat STOP action (0)."""
        try:
            values = action.detach().cpu().numpy().reshape(-1)
            return bool(values.size == 1 and int(values[0]) == 0)
        except Exception:
            return False

    def _filter_blacklisted_frontiers(self) -> None:
        """Remove frontiers that repeatedly failed to produce motion."""
        blacklist = getattr(self, "_explore_frontier_blacklist", None) or []
        if not blacklist:
            return
        cache = self._observations_cache
        raw = np.asarray(
            cache.get("frontier_sensor", np.empty((0, 2))),
            dtype=np.float64,
        )
        if raw.size == 0:
            return
        points = raw.reshape(-1, 2) if raw.ndim == 1 else raw[:, :2]
        keep = np.ones(points.shape[0], dtype=bool)
        for stuck in blacklist:
            stuck_xy = np.asarray(stuck, dtype=np.float64).reshape(-1)[:2]
            if stuck_xy.size != 2 or not np.all(np.isfinite(stuck_xy)):
                continue
            keep &= (
                np.linalg.norm(points - stuck_xy[None, :], axis=1)
                > 0.50
            )
        cache["frontier_sensor"] = points[keep]

    def _explore_safe(self, observations: "TensorDict") -> Tensor:
        """Explore via the Habitat frontier sensor and baseline explorer."""
        robot_xy = np.asarray(
            self._observations_cache.get("robot_xy", np.zeros(2)),
            dtype=np.float64,
        ).reshape(-1)[:2]

        if self._recheck_active:
            recheck_action = self._recheck_act(observations, robot_xy)
            if recheck_action is not None:
                return recheck_action

        self._filter_blacklisted_frontiers()

        # Fallback trigger: the robot has not moved for several explore steps.
        if self._last_explore_xy is not None:
            moved = float(np.linalg.norm(robot_xy - self._last_explore_xy))
            if moved <= self._explore_stuck_move_m:
                self._explore_stuck_count += 1
            else:
                self._explore_stuck_count = 0
        self._last_explore_xy = robot_xy.copy()

        self._explore_forced_fallback = False

        action = self._explore(observations)
        if not self._is_stop_action(action):
            if self._explore_stuck_count < self._explore_stuck_steps:
                self._explore_exhausted = False
                self._explore_fallback_turns = 0
                return action

            stuck_xy = np.asarray(
                getattr(self, "_last_frontier", np.zeros(2)),
                dtype=np.float64,
            ).reshape(-1)[:2]
            if (
                stuck_xy.size == 2
                and np.all(np.isfinite(stuck_xy))
                and not np.allclose(stuck_xy, 0.0)
            ):
                self._explore_frontier_blacklist.append(stuck_xy.copy())
            self._last_frontier = np.zeros(2, dtype=np.float64)
            self._explore_stuck_count = 0
            self._explore_forced_fallback = True
            self._explore_stop_suppressed += 1
            print(
                "[EXPLORE] frontier fallback: "
                f"stuck={self._explore_stuck_steps}steps "
                f"frontier={stuck_xy.tolist()}",
                flush=True,
            )
            self._filter_blacklisted_frontiers()
            retry_action = None
            try:
                retry_action = self._explore(observations)
            except Exception:
                retry_action = None
            if retry_action is not None and not self._is_stop_action(
                retry_action
            ):
                self._explore_forced_fallback = False
                self._explore_exhausted = False
                self._explore_fallback_turns = 0
                print(
                    "[EXPLORE] retry with failed frontier removed",
                    flush=True,
                )
                return retry_action
            self._explore_exhausted = True
        else:
            self._explore_forced_fallback = False
            self._explore_exhausted = True
            self._explore_stuck_count = 0
            self._explore_stop_suppressed += 1
            print(
                "[EXPLORE] frontier exhausted: STOP suppressed; "
                f"count={self._explore_stop_suppressed}",
                flush=True,
            )

        if self._recheck_begin(robot_xy):
            recheck_action = self._recheck_act(observations, robot_xy)
            if recheck_action is not None:
                return recheck_action
        self._explore_exhausted = True
        print(
            "[EXPLORE] no candidate to recheck: STOP allowed "
            f"step={int(self._num_steps)}",
            flush=True,
        )
        return self._stop_action

