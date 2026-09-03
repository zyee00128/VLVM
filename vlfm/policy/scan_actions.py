"""Scan-only wide-turn task action for the VLVM panoramic route.

Decouples the scan rotation from the explore/navigate rotation. Normal movement
keeps the standard 30° ``turn_left`` (env discrete ids 0-3); a panoramic scan
instead emits this action, which internally performs ``turns`` consecutive
standard 30° sim turns inside ONE environment step (scan angle = turns * 30°,
default turns=2 -> 60° per scan step).

Registered at import time (``vlfm.run``) so the env instantiates it while
building the task-action space. It must be appended at the END of
``habitat.task.actions`` in the experiment YAML so that the standard action ids
never shift (stop=0, move_forward=1, turn_left=2, turn_right=3, look_up=4,
look_down=5, turn_left_wide=6).

The habitat ActionConfig schema does not carry per-action scalars, so the turn
count is read from the environment variable ``TURN_LEFT_WIDE_TURNS`` (default
2 -> 60°). ``vlfm.run`` sets it automatically from the YAML hyperparameter
``panoramic_turn_steps`` (per-step scan angle = 360/turn_steps, turns =
angle/30), so the user only edits the YAML value and never exports this
variable manually.
"""

import os
from typing import Any

from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions


@registry.register_task_action
class TurnLeftWideAction(SimulatorTaskAction):
    """Turn left by ``turns`` x 30° in a single environment step (scan only)."""

    def __init__(
        self,
        *args: Any,
        config: Any,
        sim: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._turns = max(int(os.environ.get("TURN_LEFT_WIDE_TURNS", "2")), 1)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        obs = None
        for _ in range(self._turns):
            obs = self._sim.step(HabitatSimActions.turn_left)
        return obs
