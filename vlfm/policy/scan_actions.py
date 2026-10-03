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
