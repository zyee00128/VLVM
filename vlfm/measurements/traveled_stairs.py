# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any, List

import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@registry.register_measure
class TraveledStairs(Measure):
    cls_uuid: str = "traveled_stairs"

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        # Peak-to-peak elevation threshold (m) for declaring that stairs were traveled.
        # A higher value requires a larger vertical excursion to fire, a lower value
        # is more sensitive to small elevation changes.
        self._peak_to_peak_thresh = getattr(config, "peak_to_peak_thresh", 0.9)
        self._history: List[np.ndarray] = []
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return TraveledStairs.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._history = []
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        curr_z = self._sim.get_agent_state().position[1]
        self._history.append(curr_z)
        # Make self._metric True (1) if peak-to-peak distance is greater than the threshold
        self._metric = int(np.ptp(self._history) > self._peak_to_peak_thresh)


@registry.register_measure
class CumulativeElevationChange(Measure):
    """
    Cumulative vertical elevation change.
    Records the cumulative vertical travel distance of the robot during 3D active
    exploration across multi-level rooms, reflecting the agent's physical
    exploration activity in 3D.
    """
    cls_uuid: str = "cumulative_elevation_change"

    def __init__(self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._last_z = None
        self._cumulative_change = 0.0
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return CumulativeElevationChange.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        # In the Habitat simulator, position[1] usually corresponds to the Y axis
        self._last_z = self._sim.get_agent_state().position[1]
        self._cumulative_change = 0.0
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        curr_z = self._sim.get_agent_state().position[1]
        if self._last_z is not None:
            self._cumulative_change += abs(curr_z - self._last_z)
        self._last_z = curr_z
        self._metric = self._cumulative_change


@dataclass
class TraveledStairsMeasurementConfig(MeasurementConfig):
    type: str = TraveledStairs.__name__
    # Peak-to-peak elevation threshold (m) for detecting that stairs were traveled;
    # higher = requires a larger vertical excursion, lower = more sensitive.
    peak_to_peak_thresh: float = 0.9


@dataclass
class CumulativeElevationChangeMeasurementConfig(MeasurementConfig):
    type: str = CumulativeElevationChange.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.traveled_stairs",
    group="habitat/task/measurements",
    name="traveled_stairs",
    node=TraveledStairsMeasurementConfig,
)

cs.store(
    package="habitat.task.measurements.cumulative_elevation_change",
    group="habitat/task/measurements",
    name="cumulative_elevation_change",
    node=CumulativeElevationChangeMeasurementConfig,
)