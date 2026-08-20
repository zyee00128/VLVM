from dataclasses import dataclass
from typing import Any, Dict, Union
import numpy as np
import torch
from torch import Tensor
from depth_camera_filtering import filter_depth

from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import PolicyConfig
from habitat_baselines.rl.ppo.policy import PolicyActionData
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.detections import ObjectDetections
from vlfm.policy.tsp3d_objectnav_policy import TSP3DObjectNavPolicy, VLVMConfig
from vlfm.policy.itm3d_policy import ITM3DPolicyV1
from vlfm.policy.itm_policy import ITMPolicyV1, ITMPolicyV2

HM3D_ID_TO_NAME = [
    "chair",
    "bed", 
    "potted plant",
    "toilet", 
    "tv", 
    "couch"]
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",
    "framed photograph",
    "cabinet",
    "pillow",
    "couch",
    "bed",
    "nightstand",
    "potted plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class Habitat3DMixin:
    """
    This Mixin provides adapter utilities for running 3D active semantic navigation 
    policies inside the Habitat simulator, managing observation parsing, parameter routing, and action mapping.
    """

    _stop_action: Tensor = TorchActionIDs.STOP
    _turn_left_action: Tensor = TorchActionIDs.TURN_LEFT
    _start_yaw: Union[float, None] = None
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    
    def __init__(
        self,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        dataset_type: str = "hm3d",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs["camera_height"] = camera_height
        kwargs["min_depth"] = min_depth
        kwargs["max_depth"] = max_depth
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

        # When initializing the agent, if no checkpoint is detected, Habitat tries
        # to orthogonally initialize the network weights, e.g.
        # nn.init.orthogonal_(actor_critic.critic.fc.weight). This navigation
        # policy is a zero-shot, purely logic/geometry-based heuristic with no
        # critic or actor neural networks, so Habitat would crash when it tries
        # to access critic.fc.weight. The dummy nets below prevent that.
        class DummyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(1, 1)
        self.actor = DummyNet()
        self.critic = DummyNet()

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> "Habitat3DMixin":
        # Parse policy configurations from YAML and route them to the policy constructors
        policy_config = config.habitat_baselines.rl.policy
        # Safely resolve OmegaConf DictConfig to a native Python dictionary for **kwargs unpacking
        kwargs = OmegaConf.to_container(policy_config, resolve=True)
        kwargs.pop("name", None)

        # Bind Habitat sensor configurations (Focal length, height, FOV for geometric projection)
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")

        return cls(**kwargs)

    def act(
        self: Union["Habitat3DMixin", TSP3DObjectNavPolicy],
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """
        Maps incoming object IDs to string names,
        invokes the base 3D Policy, and maps predicted actions.
        """
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()
        
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
        elif self._dataset_type == "mp3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
        
        try:
            action, rnn_hidden_states = super().act(
                obs_dict, rnn_hidden_states, prev_actions, masks, deterministic
            )
        except StopIteration:
            action = self._stop_action
            
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing = not self._num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        super()._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        info = super()._get_policy_info(detections)

        if getattr(self, "_visualize", False):
            if self._start_yaw is None:
                self._start_yaw = self._observations_cache.get("habitat_start_yaw", 0.0)
            info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["Habitat3DMixin", TSP3DObjectNavPolicy], observations: TensorDict) -> None:
        """
        Parses RGB, Depth, and GPS data from Habitat, and converts them into 
        the 3D transformations and raw point clouds required by the 3D policies.
        """
        if len(self._observations_cache) > 0:
            return
            
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        
        # Process and filter the normalized depth image
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Convert Habitat GPS coordinates to episodic 3D coordinates (X, Y, Z)
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        # Construct camera-to-episodic transformation matrix (4x4)
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        # Fallback checks for frontier sensors
        if "frontier_sensor_3d" in observations:
            frontiers_3d = observations["frontier_sensor_3d"][0].cpu().numpy()
        else:
            frontiers_3d = np.array([])
            
        if "frontier_sensor" in observations:
            frontiers = observations["frontier_sensor"][0].cpu().numpy()
        else:
            frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "frontier_sensor_3d": frontiers_3d,
            "nav_depth": observations["depth"], # Target depth for base PointNav
            "robot_xy": robot_xy,               # 2D horizontal coordinates
            "robot_xy_z": camera_position,      # 3D world coordinates
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }


@baseline_registry.register_policy
class OracleFBEPolicy(Habitat3DMixin, TSP3DObjectNavPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(Habitat3DMixin, TSP3DObjectNavPolicy):
    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        oracle_key = "frontier_sensor" \
            if "frontier_sensor" in observations \
            else [k for k in observations.keys() 
                    if "explorer" in k or "frontier" in k][0]
        return PolicyActionData(
            actions=observations[oracle_key],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITM3DPolicy(Habitat3DMixin, ITM3DPolicyV1):
    """
    3D semantic value mapping: projects cosine scores into a sparse 3D
    semantic voxel grid with confidence-weighted fusion, 
    space carving, and distant voxel pruning.
    """
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV1(Habitat3DMixin, ITMPolicyV1):
    """
    2.5D BEV semantic value plane (route 1, region style): 
    VLFM 2D ValueMap + H1 (vertical passability).
    """
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV2(Habitat3DMixin, ITMPolicyV2):
    """
    2.5D BEV semantic value plane (route 2, surface style): 
    3D surface points -> 2D buckets (confidence-gated S) + H1.
    """
    pass


@dataclass
class VLVMPolicyConfig(VLVMConfig, PolicyConfig):
    pass


cs = ConfigStore.instance()
cs.store(group="habitat_baselines/rl/policy", name="vlvm_policy", node=VLVMPolicyConfig)
