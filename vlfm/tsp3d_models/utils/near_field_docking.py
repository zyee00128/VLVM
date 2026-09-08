from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class NearFieldDockingConfig:
    """Pitch decision near-field camera pitch-down hyper-parameters."""

    pitch_enable: bool = True           # enable the look-down
    pitch_trigger_dist: float = 1.50    # consider look-down only within this 2D distance (m)
    pitch_aim_z_bias: float = 0.10      # safety margin added above the aim height (m)
    pitch_min_angle_deg: float = 8.0    # skip if the required tilt is below this (deg, anti-jitter)
    pitch_step_deg: float = 30.0        # tilt per look_down step (deg, matches tilt_angle)
    pitch_max_down_steps: int = 1       # max look_down steps per trigger


class NearFieldDockingEngine:
    """Pitch decision engine.

    Args:
        ``aim_xyz``: target aim world coordinate (surface point or centroid, with z)
        ``robot_xyz`` / ``camera_z``: robot pose and camera height
    """

    def __init__(self, config: Optional[NearFieldDockingConfig] = None) -> None:
        self.config = config or NearFieldDockingConfig()

    def desired_pitch_at(
        self,
        aim_xyz: np.ndarray,
        robot_xyz: np.ndarray,
        camera_z: float = 0.88,
    ) -> Tuple[int, Dict[str, Any]]:
        """Look-down steps needed to keep a low aim point inside the FOV.

        Triggered when the 2D distance <= pitch_trigger_dist, the aim is below the
        camera line, and the required tilt exceeds pitch_min_angle_deg (jitter guard).

        Returns:
            down_steps: look_down steps (0 = none)
            diag: diagnostics (distance / required angle / height drop)
        """
        if not self.config.pitch_enable:
            return 0, {"pitch_enabled": False}
        
        aim = np.asarray(aim_xyz, dtype=np.float64).reshape(-1)
        robot = np.asarray(robot_xyz, dtype=np.float64).reshape(-1)
        if aim.shape[0] < 3 or robot.shape[0] < 2:
            return 0, {"reason": "bad_input"}
        
        dist2 = float(np.linalg.norm(aim[:2] - robot[:2]))
        if dist2 > float(self.config.pitch_trigger_dist):
            return 0, {"reason": "too_far", "dist": round(dist2, 3)}
        
        cam_z_abs = float(robot[2]) + float(camera_z) if robot.shape[0] >= 3 else float(camera_z)
        aim_z = float(aim[2]) + float(self.config.pitch_aim_z_bias)
        drop = max(0.0, cam_z_abs - aim_z)
        if drop <= 1e-6:
            return 0, {"reason": "aim_above_camera", "dist": round(dist2, 3)}
        
        angle_deg = float(np.degrees(np.arctan2(drop, max(dist2, 1e-3))))
        if angle_deg < float(self.config.pitch_min_angle_deg):
            return 0, {"reason": "below_min_angle", "angle": round(angle_deg, 2)}
        
        steps = int(np.ceil(angle_deg / float(self.config.pitch_step_deg)))
        steps = max(1, min(int(self.config.pitch_max_down_steps), steps))

        return steps, {"dist": round(dist2, 3), "angle_deg": round(angle_deg, 2),
                       "steps": steps, "drop": round(drop, 3)}
    