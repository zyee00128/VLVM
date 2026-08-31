"""TSP3D input adaptation utilities (VLVM 4.3c).

Modules:
  sliding_window : camera-canonical N-frame sliding-window fusion (baseline)
  local_map      : world-frame local map accumulation (temporal fusion upgrade)
  sampling       : send-side post-processing (voxelization / distance sampling / cap)
  pipeline       : unified TSP3DInputPreprocessor (camera / world two routes)
"""

from .local_map import WorldLocalMap
from .pipeline import TSP3DInputPreprocessor
from .sampling import cap_point_count, distance_adaptive_sample, voxelize_world
from .sliding_window import TemporalPcdWindow
from .vqa_confirmation import vqa_confirm_detections

__all__ = [
    "WorldLocalMap",
    "TSP3DInputPreprocessor",
    "TemporalPcdWindow",
    "voxelize_world",
    "distance_adaptive_sample",
    "cap_point_count",
    "vqa_confirm_detections",
]
