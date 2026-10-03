"""TSP3D input adaptation utilities (VLVM 4.3c).

Modules:
  sliding_window : camera-canonical N-frame sliding-window fusion (camera route)
  world_map      : world-frame local map accumulation (world / world+scan route)
  panoramic      : single-position 360° spatial fusion (panoramic route)
  sampling       : send-side post-processing (voxelization / distance sampling / cap)
  pipeline       : unified TSP3DInputPreprocessor (camera / world / panoramic / none)
  s_penalty      : semantic-field cross-validation (c' = c * w_S)
  target_geometric_gating : occupancy-consistency admission gate
  target_memory_manager   : target lifecycle (merge / fallback)
"""

from .world_map import WorldLocalMap
from .pipeline import TSP3DInputPreprocessor
from .sampling import cap_point_count, distance_adaptive_sample, voxelize_world
from .camera_sliding_window import TemporalPcdWindow

__all__ = [
    "WorldLocalMap",
    "TSP3DInputPreprocessor",
    "TemporalPcdWindow",
    "voxelize_world",
    "distance_adaptive_sample",
    "cap_point_count",
]
