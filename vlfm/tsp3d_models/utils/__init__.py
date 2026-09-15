"""TSP3D input adaptation utilities (VLVM 4.3c).

Modules:
  world_map      : world-frame local map accumulation (the final fusion route)
  sampling       : send-side post-processing (voxelization / distance sampling / cap)
  pipeline       : unified TSP3DInputPreprocessor (world route)
  s_penalty      : semantic-field cross-validation (c' = c * w_S)
  target_geometric_gating : occ-consistency / density admission gates
  target_memory_manager   : target lifecycle (merge / fallback)
"""

from .world_map import WorldLocalMap
from .pipeline import TSP3DInputPreprocessor
from .sampling import cap_point_count, distance_adaptive_sample, voxelize_world

__all__ = [
    "WorldLocalMap",
    "TSP3DInputPreprocessor",
    "voxelize_world",
    "distance_adaptive_sample",
    "cap_point_count",
]
