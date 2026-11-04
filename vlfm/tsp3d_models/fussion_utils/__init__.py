"""TSP3D input adaptation utilities.

Modules:
  world_map      : world-frame local map accumulation (world route)
  sampling       : send-side post-processing (voxelization / distance sampling / cap)
  pipeline       : unified TSP3DInputPreprocessor (world / none)
"""

from .pipeline import TSP3DInputPreprocessor
from .sampling import cap_point_count, distance_adaptive_sample, voxelize_world
from .world_map import WorldLocalMap

__all__ = [
    "WorldLocalMap",
    "TSP3DInputPreprocessor",
    "voxelize_world",
    "distance_adaptive_sample",
    "cap_point_count",
]
