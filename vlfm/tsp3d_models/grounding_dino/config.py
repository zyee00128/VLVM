"""Knobs of the GD proposal source (single place, no logic)."""

from dataclasses import dataclass


@dataclass
class GdProposalConfig:
    """Every parameter of the mechanism, grouped by the stage it belongs to."""

    # ---- frame alignment ----
    enabled: bool = True
    every_n: int = 1                  # GD runs on every N-th frame (1 = every frame)
    
    # ---- GD detection side ----
    port: int = 12181
    box_thr: float = 0.4              # VLFM non-COCO logits threshold (GD scale)
    caption_style: str = "vocab"      # "vocab" = class list caption / "target" = target classes only
    max_boxes: int = 2                # proposals built per frame (cost guard)

    # ---- 2D box -> 3D region (VLFM `_extract_object_cloud` chain) ----
    erode_iters: int = 5              # VLFM `object_map_erosion_size`
    min_pixels: int = 256             # eroded rectangle below this is too small to land
    rand_subarray: int = 5000         # VLFM random subsample size
    dbscan_eps: float = 0.2           # VLFM DBSCAN eps (m)
    dbscan_min_points: int = 100      # VLFM DBSCAN min_points
    min_dist: float = 1.0             # VLFM near rejection (closest cluster point to camera)

    # ---- candidate-level two-vote merge (rules used by the caller) ----
    merge_dist: float = 0.5           # centroid distance (m) that counts as "same object"
    merge_iau: float = 0.3            # AABB intersection-over-union that counts as overlap

    # ---- single-vote policy (accuracy-first defaults) ----
    weak_guard: bool = True           # single-vote (`GD`) proposals are not lockable
