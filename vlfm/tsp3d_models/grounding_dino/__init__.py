"""Grounding-DINO proposal source.

Role
----
GD is a *second, independent proposer*: every `every_n` frames it consumes the raw
observation frame (`rgb` + target text) and proposes 3D candidates on its own. 
The two models stay **frame-aligned**: GD reads the same frame the pre-TSP3D fusion
chain just consumed, but its point set is never injected into that chain and its
boxes are never sent to TSP3D as a re-query.

Vote model
----------
A frame's candidates are `TSP3D boxes ∪ GD proposals`:
  * `BOTH`  — a GD proposal co-locates with an admitted TSP3D candidate -> strong;
  * `GD`    — GD alone -> written as suspicious (single vote);
  * `TSP3D` — TSP3D alone -> unchanged existing behaviour.

Module map (no policy / habitat imports at any level)
----------------------------------------------------
  `config.py`     `GdProposalConfig` — every knob of this mechanism
  `types.py`      dataclasses (`CameraView`, `GdProposal`, `Tsp3dCandidate`, ...)
  `vocab.py`      class-level text alignment + caption builders (T1/T2)
  `client.py`     guarded GD server access (failure = "no info", never negative)
  `region.py`     2D box -> 3D cluster = VLFM `_extract_object_cloud` chain
  `boxes.py`      3D AABB geometry (corners / IaU / containment / nearest point)
  `proposer.py`   `GdProposer` — frame -> `List[GdProposal]` (+ stats)
  `provenance.py` candidate-level vote merge + single-vote admission/lock rules
"""

from .boxes import aabb_center, aabb_corners, containment, iau_3d, nearest_point
from .client import GdDetector
from .config import GdProposalConfig
from .proposer import GdProposer, should_run
from .provenance import (
    BOTH,
    GD,
    TSP3D,
    attach_votes,
    admit_as_suspicious,
    is_lockable,
    merge_vote,
)
from .region import (
    backproject_mask,
    depth_holes_to_far,
    erode_mask,
    largest_cluster,
    rect_mask,
    region_points_world,
    too_offset_rect,
)
from .types import CameraView, GdProposal, MatchResult, ProposalStats, Tsp3dCandidate
from .vocab import (
    DEFAULT_ALIASES,
    DEFAULT_VOCAB,
    build_alias_map,
    build_caption,
    match_phrase,
    normalize_phrase,
    phrase_classes,
    singularize,
)

__all__ = [
    # config / types
    "GdProposalConfig",
    "CameraView",
    "GdProposal",
    "ProposalStats",
    "Tsp3dCandidate",
    "MatchResult",
    # vocab (T)
    "DEFAULT_VOCAB",
    "DEFAULT_ALIASES",
    "normalize_phrase",
    "singularize",
    "build_alias_map",
    "phrase_classes",
    "match_phrase",
    "build_caption",
    # client / region / boxes
    "GdDetector",
    "rect_mask",
    "erode_mask",
    "depth_holes_to_far",
    "backproject_mask",
    "largest_cluster",
    "region_points_world",
    "too_offset_rect",
    "aabb_corners",
    "aabb_center",
    "nearest_point",
    "iau_3d",
    "containment",
    # proposer / provenance
    "GdProposer",
    "should_run",
    "attach_votes",
    "merge_vote",
    "admit_as_suspicious",
    "is_lockable",
    "BOTH",
    "GD",
    "TSP3D",
]
