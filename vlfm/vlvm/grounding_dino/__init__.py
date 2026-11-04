"""Grounding-DINO proposal source used by the online GD calibration path.

Current runtime path::

    TSP3DObjectNavPolicy
      -> GdCalibrationLogger (calibration.py)
      -> GdProposer (proposer.py)
      -> GdMechanism (mechanism.py)

``calibration.py`` owns the per-frame GD call and candidate/track association;
``mechanism.py`` consumes its JSON-safe record for track-only gd-tsp routing and
the gd-only fallback pool.  The older candidate-level vote model
(``provenance.py`` and ``Tsp3dCandidate``) has been removed.

Active modules
--------------
``config.py``    ``GdProposalConfig``
``types.py``     ``CameraView`` / ``GdProposal`` / ``ProposalStats``
``vocab.py``     class text alignment and caption builders
``client.py``    guarded GD server access (failure = no information)
``region.py``    2D box -> world point cluster
``boxes.py``     3D AABB geometry
``proposer.py``  ``GdProposer`` / ``should_run``
``calibration.py`` ``GdCalibrationLogger`` / ``GdCalibConfig``
``mechanism.py`` ``GdMechanism`` / ``GdMechanismConfig``
``gd_offline_cal.py`` offline replay of the calibration logs
"""

from .boxes import aabb_center, aabb_corners, containment, iau_3d, nearest_point
from .client import GdDetector
from .config import GdProposalConfig
from .proposer import GdProposer, should_run
from .region import (
    backproject_mask,
    depth_holes_to_far,
    erode_mask,
    largest_cluster,
    rect_mask,
    region_points_world,
    too_offset_rect,
)
from .types import CameraView, GdProposal, MatchResult, ProposalStats
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
    "GdProposalConfig",
    "CameraView",
    "GdProposal",
    "ProposalStats",
    "MatchResult",
    "DEFAULT_VOCAB",
    "DEFAULT_ALIASES",
    "normalize_phrase",
    "singularize",
    "build_alias_map",
    "phrase_classes",
    "match_phrase",
    "build_caption",
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
    "GdProposer",
    "should_run",
]
