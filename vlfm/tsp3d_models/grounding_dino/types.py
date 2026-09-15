"""Data carriers of the GD proposal source (pure data, no behaviour)."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CameraView:
    """Camera intrinsics + pose of one observation frame (base = forward, left, up)."""

    fx: float
    fy: float
    height: int
    width: int
    tf_camera_to_episodic: np.ndarray
    min_depth: float = 0.5
    max_depth: float = 5.0


@dataclass
class MatchResult:
    """T: class-level alignment of one GD phrase against the target classes."""

    canon: Optional[str] = None       # canonical class name (None = ambiguous/unknown)
    strength: float = 0.0             # partial-match grade in [0, 1]
    is_target: bool = False           # strict single-class match of the target class
    n_named: int = 0                  # how many classes the phrase names (>=2 = ambiguous)


@dataclass
class Tsp3dCandidate:
    """Minimal view of a TSP3D candidate, used only for the candidate-level merge."""

    centroid: np.ndarray
    box8: Optional[np.ndarray] = None
    conf: float = 0.0
    admitted: bool = True             # survived S-penalty (admitted = "the TSP3D vote")


@dataclass
class GdProposal:
    """One GD-only 3D candidate of the current frame (single vote until merged)."""

    box_px: np.ndarray                # 2D box in pixels (x1, y1, x2, y2)
    conf: float                       # GD logits (already >= `box_thr`)
    phrase: str
    canon: Optional[str]
    points_world: np.ndarray          # largest-cluster points, world frame (N, 3)
    centroid: np.ndarray              # AABB centre (world)
    surface: np.ndarray               # cluster point closest to the robot (world)
    box8: np.ndarray                  # AABB corners (8, 3), world
    n_points: int
    suspect: bool = False             # `too_offset` -> admit as suspicious
    diag: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposalStats:
    """Per-episode counters (coverage / cost / rejection reasons)."""

    frames: int = 0                   # frames on which GD actually ran
    boxes_above_thr: int = 0          # GD boxes above `box_thr` AND target-class matched
    proposals: int = 0                # boxes that survived the full 2D -> 3D chain
    rejects: Dict[str, int] = field(default_factory=dict)
    # A4 measurement #6 (§3.7.10): phrase distribution of the boxes ABOVE `box_thr`
    # (before the strict single-class rule), so multi-class naming can be quantified.
    phrases: Dict[str, int] = field(default_factory=dict)
    multi_named: int = 0              # boxes whose phrase names >= 2 classes
    empty_phrase: int = 0             # boxes with an empty phrase
    target_hit: int = 0               # boxes passing the strict single-class rule

    def bump(self, key: str) -> None:
        self.rejects[key] = self.rejects.get(key, 0) + 1

    def bump_phrase(self, phrase: str, n_named: int, is_target: bool, empty: bool) -> None:
        """Count one above-threshold box's phrase and its naming shape."""
        key = str(phrase).strip().lower() or "<empty>"
        self.phrases[key] = self.phrases.get(key, 0) + 1
        if int(n_named) >= 2:
            self.multi_named += 1
        if empty:
            self.empty_phrase += 1
        if is_target:
            self.target_hit += 1

    def top_phrases(self, k: int = 20) -> Dict[str, int]:
        """Most frequent phrases (keeps the per-episode log line bounded)."""
        return dict(sorted(self.phrases.items(), key=lambda kv: -kv[1])[: max(1, int(k))])

    def reset(self) -> None:
        self.frames = 0
        self.boxes_above_thr = 0
        self.proposals = 0
        self.rejects = {}
        self.phrases = {}
        self.multi_named = 0
        self.empty_phrase = 0
        self.target_hit = 0
