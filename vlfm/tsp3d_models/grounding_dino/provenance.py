"""Two-vote provenance: candidate-level merge + single-vote admission / lock rules.

A frame's candidate set is `TSP3D boxes ∪ GD proposals`. The rules below are pure
functions so the policy can consume them without importing any of its own state:

  * `attach_votes`        — a GD proposal that co-locates with an **admitted** TSP3D
                            candidate becomes `BOTH`, otherwise it stays `GD`;
  * `admit_as_suspicious` — single-vote GD entries are written as suspicious
                            (pending/soft), i.e. the existing lifecycle may delete
                            them again;
  * `is_lockable`         — with `weak_guard` a single-vote GD entry never triggers
                            the explore -> navigate switch (accuracy-first default).
"""

from typing import List, Optional, Sequence, Tuple

from .boxes import same_object
from .config import GdProposalConfig
from .types import GdProposal, Tsp3dCandidate

TSP3D = "tsp3d"
GD = "gd"
BOTH = "both"


def merge_vote(old: Optional[str], new: Optional[str]) -> str:
    """Vote of an entry seen by both models: any pair of different votes = `BOTH`."""
    a = TSP3D if old in (None, "") else str(old)
    b = TSP3D if new in (None, "") else str(new)
    if a == b:
        return a
    if BOTH in (a, b):
        return BOTH
    return BOTH


def attach_votes(
    proposals: Sequence[GdProposal],
    tsp3d_candidates: Sequence[Tsp3dCandidate],
    cfg: Optional[GdProposalConfig] = None,
) -> List[Tuple[GdProposal, str, Optional[int]]]:
    """Vote of every GD proposal against this frame's TSP3D candidates.

    Returns `(proposal, vote, matched_index)`; only candidates with
    `admitted=True` count as the TSP3D vote (a box rejected by the admission gates
    must not make a GD proposal look strong).
    """
    c = cfg or GdProposalConfig()
    valid = [k for k, cand in enumerate(tsp3d_candidates) if bool(getattr(cand, "admitted", True))]
    out: List[Tuple[GdProposal, str, Optional[int]]] = []
    for prop in proposals:
        best: Optional[int] = None
        best_dist = float("inf")
        for k in valid:
            cand = tsp3d_candidates[k]
            box8 = getattr(cand, "box8", None)
            if not same_object(prop.box8, prop.centroid, box8, cand.centroid, c.merge_dist, c.merge_iau):
                continue
            d = float(((prop.centroid[:2] - cand.centroid[:2]) ** 2).sum() ** 0.5)
            if d < best_dist:
                best, best_dist = k, d
        out.append((prop, BOTH if best is not None else GD, best))
    return out


def admit_as_suspicious(vote: str) -> bool:
    """Single-vote GD proposals enter memory as suspicious; `BOTH` entries do not."""
    return str(vote) != BOTH


def is_lockable(
    vote: str,
    num_obs: int,
    lock_min_obs: int,
    weak_guard: bool = True,
) -> bool:
    """Single-vote GD entries cannot be locked before enough evidence.

    The criterion is the raw observation count `num_obs` of the entry; `tsp3d` /
    `both` entries are never gated here.
    """
    if not weak_guard:
        return True
    if str(vote) == GD:
        return int(num_obs) >= max(1, int(lock_min_obs))
    return True
