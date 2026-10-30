"""Policy-independent inference for the target-conditioned field.

This module deliberately has no import from ``vlfm.policy`` and never chooses a
navigation action.  Callers supply region features, candidate regions, view
geometry/visibility, and route costs; the module returns posteriors and
information-field scores that can be logged, replayed, or tested offline.

The companion ``offline_data`` / ``offline_analysis`` modules build these
inputs from a field-observation archive.  Keeping the inference layer separate
makes it possible to validate the field without touching the policy or TSP3D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .core import (
    FieldResult,
    FieldSnapshot,
    bayesian_update,
    compute_field_snapshot,
    rank_inspection_views,
    region_posterior,
)


# ---------------------------------------------------------------------------
# observation model
# ---------------------------------------------------------------------------
def observation_likelihoods(
    hypothesis_count: int,
    *,
    hit_probabilities: Optional[Sequence[float]] = None,
    false_alarm_probability: float = 0.10,
    ambiguous_probability: float = 0.0,
) -> np.ndarray:
    """Return a ``(hypotheses, 3)`` matrix for outcomes ``[+, -, ?]``.

    ``hit_probabilities[h]`` is ``P(o=+|h, a)``.  When omitted, every
    hypothesis uses ``false_alarm_probability``.  The negative outcome is
    ``1 - P(+) - P(?)``.  Rows are normalized defensively by the core update
    functions, so this function only validates the probability constraints.
    """
    if hypothesis_count < 1:
        raise ValueError("hypothesis_count must be positive")
    if not np.isfinite(false_alarm_probability) or not 0.0 <= false_alarm_probability <= 1.0:
        raise ValueError("false_alarm_probability must be in [0, 1]")
    if not np.isfinite(ambiguous_probability) or not 0.0 <= ambiguous_probability <= 1.0:
        raise ValueError("ambiguous_probability must be in [0, 1]")
    if hit_probabilities is None:
        positive = np.full(hypothesis_count, float(false_alarm_probability), dtype=np.float64)
    else:
        positive = np.asarray(hit_probabilities, dtype=np.float64).reshape(-1)
        if positive.size != hypothesis_count:
            raise ValueError("hit_probabilities must have one entry per hypothesis")
    if not np.all(np.isfinite(positive)) or np.any(positive < 0.0) or np.any(positive > 1.0):
        raise ValueError("hit_probabilities must be finite and in [0, 1]")

    ambiguous = np.full(hypothesis_count, float(ambiguous_probability), dtype=np.float64)
    if np.any(positive + ambiguous > 1.0 + 1e-12):
        raise ValueError("P(+) + P(?) cannot exceed one for any hypothesis")
    negative = np.clip(1.0 - positive - ambiguous, 0.0, 1.0)
    return np.stack((positive, negative, ambiguous), axis=1)


@dataclass(frozen=True)
class InspectionView:
    """One candidate observation action.

    ``target_region`` is the index of the known region that the action is
    designed to inspect.  For a wider action, provide explicit
    ``hit_probabilities`` over all known regions plus the unknown hypothesis.
    ``visibility`` is the geometric observability fraction ``nu(a, r)``;
    below ``min_visibility`` the action is treated as non-informative rather
    than as negative evidence.
    """

    view_id: str
    route_cost: float
    target_region: int
    visibility: float = 1.0
    hit_probability: float = 0.70
    false_alarm_probability: float = 0.10
    ambiguous_probability: float = 0.0
    hit_probabilities: Optional[Tuple[float, ...]] = None

    def likelihoods(
        self,
        hypothesis_count: int,
        *,
        min_visibility: float = 0.15,
    ) -> Tuple[np.ndarray, bool]:
        """Return ``(likelihoods, valid)`` for this view.

        Invalid/low-visibility views produce an all-ambiguous row, which has
        zero expected information gain and is excluded from route-cost ranking
        by the caller.
        """
        if hypothesis_count < 2:
            raise ValueError("hypothesis_count must include unknown")
        if self.target_region < 0 or self.target_region >= hypothesis_count - 1:
            raise ValueError("target_region must index a known region")
        visibility = float(self.visibility)
        valid = bool(np.isfinite(visibility) and visibility >= min_visibility)
        if not valid:
            return observation_likelihoods(
                hypothesis_count,
                hit_probabilities=np.zeros(hypothesis_count, dtype=np.float64),
                false_alarm_probability=0.0,
                ambiguous_probability=1.0,
            ), False

        if self.hit_probabilities is not None:
            positive = np.asarray(self.hit_probabilities, dtype=np.float64).reshape(-1)
            if positive.size != hypothesis_count:
                raise ValueError("hit_probabilities must include unknown hypothesis")
        else:
            positive = np.full(hypothesis_count, float(self.false_alarm_probability), dtype=np.float64)
            positive[self.target_region] = float(self.hit_probability)

        # A partially visible view can only produce a weaker positive signal;
        # the remaining probability mass is treated as ambiguous, never as a
        # confident negative observation.
        ambiguous = max(float(self.ambiguous_probability), 0.0)
        ambiguous = max(ambiguous, 1.0 - min(visibility, 1.0))
        positive = np.clip(positive * min(visibility, 1.0), 0.0, 1.0)
        ambiguous = np.clip(ambiguous, 0.0, 1.0)
        if np.any(positive + ambiguous > 1.0 + 1e-12):
            ambiguous = np.maximum(0.0, 1.0 - positive)
        return observation_likelihoods(
            hypothesis_count,
            hit_probabilities=positive,
            false_alarm_probability=float(self.false_alarm_probability),
            ambiguous_probability=float(ambiguous),
        ), True


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldInferenceConfig:
    """Calibrated parameters used by :class:`FieldInference`."""

    weights: Tuple[float, ...] = ()
    region_bias: float = 0.0
    unknown_logit: float = 0.0
    temperature: float = 1.0
    min_visibility: float = 0.15
    feature_names: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateFieldScore:
    """Field score attached to one candidate track/region."""

    candidate_id: str
    region_index: Optional[int]
    best_view_id: Optional[str]
    field_score: float
    entropy_bits: float
    expected_gain_bits: float
    route_cost: float
    valid: bool
    reason: str
    view_results: Tuple[FieldResult, ...] = ()


class FieldInference:
    """Compute posterior and information-field scores from supplied inputs."""

    def __init__(self, config: FieldInferenceConfig):
        if not np.isfinite(config.region_bias) or not np.isfinite(config.unknown_logit):
            raise ValueError("region_bias/unknown_logit must be finite")
        if not np.isfinite(config.temperature) or config.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= config.min_visibility <= 1.0:
            raise ValueError("min_visibility must be in [0, 1]")
        self.config = config

    def posterior(self, region_features: np.ndarray) -> np.ndarray:
        """Return ``[b(r_1), ..., b(r_R), b(unknown)]``."""
        features = np.asarray(region_features, dtype=np.float64)
        weights = np.asarray(self.config.weights, dtype=np.float64).reshape(-1)
        if weights.size == 0:
            # An unfitted geometry prior is still useful for smoke tests and
            # diagnostics, but callers should normally fit weights offline.
            weights = np.zeros(features.shape[1], dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != weights.size:
            raise ValueError("region_features must match configured weights")
        return region_posterior(
            features,
            weights,
            unknown_logit=self.config.unknown_logit,
            region_bias=self.config.region_bias,
            temperature=self.config.temperature,
        )

    def update(
        self,
        prior: Sequence[float],
        view: InspectionView,
        outcome: int,
    ) -> np.ndarray:
        """Apply one observed outcome to a prior and return the posterior.

        ``outcome`` uses the core ordering: 0=positive, 1=negative,
        2=ambiguous.  Low-visibility views return the prior unchanged, which
        prevents missing evidence from being treated as a negative detection.
        """
        likelihoods, valid = view.likelihoods(
            len(prior), min_visibility=self.config.min_visibility
        )
        if not valid:
            return np.asarray(prior, dtype=np.float64).reshape(-1).copy()
        return bayesian_update(prior, likelihoods, int(outcome))

    def snapshot(
        self,
        region_features: np.ndarray,
        views: Sequence[InspectionView],
    ) -> FieldSnapshot:
        """Score all views against the current region belief."""
        posterior = self.posterior(region_features)
        likelihoods = []
        costs = []
        for view in views:
            likelihood, valid = view.likelihoods(
                posterior.size, min_visibility=self.config.min_visibility
            )
            likelihoods.append(likelihood)
            costs.append(float(view.route_cost) if valid else float("inf"))
        return compute_field_snapshot(
            np.asarray(region_features, dtype=np.float64),
            self.config.weights if self.config.weights else np.zeros(
                np.asarray(region_features, dtype=np.float64).shape[1],
                dtype=np.float64,
            ),
            likelihoods,
            costs,
            unknown_logit=self.config.unknown_logit,
            region_bias=self.config.region_bias,
            temperature=self.config.temperature,
        )

    def score_candidates(
        self,
        region_features: np.ndarray,
        candidates: Sequence[Tuple[str, int, Sequence[InspectionView]]],
    ) -> Tuple[CandidateFieldScore, ...]:
        """Return ``F_cand(j) = max_a F(a)`` for each candidate.

        ``candidates`` contains ``(candidate_id, region_index, views)``.
        A candidate with no valid view is returned with ``valid=False`` and
        zero score rather than being assigned a misleading low confidence.
        """
        posterior = self.posterior(region_features)
        results = []
        for candidate_id, region_index, views in candidates:
            if region_index < 0 or region_index >= posterior.size - 1:
                results.append(
                    CandidateFieldScore(
                        candidate_id=str(candidate_id),
                        region_index=None,
                        best_view_id=None,
                        field_score=0.0,
                        entropy_bits=float("-inf"),
                        expected_gain_bits=0.0,
                        route_cost=float("inf"),
                        valid=False,
                        reason="invalid_region",
                    )
                )
                continue
            likelihoods = []
            costs = []
            valid_indices = []
            for i, view in enumerate(views):
                likelihood, valid = view.likelihoods(
                    posterior.size, min_visibility=self.config.min_visibility
                )
                likelihoods.append(likelihood)
                costs.append(float(view.route_cost) if valid else float("inf"))
                if valid:
                    valid_indices.append(i)
            if not valid_indices:
                results.append(
                    CandidateFieldScore(
                        candidate_id=str(candidate_id),
                        region_index=int(region_index),
                        best_view_id=None,
                        field_score=0.0,
                        entropy_bits=float("-inf"),
                        expected_gain_bits=0.0,
                        route_cost=float("inf"),
                        valid=False,
                        reason="no_valid_view",
                    )
                )
                continue
            best, view_results = rank_inspection_views(
                posterior, likelihoods, costs
            )
            if best is None:
                results.append(
                    CandidateFieldScore(
                        candidate_id=str(candidate_id),
                        region_index=int(region_index),
                        best_view_id=None,
                        field_score=0.0,
                        entropy_bits=float("-inf"),
                        expected_gain_bits=0.0,
                        route_cost=float("inf"),
                        valid=False,
                        reason="no_valid_view",
                        view_results=view_results,
                    )
                )
                continue
            best_result = view_results[best]
            results.append(
                CandidateFieldScore(
                    candidate_id=str(candidate_id),
                    region_index=int(region_index),
                    best_view_id=str(views[best].view_id),
                    field_score=float(best_result.information_value),
                    entropy_bits=float(best_result.entropy_before_bits),
                    expected_gain_bits=float(best_result.expected_entropy_reduction_bits),
                    route_cost=float(best_result.route_cost),
                    valid=bool(best_result.valid),
                    reason="ok" if best_result.valid else "invalid_route",
                    view_results=view_results,
                )
            )
        return tuple(results)


def field_config_from_mapping(mapping: Dict[str, object]) -> FieldInferenceConfig:
    """Build a config from the JSON produced by the offline analyzer."""
    return FieldInferenceConfig(
        weights=tuple(float(v) for v in mapping.get("weights", ())),  # type: ignore[arg-type]
        region_bias=float(mapping.get("region_bias", 0.0) or 0.0),
        unknown_logit=float(mapping.get("unknown_logit", 0.0) or 0.0),
        temperature=float(mapping.get("temperature", 1.0) or 1.0),
        min_visibility=float(mapping.get("min_visibility", 0.15) or 0.0),
        feature_names=tuple(str(v) for v in mapping.get("feature_names", ())),  # type: ignore[arg-type]
    )
