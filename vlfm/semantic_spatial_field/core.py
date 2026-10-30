"""Pure NumPy primitives for the target-conditioned information field.

No model inference or policy action selection happens here.  Inputs are
precomputed region features and observation likelihoods; callers can run this
module offline and inspect every intermediate quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def _probability_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be a non-empty finite non-negative vector")
    total = float(result.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive sum")
    return result / total


def entropy_bits(probabilities: Sequence[float], *, epsilon: float = 1e-12) -> float:
    """Categorical entropy in bits; zero-probability categories contribute 0."""
    p = _probability_vector(probabilities, name="probabilities")
    p = p[p > epsilon]
    return float(-np.sum(p * np.log2(p)))


def region_posterior(
    region_features: np.ndarray,
    weights: Sequence[float],
    *,
    unknown_logit: float = 0.0,
    region_bias: float = 0.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Softmax region logits plus an explicit unknown-target hypothesis.

    ``region_features`` has shape ``(n_regions, n_features)`` and the weights
    have shape ``(n_features,)``. The unknown hypothesis is appended last.
    ``region_bias`` is the target-class prior/intercept shared by known
    regions; pass the intercept returned by ``fit_logistic_weights`` here.
    """
    features = np.asarray(region_features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[1] != w.size:
        raise ValueError("region_features must be 2D and match weights")
    if features.shape[0] == 0 or not np.all(np.isfinite(features)):
        raise ValueError("region_features must contain finite region rows")
    if (
        not np.isfinite(unknown_logit)
        or not np.isfinite(region_bias)
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            "unknown_logit/region_bias must be finite and temperature positive"
        )
    logits = np.concatenate(
        (features @ w + float(region_bias), [float(unknown_logit)])
    ) / temperature
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def compose_region_features(
    context_scores: np.ndarray,
    context_reliability: np.ndarray,
    relation_scores: Optional[np.ndarray] = None,
    geometry_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build one transparent feature row per candidate region.

    Context scores are ``(regions, contexts)`` and reliability is either a
    matching matrix or a context vector. The first output feature is the
    reliability-weighted context score; the second is its weighted coverage.
    Optional relation features are averaged across their last axis, and
    geometry features are appended unchanged for offline weight calibration.
    """
    scores = np.asarray(context_scores, dtype=np.float64)
    reliability = np.asarray(context_reliability, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("context_scores must be a finite (regions, contexts) matrix")
    if reliability.ndim == 1:
        if reliability.size != scores.shape[1]:
            raise ValueError("context reliability vector must match context count")
        reliability = np.broadcast_to(reliability[None, :], scores.shape)
    if reliability.shape != scores.shape or not np.all(np.isfinite(reliability)):
        raise ValueError("context_reliability must match context_scores")
    reliability = np.clip(reliability, 0.0, 1.0)
    denom = reliability.sum(axis=1)
    weighted_context = np.divide(
        (scores * reliability).sum(axis=1), denom,
        out=np.zeros(scores.shape[0], dtype=np.float64), where=denom > 0,
    )
    coverage = denom / max(1, scores.shape[1])
    blocks = [weighted_context[:, None], coverage[:, None]]
    if relation_scores is not None:
        relation = np.asarray(relation_scores, dtype=np.float64)
        if relation.ndim == 2:
            relation = relation[:, :, None]
        if relation.ndim != 3 or relation.shape[0] != scores.shape[0] or not np.all(np.isfinite(relation)):
            raise ValueError("relation_scores must have shape (regions, relations[, channels])")
        blocks.append(relation.mean(axis=2))
    if geometry_features is not None:
        geometry = np.asarray(geometry_features, dtype=np.float64)
        if geometry.ndim == 1:
            geometry = geometry[:, None]
        if geometry.ndim != 2 or geometry.shape[0] != scores.shape[0] or not np.all(np.isfinite(geometry)):
            raise ValueError("geometry_features must have one finite row per region")
        blocks.append(geometry)
    return np.concatenate(blocks, axis=1)


def bayesian_update(
    prior: Sequence[float],
    likelihoods: np.ndarray,
    outcome: int,
) -> np.ndarray:
    """Update hypotheses for one observed outcome.

    ``likelihoods[h, o] = P(outcome=o | hypothesis=h)``. Each row is
    normalized defensively before applying Bayes' rule.
    """
    p = _probability_vector(prior, name="prior")
    likelihood = np.asarray(likelihoods, dtype=np.float64)
    if likelihood.ndim != 2 or likelihood.shape[0] != p.size:
        raise ValueError("likelihoods must have one row per hypothesis")
    if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0):
        raise ValueError("likelihoods must be finite and non-negative")
    if outcome < 0 or outcome >= likelihood.shape[1]:
        raise ValueError("outcome index is outside likelihood columns")
    row_sums = likelihood.sum(axis=1, keepdims=True)
    likelihood = np.divide(likelihood, row_sums, out=np.zeros_like(likelihood), where=row_sums > 0)
    unnormalized = p * likelihood[:, outcome]
    total = float(unnormalized.sum())
    if total <= 0:
        return p.copy()
    return unnormalized / total


def expected_information_gain(
    prior: Sequence[float], likelihoods: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Return ``H_before, E[H_after], max(0, deltaH), P(outcome)`` in bits."""
    p = _probability_vector(prior, name="prior")
    likelihood = np.asarray(likelihoods, dtype=np.float64)
    if likelihood.ndim != 2 or likelihood.shape[0] != p.size:
        raise ValueError("likelihoods must have one row per hypothesis")
    if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0):
        raise ValueError("likelihoods must be finite and non-negative")
    row_sums = likelihood.sum(axis=1, keepdims=True)
    likelihood = np.divide(likelihood, row_sums, out=np.zeros_like(likelihood), where=row_sums > 0)
    outcome_probability = p @ likelihood
    before = entropy_bits(p)
    expected_after = 0.0
    for outcome, p_outcome in enumerate(outcome_probability):
        if p_outcome > 0:
            posterior = bayesian_update(p, likelihood, outcome)
            expected_after += float(p_outcome) * entropy_bits(posterior)
    gain = max(0.0, before - expected_after)
    return before, float(expected_after), float(gain), outcome_probability


def binary_inspection_likelihoods(
    hypothesis_count: int,
    target_hypothesis_index: int,
    *,
    hit_probability: float,
    false_alarm_probability: float,
    ambiguous_probability: float = 0.0,
) -> np.ndarray:
    """Create ``[positive, negative, ambiguous]`` likelihood rows for a view.

    The selected region hypothesis receives the calibrated hit rate. Other
    region hypotheses and the final unknown-target hypothesis receive the
    calibrated false-alarm rate. Ambiguous observations are represented
    explicitly rather than silently counted as misses.
    """
    if hypothesis_count < 2 or not 0 <= target_hypothesis_index < hypothesis_count - 1:
        raise ValueError("hypothesis_count must include unknown and index a known region")
    values = np.asarray(
        [hit_probability, false_alarm_probability, ambiguous_probability], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("observation probabilities must be in [0, 1]")
    if hit_probability + ambiguous_probability > 1 or false_alarm_probability + ambiguous_probability > 1:
        raise ValueError("positive and ambiguous outcome probabilities cannot exceed one")
    likelihoods = np.empty((hypothesis_count, 3), dtype=np.float64)
    likelihoods[:, 0] = false_alarm_probability
    likelihoods[:, 1] = 1.0 - false_alarm_probability - ambiguous_probability
    likelihoods[:, 2] = ambiguous_probability
    likelihoods[target_hypothesis_index, 0] = hit_probability
    likelihoods[target_hypothesis_index, 1] = 1.0 - hit_probability - ambiguous_probability
    return likelihoods


def action_information_value(
    prior: Sequence[float],
    likelihoods: np.ndarray,
    route_cost: float,
    *,
    epsilon: float = 1e-6,
) -> "FieldResult":
    """Score an inspection action as expected entropy reduction / route cost."""
    before, after, gain, outcome_probability = expected_information_gain(prior, likelihoods)
    valid = bool(np.isfinite(route_cost) and route_cost >= 0)
    value = gain / (float(route_cost) + epsilon) if valid else 0.0
    return FieldResult(
        entropy_before_bits=before,
        expected_entropy_after_bits=after,
        expected_entropy_reduction_bits=gain,
        route_cost=float(route_cost),
        information_value=float(value),
        outcome_probabilities=tuple(float(x) for x in outcome_probability),
        valid=valid,
    )


def rank_inspection_views(
    prior: Sequence[float],
    view_likelihoods: Sequence[np.ndarray],
    route_costs: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> Tuple[Optional[int], Tuple[FieldResult, ...]]:
    """Rank offline candidate viewpoints by expected information per cost."""
    if len(view_likelihoods) != len(route_costs):
        raise ValueError("each candidate view needs one route cost")
    results = tuple(
        action_information_value(prior, likelihood, cost, epsilon=epsilon)
        for likelihood, cost in zip(view_likelihoods, route_costs)
    )
    valid_indices = [i for i, result in enumerate(results) if result.valid]
    best = max(valid_indices, key=lambda i: results[i].information_value) if valid_indices else None
    return best, results


@dataclass(frozen=True)
class FieldSnapshot:
    """One target-conditioned belief and its scored inspection viewpoints."""

    region_probabilities: Tuple[float, ...]
    entropy_bits: float
    best_view_index: Optional[int]
    view_results: Tuple[FieldResult, ...]


def compute_field_snapshot(
    region_features: np.ndarray,
    semantic_spatial_weights: Sequence[float],
    view_likelihoods: Sequence[np.ndarray],
    route_costs: Sequence[float],
    *,
    unknown_logit: float = 0.0,
    region_bias: float = 0.0,
    temperature: float = 1.0,
    epsilon: float = 1e-6,
) -> FieldSnapshot:
    """Compute ``B_t``, ``H``, ``DeltaH``, ``C`` and ``F`` for supplied views.

    All visual-language inference, region generation, and route-cost
    estimation are inputs computed by the caller. This is suitable for the
    proposed offline side-channel and has no policy dependency.
    """
    posterior = region_posterior(
        region_features,
        semantic_spatial_weights,
        unknown_logit=unknown_logit,
        region_bias=region_bias,
        temperature=temperature,
    )
    best, results = rank_inspection_views(
        posterior, view_likelihoods, route_costs, epsilon=epsilon
    )
    return FieldSnapshot(
        region_probabilities=tuple(float(value) for value in posterior),
        entropy_bits=entropy_bits(posterior),
        best_view_index=best,
        view_results=results,
    )


@dataclass(frozen=True)
class FieldResult:
    """Transparent per-viewpoint information-field outputs."""

    entropy_before_bits: float
    expected_entropy_after_bits: float
    expected_entropy_reduction_bits: float
    route_cost: float
    information_value: float
    outcome_probabilities: Tuple[float, ...]
    valid: bool


def observation_rates_beta(
    positives: Sequence[int],
    trials: Sequence[int],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Beta-smoothed per-stratum probability of a positive observation."""
    pos = np.asarray(positives, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    if pos.shape != n.shape or np.any(pos < 0) or np.any(n < pos):
        raise ValueError("positives/trials must match and satisfy 0 <= positives <= trials")
    if alpha <= 0 or beta <= 0 or not np.isfinite(alpha + beta):
        raise ValueError("Beta prior parameters must be positive and finite")
    return (pos + alpha) / (n + alpha + beta)


def fit_logistic_weights(
    features: np.ndarray,
    labels: Sequence[int],
    *,
    l2: float = 1.0,
    sample_weights: Optional[Sequence[float]] = None,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """Fit small offline logistic calibration weights with damped Newton steps.

    Returns ``(weights, intercept)``. This is scalar calibration over supplied
    features, not neural-network training. The intercept is not L2 penalized.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D matrix aligned with labels")
    if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("features must be finite and labels must be binary")
    if l2 < 0 or max_iter <= 0:
        raise ValueError("l2 must be non-negative and max_iter positive")
    sw = np.ones(y.size, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if sw.size != y.size or not np.all(np.isfinite(sw)) or np.any(sw < 0) or sw.sum() <= 0:
        raise ValueError("sample_weights must be finite, non-negative, and aligned")
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    theta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(theta.size, dtype=np.float64) * float(l2)
    penalty[-1, -1] = 0.0
    for _ in range(max_iter):
        logits = np.clip(design @ theta, -35.0, 35.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = design.T @ (sw * (probs - y)) + penalty @ theta
        curvature = sw * probs * (1.0 - probs)
        hessian = (design.T * curvature) @ design + penalty
        try:
            step = np.linalg.solve(hessian + np.eye(theta.size) * 1e-10, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        theta_next = theta - step
        if np.linalg.norm(theta_next - theta) <= tolerance * (1.0 + np.linalg.norm(theta)):
            theta = theta_next
            break
        theta = theta_next
    return theta[:-1], float(theta[-1])
