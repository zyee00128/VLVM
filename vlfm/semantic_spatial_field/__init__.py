"""Offline semantic-spatial information field.

This package is deliberately independent of policy code.  It provides pure
computations and an optional episode-data archive for offline analysis.
"""

from .core import (
    FieldResult,
    FieldSnapshot,
    action_information_value,
    bayesian_update,
    binary_inspection_likelihoods,
    compose_region_features,
    compute_field_snapshot,
    entropy_bits,
    expected_information_gain,
    fit_logistic_weights,
    observation_rates_beta,
    rank_inspection_views,
    region_posterior,
)
from .inference import (
    CandidateFieldScore,
    FieldInference,
    FieldInferenceConfig,
    InspectionView,
    field_config_from_mapping,
    observation_likelihoods,
)
from .observation_logger import FieldObservationLogger
from .shadow import FieldExplorationShadow, FieldShadowConfig

__all__ = [
    "CandidateFieldScore",
    "FieldInference",
    "FieldInferenceConfig",
    "FieldObservationLogger",
    "FieldExplorationShadow",
    "FieldShadowConfig",
    "FieldResult",
    "FieldSnapshot",
    "InspectionView",
    "field_config_from_mapping",
    "observation_likelihoods",
    "action_information_value",
    "bayesian_update",
    "binary_inspection_likelihoods",
    "compose_region_features",
    "compute_field_snapshot",
    "entropy_bits",
    "expected_information_gain",
    "fit_logistic_weights",
    "observation_rates_beta",
    "rank_inspection_views",
    "region_posterior",
]
