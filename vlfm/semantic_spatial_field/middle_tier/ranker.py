"""Policy-safe middle-tier probability scoring.

The ranker loads an offline-fitted JSON model and only computes scores.  It
does not import the offline fitter and never changes navigation state.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Mapping, Optional

from .features import FEATURE_GROUPS, MIDDLE_TIER_TYPES

MODEL_FORMAT = "vlvm_middle_tier_ranker_v1"
_EPS = 1e-6


def sigmoid(value: float) -> float:
    value = float(max(-35.0, min(35.0, value)))
    return 1.0 / (1.0 + math.exp(-value))


def logit(probability: float) -> float:
    probability = float(max(_EPS, min(1.0 - _EPS, probability)))
    return math.log(probability / (1.0 - probability))


def _feature_value(features: Mapping[str, Any], name: str) -> Optional[float]:
    if name.startswith("__missing__."):
        base = name[len("__missing__.") :]
        return 0.0 if _feature_value(features, base) is not None else 1.0
    if "." not in name:
        return None
    group, key = name.split(".", 1)
    value = (features.get(group) or {}).get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def submodel_predict(
    submodel: Mapping[str, Any], features: Mapping[str, Any]
) -> float:
    names = list(submodel.get("features") or [])
    means = list(submodel.get("mean") or [])
    scales = list(submodel.get("scale") or [])
    coefficients = list(submodel.get("coef") or [])
    if not names:
        return 0.5
    z = float(submodel.get("intercept", 0.0) or 0.0)
    for index, name in enumerate(names):
        value = _feature_value(features, name)
        mean = float(means[index]) if index < len(means) else 0.0
        if value is None:
            value = mean
        scale = float(scales[index]) if index < len(scales) else 1.0
        z += float(coefficients[index] if index < len(coefficients) else 0.0) * (
            (value - mean) / scale if scale else 0.0
        )
    return sigmoid(z)


class MiddleTierRanker:
    """Load a fitted model and score one shadow row."""

    def __init__(self, model_path: Optional[str] = None, enabled: bool = True) -> None:
        self.model_path = None if model_path in (None, "", "null") else str(model_path)
        self.model: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.version: Optional[str] = None
        if not enabled:
            self.error = "disabled"
            return
        if not self.model_path:
            return
        if not os.path.exists(self.model_path):
            self.error = f"model_missing:{self.model_path}"
            return
        try:
            with open(self.model_path, "r", encoding="utf-8") as handle:
                model = json.load(handle)
            if str(model.get("format") or "") != MODEL_FORMAT:
                raise ValueError(f"unsupported format: {model.get('format')}")
            self.model = model
            metadata = model.get("metadata") or {}
            self.version = str(metadata.get("version") or "")
        except Exception as exc:  # noqa: BLE001 - shadow must not kill nav
            self.error = f"{type(exc).__name__}: {exc}"
            self.model = None

    @property
    def available(self) -> bool:
        return bool(self.model)

    def _empty_scores(self) -> Dict[str, Any]:
        return {
            "s_cross": None,
            "s_repeat": None,
            "s_geo": None,
            "s_ctx": None,
            "p": None,
            "model_version": self.version,
        }

    def score(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            return self._empty_scores()
        type_name = str(row.get("type") or "")
        if type_name not in MIDDLE_TIER_TYPES:
            return self._empty_scores()
        type_model = (self.model.get("types") or {}).get(type_name)
        if not type_model:
            return self._empty_scores()
        features = row.get("features") or {}
        groups = type_model.get("groups") or {}
        sub_scores: Dict[str, float] = {}
        sub_logits: Dict[str, float] = {}
        for group in FEATURE_GROUPS:
            submodel = groups.get(group) or {}
            probability = submodel_predict(submodel, features)
            sub_scores[group] = probability
            sub_logits[group] = logit(probability)
        combine = type_model.get("combine") or {}
        coefficients = combine.get("coef") or {}
        z = float(combine.get("intercept", 0.0) or 0.0)
        for group in FEATURE_GROUPS:
            z += float(coefficients.get(group, 0.0) or 0.0) * sub_logits[group]
        probability = sigmoid(z)
        return {
            "s_cross": sub_scores["cross"],
            "s_repeat": sub_scores["repeat"],
            "s_geo": sub_scores["geo"],
            "s_ctx": sub_scores["ctx"],
            "p": probability,
            "model_version": self.version,
        }
