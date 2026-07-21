from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.schemas import ModelPrediction
from labs.common.artifacts import get_lab_model_dir


PROFILE_FILENAME = "inference_profile.json"
PROFILE_VERSION = 1


@dataclass(frozen=True)
class ReliabilityAssessment:
    accepted: bool
    status: str
    warnings: tuple[str, ...]
    confidence: float
    probability_margin: float
    out_of_range_features: tuple[str, ...]
    max_robust_distance: float


def build_feature_profile(
    features: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    fence_multiplier: float = 1.5,
) -> dict[str, Any]:
    if features.empty:
        raise ValueError("Cannot build an inference profile from an empty dataset")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Invalid quantile range")
    if fence_multiplier <= 0.0:
        raise ValueError("fence_multiplier must be positive")

    feature_profiles: dict[str, dict[str, float]] = {}
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float))]
        if values.empty:
            raise ValueError(f"Feature '{column}' has no finite values")

        q_low = float(values.quantile(lower_quantile))
        q25 = float(values.quantile(0.25))
        median = float(values.quantile(0.50))
        q75 = float(values.quantile(0.75))
        q_high = float(values.quantile(upper_quantile))
        iqr = q75 - q25
        central_span = q_high - q_low
        scale = max(iqr, central_span / 4.0, abs(median) * 1e-6, 1e-9)

        feature_profiles[str(column)] = {
            "median": median,
            "q01": q_low,
            "q25": q25,
            "q75": q75,
            "q99": q_high,
            "scale": float(scale),
            "lower_bound": float(q_low - fence_multiplier * scale),
            "upper_bound": float(q_high + fence_multiplier * scale),
        }

    return {
        "version": PROFILE_VERSION,
        "n_samples": int(len(features)),
        "feature_names": [str(column) for column in features.columns],
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "fence_multiplier": float(fence_multiplier),
        "features": feature_profiles,
    }


def save_feature_profile(profile: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_feature_profile(lab_id: str) -> dict[str, Any]:
    path = get_lab_model_dir(lab_id) / PROFILE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Inference profile is missing for '{lab_id}'. "
            "Run: python build_inference_profiles.py"
        )
    profile = json.loads(path.read_text(encoding="utf-8"))
    if int(profile.get("version", -1)) != PROFILE_VERSION:
        raise ValueError(f"Unsupported inference profile version: {path}")
    return profile


def assess_reliability(
    feature_values: Mapping[str, float],
    probabilities: Sequence[float],
    profile: Mapping[str, Any],
    *,
    min_confidence: float = 0.55,
    min_probability_margin: float = 0.08,
    max_out_of_range_fraction: float = 0.15,
    severe_robust_distance: float = 4.0,
) -> ReliabilityAssessment:
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.ndim != 1 or probability_array.size < 2:
        raise ValueError("At least two class probabilities are required")
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("Class probabilities must be finite")

    sorted_probabilities = np.sort(probability_array)[::-1]
    confidence = float(sorted_probabilities[0])
    probability_margin = float(sorted_probabilities[0] - sorted_probabilities[1])

    expected_names = [str(name) for name in profile.get("feature_names", [])]
    missing = [name for name in expected_names if name not in feature_values]
    if missing:
        raise ValueError("Missing features for reliability check: " + ", ".join(missing))

    outside: list[str] = []
    severe: list[str] = []
    max_distance = 0.0
    profile_features = profile.get("features", {})

    for name in expected_names:
        value = float(feature_values[name])
        parameters = profile_features.get(name)
        if parameters is None:
            raise ValueError(f"Feature '{name}' is missing from the inference profile")

        lower_bound = float(parameters["lower_bound"])
        upper_bound = float(parameters["upper_bound"])
        scale = max(float(parameters["scale"]), 1e-9)

        if value < lower_bound:
            distance = (lower_bound - value) / scale
        elif value > upper_bound:
            distance = (value - upper_bound) / scale
        else:
            distance = 0.0

        max_distance = max(max_distance, float(distance))
        if distance > 0.0:
            outside.append(name)
        if distance >= severe_robust_distance:
            severe.append(name)

    allowed_outside = max(1, int(ceil(len(expected_names) * max_out_of_range_fraction)))
    warnings: list[str] = []

    if len(outside) > allowed_outside:
        warnings.append(
            "Слишком много признаков находятся за пределами диапазона, "
            "использованного при обучении модели."
        )
    if severe:
        displayed = ", ".join(severe[:5])
        warnings.append(
            "Обнаружены признаки с существенным выходом за обучающий диапазон: "
            f"{displayed}."
        )
    if confidence < min_confidence:
        warnings.append(
            "Модель не выделяет одну диагностическую гипотезу с достаточной уверенностью."
        )
    if probability_margin < min_probability_margin:
        warnings.append(
            "Две наиболее вероятные диагностические гипотезы имеют близкие оценки."
        )

    distribution_issue = len(outside) > allowed_outside or bool(severe)
    ambiguity_issue = (
        confidence < min_confidence
        or probability_margin < min_probability_margin
    )

    if distribution_issue:
        status = "out_of_distribution"
    elif ambiguity_issue:
        status = "ambiguous"
    else:
        status = "accepted"

    return ReliabilityAssessment(
        accepted=not warnings,
        status=status,
        warnings=tuple(warnings),
        confidence=confidence,
        probability_margin=probability_margin,
        out_of_range_features=tuple(outside),
        max_robust_distance=float(max_distance),
    )


def predict_with_reliability(
    *,
    lab_id: str,
    model: Any,
    feature_names: Sequence[str],
    features: Mapping[str, float],
) -> ModelPrediction:
    ordered_names = [str(name) for name in feature_names]
    matrix = pd.DataFrame(
        [[float(features[name]) for name in ordered_names]],
        columns=ordered_names,
    )
    probabilities = np.asarray(model.predict_proba(matrix)[0], dtype=float)
    classes = np.asarray(model.classes_, dtype=object)
    best_index = int(np.argmax(probabilities))
    candidate_class = str(classes[best_index])

    profile = load_feature_profile(lab_id)
    profile_names = [str(name) for name in profile.get("feature_names", [])]
    if profile_names != ordered_names:
        raise ValueError(
            f"Inference profile features do not match the model for '{lab_id}'"
        )

    assessment = assess_reliability(
        feature_values=features,
        probabilities=probabilities,
        profile=profile,
    )

    return ModelPrediction(
        lab_id=lab_id,
        predicted_class=candidate_class if assessment.accepted else "unknown",
        candidate_class=candidate_class,
        confidence=assessment.confidence,
        probabilities={
            str(class_name): float(probability)
            for class_name, probability in zip(classes, probabilities)
        },
        features={str(name): float(value) for name, value in features.items()},
        accepted=assessment.accepted,
        reliability_status=assessment.status,
        reliability_warnings=list(assessment.warnings),
    )
