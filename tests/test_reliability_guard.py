from __future__ import annotations

import numpy as np
import pandas as pd

from labs.common.reliability import assess_reliability, build_feature_profile


def _profile() -> dict:
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 10.0, 200),
            "feature_b": np.linspace(-2.0, 2.0, 200),
            "feature_c": np.linspace(100.0, 120.0, 200),
        }
    )
    return build_feature_profile(frame)


def test_in_distribution_prediction_is_accepted() -> None:
    assessment = assess_reliability(
        feature_values={"feature_a": 5.0, "feature_b": 0.0, "feature_c": 110.0},
        probabilities=[0.78, 0.12, 0.07, 0.03],
        profile=_profile(),
    )

    assert assessment.accepted is True
    assert assessment.status == "accepted"
    assert assessment.warnings == ()


def test_low_probability_margin_is_rejected() -> None:
    assessment = assess_reliability(
        feature_values={"feature_a": 5.0, "feature_b": 0.0, "feature_c": 110.0},
        probabilities=[0.42, 0.39, 0.11, 0.08],
        profile=_profile(),
    )

    assert assessment.accepted is False
    assert assessment.status == "ambiguous"
    assert any("близкие оценки" in warning for warning in assessment.warnings)


def test_extreme_feature_values_are_rejected() -> None:
    assessment = assess_reliability(
        feature_values={
            "feature_a": 1000.0,
            "feature_b": -500.0,
            "feature_c": 110.0,
        },
        probabilities=[0.99, 0.005, 0.003, 0.002],
        profile=_profile(),
    )

    assert assessment.accepted is False
    assert assessment.status == "out_of_distribution"
    assert set(assessment.out_of_range_features) == {"feature_a", "feature_b"}
    assert assessment.max_robust_distance > 4.0


def test_feature_profile_rejects_empty_dataset() -> None:
    try:
        build_feature_profile(pd.DataFrame())
    except ValueError as error:
        assert "empty dataset" in str(error)
    else:
        raise AssertionError("Expected ValueError")
