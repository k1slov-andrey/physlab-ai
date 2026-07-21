from __future__ import annotations

import importlib

import pandas as pd
import pytest

from labs.common.realism import generation_family


LAB_IDS = (
    "cooling",
    "boyle_mariotte",
    "isochoric",
    "heat_balance",
)


def test_generation_family_identifier_is_explicit_and_stable() -> None:
    assert generation_family("cooling", 7) == "cooling_family_0007"

    with pytest.raises(ValueError):
        generation_family("", 0)
    with pytest.raises(ValueError):
        generation_family("cooling", -1)


@pytest.mark.parametrize("lab_id", LAB_IDS)
def test_each_generation_family_contains_all_class_variants(lab_id: str) -> None:
    module = importlib.import_module(f"labs.{lab_id}.module")
    _, features = module.generate_dataset(n_per_class=8, seed=123)

    assert len(features) == 8 * len(module.CLASSES)
    assert features["generation_group"].nunique() == 8
    assert features["generation_group"].str.startswith(
        f"{lab_id}_family_"
    ).all()

    family_class_counts = pd.crosstab(
        features["generation_group"],
        features["class_name"],
    )
    assert list(family_class_counts.columns) == sorted(module.CLASSES)
    assert (family_class_counts == 1).all().all()


@pytest.mark.parametrize("lab_id", LAB_IDS)
def test_family_variants_share_measurement_context(lab_id: str) -> None:
    module = importlib.import_module(f"labs.{lab_id}.module")
    _, features = module.generate_dataset(n_per_class=8, seed=321)

    per_family = features.groupby("generation_group").agg(
        device_profiles=("device_profile", "nunique"),
        environment_profiles=("environment_profile", "nunique"),
    )

    assert (per_family["device_profiles"] == 1).all()
    assert (per_family["environment_profiles"] == 1).all()
