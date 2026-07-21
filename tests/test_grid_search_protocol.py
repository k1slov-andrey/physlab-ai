from __future__ import annotations

import numpy as np
import pandas as pd

from labs.common.splitting import DatasetSplit
from validate_and_gridsearch import _select_configuration


def test_tuned_configuration_requires_cv_improvement() -> None:
    assert _select_configuration(0.800, 0.803, 0.002) == "tuned"
    assert _select_configuration(0.800, 0.801, 0.002) == "baseline"


def test_configuration_selection_does_not_accept_test_metrics() -> None:
    parameters = _select_configuration.__annotations__
    assert "test_score" not in parameters
    assert "holdout_score" not in parameters


def test_development_and_test_indices_are_disjoint() -> None:
    split = DatasetSplit(
        train_index=np.array([0, 1, 2]),
        validation_index=np.array([3, 4]),
        test_index=np.array([5, 6]),
        strategy="test",
    )
    frame = pd.DataFrame(
        {
            "class_name": ["a", "b", "a", "a", "b", "a", "b"],
        }
    )
    development = set(
        np.concatenate([split.train_index, split.validation_index]).tolist()
    )
    test = set(split.test_index.tolist())
    assert development.isdisjoint(test)
