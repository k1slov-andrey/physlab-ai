from __future__ import annotations

import importlib

import pandas as pd
import pytest


LAB_CASES = [
    ("cooling", "labs.cooling.module", "normal"),
    ("boyle_mariotte", "labs.boyle_mariotte.module", "normal"),
    ("isochoric", "labs.isochoric.module", "normal"),
    ("heat_balance", "labs.heat_balance.module", "normal"),
]


@pytest.mark.parametrize("lab_id,module_name,class_name", LAB_CASES)
def test_lab_simulation_and_prediction(
    lab_id: str,
    module_name: str,
    class_name: str,
) -> None:
    module = importlib.import_module(module_name)

    dataframe = module.simulate(class_name, seed=123)

    assert isinstance(dataframe, pd.DataFrame)
    assert not dataframe.empty
    assert len(dataframe) >= 3

    prediction = module.predict(dataframe)

    assert getattr(prediction, "lab_id", None) == lab_id
    assert isinstance(getattr(prediction, "predicted_class", None), str)

    confidence = float(getattr(prediction, "confidence", -1.0))
    assert 0.0 <= confidence <= 1.0

    probabilities = getattr(prediction, "probabilities", None)
    assert isinstance(probabilities, dict)
    assert probabilities
    assert all(0.0 <= float(value) <= 1.0 for value in probabilities.values())
