from __future__ import annotations

import pandas as pd

from evaluate_deployed_models import build_summary, check_outputs


LAB_IDS = {"cooling", "boyle_mariotte", "isochoric", "heat_balance"}


def test_deployed_summary_covers_all_labs() -> None:
    summary, _ = build_summary()
    assert set(summary["lab_id"]) == LAB_IDS
    assert (summary["test_samples"] == 160).all()
    assert (summary["test_groups"] == 40).all()


def test_deployed_metrics_use_saved_test_partition() -> None:
    summary, predictions = build_summary()
    assert set(predictions) == LAB_IDS
    for lab_id, frame in predictions.items():
        assert len(frame) == 160, lab_id
        assert frame["generation_group"].nunique() == 40, lab_id
        assert frame["target"].notna().all(), lab_id
        assert frame["prediction"].notna().all(), lab_id


def test_committed_deployed_evaluation_is_current() -> None:
    summary, predictions = build_summary()
    check_outputs(summary, predictions)
    stored = pd.read_csv("evaluation/final_model_summary.csv")
    assert set(stored["lab_id"]) == LAB_IDS
