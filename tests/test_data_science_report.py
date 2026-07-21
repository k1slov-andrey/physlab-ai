from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_data_science_report import check_committed_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DS_DIR = PROJECT_ROOT / "evaluation" / "data_science"


def test_data_science_artifacts_are_current() -> None:
    check_committed_outputs()


def test_eda_confirms_balanced_complete_feature_tables() -> None:
    overview = pd.read_csv(DS_DIR / "eda_overview.csv")
    assert len(overview) == 4
    assert (overview["feature_rows"] == 640).all()
    assert (overview["generation_groups"] == 160).all()
    assert (overview["classes"] == 4).all()
    assert (overview["min_class_samples"] == 160).all()
    assert (overview["max_class_samples"] == 160).all()
    assert (overview["missing_model_values"] == 0).all()
    assert (overview["duplicate_experiment_ids"] == 0).all()


def test_model_selection_documents_baseline_cv_and_locked_test() -> None:
    selection = pd.read_csv(DS_DIR / "model_selection_summary.csv")
    assert set(selection["selected_configuration"]) == {"baseline", "tuned"}
    assert (selection["dummy_validation_macro_f1"] == 0.1).all()
    assert (selection["development_samples"] == 480).all()
    assert (selection["test_samples"] == 160).all()
    assert selection["final_test_macro_f1"].between(0.0, 1.0).all()


def test_per_class_metrics_cover_all_sixteen_scenarios() -> None:
    metrics = pd.read_csv(DS_DIR / "per_class_metrics.csv")
    assert len(metrics) == 16
    assert (metrics.groupby("lab_id")["class_name"].nunique() == 4).all()
    assert (metrics["support"] == 40).all()
    assert metrics["f1_score"].between(0.0, 1.0).all()


def test_reliability_evidence_exposes_coverage_and_stress_limits() -> None:
    reliability = pd.read_csv(DS_DIR / "reliability_summary.csv")
    assert reliability["coverage"].between(0.0, 1.0).all()
    assert reliability["selective_accuracy"].between(0.0, 1.0).all()
    assert (reliability["single_severe_shift_rejection_rate"] == 1.0).all()
    assert (reliability["multifeature_shift_rejection_rate"] == 1.0).all()
