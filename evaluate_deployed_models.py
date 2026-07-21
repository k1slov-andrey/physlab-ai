from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from core.lab_registry import list_labs


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = PROJECT_ROOT / "evaluation" / "final_model_summary.csv"
EVALUATION_SOURCE = "deployed_model_on_saved_test_split"


def _load_protocol(lab_id: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "evaluation" / lab_id / "evaluation_protocol.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_lab(lab_id: str) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    data_dir = PROJECT_ROOT / "data" / lab_id
    model_dir = PROJECT_ROOT / "models" / lab_id
    evaluation_dir = PROJECT_ROOT / "evaluation" / lab_id

    features = pd.read_csv(data_dir / "features.csv")
    manifest = pd.read_csv(evaluation_dir / "split_manifest.csv")
    feature_names = [
        str(value) for value in joblib.load(model_dir / "feature_names.joblib")
    ]
    model = joblib.load(model_dir / "best_model.joblib")

    test_manifest = manifest.loc[
        manifest["dataset_role"].astype(str) == "test"
    ].copy()
    test_manifest = test_manifest.sort_values("row_index").reset_index(drop=True)
    row_indices = test_manifest["row_index"].astype(int).to_numpy()

    x_test = features.iloc[row_indices][feature_names]
    y_true = test_manifest["target"].astype(str).to_numpy()
    y_pred = np.asarray(model.predict(x_test), dtype=str)

    if not callable(getattr(model, "predict_proba", None)):
        raise RuntimeError(f"{lab_id}: deployed model has no predict_proba method")
    probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
    classes = [str(value) for value in model.classes_]
    confidence = probabilities.max(axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    protocol = _load_protocol(lab_id)
    split_strategy = str(
        protocol.get("split_strategy", "stratified_group_train_validation_test")
    )
    test_groups = int(test_manifest["generation_group"].astype(str).nunique())

    summary_row = {
        "lab_id": lab_id,
        "model_class": type(model).__name__,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "test_samples": int(len(test_manifest)),
        "test_groups": test_groups,
        "split_strategy": split_strategy,
        "evaluation_source": EVALUATION_SOURCE,
    }

    metrics = {
        **summary_row,
        "feature_count": len(feature_names),
        "class_labels": classes,
    }

    predictions = pd.DataFrame(
        {
            "row_index": row_indices,
            "experiment_id": test_manifest["experiment_id"].astype(str),
            "generation_group": test_manifest["generation_group"].astype(str),
            "target": y_true,
            "prediction": y_pred,
            "correct": y_true == y_pred,
            "confidence": confidence,
        }
    )
    for class_index, class_name in enumerate(classes):
        predictions[f"probability_{class_name}"] = probabilities[:, class_index]

    return summary_row, metrics, predictions


def build_summary() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the deployed-model summary and one prediction frame per laboratory.

    The public return shape is intentionally stable because project tests and
    downstream checks consume the prediction frames directly.
    """
    rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for lab in sorted(list_labs(), key=lambda item: item.lab_id):
        summary_row, _, lab_predictions = _evaluate_lab(lab.lab_id)
        rows.append(summary_row)
        predictions[lab.lab_id] = lab_predictions

    return pd.DataFrame(rows), predictions


def _assert_frames_equal(expected: pd.DataFrame, actual: pd.DataFrame, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise SystemExit(f"{name} is stale: {error}") from error


def write_outputs(
    summary: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    for lab_id, lab_predictions in predictions.items():
        _, metrics, _ = _evaluate_lab(lab_id)
        evaluation_dir = PROJECT_ROOT / "evaluation" / lab_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        (evaluation_dir / "deployed_model_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        lab_predictions.to_csv(
            evaluation_dir / "deployed_model_predictions.csv",
            index=False,
        )


def check_outputs(
    summary: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> None:
    if not SUMMARY_PATH.is_file():
        raise SystemExit("Missing evaluation/final_model_summary.csv")
    _assert_frames_equal(
        summary,
        pd.read_csv(SUMMARY_PATH),
        "evaluation/final_model_summary.csv",
    )

    for lab_id, expected_predictions in predictions.items():
        evaluation_dir = PROJECT_ROOT / "evaluation" / lab_id
        metrics_path = evaluation_dir / "deployed_model_metrics.json"
        predictions_path = evaluation_dir / "deployed_model_predictions.csv"

        if not metrics_path.is_file():
            raise SystemExit(f"Missing {metrics_path.relative_to(PROJECT_ROOT)}")
        if not predictions_path.is_file():
            raise SystemExit(f"Missing {predictions_path.relative_to(PROJECT_ROOT)}")

        _, expected_metrics, _ = _evaluate_lab(lab_id)
        actual_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if actual_metrics != expected_metrics:
            raise SystemExit(f"{metrics_path.relative_to(PROJECT_ROOT)} is stale")

        _assert_frames_equal(
            expected_predictions,
            pd.read_csv(predictions_path),
            str(predictions_path.relative_to(PROJECT_ROOT)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the deployed models on their locked test splits."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that committed deployed-model evaluation artifacts are current",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, predictions = build_summary()
    if args.check:
        check_outputs(summary, predictions)
        print("Deployed model evaluation is current")
        return

    write_outputs(summary, predictions)
    print(f"Saved {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved deployed-model outputs for {len(predictions)} laboratories")


if __name__ == "__main__":
    main()
