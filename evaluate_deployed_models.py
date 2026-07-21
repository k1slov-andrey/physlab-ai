from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from core.lab_registry import list_labs


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = PROJECT_ROOT / "evaluation" / "final_model_summary.csv"
FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DeployedModelMetrics:
    lab_id: str
    model_class: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    test_samples: int
    test_groups: int
    split_strategy: str
    evaluation_source: str = "deployed_model_on_saved_test_split"


def _load_test_partition(lab_id: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features_path = PROJECT_ROOT / "data" / lab_id / "features.csv"
    manifest_path = PROJECT_ROOT / "evaluation" / lab_id / "split_manifest.csv"
    feature_names_path = PROJECT_ROOT / "models" / lab_id / "feature_names.joblib"

    features = pd.read_csv(features_path)
    manifest = pd.read_csv(manifest_path)

    required_manifest_columns = {
        "row_index",
        "dataset_role",
        "target",
        "generation_group",
    }
    missing = required_manifest_columns.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"{lab_id}: split manifest is missing columns: {sorted(missing)}"
        )

    if manifest["row_index"].duplicated().any():
        raise ValueError(f"{lab_id}: row_index values must be unique")

    row_indices = manifest["row_index"].astype(int)
    if row_indices.min() < 0 or row_indices.max() >= len(features):
        raise ValueError(f"{lab_id}: split manifest contains an invalid row_index")

    test_manifest = manifest.loc[manifest["dataset_role"] == "test"].copy()
    if test_manifest.empty:
        raise ValueError(f"{lab_id}: test partition is empty")

    feature_names = [str(value) for value in joblib.load(feature_names_path)]
    missing_features = [name for name in feature_names if name not in features.columns]
    if missing_features:
        raise ValueError(
            f"{lab_id}: features.csv is missing model inputs: {missing_features}"
        )

    test_rows = test_manifest["row_index"].astype(int).to_numpy()
    x_test = features.iloc[test_rows][feature_names].copy()
    y_test = test_manifest["target"].astype(str).reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    test_manifest = test_manifest.reset_index(drop=True)
    return x_test, y_test, test_manifest


def evaluate_lab(lab_id: str) -> tuple[DeployedModelMetrics, pd.DataFrame]:
    model_path = PROJECT_ROOT / "models" / lab_id / "best_model.joblib"
    protocol_path = PROJECT_ROOT / "evaluation" / lab_id / "evaluation_protocol.json"

    model = joblib.load(model_path)
    x_test, y_test, test_manifest = _load_test_partition(lab_id)
    predictions = pd.Series(model.predict(x_test), dtype="string")

    with protocol_path.open(encoding="utf-8") as source:
        protocol = json.load(source)

    metrics = DeployedModelMetrics(
        lab_id=lab_id,
        model_class=type(model).__name__,
        accuracy=float(accuracy_score(y_test, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        test_samples=int(len(test_manifest)),
        test_groups=int(test_manifest["generation_group"].nunique()),
        split_strategy=str(protocol.get("split_strategy", "unknown")),
    )

    prediction_frame = test_manifest[
        ["row_index", "generation_group", "target"]
    ].copy()
    if "experiment_id" in test_manifest.columns:
        prediction_frame.insert(1, "experiment_id", test_manifest["experiment_id"])
    prediction_frame["prediction"] = predictions
    prediction_frame["is_correct"] = prediction_frame["target"] == predictions
    return metrics, prediction_frame


def build_summary(lab_ids: Iterable[str] | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    selected_labs = list(lab_ids) if lab_ids is not None else [lab.lab_id for lab in list_labs()]
    metrics_rows: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for lab_id in selected_labs:
        metrics, lab_predictions = evaluate_lab(lab_id)
        metrics_rows.append(asdict(metrics))
        predictions[lab_id] = lab_predictions

    summary = pd.DataFrame(metrics_rows).sort_values("lab_id").reset_index(drop=True)
    return summary, predictions


def write_outputs(summary: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    for row in summary.to_dict(orient="records"):
        lab_id = str(row["lab_id"])
        evaluation_dir = PROJECT_ROOT / "evaluation" / lab_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = evaluation_dir / "deployed_model_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as target:
            json.dump(row, target, ensure_ascii=False, indent=2)
            target.write("\n")

        predictions[lab_id].to_csv(
            evaluation_dir / "deployed_model_predictions.csv",
            index=False,
        )


def _assert_frames_equal(expected: pd.DataFrame, actual: pd.DataFrame, source: str) -> None:
    if list(expected.columns) != list(actual.columns):
        raise SystemExit(f"{source}: column order does not match the current evaluator")
    if len(expected) != len(actual):
        raise SystemExit(f"{source}: row count does not match the current evaluator")

    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left):
            difference = (left.astype(float) - right.astype(float)).abs().max()
            if pd.isna(difference):
                difference = 0.0
            if difference > FLOAT_TOLERANCE:
                raise SystemExit(
                    f"{source}: numeric column {column!r} is stale "
                    f"(max difference {difference})"
                )
        elif not left.astype(str).equals(right.astype(str)):
            raise SystemExit(f"{source}: column {column!r} is stale")


def check_outputs(summary: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> None:
    if not SUMMARY_PATH.is_file():
        raise SystemExit(
            "Missing evaluation/final_model_summary.csv. "
            "Run: python evaluate_deployed_models.py"
        )

    stored_summary = pd.read_csv(SUMMARY_PATH)
    _assert_frames_equal(summary, stored_summary, str(SUMMARY_PATH.relative_to(PROJECT_ROOT)))

    for lab_id, expected_predictions in predictions.items():
        evaluation_dir = PROJECT_ROOT / "evaluation" / lab_id
        metrics_path = evaluation_dir / "deployed_model_metrics.json"
        predictions_path = evaluation_dir / "deployed_model_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise SystemExit(
                f"{lab_id}: deployed evaluation artifacts are missing. "
                "Run: python evaluate_deployed_models.py"
            )

        stored_predictions = pd.read_csv(predictions_path)
        _assert_frames_equal(
            expected_predictions,
            stored_predictions,
            str(predictions_path.relative_to(PROJECT_ROOT)),
        )

        with metrics_path.open(encoding="utf-8") as source:
            stored_metrics = json.load(source)
        current_metrics = summary.loc[summary["lab_id"] == lab_id].iloc[0].to_dict()
        if set(stored_metrics) != set(current_metrics):
            raise SystemExit(f"{lab_id}: deployed_model_metrics.json has stale fields")
        for key, current_value in current_metrics.items():
            stored_value = stored_metrics[key]
            if isinstance(current_value, float):
                if abs(float(current_value) - float(stored_value)) > FLOAT_TOLERANCE:
                    raise SystemExit(f"{lab_id}: metric {key!r} is stale")
            elif str(current_value) != str(stored_value):
                raise SystemExit(f"{lab_id}: field {key!r} is stale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deployed model files on the saved test partitions."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that committed evaluation artifacts match the deployed models",
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
    print(summary.to_string(index=False))
    print(f"\nSaved {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
