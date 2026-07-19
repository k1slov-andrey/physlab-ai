from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from labs.common.artifacts import (
    get_lab_data_dir,
    get_lab_evaluation_dir,
    get_lab_model_dir,
)
from labs.common.model_training import evaluate_models, results_to_dataframe


NON_FEATURE_COLUMNS = {
    "class_name",
    "experiment_id",
    "generation_group",
    "device_profile",
    "environment_profile",
    "secondary_errors",
    "severity",
}


def _split_dataset(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    random_state: int,
):
    x = features_df[feature_columns]
    y = features_df[target_column]

    if (
        "generation_group" in features_df.columns
        and features_df["generation_group"].nunique() >= 4
    ):
        groups = features_df["generation_group"]
        splitter = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )
        train_indices, test_indices = next(splitter.split(x, y, groups))
        return (
            x.iloc[train_indices],
            x.iloc[test_indices],
            y.iloc[train_indices],
            y.iloc[test_indices],
            "stratified_group_holdout",
            groups.iloc[train_indices],
            groups.iloc[test_indices],
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )
    return (
        x_train,
        x_test,
        y_train,
        y_test,
        "stratified_random_holdout",
        None,
        None,
    )


def train_and_save(
    lab_id: str,
    features_df: pd.DataFrame,
    target_col: str = "class_name",
    id_col: str = "experiment_id",
    random_state: int = 42,
):
    excluded = NON_FEATURE_COLUMNS | {target_col, id_col}
    feature_cols = [
        column
        for column in features_df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(features_df[column])
    ]
    if not feature_cols:
        raise ValueError(f"No numeric model features found for {lab_id}")

    (
        x_train,
        x_test,
        y_train,
        y_test,
        validation_strategy,
        train_groups,
        test_groups,
    ) = _split_dataset(
        features_df,
        feature_cols,
        target_col,
        random_state,
    )

    results, best = evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
        random_state,
    )
    predictions = best.model.predict(x_test)

    model_dir = get_lab_model_dir(lab_id)
    evaluation_dir = get_lab_evaluation_dir(lab_id)
    data_dir = get_lab_data_dir(lab_id)

    joblib.dump(best.model, model_dir / "best_model.joblib")
    joblib.dump(feature_cols, model_dir / "feature_names.joblib")

    results_to_dataframe(results).to_csv(
        evaluation_dir / "model_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    ).T.to_csv(evaluation_dir / "classification_report.csv")

    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=labels,
    ).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    figure.savefig(
        evaluation_dir / "confusion_matrix.png",
        dpi=180,
    )
    plt.close(figure)

    permutation = permutation_importance(
        best.model,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(
        evaluation_dir / "feature_importance.csv",
        index=False,
    )

    top_importance = importance_df.head(15).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(
        top_importance["feature"],
        top_importance["importance_mean"],
        xerr=top_importance["importance_std"],
    )
    axis.set_xlabel("Permutation importance, Macro F1 decrease")
    axis.set_title(f"Feature importance: {lab_id}")
    figure.tight_layout()
    figure.savefig(
        evaluation_dir / "feature_importance.png",
        dpi=180,
    )
    plt.close(figure)

    holdout_df = pd.DataFrame(
        {
            "row_index": x_test.index,
            "true_class": y_test.to_numpy(),
            "predicted_class": predictions,
        }
    )
    if hasattr(best.model, "predict_proba"):
        probabilities = best.model.predict_proba(x_test)
        holdout_df["confidence"] = np.max(probabilities, axis=1)
    holdout_df.to_csv(
        evaluation_dir / "holdout_predictions.csv",
        index=False,
    )

    features_df.to_csv(data_dir / "features.csv", index=False)

    summary = {
        "lab_id": lab_id,
        "best_model": best.name,
        "accuracy": best.accuracy,
        "macro_f1": best.macro_f1,
        "n_samples": len(features_df),
        "n_features": len(feature_cols),
        "validation_strategy": validation_strategy,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "train_groups": 0 if train_groups is None else train_groups.nunique(),
        "test_groups": 0 if test_groups is None else test_groups.nunique(),
    }
    pd.DataFrame([summary]).to_csv(
        evaluation_dir / "summary.csv",
        index=False,
    )
    return summary


def load_model_and_features(lab_id: str):
    model_dir = get_lab_model_dir(lab_id)
    return (
        joblib.load(model_dir / "best_model.joblib"),
        joblib.load(model_dir / "feature_names.joblib"),
    )
