from __future__ import annotations

import json

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

from labs.common.artifacts import (
    get_lab_data_dir,
    get_lab_evaluation_dir,
    get_lab_model_dir,
)
from labs.common.model_training import (
    evaluate_models,
    fit_selected_model,
    results_to_dataframe,
    score_model,
)
from labs.common.reliability import build_feature_profile, save_feature_profile
from labs.common.splitting import build_split_manifest, split_train_validation_test


NON_FEATURE_COLUMNS = {
    "class_name",
    "experiment_id",
    "generation_group",
    "device_profile",
    "environment_profile",
    "secondary_errors",
    "severity",
}


def train_and_save(
    lab_id: str,
    features_df: pd.DataFrame,
    target_col: str = "class_name",
    id_col: str = "experiment_id",
    random_state: int = 42,
) -> dict[str, object]:
    excluded = NON_FEATURE_COLUMNS | {target_col, id_col}
    feature_cols = [
        column
        for column in features_df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(features_df[column])
    ]
    if not feature_cols:
        raise ValueError(f"No numeric model features found for {lab_id}")

    split = split_train_validation_test(
        frame=features_df,
        target_column=target_col,
        group_column="generation_group",
        random_state=random_state,
        test_fraction=0.25,
        validation_fraction=0.25,
    )

    x = features_df[feature_cols]
    y = features_df[target_col]
    x_train = x.iloc[split.train_index]
    y_train = y.iloc[split.train_index]
    x_validation = x.iloc[split.validation_index]
    y_validation = y.iloc[split.validation_index]
    x_test = x.iloc[split.test_index]
    y_test = y.iloc[split.test_index]

    candidate_results, selected = evaluate_models(
        x_train=x_train,
        x_validation=x_validation,
        y_train=y_train,
        y_validation=y_validation,
        random_state=random_state,
    )

    development_index = np.concatenate(
        [split.train_index, split.validation_index]
    )

    model_dir = get_lab_model_dir(lab_id)
    evaluation_dir = get_lab_evaluation_dir(lab_id)
    data_dir = get_lab_data_dir(lab_id)

    inference_profile = build_feature_profile(x.iloc[development_index])
    inference_profile["lab_id"] = lab_id
    inference_profile["dataset_roles"] = ["train", "validation"]
    save_feature_profile(
        inference_profile,
        model_dir / "inference_profile.json",
    )

    final_model = fit_selected_model(
        model_name=selected.name,
        features=x.iloc[development_index],
        target=y.iloc[development_index],
        random_state=random_state,
    )
    test_scores = score_model(final_model, x_test, y_test)
    predictions = final_model.predict(x_test)

    joblib.dump(final_model, model_dir / "best_model.joblib")
    joblib.dump(feature_cols, model_dir / "feature_names.joblib")

    candidate_frame = results_to_dataframe(candidate_results)
    candidate_frame["selected"] = candidate_frame["model"].eq(selected.name)
    candidate_frame.to_csv(
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
        final_model,
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
            "row_index": split.test_index,
            "true_class": y_test.to_numpy(),
            "predicted_class": predictions,
        }
    )
    if "generation_group" in features_df.columns:
        holdout_df["generation_group"] = features_df.iloc[
            split.test_index
        ]["generation_group"].to_numpy()
    if hasattr(final_model, "predict_proba"):
        probabilities = final_model.predict_proba(x_test)
        holdout_df["confidence"] = np.max(probabilities, axis=1)
    holdout_df.to_csv(
        evaluation_dir / "holdout_predictions.csv",
        index=False,
    )

    split_manifest = build_split_manifest(
        frame=features_df,
        split=split,
        target_column=target_col,
        group_column="generation_group",
    )
    split_manifest.to_csv(
        evaluation_dir / "split_manifest.csv",
        index=False,
    )

    features_df.to_csv(data_dir / "features.csv", index=False)

    group_column_available = "generation_group" in features_df.columns
    summary: dict[str, object] = {
        "lab_id": lab_id,
        "best_model": selected.name,
        "accuracy": test_scores["accuracy"],
        "balanced_accuracy": test_scores["balanced_accuracy"],
        "macro_f1": test_scores["macro_f1"],
        "selection_validation_accuracy": selected.accuracy,
        "selection_validation_balanced_accuracy": selected.balanced_accuracy,
        "selection_validation_macro_f1": selected.macro_f1,
        "n_samples": len(features_df),
        "n_features": len(feature_cols),
        "validation_strategy": split.strategy,
        "train_samples": len(split.train_index),
        "validation_samples": len(split.validation_index),
        "test_samples": len(split.test_index),
        "train_groups": (
            features_df.iloc[split.train_index]["generation_group"].nunique()
            if group_column_available
            else 0
        ),
        "validation_groups": (
            features_df.iloc[split.validation_index]["generation_group"].nunique()
            if group_column_available
            else 0
        ),
        "test_groups": (
            features_df.iloc[split.test_index]["generation_group"].nunique()
            if group_column_available
            else 0
        ),
    }
    pd.DataFrame([summary]).to_csv(
        evaluation_dir / "summary.csv",
        index=False,
    )

    protocol = {
        "model_selection": "Candidate model family selected on validation Macro F1.",
        "final_fit": "Selected family fitted on train and validation partitions.",
        "final_evaluation": "Test partition used once for the reported metrics.",
        "split_strategy": split.strategy,
        "group_definition": (
            "Each generation group is one latent experimental setup. "
            "It contains one counterfactual variant per target class generated "
            "from the same random seed, device profile, environment profile "
            "and base measurement plan."
        ),
        "random_state": random_state,
        "test_fraction": 0.25,
        "validation_fraction": 0.25,
    }
    (evaluation_dir / "evaluation_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def load_model_and_features(lab_id: str):
    model_dir = get_lab_model_dir(lab_id)
    return (
        joblib.load(model_dir / "best_model.joblib"),
        joblib.load(model_dir / "feature_names.joblib"),
    )
