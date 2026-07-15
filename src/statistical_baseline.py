from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from feature_engineering import create_feature_table
from strict_validation import generate_strict_dataset


RANDOM_SEED = 2026


def midpoint_threshold(
    negative_values: pd.Series,
    positive_values: pd.Series,
) -> float:
    """
    Определяет порог между двумя распределениями.

    Используются 95-й процентиль отрицательного класса
    и 5-й процентиль положительного класса.
    """

    negative_boundary = float(
        negative_values.quantile(0.95)
    )

    positive_boundary = float(
        positive_values.quantile(0.05)
    )

    return (
        negative_boundary
        + positive_boundary
    ) / 2.0


def calculate_thresholds(
    train_features: pd.DataFrame,
) -> dict:
    """Подбирает пороги только по обучающей выборке."""

    outlier_threshold = midpoint_threshold(
        negative_values=train_features.loc[
            train_features["class_name"]
            != "single_outlier",
            "max_abs_jump",
        ],
        positive_values=train_features.loc[
            train_features["class_name"]
            == "single_outlier",
            "max_abs_jump",
        ],
    )

    noise_threshold = midpoint_threshold(
        negative_values=train_features.loc[
            train_features["class_name"]
            != "high_noise",
            "std_residual",
        ],
        positive_values=train_features.loc[
            train_features["class_name"]
            == "high_noise",
            "std_residual",
        ],
    )

    drift_threshold = midpoint_threshold(
        negative_values=train_features.loc[
            train_features["class_name"]
            != "sensor_drift",
            "residual_mean_change",
        ].abs(),
        positive_values=train_features.loc[
            train_features["class_name"]
            == "sensor_drift",
            "residual_mean_change",
        ].abs(),
    )

    return {
        "max_abs_jump": outlier_threshold,
        "std_residual": noise_threshold,
        "residual_mean_change": drift_threshold,
    }


def rule_based_predict(
    row: pd.Series,
    thresholds: dict,
) -> str:
    """
    Классифицирует эксперимент с помощью
    последовательности пороговых правил.
    """

    if (
        row["max_abs_jump"]
        >= thresholds["max_abs_jump"]
    ):
        return "single_outlier"

    if (
        row["std_residual"]
        >= thresholds["std_residual"]
    ):
        return "high_noise"

    if (
        abs(row["residual_mean_change"])
        >= thresholds["residual_mean_change"]
    ):
        return "sensor_drift"

    return "normal"


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    features_path = (
        project_root
        / "data"
        / "features.csv"
    )

    model_path = (
        project_root
        / "models"
        / "best_model.joblib"
    )

    feature_names_path = (
        project_root
        / "models"
        / "feature_names.joblib"
    )

    output_path = (
        project_root
        / "data"
        / "baseline_comparison.csv"
    )

    training_features = pd.read_csv(
        features_path
    )

    train_part, _ = train_test_split(
        training_features,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=training_features["class_name"],
    )

    thresholds = calculate_thresholds(
        train_part
    )

    print("Пороги статистического baseline:")

    for feature_name, threshold in thresholds.items():
        print(
            f"{feature_name}: {threshold:.6f}"
        )

    strict_dataset = generate_strict_dataset()

    strict_features = create_feature_table(
        strict_dataset
    )

    y_true = strict_features["class_name"]

    rule_predictions = strict_features.apply(
        rule_based_predict,
        axis=1,
        thresholds=thresholds,
    )

    rule_accuracy = accuracy_score(
        y_true,
        rule_predictions,
    )

    rule_macro_f1 = f1_score(
        y_true,
        rule_predictions,
        average="macro",
    )

    model = joblib.load(model_path)

    feature_names = joblib.load(
        feature_names_path
    )

    ml_predictions = model.predict(
        strict_features[feature_names]
    )

    ml_accuracy = accuracy_score(
        y_true,
        ml_predictions,
    )

    ml_macro_f1 = f1_score(
        y_true,
        ml_predictions,
        average="macro",
    )

    comparison = pd.DataFrame(
        [
            {
                "approach": (
                    "Statistical rules"
                ),
                "accuracy": rule_accuracy,
                "macro_f1": rule_macro_f1,
            },
            {
                "approach": (
                    "Best ML model"
                ),
                "accuracy": ml_accuracy,
                "macro_f1": ml_macro_f1,
            },
        ]
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print("\nСравнение подходов:")
    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nОтчет статистического baseline:"
    )

    print(
        classification_report(
            y_true,
            rule_predictions,
            digits=4,
        )
    )

    labels = sorted(
        y_true.unique()
    )

    matrix = confusion_matrix(
        y_true,
        rule_predictions,
        labels=labels,
    )

    print(
        "\nМатрица ошибок baseline:"
    )

    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )
    )

    print(
        "\nФайл сравнения сохранен:"
    )

    print(output_path)