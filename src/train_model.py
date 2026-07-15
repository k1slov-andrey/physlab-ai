from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 2026


def evaluate_model(
    model_name: str,
    model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:
    """Оценивает модель на валидационной выборке."""

    predictions = model.predict(x_validation)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_validation, predictions),
        "macro_f1": f1_score(
            y_validation,
            predictions,
            average="macro",
        ),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    features_path = project_root / "data" / "features.csv"
    models_directory = project_root / "models"
    data_directory = project_root / "data"

    models_directory.mkdir(exist_ok=True)

    features = pd.read_csv(features_path)

    x = features.drop(
        columns=[
            "experiment_id",
            "class_name",
        ]
    )

    y = features["class_name"]

    # 70% обучение, 15% валидация, 15% тест
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    models = {
        "dummy_baseline": DummyClassifier(
            strategy="most_frequent",
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
    }

    validation_results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        trained_models[model_name] = model

        result = evaluate_model(
            model_name=model_name,
            model=model,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        validation_results.append(result)

    metrics_table = pd.DataFrame(validation_results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    print("Результаты на валидационной выборке:")
    print(metrics_table.to_string(index=False))

    best_model_name = metrics_table.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    test_predictions = best_model.predict(x_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
    )

    print("\nЛучшая модель:")
    print(best_model_name)

    print("\nРезультаты на тестовой выборке:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")

    print("\nПодробный отчет:")
    print(
        classification_report(
            y_test,
            test_predictions,
            digits=4,
        )
    )

    metrics_table["selected_as_best"] = (
        metrics_table["model"] == best_model_name
    )

    metrics_table.to_csv(
        data_directory / "model_metrics.csv",
        index=False,
    )

    joblib.dump(
        best_model,
        models_directory / "best_model.joblib",
    )

    feature_names = list(x.columns)

    joblib.dump(
        feature_names,
        models_directory / "feature_names.joblib",
    )

    labels = sorted(y.unique())

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=labels,
        xticks_rotation=30,
    )

    plt.title(
        f"Матрица ошибок: {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        data_directory / "confusion_matrix.png",
        dpi=200,
    )

    plt.show()

    print("\nФайлы сохранены:")
    print(data_directory / "model_metrics.csv")
    print(data_directory / "confusion_matrix.png")
    print(models_directory / "best_model.joblib")