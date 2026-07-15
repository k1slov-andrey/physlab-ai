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

from feature_engineering import create_feature_table
from generate_cooling import (
    add_high_noise,
    add_sensor_drift,
    add_single_outlier,
    generate_cooling_experiment,
)


RANDOM_SEED = 777
EXPERIMENTS_PER_CLASS = 80


def create_strict_experiment(
    experiment_id: int,
    class_name: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Создает эксперимент с параметрами,
    отличающимися от обучающей выборки.
    """

    initial_temperature = float(
        rng.uniform(60.0, 100.0)
    )

    room_temperature = float(
        rng.uniform(15.0, 30.0)
    )

    cooling_coefficient = float(
        rng.uniform(0.004, 0.020)
    )

    noise_std = float(
        rng.uniform(0.08, 0.55)
    )

    experiment = generate_cooling_experiment(
        initial_temperature=initial_temperature,
        room_temperature=room_temperature,
        cooling_coefficient=cooling_coefficient,
        noise_std=noise_std,
        random_seed=int(rng.integers(0, 1_000_000)),
    )

    if class_name == "single_outlier":
        experiment = add_single_outlier(
            experiment,
            rng,
        )

    elif class_name == "sensor_drift":
        experiment = add_sensor_drift(
            experiment,
            rng,
        )

    elif class_name == "high_noise":
        experiment = add_high_noise(
            experiment,
            rng,
        )

    experiment["experiment_id"] = experiment_id
    experiment["class_name"] = class_name

    return experiment


def generate_strict_dataset() -> pd.DataFrame:
    """Создает независимую строгую тестовую выборку."""

    rng = np.random.default_rng(RANDOM_SEED)

    classes = [
        "normal",
        "single_outlier",
        "sensor_drift",
        "high_noise",
    ]

    experiments = []
    experiment_id = 0

    for class_name in classes:
        for _ in range(EXPERIMENTS_PER_CLASS):
            experiment = create_strict_experiment(
                experiment_id=experiment_id,
                class_name=class_name,
                rng=rng,
            )

            experiments.append(experiment)
            experiment_id += 1

    return pd.concat(
        experiments,
        ignore_index=True,
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

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
        / "strict_validation_results.csv"
    )

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)

    strict_dataset = generate_strict_dataset()
    strict_features = create_feature_table(
        strict_dataset
    )

    x_strict = strict_features[feature_names]
    y_strict = strict_features["class_name"]

    predictions = model.predict(x_strict)

    accuracy = accuracy_score(
        y_strict,
        predictions,
    )

    macro_f1 = f1_score(
        y_strict,
        predictions,
        average="macro",
    )

    print("Строгая независимая проверка")
    print(f"Количество экспериментов: {len(strict_features)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nОтчет по классам:")
    print(
        classification_report(
            y_strict,
            predictions,
            digits=4,
        )
    )

    labels = sorted(y_strict.unique())

    matrix = confusion_matrix(
        y_strict,
        predictions,
        labels=labels,
    )

    print("\nМатрица ошибок:")
    print(pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ))

    results = strict_features[
        [
            "experiment_id",
            "class_name",
        ]
    ].copy()

    results["predicted_class"] = predictions
    results["is_correct"] = (
        results["class_name"]
        == results["predicted_class"]
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nРезультаты сохранены:")
    print(output_path)