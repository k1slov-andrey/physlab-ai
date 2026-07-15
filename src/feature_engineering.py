from pathlib import Path

import numpy as np
import pandas as pd


def extract_features_from_experiment(
    experiment: pd.DataFrame,
) -> dict:
    """Преобразует один временной ряд в набор числовых признаков."""

    measured = experiment["measured_temperature"].to_numpy()
    ideal = experiment["ideal_temperature"].to_numpy()
    time = experiment["time_seconds"].to_numpy()

    temperature_diff = np.diff(measured)
    residuals = measured - ideal

    duration = time[-1] - time[0]

    features = {
        "experiment_id": int(experiment["experiment_id"].iloc[0]),
        "mean_temperature": float(np.mean(measured)),
        "std_temperature": float(np.std(measured)),
        "max_temperature": float(np.max(measured)),
        "min_temperature": float(np.min(measured)),
        "temperature_range": float(np.max(measured) - np.min(measured)),
        "max_abs_jump": float(np.max(np.abs(temperature_diff))),
        "mean_abs_jump": float(np.mean(np.abs(temperature_diff))),
        "mean_residual": float(np.mean(residuals)),
        "mean_abs_residual": float(np.mean(np.abs(residuals))),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "std_residual": float(np.std(residuals)),
        "residual_slope": float(
            np.polyfit(time, residuals, 1)[0]
        ),
        "average_cooling_rate": float(
            (measured[-1] - measured[0]) / duration
        ),
        "class_name": experiment["class_name"].iloc[0],
    }

    return features


def create_feature_table(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Создает таблицу: одна строка — один эксперимент."""

    feature_rows = []

    for _, experiment in dataset.groupby("experiment_id"):
        features = extract_features_from_experiment(
            experiment
        )
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "ml_dataset.csv"
    output_path = project_root / "data" / "features.csv"

    dataset = pd.read_csv(input_path)

    feature_table = create_feature_table(dataset)

    feature_table.to_csv(
        output_path,
        index=False,
    )

    print("Таблица признаков создана:")
    print(output_path)

    print("\nРазмер таблицы:")
    print(feature_table.shape)

    print("\nКоличество экспериментов по классам:")
    print(feature_table["class_name"].value_counts())

    print("\nПервые строки:")
    print(feature_table.head())