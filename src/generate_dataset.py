from pathlib import Path

import numpy as np
import pandas as pd

from generate_cooling import (
    add_high_noise,
    add_sensor_drift,
    add_single_outlier,
    generate_cooling_experiment,
)


EXPERIMENTS_PER_CLASS = 150
RANDOM_SEED = 2026


def create_experiment(
    experiment_id: int,
    class_name: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Создает один размеченный эксперимент."""

    initial_temperature = float(
        rng.uniform(70.0, 95.0)
    )

    room_temperature = float(
        rng.uniform(18.0, 27.0)
    )

    cooling_coefficient = float(
        rng.uniform(0.006, 0.016)
    )

    noise_std = float(
        rng.uniform(0.12, 0.40)
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

    experiment["initial_temperature"] = initial_temperature
    experiment["room_temperature"] = room_temperature
    experiment["cooling_coefficient"] = cooling_coefficient
    experiment["base_noise_std"] = noise_std

    return experiment


def generate_dataset() -> pd.DataFrame:
    """Создает единый датасет из четырех классов."""

    rng = np.random.default_rng(RANDOM_SEED)

    classes = [
        "normal",
        "single_outlier",
        "sensor_drift",
        "high_noise",
    ]

    experiments: list[pd.DataFrame] = []

    experiment_id = 0

    for class_name in classes:
        for _ in range(EXPERIMENTS_PER_CLASS):
            experiment = create_experiment(
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
    dataset = generate_dataset()

    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data"
    data_directory.mkdir(exist_ok=True)

    output_path = data_directory / "ml_dataset.csv"

    dataset.to_csv(
        output_path,
        index=False,
    )

    experiment_counts = (
        dataset[
            [
                "experiment_id",
                "class_name",
            ]
        ]
        .drop_duplicates()
        ["class_name"]
        .value_counts()
    )

    print("Датасет создан:")
    print(output_path)

    print("\nКоличество экспериментов по классам:")
    print(experiment_counts)

    print("\nКоличество строк:")
    print(len(dataset))

    print("\nКоличество уникальных экспериментов:")
    print(dataset["experiment_id"].nunique())