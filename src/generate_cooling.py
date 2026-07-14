from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_cooling_experiment() -> pd.DataFrame:
    """Создает нормальный эксперимент охлаждения жидкости."""

    initial_temperature = 85.0
    room_temperature = 22.0
    cooling_coefficient = 0.01

    time = np.arange(0, 601, 5)

    ideal_temperature = room_temperature + (
        initial_temperature - room_temperature
    ) * np.exp(-cooling_coefficient * time)

    rng = np.random.default_rng(42)

    measured_temperature = ideal_temperature + rng.normal(
        0,
        0.25,
        len(time),
    )

    return pd.DataFrame(
        {
            "time_seconds": time,
            "ideal_temperature": ideal_temperature,
            "measured_temperature": measured_temperature,
        }
    )


def add_single_outlier(
    experiment: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Добавляет один резкий ошибочный скачок температуры."""

    result = experiment.copy()
    outlier_index = 45

    result.loc[
        outlier_index,
        "measured_temperature",
    ] += 7.0

    result["is_anomaly"] = 0
    result.loc[outlier_index, "is_anomaly"] = 1

    return result, outlier_index


if __name__ == "__main__":
    normal_experiment = generate_cooling_experiment()

    experiment_with_outlier, outlier_index = add_single_outlier(
        normal_experiment
    )

    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data"
    data_directory.mkdir(exist_ok=True)

    normal_experiment.to_csv(
        data_directory / "normal_cooling_experiment.csv",
        index=False,
    )

    experiment_with_outlier.to_csv(
        data_directory / "single_outlier_experiment.csv",
        index=False,
    )

    plt.plot(
        experiment_with_outlier["time_seconds"],
        experiment_with_outlier["measured_temperature"],
        label="Показания датчика",
    )

    plt.plot(
        experiment_with_outlier["time_seconds"],
        experiment_with_outlier["ideal_temperature"],
        label="Физическая модель",
    )

    plt.scatter(
        experiment_with_outlier.loc[outlier_index, "time_seconds"],
        experiment_with_outlier.loc[
            outlier_index,
            "measured_temperature",
        ],
        label="Единичный выброс",
        s=80,
    )

    plt.xlabel("Время, с")
    plt.ylabel("Температура, °C")
    plt.title("Охлаждение жидкости с ошибкой датчика")
    plt.legend()
    plt.grid()
    plt.show()