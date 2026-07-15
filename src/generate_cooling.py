from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_cooling_experiment(
    initial_temperature: float = 85.0,
    room_temperature: float = 22.0,
    cooling_coefficient: float = 0.01,
    duration_seconds: int = 600,
    measurement_interval: int = 5,
    noise_std: float = 0.25,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Создает синтетический временной ряд охлаждения жидкости."""

    rng = np.random.default_rng(random_seed)

    time = np.arange(
        0,
        duration_seconds + measurement_interval,
        measurement_interval,
    )

    ideal_temperature = room_temperature + (
        initial_temperature - room_temperature
    ) * np.exp(-cooling_coefficient * time)

    measured_temperature = ideal_temperature + rng.normal(
        loc=0.0,
        scale=noise_std,
        size=len(time),
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
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Добавляет один резкий выброс датчика."""

    result = experiment.copy()

    outlier_index = int(
        rng.integers(
            low=15,
            high=len(result) - 15,
        )
    )

    outlier_value = float(
        rng.choice([-1, 1]) * rng.uniform(4.0, 9.0)
    )

    result.loc[
        outlier_index,
        "measured_temperature",
    ] += outlier_value

    return result


def add_sensor_drift(
    experiment: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Добавляет постепенный дрейф показаний датчика."""

    result = experiment.copy()

    drift_start_index = int(
        rng.integers(
            low=35,
            high=75,
        )
    )

    drift_amplitude = float(
        rng.uniform(2.5, 6.0)
    )

    drift = np.zeros(len(result))

    drift[drift_start_index:] = np.linspace(
        0.0,
        drift_amplitude,
        len(result) - drift_start_index,
    )

    result["measured_temperature"] += drift

    return result


def add_high_noise(
    experiment: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Добавляет повышенный случайный шум датчика."""

    result = experiment.copy()

    additional_noise = rng.normal(
        loc=0.0,
        scale=rng.uniform(0.8, 1.8),
        size=len(result),
    )

    result["measured_temperature"] += additional_noise

    return result


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    experiment = generate_cooling_experiment(
        random_seed=42,
    )

    experiment_with_outlier = add_single_outlier(
        experiment,
        rng,
    )

    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data"
    data_directory.mkdir(exist_ok=True)

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

    plt.xlabel("Время, с")
    plt.ylabel("Температура, °C")
    plt.title("Охлаждение жидкости с единичным выбросом")
    plt.legend()
    plt.grid()
    plt.show()