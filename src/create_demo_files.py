from pathlib import Path

import numpy as np

from src.generate_cooling import (
    add_high_noise,
    add_sensor_drift,
    add_single_outlier,
    generate_cooling_experiment,
)


RANDOM_SEED = 42


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data"
    data_directory.mkdir(exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    normal_experiment = generate_cooling_experiment(
        initial_temperature=85.0,
        room_temperature=22.0,
        cooling_coefficient=0.01,
        noise_std=0.25,
        random_seed=RANDOM_SEED,
    )

    single_outlier_experiment = add_single_outlier(
        normal_experiment,
        rng,
    )

    sensor_drift_experiment = add_sensor_drift(
        normal_experiment,
        rng,
    )

    high_noise_experiment = add_high_noise(
        normal_experiment,
        rng,
    )

    demo_files = {
        "normal_cooling_experiment.csv": normal_experiment,
        "single_outlier_experiment.csv": single_outlier_experiment,
        "sensor_drift_experiment.csv": sensor_drift_experiment,
        "high_noise_experiment.csv": high_noise_experiment,
    }

    for filename, experiment in demo_files.items():
        output_path = data_directory / filename

        experiment[
            [
                "time_seconds",
                "measured_temperature",
            ]
        ].to_csv(
            output_path,
            index=False,
        )

        print(f"Создан файл: {output_path}")

    print("\nДемонстрационные файлы готовы.")