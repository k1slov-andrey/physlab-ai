from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_cooling_experiment() -> pd.DataFrame:
    """Создает синтетические данные охлаждения жидкости."""

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

experiment = generate_cooling_experiment()
project_root = Path(__file__).resolve().parent.parent
data_directory = project_root / "data"
data_directory.mkdir(exist_ok=True)

experiment.to_csv(
    data_directory / "normal_cooling_experiment.csv",
    index=False,
)

plt.plot(
    experiment["time_seconds"],
    experiment["measured_temperature"],
    label="Измеренная температура",
)

plt.plot(
    experiment["time_seconds"],
    experiment["ideal_temperature"],
    label="Физическая модель",
)

plt.xlabel("Время, с")
plt.ylabel("Температура, °C")
plt.title("Охлаждение жидкости")
plt.legend()
plt.grid()
plt.show()