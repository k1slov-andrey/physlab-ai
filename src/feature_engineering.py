from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def newton_cooling_model(
    time: np.ndarray,
    initial_temperature: float,
    room_temperature: float,
    cooling_coefficient: float,
) -> np.ndarray:
    """Рассчитывает температуру по закону охлаждения Ньютона."""

    relative_time = time - time[0]

    return room_temperature + (
        initial_temperature - room_temperature
    ) * np.exp(-cooling_coefficient * relative_time)


def fit_cooling_model(
    time: np.ndarray,
    measured: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Подбирает параметры физической модели по измеренным данным.

    Используется робастная оптимизация, чтобы отдельные выбросы
    меньше влияли на итоговую аппроксимацию.
    """

    initial_guess = np.array(
        [
            measured[0],
            np.median(measured[-10:]),
            0.01,
        ],
        dtype=float,
    )

    lower_bounds = np.array(
        [
            measured.min() - 10.0,
            -20.0,
            0.00001,
        ]
    )

    upper_bounds = np.array(
        [
            measured.max() + 10.0,
            measured.max() + 10.0,
            0.10,
        ]
    )

    def residual_function(
        parameters: np.ndarray,
    ) -> np.ndarray:
        predicted = newton_cooling_model(
            time=time,
            initial_temperature=parameters[0],
            room_temperature=parameters[1],
            cooling_coefficient=parameters[2],
        )

        return measured - predicted

    result = least_squares(
        residual_function,
        x0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=0.5,
        max_nfev=3000,
    )

    fitted_parameters = result.x

    fitted_temperature = newton_cooling_model(
        time=time,
        initial_temperature=fitted_parameters[0],
        room_temperature=fitted_parameters[1],
        cooling_coefficient=fitted_parameters[2],
    )

    parameter_values = {
        "fitted_initial_temperature": float(
            fitted_parameters[0]
        ),
        "fitted_room_temperature": float(
            fitted_parameters[1]
        ),
        "fitted_cooling_coefficient": float(
            fitted_parameters[2]
        ),
    }

    return fitted_temperature, parameter_values


def extract_features_from_experiment(
    experiment: pd.DataFrame,
) -> dict:
    """Преобразует один временной ряд в набор числовых признаков."""

    required_columns = {
        "time_seconds",
        "measured_temperature",
    }

    missing_columns = required_columns - set(experiment.columns)

    if missing_columns:
        raise ValueError(
            "Отсутствуют обязательные столбцы: "
            + ", ".join(sorted(missing_columns))
        )

    clean_experiment = (
        experiment[
            [
                "time_seconds",
                "measured_temperature",
            ]
        ]
        .dropna()
        .sort_values("time_seconds")
    )

    if len(clean_experiment) < 10:
        raise ValueError(
            "Для анализа требуется минимум 10 измерений."
        )

    time = clean_experiment[
        "time_seconds"
    ].to_numpy(dtype=float)

    measured = clean_experiment[
        "measured_temperature"
    ].to_numpy(dtype=float)

    if np.any(np.diff(time) <= 0):
        raise ValueError(
            "Значения времени должны строго возрастать."
        )

    fitted_temperature, fitted_parameters = fit_cooling_model(
        time=time,
        measured=measured,
    )

    temperature_diff = np.diff(measured)
    time_diff = np.diff(time)

    temperature_rate = temperature_diff / time_diff
    residuals = measured - fitted_temperature

    duration = time[-1] - time[0]

    residual_sum_of_squares = float(
        np.sum(residuals ** 2)
    )

    total_sum_of_squares = float(
        np.sum(
            (
                measured
                - np.mean(measured)
            ) ** 2
        )
    )

    if total_sum_of_squares > 0:
        fit_r2 = (
            1.0
            - residual_sum_of_squares
            / total_sum_of_squares
        )
    else:
        fit_r2 = 0.0

    midpoint = len(residuals) // 2

    experiment_id = (
        int(experiment["experiment_id"].iloc[0])
        if "experiment_id" in experiment.columns
        else 0
    )

    class_name = (
        experiment["class_name"].iloc[0]
        if "class_name" in experiment.columns
        else "unknown"
    )

    features = {
        "experiment_id": experiment_id,
        "mean_temperature": float(
            np.mean(measured)
        ),
        "std_temperature": float(
            np.std(measured)
        ),
        "max_temperature": float(
            np.max(measured)
        ),
        "min_temperature": float(
            np.min(measured)
        ),
        "temperature_range": float(
            np.max(measured)
            - np.min(measured)
        ),
        "max_abs_jump": float(
            np.max(
                np.abs(temperature_diff)
            )
        ),
        "mean_abs_jump": float(
            np.mean(
                np.abs(temperature_diff)
            )
        ),
        "max_abs_rate": float(
            np.max(
                np.abs(temperature_rate)
            )
        ),
        "mean_residual": float(
            np.mean(residuals)
        ),
        "mean_abs_residual": float(
            np.mean(
                np.abs(residuals)
            )
        ),
        "max_abs_residual": float(
            np.max(
                np.abs(residuals)
            )
        ),
        "std_residual": float(
            np.std(residuals)
        ),
        "residual_slope": float(
            np.polyfit(
                time,
                residuals,
                1,
            )[0]
        ),
        "early_residual_mean": float(
            np.mean(
                residuals[:midpoint]
            )
        ),
        "late_residual_mean": float(
            np.mean(
                residuals[midpoint:]
            )
        ),
        "residual_mean_change": float(
            np.mean(
                residuals[midpoint:]
            )
            - np.mean(
                residuals[:midpoint]
            )
        ),
        "fit_rmse": float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        ),
        "fit_r2": float(fit_r2),
        "average_cooling_rate": float(
            (
                measured[-1]
                - measured[0]
            )
            / duration
        ),
        **fitted_parameters,
        "class_name": class_name,
    }

    return features


def create_feature_table(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Создает таблицу: одна строка — один эксперимент."""

    feature_rows = []

    for _, experiment in dataset.groupby(
        "experiment_id"
    ):
        features = extract_features_from_experiment(
            experiment
        )

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root
        / "data"
        / "ml_dataset.csv"
    )

    output_path = (
        project_root
        / "data"
        / "features.csv"
    )

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
    print(
        feature_table[
            "class_name"
        ].value_counts()
    )

    print("\nИспользование ideal_temperature:")
    print("Нет — модель оценивается по измеренным данным.")

    print("\nПервые строки:")
    print(feature_table.head())