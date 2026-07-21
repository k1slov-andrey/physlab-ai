from __future__ import annotations

import numpy as np
import pandas as pd
from core.schemas import ModelPrediction
from labs.common.pipeline import load_model_and_features
from labs.common.reliability import predict_with_reliability
from labs.common.realism import (
    choose_device_profile,
    choose_environment_profile,
    first_order_lag,
    generation_family,
    normalized_slope,
    quantize,
    random_walk,
    robust_scale,
    safe_correlation,
)

CLASSES = (
    "normal",
    "single_outlier",
    "sensor_drift",
    "high_noise",
)


def _simulate_body_temperature(
    time_seconds: np.ndarray,
    initial_temperature_c: float,
    environment_temperature_c: np.ndarray,
    cooling_coefficient_per_s: float,
    radiation_strength: float,
) -> np.ndarray:
    result = np.empty_like(time_seconds, dtype=float)
    result[0] = initial_temperature_c
    for index in range(1, len(time_seconds)):
        dt = float(time_seconds[index] - time_seconds[index - 1])
        body_k = result[index - 1] + 273.15
        env_k = environment_temperature_c[index] + 273.15
        convective = cooling_coefficient_per_s * (
            result[index - 1] - environment_temperature_c[index]
        )
        radiative = radiation_strength * (body_k**4 - env_k**4)
        result[index] = result[index - 1] - dt * (convective + radiative)
    return result


def simulate(
    class_name: str = "normal",
    experiment_id: str = "CL_0001",
    n_points: int = 180,
    seed: int = 42,
    group_name: str = "group_00",
) -> pd.DataFrame:
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class_name: {class_name}")

    rng = np.random.default_rng(seed)
    device = choose_device_profile(rng)
    environment = choose_environment_profile(rng)

    n_points = int(np.clip(n_points, 90, 260))
    sample_interval_s = float(rng.uniform(6.0, 14.0))
    time_seconds = np.arange(n_points, dtype=float) * sample_interval_s

    room_c = float(environment.room_temperature_c)
    environment_temperature_c = np.full(n_points, room_c, dtype=float)
    environment_temperature_c += (
        environment.temperature_drift_c_per_hour * time_seconds / 3600.0
    )

    if rng.random() < environment.disturbance_probability:
        event_index = int(rng.integers(n_points // 4, 3 * n_points // 4))
        event_amplitude = float(rng.uniform(-1.6, 1.4))
        decay_s = float(rng.uniform(80.0, 260.0))
        event_time = time_seconds[event_index:]-time_seconds[event_index]
        environment_temperature_c[event_index:] += event_amplitude * np.exp(
            -event_time / decay_s
        )

    initial_temperature_c = float(rng.uniform(62.0, 91.0))
    cooling_coefficient_per_s = float(rng.uniform(0.00065, 0.00175))
    radiation_strength = float(rng.uniform(0.8e-12, 2.2e-12))
    true_temperature_c = _simulate_body_temperature(
        time_seconds,
        initial_temperature_c,
        environment_temperature_c,
        cooling_coefficient_per_s,
        radiation_strength,
    )

    sensor_temperature_c = first_order_lag(
        true_temperature_c,
        time_seconds,
        device.temperature_lag_s,
        initial_value=true_temperature_c[0],
    )
    sensor_temperature_c += device.temperature_offset_c

    base_noise = rng.normal(0.0, device.temperature_noise_c, n_points)
    measured_temperature_c = sensor_temperature_c + base_noise

    severity = float(rng.uniform(0.02, 0.16))
    if class_name == "single_outlier":
        severity = float(rng.uniform(0.45, 1.0))
        number_of_outliers = int(rng.integers(1, 3))
        indices = rng.choice(
            np.arange(8, n_points - 8),
            size=number_of_outliers,
            replace=False,
        )
        for index in indices:
            measured_temperature_c[index] += rng.choice([-1.0, 1.0]) * rng.uniform(
                2.0,
                7.0,
            ) * severity
    elif class_name == "sensor_drift":
        severity = float(rng.uniform(0.45, 1.0))
        drift_total_c = rng.choice([-1.0, 1.0]) * rng.uniform(1.2, 4.5) * severity
        deterministic_drift = np.linspace(0.0, drift_total_c, n_points)
        stochastic_drift = random_walk(
            n_points,
            step_std=0.012 * severity,
            rng=rng,
            center=False,
        )
        measured_temperature_c += deterministic_drift + stochastic_drift
    elif class_name == "high_noise":
        severity = float(rng.uniform(0.45, 1.0))
        extra_noise_std = rng.uniform(0.35, 1.25) * severity
        measured_temperature_c += rng.normal(0.0, extra_noise_std, n_points)
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(5, n_points - 12))
            length = int(rng.integers(3, 10))
            measured_temperature_c[start:start+length] += rng.normal(
                0.0,
                extra_noise_std * 1.8,
                min(length, n_points - start),
            )

    measured_temperature_c = quantize(
        measured_temperature_c,
        device.temperature_resolution_c,
    )

    secondary_errors = []
    if device.temperature_lag_s > 8.0:
        secondary_errors.append("sensor_lag")
    if environment.name == "variable_room":
        secondary_errors.append("ambient_drift")
    if abs(device.temperature_offset_c) > 0.2:
        secondary_errors.append("sensor_offset")

    return pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "time_seconds": np.round(time_seconds, 2),
            "measured_temperature": measured_temperature_c,
            "ambient_temperature_c": quantize(
                environment_temperature_c,
                device.temperature_resolution_c,
            ),
            "true_temperature_c": np.round(true_temperature_c, 4),
            "device_profile": device.name,
            "environment_profile": environment.name,
            "generation_group": group_name,
            "secondary_errors": ";".join(secondary_errors) if secondary_errors else "none",
            "severity": severity,
            "class_name": class_name,
        }
    )


def _fit_newton_model(
    time_seconds: np.ndarray,
    measured_temperature: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    relative_time = time_seconds - time_seconds[0]
    tail_median = float(np.median(measured_temperature[-max(8, len(measured_temperature) // 8):]))
    minimum = float(np.min(measured_temperature))
    room_candidates = np.linspace(
        min(tail_median - 8.0, minimum - 0.2),
        min(tail_median + 1.5, measured_temperature[-1] - 0.05),
        28,
    )

    best_rmse = float("inf")
    best_fitted = None
    best_parameters = None

    for room in room_candidates:
        delta = measured_temperature - room
        valid = delta > 0.05
        if int(np.sum(valid)) < max(10, len(measured_temperature) // 3):
            continue
        x = relative_time[valid]
        y = np.log(delta[valid])
        slope, intercept = np.polyfit(x, y, 1)
        coefficient = float(np.clip(-slope, 0.00001, 0.02))
        initial = float(room + np.exp(intercept))
        fitted = room + (initial - room) * np.exp(-coefficient * relative_time)
        rmse = float(np.sqrt(np.mean((measured_temperature - fitted) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_fitted = fitted
            best_parameters = {
                "fitted_initial_temperature": initial,
                "fitted_room_temperature": float(room),
                "fitted_cooling_coefficient": coefficient,
            }

    if best_fitted is None or best_parameters is None:
        room = float(np.min(measured_temperature) - 0.5)
        coefficient = 0.001
        initial = float(measured_temperature[0])
        best_fitted = room + (initial - room) * np.exp(-coefficient * relative_time)
        best_parameters = {
            "fitted_initial_temperature": initial,
            "fitted_room_temperature": room,
            "fitted_cooling_coefficient": coefficient,
        }

    return best_fitted, best_parameters


def extract_features(df: pd.DataFrame) -> dict[str, float]:
    temperature_column = (
        "measured_temperature"
        if "measured_temperature" in df.columns
        else "temperature_c"
        if "temperature_c" in df.columns
        else None
    )
    if temperature_column is None or "time_seconds" not in df.columns:
        raise ValueError("Required columns: time_seconds and measured_temperature")

    clean = df.dropna(subset=["time_seconds", temperature_column]).sort_values(
        "time_seconds"
    )
    if len(clean) < 20:
        raise ValueError("At least 20 measurements are required")

    time = clean["time_seconds"].to_numpy(dtype=float)
    measured = clean[temperature_column].to_numpy(dtype=float)
    fitted, fitted_parameters = _fit_newton_model(time, measured)
    residuals = measured - fitted
    temperature_diff = np.diff(measured)
    time_diff = np.diff(time)
    rates = temperature_diff / np.clip(time_diff, 1e-9, None)

    midpoint = len(residuals) // 2
    denominator = float(np.sum((measured - np.mean(measured)) ** 2))
    fit_r2 = 0.0 if denominator < 1e-12 else float(
        1.0 - np.sum(residuals**2) / denominator
    )

    outlier_threshold = max(0.45, 4.5 * robust_scale(residuals))
    outlier_fraction = float(np.mean(np.abs(residuals) > outlier_threshold))

    return {
        "mean_temperature": float(np.mean(measured)),
        "std_temperature": float(np.std(measured)),
        "temperature_range": float(np.ptp(measured)),
        "max_abs_jump": float(np.max(np.abs(temperature_diff))),
        "mean_abs_jump": float(np.mean(np.abs(temperature_diff))),
        "max_abs_rate": float(np.max(np.abs(rates))),
        "mean_abs_rate": float(np.mean(np.abs(rates))),
        "fit_rmse": float(np.sqrt(np.mean(residuals**2))),
        "fit_r2": fit_r2,
        "residual_std": float(np.std(residuals)),
        "residual_robust_scale": robust_scale(residuals),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "residual_normalized_slope": normalized_slope(residuals),
        "residual_time_corr": safe_correlation(residuals, time),
        "early_residual_mean": float(np.mean(residuals[:midpoint])),
        "late_residual_mean": float(np.mean(residuals[midpoint:])),
        "residual_mean_change": float(
            np.mean(residuals[midpoint:]) - np.mean(residuals[:midpoint])
        ),
        "outlier_fraction": outlier_fraction,
        "second_difference_scale": robust_scale(np.diff(measured, n=2)),
        "average_cooling_rate": float(
            (measured[-1] - measured[0]) / (time[-1] - time[0] + 1e-9)
        ),
        **fitted_parameters,
    }


def generate_dataset(
    n_per_class: int = 220,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_per_class < 1:
        raise ValueError("n_per_class must be at least 1")

    rng = np.random.default_rng(seed)
    raw_parts: list[pd.DataFrame] = []
    feature_rows: list[dict] = []
    counter = 0

    for family_index in range(n_per_class):
        family_seed = int(rng.integers(0, 2_000_000_000))
        n_points = int(rng.integers(120, 230))
        group_name = generation_family("cooling", family_index)

        for class_name in CLASSES:
            counter += 1
            experiment_id = f"CL_{counter:05d}"
            experiment = simulate(
                class_name=class_name,
                experiment_id=experiment_id,
                n_points=n_points,
                seed=family_seed,
                group_name=group_name,
            )
            raw_parts.append(experiment)
            features = extract_features(experiment)
            features.update(
                {
                    "experiment_id": experiment_id,
                    "class_name": class_name,
                    "generation_group": group_name,
                    "device_profile": str(experiment["device_profile"].iloc[0]),
                    "environment_profile": str(
                        experiment["environment_profile"].iloc[0]
                    ),
                    "severity": float(experiment["severity"].iloc[0]),
                }
            )
            feature_rows.append(features)

    return pd.concat(raw_parts, ignore_index=True), pd.DataFrame(feature_rows)


def predict(df: pd.DataFrame) -> ModelPrediction:
    model, feature_names = load_model_and_features("cooling")
    features = extract_features(df)
    return predict_with_reliability(
        lab_id="cooling",
        model=model,
        feature_names=feature_names,
        features=features,
    )
