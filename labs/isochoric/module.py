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
    linear_fit_r2,
    normalized_slope,
    quantize,
    robust_scale,
    safe_correlation,
)

R_GAS = 8.314462618
CLASSES = (
    "normal",
    "air_leak",
    "volume_instability",
    "temperature_sensor_lag",
)


def simulate(
    class_name: str = "normal",
    experiment_id: str = "IS_0001",
    n_points: int = 52,
    seed: int = 42,
    group_name: str = "group_00",
) -> pd.DataFrame:
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class_name: {class_name}")

    rng = np.random.default_rng(seed)
    device = choose_device_profile(rng)
    environment = choose_environment_profile(rng)

    n_points = int(np.clip(n_points, 36, 75))
    duration_s = float(rng.uniform(420.0, 760.0))
    time_seconds = np.linspace(0.0, duration_s, n_points)

    room_c = float(environment.room_temperature_c)
    bath_final_c = float(rng.uniform(65.0, 88.0))
    bath_tau_s = float(rng.uniform(120.0, 240.0))
    bath_temperature_c = room_c + (bath_final_c - room_c) * (
        1.0 - np.exp(-time_seconds / bath_tau_s)
    )
    bath_temperature_c += (
        environment.temperature_drift_c_per_hour * time_seconds / 3600.0
    )

    gas_tau_s = float(rng.uniform(28.0, 65.0))
    true_gas_temperature_c = first_order_lag(
        bath_temperature_c,
        time_seconds,
        gas_tau_s,
        initial_value=room_c,
    )

    nominal_volume_ml = float(rng.uniform(180.0, 360.0))
    true_volume_ml = np.full(n_points, nominal_volume_ml, dtype=float)
    volume_change_fraction = 0.0
    if class_name == "volume_instability":
        volume_change_fraction = float(rng.uniform(0.018, 0.075))
        direction = rng.choice([-1.0, 1.0])
        true_volume_ml *= 1.0 + direction * volume_change_fraction * (
            time_seconds / duration_s
        ) ** rng.uniform(0.8, 1.4)
        true_volume_ml += rng.normal(0.0, nominal_volume_ml * 0.0015, n_points)

    atmospheric_pressure_kpa = float(environment.atmospheric_pressure_kpa)
    initial_pressure_kpa = atmospheric_pressure_kpa + rng.normal(0.0, 0.25)
    initial_temperature_k = true_gas_temperature_c[0] + 273.15
    amount_mol = (
        initial_pressure_kpa
        * 1000.0
        * nominal_volume_ml
        * 1e-6
        / (R_GAS * initial_temperature_k)
    )

    leak_rate_per_s = 0.0
    amount_series = np.full(n_points, amount_mol, dtype=float)
    if class_name == "air_leak":
        leak_rate_per_s = float(rng.uniform(0.000035, 0.00028))
        amount_series = amount_mol * np.exp(-leak_rate_per_s * time_seconds)

    true_pressure_kpa = (
        amount_series
        * R_GAS
        * (true_gas_temperature_c + 273.15)
        / (true_volume_ml * 1e-6)
        / 1000.0
    )

    pressure_measured_kpa = (
        true_pressure_kpa
        + device.pressure_offset_kpa
        + device.pressure_drift_kpa_per_min * time_seconds / 60.0
        + rng.normal(0.0, device.pressure_noise_kpa, n_points)
    )
    pressure_measured_kpa = quantize(
        pressure_measured_kpa,
        device.pressure_resolution_kpa,
    )

    sensor_tau_s = device.temperature_lag_s * rng.uniform(0.8, 1.3)
    if class_name == "temperature_sensor_lag":
        sensor_tau_s += float(rng.uniform(12.0, 52.0))
    temperature_sensor_c = first_order_lag(
        true_gas_temperature_c,
        time_seconds,
        sensor_tau_s,
        initial_value=room_c + device.temperature_offset_c,
    )
    temperature_sensor_c += device.temperature_offset_c
    temperature_sensor_c += rng.normal(0.0, device.temperature_noise_c, n_points)
    temperature_sensor_c = quantize(
        temperature_sensor_c,
        device.temperature_resolution_c,
    )

    measured_volume_ml = (
        true_volume_ml
        + device.volume_offset_ml
        + rng.normal(0.0, device.volume_noise_ml, n_points)
    )
    measured_volume_ml = quantize(
        measured_volume_ml,
        device.volume_resolution_ml,
    )

    if class_name == "air_leak":
        severity = min(1.0, leak_rate_per_s / 0.00028)
    elif class_name == "volume_instability":
        severity = min(1.0, volume_change_fraction / 0.075)
    elif class_name == "temperature_sensor_lag":
        severity = min(1.0, sensor_tau_s / 65.0)
    else:
        severity = float(rng.uniform(0.02, 0.18))

    secondary_errors = []
    if abs(device.pressure_offset_kpa) > 0.45:
        secondary_errors.append("pressure_zero_offset")
    if device.temperature_lag_s > 8.0 and class_name != "temperature_sensor_lag":
        secondary_errors.append("small_sensor_lag")
    if environment.name == "variable_room":
        secondary_errors.append("room_temperature_drift")

    return pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "time_seconds": np.round(time_seconds, 2),
            "temperature_c": temperature_sensor_c,
            "pressure_kpa": pressure_measured_kpa,
            "volume_ml": measured_volume_ml,
            "bath_temperature_c": quantize(
                bath_temperature_c,
                device.temperature_resolution_c,
            ),
            "atmospheric_pressure_kpa": atmospheric_pressure_kpa,
            "true_temperature_c": np.round(true_gas_temperature_c, 4),
            "true_pressure_absolute_kpa": np.round(true_pressure_kpa, 4),
            "true_volume_ml": np.round(true_volume_ml, 4),
            "device_profile": device.name,
            "environment_profile": environment.name,
            "generation_group": group_name,
            "secondary_errors": ";".join(secondary_errors) if secondary_errors else "none",
            "severity": severity,
            "class_name": class_name,
        }
    )


def _max_lagged_correlation(
    pressure: np.ndarray,
    temperature: np.ndarray,
    max_lag: int = 8,
) -> tuple[float, int]:
    best_correlation = -1.0
    best_lag = 0
    for lag in range(max_lag + 1):
        if lag == 0:
            p_part = pressure
            t_part = temperature
        else:
            p_part = pressure[lag:]
            t_part = temperature[:-lag]
        if len(p_part) < 5:
            continue
        correlation = safe_correlation(p_part, t_part)
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return float(best_correlation), int(best_lag)


def extract_features(df: pd.DataFrame) -> dict[str, float]:
    required = {"temperature_c", "pressure_kpa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    clean = df.dropna(subset=["temperature_c", "pressure_kpa"]).copy()
    if len(clean) < 10:
        raise ValueError("At least 10 measurements are required")

    temperature_c = clean["temperature_c"].to_numpy(dtype=float)
    temperature_k = temperature_c + 273.15
    pressure = clean["pressure_kpa"].to_numpy(dtype=float)
    volume = (
        clean["volume_ml"].to_numpy(dtype=float)
        if "volume_ml" in clean.columns
        else np.full(len(clean), 250.0)
    )
    time = (
        clean["time_seconds"].to_numpy(dtype=float)
        if "time_seconds" in clean.columns
        else np.arange(len(clean), dtype=float)
    )

    ratio = pressure / np.clip(temperature_k, 1e-6, None)
    coefficients = np.polyfit(temperature_k, pressure, 1)
    fitted_pressure = np.polyval(coefficients, temperature_k)
    residuals = pressure - fitted_pressure
    lagged_corr, best_lag = _max_lagged_correlation(pressure, temperature_c)

    pressure_diff = np.diff(pressure)
    temperature_diff = np.diff(temperature_c)
    volume_diff = np.diff(volume)

    return {
        "pressure_mean": float(np.mean(pressure)),
        "pressure_std": float(np.std(pressure)),
        "pressure_range": float(np.ptp(pressure)),
        "temperature_mean": float(np.mean(temperature_c)),
        "temperature_std": float(np.std(temperature_c)),
        "temperature_range": float(np.ptp(temperature_c)),
        "volume_mean": float(np.mean(volume)),
        "volume_std": float(np.std(volume)),
        "volume_range_relative": float(np.ptp(volume) / (abs(np.mean(volume)) + 1e-9)),
        "volume_normalized_slope": normalized_slope(volume),
        "pt_ratio_mean": float(np.mean(ratio)),
        "pt_ratio_std": float(np.std(ratio)),
        "pt_ratio_robust_scale": robust_scale(ratio),
        "pt_ratio_cv": float(np.std(ratio) / (abs(np.mean(ratio)) + 1e-9)),
        "pt_ratio_normalized_slope": normalized_slope(ratio),
        "pt_ratio_first_last_change": float(
            (ratio[-1] - ratio[0]) / (abs(ratio[0]) + 1e-9)
        ),
        "pressure_temperature_corr": safe_correlation(pressure, temperature_c),
        "linear_fit_r2": linear_fit_r2(temperature_k, pressure),
        "fit_rmse_relative": float(
            np.sqrt(np.mean(residuals**2)) / (abs(np.mean(pressure)) + 1e-9)
        ),
        "residual_normalized_slope": normalized_slope(residuals),
        "residual_robust_scale": robust_scale(residuals),
        "lagged_corr_max": lagged_corr,
        "best_lag_fraction": float(best_lag / max(len(clean) - 1, 1)),
        "max_pressure_jump_relative": float(
            np.max(np.abs(pressure_diff)) / (abs(np.mean(pressure)) + 1e-9)
        ),
        "max_temperature_jump_relative": float(
            np.max(np.abs(temperature_diff)) / (abs(np.mean(temperature_c)) + 1e-9)
        ),
        "max_volume_jump_relative": float(
            np.max(np.abs(volume_diff)) / (abs(np.mean(volume)) + 1e-9)
        ),
        "heating_duration_s": float(time[-1] - time[0]),
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
        n_points = int(rng.integers(40, 68))
        group_name = generation_family("isochoric", family_index)

        for class_name in CLASSES:
            counter += 1
            experiment_id = f"IS_{counter:05d}"
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
    model, feature_names = load_model_and_features("isochoric")
    features = extract_features(df)
    return predict_with_reliability(
        lab_id="isochoric",
        model=model,
        feature_names=feature_names,
        features=features,
    )
