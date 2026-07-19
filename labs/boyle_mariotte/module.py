from __future__ import annotations

import numpy as np
import pandas as pd

from core.schemas import ModelPrediction
from labs.common.pipeline import load_model_and_features
from labs.common.realism import (
    choose_device_profile,
    choose_environment_profile,
    generation_group,
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
    "temperature_change",
    "volume_measurement_error",
)


def simulate(
    class_name: str = "normal",
    experiment_id: str = "BM_0001",
    n_points: int = 28,
    seed: int = 42,
    group_name: str = "group_00",
) -> pd.DataFrame:
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class_name: {class_name}")

    rng = np.random.default_rng(seed)
    device = choose_device_profile(rng)
    environment = choose_environment_profile(rng)

    n_points = int(np.clip(n_points, 18, 40))
    measurement_number = np.arange(1, n_points + 1)
    wait_time_s = rng.uniform(5.0, 14.0, n_points)
    if class_name == "temperature_change":
        wait_time_s *= rng.uniform(0.18, 0.45)
    time_seconds = np.cumsum(wait_time_s) - wait_time_s[0]

    volume_max_ml = rng.uniform(70.0, 100.0)
    volume_min_ml = rng.uniform(22.0, 38.0)
    if rng.random() < 0.82:
        syringe_volume_true_ml = np.linspace(volume_max_ml, volume_min_ml, n_points)
    else:
        syringe_volume_true_ml = np.linspace(volume_min_ml, volume_max_ml, n_points)

    dead_volume_ml = float(rng.uniform(1.2, 5.5))
    true_gas_volume_ml = syringe_volume_true_ml + dead_volume_ml

    room_temperature_c = float(environment.room_temperature_c)
    room_temperature_k = room_temperature_c + 273.15
    atmospheric_pressure_kpa = float(environment.atmospheric_pressure_kpa)
    initial_absolute_pressure_kpa = atmospheric_pressure_kpa + rng.normal(0.0, 0.35)

    initial_volume_m3 = true_gas_volume_ml[0] * 1e-6
    amount_mol = initial_absolute_pressure_kpa * 1000.0 * initial_volume_m3 / (
        R_GAS * room_temperature_k
    )

    compression_ratio = true_gas_volume_ml[0] / true_gas_volume_ml
    adiabatic_temperature_k = room_temperature_k * np.power(compression_ratio, 0.4)
    thermal_relaxation = 1.0 - np.exp(-wait_time_s / rng.uniform(5.0, 11.0))
    true_temperature_k = adiabatic_temperature_k + thermal_relaxation * (
        room_temperature_k - adiabatic_temperature_k
    )

    if class_name != "temperature_change":
        true_temperature_k = room_temperature_k + (
            true_temperature_k - room_temperature_k
        ) * rng.uniform(0.05, 0.22)
    else:
        imposed_drift_k = np.linspace(0.0, rng.uniform(2.5, 8.0), n_points)
        true_temperature_k = true_temperature_k + imposed_drift_k

    true_temperature_k += (
        environment.temperature_drift_c_per_hour * time_seconds / 3600.0
    )

    amount_series = np.full(n_points, amount_mol, dtype=float)
    leak_rate_per_s = 0.0
    if class_name == "air_leak":
        leak_rate_per_s = float(rng.uniform(0.00028, 0.00115))
        amount_series = amount_mol * np.exp(-leak_rate_per_s * time_seconds)

    true_absolute_pressure_kpa = (
        amount_series
        * R_GAS
        * true_temperature_k
        / (true_gas_volume_ml * 1e-6)
        / 1000.0
    )
    true_gauge_pressure_kpa = true_absolute_pressure_kpa - atmospheric_pressure_kpa

    minutes = time_seconds / 60.0
    pressure_offset_kpa = device.pressure_offset_kpa + rng.normal(0.0, 0.12)
    pressure_drift = device.pressure_drift_kpa_per_min * minutes
    pressure_noise = rng.normal(0.0, device.pressure_noise_kpa, n_points)
    pressure_measured_kpa = (
        true_absolute_pressure_kpa
        + pressure_offset_kpa
        + pressure_drift
        + pressure_noise
    )
    pressure_measured_kpa = quantize(
        pressure_measured_kpa,
        device.pressure_resolution_kpa,
    )

    volume_measured_ml = (
        syringe_volume_true_ml
        + device.volume_offset_ml
        + rng.normal(0.0, device.volume_noise_ml, n_points)
    )
    volume_error_severity = 0.0
    if class_name == "volume_measurement_error":
        volume_error_severity = float(rng.uniform(0.45, 1.0))
        number_of_errors = int(rng.integers(1, 3))
        error_indices = rng.choice(
            np.arange(2, n_points - 2),
            size=number_of_errors,
            replace=False,
        )
        for index in error_indices:
            volume_measured_ml[index] += rng.choice([-1.0, 1.0]) * rng.uniform(
                3.5,
                9.5,
            ) * volume_error_severity
    volume_measured_ml = quantize(
        volume_measured_ml,
        device.volume_resolution_ml,
    )

    temperature_measured_c = (
        true_temperature_k
        - 273.15
        + device.temperature_offset_c
        + rng.normal(0.0, device.temperature_noise_c, n_points)
    )
    temperature_measured_c = quantize(
        temperature_measured_c,
        device.temperature_resolution_c,
    )

    if class_name == "air_leak":
        severity = min(1.0, leak_rate_per_s / 0.00115)
    elif class_name == "temperature_change":
        severity = min(
            1.0,
            float(np.ptp(true_temperature_k)) / 8.0,
        )
    elif class_name == "volume_measurement_error":
        severity = volume_error_severity
    else:
        severity = float(rng.uniform(0.02, 0.18))

    secondary_errors = []
    if dead_volume_ml > 4.2:
        secondary_errors.append("large_dead_volume")
    if abs(pressure_offset_kpa) > 0.45:
        secondary_errors.append("pressure_zero_offset")
    if np.ptp(true_temperature_k) > 1.2 and class_name != "temperature_change":
        secondary_errors.append("small_temperature_drift")

    return pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "measurement_number": measurement_number,
            "time_seconds": np.round(time_seconds, 2),
            "volume_ml": volume_measured_ml,
            "pressure_kpa": pressure_measured_kpa,
            "temperature_c": temperature_measured_c,
            "atmospheric_pressure_kpa": atmospheric_pressure_kpa,
            "pressure_gauge_kpa": quantize(
                pressure_measured_kpa - atmospheric_pressure_kpa,
                device.pressure_resolution_kpa,
            ),
            "true_volume_ml": np.round(true_gas_volume_ml, 4),
            "true_pressure_absolute_kpa": np.round(true_absolute_pressure_kpa, 4),
            "true_temperature_c": np.round(true_temperature_k - 273.15, 4),
            "dead_volume_ml": dead_volume_ml,
            "device_profile": device.name,
            "environment_profile": environment.name,
            "generation_group": group_name,
            "secondary_errors": ";".join(secondary_errors) if secondary_errors else "none",
            "severity": severity,
            "class_name": class_name,
        }
    )


def extract_features(df: pd.DataFrame) -> dict[str, float]:
    required = {"volume_ml", "pressure_kpa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    clean = df.dropna(subset=["volume_ml", "pressure_kpa"]).copy()
    if len(clean) < 8:
        raise ValueError("At least 8 measurements are required")

    volume = clean["volume_ml"].to_numpy(dtype=float)
    pressure = clean["pressure_kpa"].to_numpy(dtype=float)
    temperature = (
        clean["temperature_c"].to_numpy(dtype=float)
        if "temperature_c" in clean.columns
        else np.full(len(clean), 22.0)
    )
    atmospheric = (
        clean["atmospheric_pressure_kpa"].to_numpy(dtype=float)
        if "atmospheric_pressure_kpa" in clean.columns
        else np.full(len(clean), 101.325)
    )

    if float(np.nanmedian(pressure)) < 60.0:
        absolute_pressure = pressure + atmospheric
    else:
        absolute_pressure = pressure

    pv = absolute_pressure * volume
    inverse_volume = 1.0 / np.clip(volume, 1e-6, None)
    coefficients = np.polyfit(inverse_volume, absolute_pressure, 1)
    fitted_pressure = np.polyval(coefficients, inverse_volume)
    pressure_residual = absolute_pressure - fitted_pressure
    pv_centered = pv / (np.mean(pv) + 1e-9)

    volume_diff = np.diff(volume)
    pressure_diff = np.diff(absolute_pressure)
    temperature_diff = np.diff(temperature)

    return {
        "pressure_mean": float(np.mean(absolute_pressure)),
        "pressure_std": float(np.std(absolute_pressure)),
        "pressure_range": float(np.ptp(absolute_pressure)),
        "volume_mean": float(np.mean(volume)),
        "volume_std": float(np.std(volume)),
        "volume_range": float(np.ptp(volume)),
        "temperature_mean": float(np.mean(temperature)),
        "temperature_std": float(np.std(temperature)),
        "temperature_range": float(np.ptp(temperature)),
        "temperature_normalized_slope": normalized_slope(temperature),
        "pv_mean": float(np.mean(pv)),
        "pv_std": float(np.std(pv)),
        "pv_robust_scale": robust_scale(pv),
        "pv_cv": float(np.std(pv) / (abs(np.mean(pv)) + 1e-9)),
        "pv_range_relative": float(np.ptp(pv) / (abs(np.mean(pv)) + 1e-9)),
        "pv_normalized_slope": normalized_slope(pv),
        "pv_first_last_change": float(
            (pv[-1] - pv[0]) / (abs(pv[0]) + 1e-9)
        ),
        "pressure_volume_corr": safe_correlation(absolute_pressure, volume),
        "pressure_inverse_volume_corr": safe_correlation(
            absolute_pressure,
            inverse_volume,
        ),
        "inverse_fit_r2": linear_fit_r2(inverse_volume, absolute_pressure),
        "inverse_fit_rmse_relative": float(
            np.sqrt(np.mean(pressure_residual**2))
            / (abs(np.mean(absolute_pressure)) + 1e-9)
        ),
        "max_abs_pressure_residual_relative": float(
            np.max(np.abs(pressure_residual))
            / (abs(np.mean(absolute_pressure)) + 1e-9)
        ),
        "pressure_residual_robust_scale": robust_scale(pressure_residual),
        "pv_temperature_corr": safe_correlation(pv_centered, temperature),
        "max_volume_jump_ratio": float(
            np.max(np.abs(volume_diff)) / (abs(np.median(volume_diff)) + 1e-6)
        ),
        "max_pressure_jump_ratio": float(
            np.max(np.abs(pressure_diff))
            / (abs(np.median(pressure_diff)) + 1e-6)
        ),
        "max_temperature_jump": float(
            np.max(np.abs(temperature_diff)) if len(temperature_diff) else 0.0
        ),
    }


def generate_dataset(
    n_per_class: int = 220,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    raw_parts: list[pd.DataFrame] = []
    feature_rows: list[dict] = []
    counter = 0

    for class_name in CLASSES:
        for local_index in range(n_per_class):
            counter += 1
            experiment_id = f"BM_{counter:05d}"
            group_name = generation_group(local_index)
            experiment = simulate(
                class_name=class_name,
                experiment_id=experiment_id,
                n_points=int(rng.integers(20, 36)),
                seed=int(rng.integers(0, 2_000_000_000)),
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
                    "severity": float(experiment["severity"].iloc[0]),
                }
            )
            feature_rows.append(features)

    return pd.concat(raw_parts, ignore_index=True), pd.DataFrame(feature_rows)


def predict(df: pd.DataFrame) -> ModelPrediction:
    model, feature_names = load_model_and_features("boyle_mariotte")
    features = extract_features(df)
    matrix = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )
    probabilities = model.predict_proba(matrix)[0]
    classes = model.classes_
    best_index = int(np.argmax(probabilities))
    return ModelPrediction(
        lab_id="boyle_mariotte",
        predicted_class=str(classes[best_index]),
        confidence=float(probabilities[best_index]),
        probabilities={
            str(class_name): float(probability)
            for class_name, probability in zip(classes, probabilities)
        },
        features=features,
    )
