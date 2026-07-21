from __future__ import annotations

import numpy as np
import pandas as pd

from core.schemas import ModelPrediction
from labs.common.materials import MATERIALS, WATER_SPECIFIC_HEAT_J_KG_K
from labs.common.pipeline import load_model_and_features
from labs.common.reliability import predict_with_reliability
from labs.common.realism import (
    choose_device_profile,
    choose_environment_profile,
    first_order_lag,
    generation_family,
    normalized_slope,
    quantize,
    robust_scale,
    safe_correlation,
)

CLASSES = (
    "normal",
    "heat_loss",
    "mass_measurement_error",
    "insufficient_mixing",
)


def simulate(
    class_name: str = "normal",
    experiment_id: str = "HB_0001",
    n_points: int = 90,
    seed: int = 42,
    group_name: str = "group_00",
) -> pd.DataFrame:
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class_name: {class_name}")

    rng = np.random.default_rng(seed)
    device = choose_device_profile(rng)
    environment = choose_environment_profile(rng)

    n_points = int(np.clip(n_points, 60, 140))
    duration_s = float(rng.uniform(260.0, 520.0))
    time_seconds = np.linspace(0.0, duration_s, n_points)

    material_key = str(rng.choice(list(MATERIALS)))
    material = MATERIALS[material_key]
    true_specific_heat = float(
        material["specific_heat_j_kg_k"]
        * rng.normal(1.0, material["specific_heat_relative_std"])
    )

    room_temperature_c = float(environment.room_temperature_c)
    true_sample_mass_g = float(rng.uniform(45.0, 120.0))
    true_water_mass_g = float(rng.uniform(120.0, 260.0))
    hot_initial_c = float(rng.uniform(76.0, 97.0))
    cold_initial_c = float(rng.uniform(room_temperature_c - 2.0, room_temperature_c + 3.0))
    calorimeter_heat_capacity_j_k = float(rng.uniform(35.0, 120.0))

    transfer_delay_s = float(rng.uniform(3.0, 10.0))
    sample_transfer_loss_per_s = float(rng.uniform(0.0012, 0.0035))
    post_mix_loss_per_s = float(rng.uniform(0.00010, 0.00032))
    if class_name == "heat_loss":
        transfer_delay_s *= rng.uniform(2.0, 5.0)
        sample_transfer_loss_per_s *= rng.uniform(2.0, 5.0)
        post_mix_loss_per_s *= rng.uniform(4.0, 8.0)

    sample_at_mixing_c = room_temperature_c + (
        hot_initial_c - room_temperature_c
    ) * np.exp(-sample_transfer_loss_per_s * transfer_delay_s)

    sample_heat_capacity_j_k = true_specific_heat * true_sample_mass_g / 1000.0
    water_heat_capacity_j_k = (
        WATER_SPECIFIC_HEAT_J_KG_K * true_water_mass_g / 1000.0
    )
    cold_side_heat_capacity_j_k = (
        water_heat_capacity_j_k + calorimeter_heat_capacity_j_k
    )
    ideal_equilibrium_c = (
        sample_heat_capacity_j_k * sample_at_mixing_c
        + cold_side_heat_capacity_j_k * cold_initial_c
    ) / (sample_heat_capacity_j_k + cold_side_heat_capacity_j_k)

    mixing_tau_s = float(rng.uniform(10.0, 28.0))
    if class_name == "insufficient_mixing":
        mixing_tau_s *= rng.uniform(2.2, 4.5)

    true_temperature_c = ideal_equilibrium_c + (
        cold_initial_c - ideal_equilibrium_c
    ) * np.exp(-time_seconds / mixing_tau_s)
    true_temperature_c -= (
        ideal_equilibrium_c - room_temperature_c
    ) * (1.0 - np.exp(-post_mix_loss_per_s * time_seconds))

    if class_name == "insufficient_mixing":
        oscillation = rng.uniform(0.8, 2.3) * np.exp(-time_seconds / rng.uniform(110.0, 220.0))
        oscillation *= np.sin(2.0 * np.pi * time_seconds / rng.uniform(20.0, 45.0))
        local_gradient = rng.uniform(-1.2, 1.2) * np.exp(-time_seconds / rng.uniform(80.0, 180.0))
        true_temperature_c += oscillation + local_gradient

    sensor_temperature_c = first_order_lag(
        true_temperature_c,
        time_seconds,
        device.temperature_lag_s,
        initial_value=cold_initial_c,
    )
    measured_temperature_c = (
        sensor_temperature_c
        + device.temperature_offset_c
        + rng.normal(0.0, device.temperature_noise_c, n_points)
    )
    if class_name == "insufficient_mixing":
        measured_temperature_c += rng.normal(0.0, rng.uniform(0.15, 0.45), n_points)
    measured_temperature_c = quantize(
        measured_temperature_c,
        device.temperature_resolution_c,
    )

    measured_sample_mass_g = (
        true_sample_mass_g
        + rng.normal(0.0, device.mass_noise_g)
    )
    measured_water_mass_g = (
        true_water_mass_g
        + rng.normal(0.0, device.mass_noise_g * 1.5)
    )
    mass_error_fraction = 0.0
    if class_name == "mass_measurement_error":
        mass_error_fraction = float(rng.uniform(0.18, 0.45))
        measured_sample_mass_g *= 1.0 + rng.choice([-1.0, 1.0]) * mass_error_fraction
        if rng.random() < 0.35:
            measured_water_mass_g *= 1.0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.12)

    measured_sample_mass_g = float(
        quantize([measured_sample_mass_g], device.mass_resolution_g)[0]
    )
    measured_water_mass_g = float(
        quantize([measured_water_mass_g], device.mass_resolution_g)[0]
    )

    if class_name == "heat_loss":
        severity = min(1.0, transfer_delay_s / 35.0 + post_mix_loss_per_s / 0.0025)
    elif class_name == "mass_measurement_error":
        severity = min(1.0, mass_error_fraction / 0.45)
    elif class_name == "insufficient_mixing":
        severity = min(1.0, mixing_tau_s / 110.0)
    else:
        severity = float(rng.uniform(0.02, 0.16))

    secondary_errors = []
    if calorimeter_heat_capacity_j_k > 95.0:
        secondary_errors.append("high_calorimeter_heat_capacity")
    if device.temperature_lag_s > 8.0:
        secondary_errors.append("sensor_lag")
    if transfer_delay_s > 9.0 and class_name != "heat_loss":
        secondary_errors.append("small_transfer_loss")

    return pd.DataFrame(
        {
            "experiment_id": experiment_id,
            "time_seconds": np.round(time_seconds, 2),
            "temperature_c": measured_temperature_c,
            "hot_mass_g": measured_sample_mass_g,
            "cold_mass_g": measured_water_mass_g,
            "hot_initial_c": quantize([hot_initial_c], device.temperature_resolution_c)[0],
            "cold_initial_c": quantize([cold_initial_c], device.temperature_resolution_c)[0],
            "calorimeter_heat_capacity_j_k": calorimeter_heat_capacity_j_k,
            "material": material_key,
            "true_specific_heat_j_kg_k": true_specific_heat,
            "true_hot_mass_g": true_sample_mass_g,
            "true_cold_mass_g": true_water_mass_g,
            "true_temperature_c": np.round(true_temperature_c, 4),
            "true_equilibrium_c": ideal_equilibrium_c,
            "device_profile": device.name,
            "environment_profile": environment.name,
            "generation_group": group_name,
            "secondary_errors": ";".join(secondary_errors) if secondary_errors else "none",
            "severity": severity,
            "class_name": class_name,
        }
    )


def extract_features(df: pd.DataFrame) -> dict[str, float]:
    required = {
        "time_seconds",
        "temperature_c",
        "hot_mass_g",
        "cold_mass_g",
        "hot_initial_c",
        "cold_initial_c",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    clean = df.dropna(subset=["time_seconds", "temperature_c"]).sort_values(
        "time_seconds"
    )
    if len(clean) < 20:
        raise ValueError("At least 20 measurements are required")

    time = clean["time_seconds"].to_numpy(dtype=float)
    temperature = clean["temperature_c"].to_numpy(dtype=float)
    sample_mass_g = float(clean["hot_mass_g"].iloc[0])
    water_mass_g = float(clean["cold_mass_g"].iloc[0])
    hot_initial_c = float(clean["hot_initial_c"].iloc[0])
    cold_initial_c = float(clean["cold_initial_c"].iloc[0])
    calorimeter_heat_capacity = float(
        clean["calorimeter_heat_capacity_j_k"].iloc[0]
        if "calorimeter_heat_capacity_j_k" in clean.columns
        else 70.0
    )

    peak_index = int(np.argmax(temperature))
    peak_temperature = float(temperature[peak_index])
    peak_time_s = float(time[peak_index])
    late_count = max(8, len(clean) // 6)
    late_temperature = temperature[-late_count:]
    late_time = time[-late_count:]
    final_temperature = float(np.mean(late_temperature))
    late_slope = float(np.polyfit(late_time, late_temperature, 1)[0])

    denominator = sample_mass_g / 1000.0 * max(
        hot_initial_c - peak_temperature,
        0.1,
    )
    apparent_specific_heat = (
        (
            WATER_SPECIFIC_HEAT_J_KG_K * water_mass_g / 1000.0
            + calorimeter_heat_capacity
        )
        * max(peak_temperature - cold_initial_c, 0.0)
        / denominator
    )

    material_key = str(clean["material"].iloc[0]) if "material" in clean.columns else "steel"
    reference_specific_heat = float(
        MATERIALS.get(material_key, MATERIALS["steel"])["specific_heat_j_kg_k"]
    )
    specific_heat_relative_error = float(
        (apparent_specific_heat - reference_specific_heat)
        / (reference_specific_heat + 1e-9)
    )

    first_diff = np.diff(temperature)
    second_diff = np.diff(temperature, n=2)
    pre_peak = temperature[: peak_index + 1]
    post_peak = temperature[peak_index:]

    monotonic_rise_fraction = float(
        np.mean(np.diff(pre_peak) >= -0.15) if len(pre_peak) > 2 else 1.0
    )
    post_peak_decline_fraction = float(
        np.mean(np.diff(post_peak) <= 0.15) if len(post_peak) > 2 else 1.0
    )

    return {
        "temperature_mean": float(np.mean(temperature)),
        "temperature_std": float(np.std(temperature)),
        "temperature_range": float(np.ptp(temperature)),
        "peak_temperature": peak_temperature,
        "final_temperature": final_temperature,
        "peak_time_fraction": float(peak_time_s / (time[-1] - time[0] + 1e-9)),
        "peak_to_final_drop": float(peak_temperature - final_temperature),
        "late_slope": late_slope,
        "late_std": float(np.std(late_temperature)),
        "late_robust_scale": robust_scale(late_temperature),
        "max_abs_jump": float(np.max(np.abs(first_diff))),
        "second_difference_scale": robust_scale(second_diff),
        "temperature_normalized_slope": normalized_slope(temperature),
        "time_temperature_corr": safe_correlation(time, temperature),
        "monotonic_rise_fraction": monotonic_rise_fraction,
        "post_peak_decline_fraction": post_peak_decline_fraction,
        "sample_mass_g": sample_mass_g,
        "water_mass_g": water_mass_g,
        "mass_ratio": float(sample_mass_g / (water_mass_g + 1e-9)),
        "initial_delta_temperature": float(hot_initial_c - cold_initial_c),
        "calorimeter_heat_capacity_j_k": calorimeter_heat_capacity,
        "apparent_specific_heat_j_kg_k": float(apparent_specific_heat),
        "apparent_specific_heat_log": float(np.log1p(max(apparent_specific_heat, 0.0))),
        "reference_specific_heat_j_kg_k": reference_specific_heat,
        "specific_heat_relative_error": specific_heat_relative_error,
        "specific_heat_abs_relative_error": abs(specific_heat_relative_error),
        "mixing_amplitude": float(np.ptp(temperature[: max(peak_index + 1, 3)])),
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
        n_points = int(rng.integers(70, 120))
        group_name = generation_family("heat_balance", family_index)

        for class_name in CLASSES:
            counter += 1
            experiment_id = f"HB_{counter:05d}"
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
    model, feature_names = load_model_and_features("heat_balance")
    features = extract_features(df)
    return predict_with_reliability(
        lab_id="heat_balance",
        model=model,
        feature_names=feature_names,
        features=features,
    )
