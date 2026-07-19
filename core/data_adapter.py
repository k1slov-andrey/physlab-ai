from __future__ import annotations

import re

import numpy as np
import pandas as pd


ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "cooling": {
        "time_seconds": (
            "time_seconds",
            "time_s",
            "t_sec",
            "seconds",
            "time",
            "t",
        ),
        "measured_temperature": (
            "measured_temperature",
            "temperature_c",
            "temp_c",
            "t_c",
            "temperature",
            "temp",
        ),
    },
    "boyle_mariotte": {
        "measurement_number": (
            "measurement_number",
            "measurement",
            "trial",
            "point",
            "n",
        ),
        "time_seconds": (
            "time_seconds",
            "time_s",
            "t_sec",
            "seconds",
            "time",
        ),
        "volume_ml": (
            "volume_ml",
            "v_ml",
            "volume_cm3",
            "volume_cc",
            "volume",
            "v",
        ),
        "pressure_kpa": (
            "pressure_kpa",
            "p_kpa",
            "pressure",
            "p",
        ),
        "temperature_c": (
            "temperature_c",
            "temp_c",
            "t_c",
            "temperature",
            "temp",
        ),
        "atmospheric_pressure_kpa": (
            "atmospheric_pressure_kpa",
            "atmospheric_kpa",
            "p_atm_kpa",
            "patm_kpa",
        ),
    },
    "isochoric": {
        "time_seconds": (
            "time_seconds",
            "time_s",
            "t_sec",
            "seconds",
            "time",
        ),
        "temperature_c": (
            "temperature_c",
            "temp_c",
            "t_c",
            "temperature",
            "temp",
        ),
        "pressure_kpa": (
            "pressure_kpa",
            "p_kpa",
            "pressure",
            "p",
        ),
        "volume_ml": (
            "volume_ml",
            "v_ml",
            "volume_cm3",
            "volume_cc",
            "volume",
            "v",
        ),
    },
    "heat_balance": {
        "time_seconds": (
            "time_seconds",
            "time_s",
            "t_sec",
            "seconds",
            "time",
        ),
        "temperature_c": (
            "temperature_c",
            "temp_c",
            "t_c",
            "temperature",
            "temp",
        ),
        "hot_mass_g": (
            "hot_mass_g",
            "sample_mass_g",
            "body_mass_g",
            "m_sample_g",
            "m_body_g",
        ),
        "cold_mass_g": (
            "cold_mass_g",
            "water_mass_g",
            "m_water_g",
        ),
        "hot_initial_c": (
            "hot_initial_c",
            "sample_initial_c",
            "body_initial_c",
            "t_hot_c",
            "t_sample_c",
        ),
        "cold_initial_c": (
            "cold_initial_c",
            "water_initial_c",
            "t_cold_c",
            "t_water_c",
        ),
        "calorimeter_heat_capacity_j_k": (
            "calorimeter_heat_capacity_j_k",
            "calorimeter_constant_j_k",
            "c_cal_j_k",
        ),
    },
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "cooling": ("time_seconds", "measured_temperature"),
    "boyle_mariotte": ("volume_ml", "pressure_kpa"),
    "isochoric": ("temperature_c", "pressure_kpa"),
    "heat_balance": (
        "time_seconds",
        "temperature_c",
        "hot_mass_g",
        "cold_mass_g",
        "hot_initial_c",
        "cold_initial_c",
    ),
}


def _normalize_name(name: str) -> str:
    cleaned = str(name).strip().lower()
    cleaned = cleaned.replace("°", "")
    cleaned = re.sub(r"[^a-zа-я0-9]+", "_", cleaned, flags=re.IGNORECASE)
    return cleaned.strip("_")


def normalize_experiment_dataframe(
    lab_id: str,
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if lab_id not in ALIASES:
        raise KeyError(f"Unknown laboratory: {lab_id}")

    df = dataframe.copy()
    normalized_existing = {_normalize_name(column): column for column in df.columns}
    rename_map: dict[str, str] = {}

    for canonical, candidates in ALIASES[lab_id].items():
        for candidate in candidates:
            normalized_candidate = _normalize_name(candidate)
            if normalized_candidate in normalized_existing:
                source = normalized_existing[normalized_candidate]
                rename_map[source] = canonical
                break

    df = df.rename(columns=rename_map)

    if lab_id == "boyle_mariotte":
        if "measurement_number" not in df.columns:
            df["measurement_number"] = np.arange(1, len(df) + 1)
        if "temperature_c" not in df.columns:
            df["temperature_c"] = 22.0
        if "time_seconds" not in df.columns:
            df["time_seconds"] = np.arange(len(df), dtype=float) * 10.0
        if "atmospheric_pressure_kpa" not in df.columns:
            df["atmospheric_pressure_kpa"] = 101.325

    if lab_id == "isochoric":
        if "time_seconds" not in df.columns:
            df["time_seconds"] = np.arange(len(df), dtype=float) * 10.0
        if "volume_ml" not in df.columns:
            df["volume_ml"] = 250.0

    if lab_id == "heat_balance":
        if "calorimeter_heat_capacity_j_k" not in df.columns:
            df["calorimeter_heat_capacity_j_k"] = 70.0

    missing = [
        column
        for column in REQUIRED_COLUMNS[lab_id]
        if column not in df.columns
    ]

    for column in df.columns:
        if column in {
            "experiment_id",
            "class_name",
            "material",
            "device_profile",
            "environment_profile",
            "generation_group",
            "secondary_errors",
        }:
            continue
        try:
            df[column] = pd.to_numeric(df[column])
        except (ValueError, TypeError):
            pass

    return df, missing


def visible_experiment_columns(dataframe: pd.DataFrame) -> list[str]:
    hidden_prefixes = ("true_",)
    hidden_exact = {
        "class_name",
        "severity",
        "generation_group",
        "secondary_errors",
        "device_profile",
        "environment_profile",
        "dead_volume_ml",
    }
    return [
        column
        for column in dataframe.columns
        if not column.startswith(hidden_prefixes)
        and column not in hidden_exact
    ]
