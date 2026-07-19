from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "real_data_integration"
    / "empirical_realism_profile.json"
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def load_empirical_profile() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def temperature_calibration() -> dict[str, float] | None:
    profile = load_empirical_profile()
    sensor = profile.get("temperature_sensor", {})

    resolution = sensor.get("temperature_resolution_c_median")
    noise_median = sensor.get("temperature_noise_sigma_c_median")
    noise_q90 = sensor.get("temperature_noise_sigma_c_q90")

    numeric_values = (resolution, noise_median, noise_q90)
    if any(value is None for value in numeric_values):
        return None

    try:
        return {
            "resolution_c": _clamp(float(resolution), 0.05, 0.5),
            "noise_median_c": _clamp(float(noise_median), 0.02, 0.6),
            "noise_q90_c": _clamp(float(noise_q90), 0.03, 0.9),
        }
    except (TypeError, ValueError):
        return None


def calibrate_device_profile(profile):
    calibration = temperature_calibration()
    if calibration is None:
        return profile

    profile_multiplier = {
        "school_digital": 0.75,
        "school_basic": 1.00,
        "low_cost_sensor": 1.35,
    }.get(profile.name, 1.0)

    empirical_noise = calibration["noise_median_c"] * profile_multiplier
    if profile.name == "low_cost_sensor":
        empirical_noise = max(
            empirical_noise,
            calibration["noise_q90_c"],
        )

    calibrated_noise = 0.45 * profile.temperature_noise_c + 0.55 * empirical_noise
    calibrated_resolution = max(
        profile.temperature_resolution_c,
        calibration["resolution_c"],
    )

    return replace(
        profile,
        temperature_noise_c=_clamp(calibrated_noise, 0.03, 0.8),
        temperature_resolution_c=_clamp(calibrated_resolution, 0.05, 0.5),
    )
