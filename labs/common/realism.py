from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from labs.common.empirical_calibration import calibrate_device_profile


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    temperature_noise_c: float
    temperature_resolution_c: float
    temperature_offset_c: float
    temperature_lag_s: float
    pressure_noise_kpa: float
    pressure_resolution_kpa: float
    pressure_offset_kpa: float
    pressure_drift_kpa_per_min: float
    volume_noise_ml: float
    volume_resolution_ml: float
    volume_offset_ml: float
    mass_noise_g: float
    mass_resolution_g: float


DEVICE_PROFILES: tuple[DeviceProfile, ...] = (
    DeviceProfile(
        name="school_basic",
        temperature_noise_c=0.18,
        temperature_resolution_c=0.1,
        temperature_offset_c=0.12,
        temperature_lag_s=7.0,
        pressure_noise_kpa=0.32,
        pressure_resolution_kpa=0.1,
        pressure_offset_kpa=0.35,
        pressure_drift_kpa_per_min=0.025,
        volume_noise_ml=0.30,
        volume_resolution_ml=0.5,
        volume_offset_ml=0.35,
        mass_noise_g=0.20,
        mass_resolution_g=0.1,
    ),
    DeviceProfile(
        name="school_digital",
        temperature_noise_c=0.09,
        temperature_resolution_c=0.1,
        temperature_offset_c=0.05,
        temperature_lag_s=4.0,
        pressure_noise_kpa=0.16,
        pressure_resolution_kpa=0.1,
        pressure_offset_kpa=0.16,
        pressure_drift_kpa_per_min=0.010,
        volume_noise_ml=0.16,
        volume_resolution_ml=0.1,
        volume_offset_ml=0.12,
        mass_noise_g=0.10,
        mass_resolution_g=0.1,
    ),
    DeviceProfile(
        name="low_cost_sensor",
        temperature_noise_c=0.32,
        temperature_resolution_c=0.1,
        temperature_offset_c=0.28,
        temperature_lag_s=10.0,
        pressure_noise_kpa=0.55,
        pressure_resolution_kpa=0.2,
        pressure_offset_kpa=0.65,
        pressure_drift_kpa_per_min=0.050,
        volume_noise_ml=0.45,
        volume_resolution_ml=0.5,
        volume_offset_ml=0.55,
        mass_noise_g=0.35,
        mass_resolution_g=0.1,
    ),
)


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    room_temperature_c: float
    atmospheric_pressure_kpa: float
    temperature_drift_c_per_hour: float
    disturbance_probability: float


ENVIRONMENT_PROFILES: tuple[EnvironmentProfile, ...] = (
    EnvironmentProfile("stable_room", 22.0, 101.3, 0.10, 0.05),
    EnvironmentProfile("warm_classroom", 25.0, 100.8, 0.25, 0.10),
    EnvironmentProfile("cool_laboratory", 19.0, 101.8, -0.12, 0.08),
    EnvironmentProfile("variable_room", 23.0, 99.9, 0.55, 0.25),
)


def choose_device_profile(rng: np.random.Generator) -> DeviceProfile:
    weights = np.array([0.45, 0.35, 0.20], dtype=float)
    index = int(rng.choice(len(DEVICE_PROFILES), p=weights))
    return calibrate_device_profile(DEVICE_PROFILES[index])


def choose_environment_profile(rng: np.random.Generator) -> EnvironmentProfile:
    weights = np.array([0.45, 0.20, 0.15, 0.20], dtype=float)
    index = int(rng.choice(len(ENVIRONMENT_PROFILES), p=weights))
    base = ENVIRONMENT_PROFILES[index]
    return EnvironmentProfile(
        name=base.name,
        room_temperature_c=float(base.room_temperature_c + rng.normal(0.0, 0.8)),
        atmospheric_pressure_kpa=float(base.atmospheric_pressure_kpa + rng.normal(0.0, 0.6)),
        temperature_drift_c_per_hour=float(base.temperature_drift_c_per_hour + rng.normal(0.0, 0.06)),
        disturbance_probability=base.disturbance_probability,
    )


def quantize(values: np.ndarray | Iterable[float], resolution: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if resolution <= 0:
        return array
    return np.round(array / resolution) * resolution


def first_order_lag(
    signal: np.ndarray | Iterable[float],
    time_seconds: np.ndarray | Iterable[float],
    tau_seconds: float,
    initial_value: float | None = None,
) -> np.ndarray:
    source = np.asarray(signal, dtype=float)
    time = np.asarray(time_seconds, dtype=float)
    if len(source) != len(time):
        raise ValueError("signal and time_seconds must have the same length")
    if len(source) == 0:
        return source.copy()
    if tau_seconds <= 1e-9:
        return source.copy()

    result = np.empty_like(source)
    result[0] = source[0] if initial_value is None else float(initial_value)
    for index in range(1, len(source)):
        dt = max(float(time[index] - time[index - 1]), 1e-9)
        alpha = 1.0 - np.exp(-dt / tau_seconds)
        result[index] = result[index - 1] + alpha * (source[index] - result[index - 1])
    return result


def random_walk(
    size: int,
    step_std: float,
    rng: np.random.Generator,
    center: bool = False,
) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=float)
    walk = np.cumsum(rng.normal(0.0, step_std, size=size))
    if center:
        walk = walk - float(np.mean(walk))
    return walk


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 2 or len(y_array) < 2:
        return 0.0
    if np.std(x_array) < 1e-12 or np.std(y_array) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_array, y_array)[0, 1])


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 2 or np.std(x_array) < 1e-12:
        return 0.0
    coefficients = np.polyfit(x_array, y_array, 1)
    prediction = np.polyval(coefficients, x_array)
    denominator = float(np.sum((y_array - np.mean(y_array)) ** 2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_array - prediction) ** 2) / denominator)


def robust_scale(values: np.ndarray | Iterable[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return 0.0
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return 1.4826 * mad


def normalized_slope(values: np.ndarray | Iterable[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(array))
    slope = float(np.polyfit(x, array, 1)[0])
    scale = float(np.mean(np.abs(array))) + 1e-9
    return slope / scale


def generation_group(index: int, total_groups: int = 8) -> str:
    return f"group_{index % total_groups:02d}"
