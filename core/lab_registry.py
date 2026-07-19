from core.schemas import LabConfig

LAB_REGISTRY = {
    "cooling": LabConfig(
        "cooling",
        "Исследование нагревания и охлаждения жидкости",
        "Нагревание и охлаждение",
        "Анализ изменения температуры жидкости во времени.",
        "dT/dt = -k(T - T_env)",
        ("time_seconds", "measured_temperature"),
        ("normal", "single_outlier", "sensor_drift", "high_noise"),
        "Интерпретация временного ряда и оценка качества измерений.",
        "ml_ready",
    ),
    "boyle_mariotte": LabConfig(
        "boyle_mariotte",
        "Исследование закона Бойля — Мариотта",
        "Закон Бойля — Мариотта",
        "Анализ зависимости абсолютного давления газа от объема при постоянной температуре.",
        "p_abs · V = const",
        (
            "measurement_number",
            "volume_ml",
            "pressure_kpa",
            "temperature_c",
        ),
        (
            "normal",
            "air_leak",
            "temperature_change",
            "volume_measurement_error",
        ),
        "Контроль герметичности, температуры, типа давления и постоянства pV.",
        "ml_ready",
    ),
    "isochoric": LabConfig(
        "isochoric",
        "Исследование изохорного процесса",
        "Изохорный процесс",
        "Анализ зависимости абсолютного давления от абсолютной температуры при постоянном объеме.",
        "p_abs / T = const",
        ("time_seconds", "temperature_c", "pressure_kpa", "volume_ml"),
        (
            "normal",
            "air_leak",
            "volume_instability",
            "temperature_sensor_lag",
        ),
        "Использование абсолютной температуры и контроль постоянства количества газа и объема.",
        "ml_ready",
    ),
    "heat_balance": LabConfig(
        "heat_balance",
        "Определение удельной теплоемкости методом теплового баланса",
        "Тепловой баланс",
        "Анализ смешения, теплопотерь, масс и достижения температурного равновесия.",
        "Q_sample = Q_water + Q_calorimeter",
        (
            "time_seconds",
            "temperature_c",
            "hot_mass_g",
            "cold_mass_g",
            "hot_initial_c",
            "cold_initial_c",
        ),
        (
            "normal",
            "heat_loss",
            "mass_measurement_error",
            "insufficient_mixing",
        ),
        "Многопараметрический анализ теплового баланса и качества измерений.",
        "ml_ready",
    ),
}


def get_lab_config(lab_id: str) -> LabConfig:
    if lab_id not in LAB_REGISTRY:
        raise KeyError(f"Неизвестная лабораторная работа: {lab_id}")
    return LAB_REGISTRY[lab_id]


def list_labs():
    return list(LAB_REGISTRY.values())


def get_lab_titles():
    return {key: value.short_title for key, value in LAB_REGISTRY.items()}
