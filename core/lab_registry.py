from core.schemas import LabConfig


LAB_REGISTRY: dict[str, LabConfig] = {
    "cooling": LabConfig(
        lab_id="cooling",
        title="Исследование нагревания и охлаждения жидкости",
        short_title="Нагревание и охлаждение",
        description=(
            "Исследование изменения температуры жидкости во времени "
            "и сопоставление экспериментальных данных с законом охлаждения Ньютона."
        ),
        physics_model=(
            "T(t) = T_env + (T0 - T_env) * exp(-k * t)"
        ),
        input_columns=(
            "time_seconds",
            "measured_temperature",
        ),
        classes=(
            "normal",
            "single_outlier",
            "sensor_drift",
            "high_noise",
        ),
        educational_goal=(
            "Научиться интерпретировать временной ряд, обнаруживать "
            "отклонения и оценивать качество экспериментальных данных."
        ),
        implementation_status="ml_ready",
    ),

    "boyle_mariotte": LabConfig(
        lab_id="boyle_mariotte",
        title="Исследование закона Бойля — Мариотта",
        short_title="Закон Бойля — Мариотта",
        description=(
            "Исследование зависимости давления газа от его объёма "
            "при постоянной температуре."
        ),
        physics_model="p * V = const",
        input_columns=(
            "measurement_number",
            "volume_ml",
            "pressure_kpa",
            "temperature_c",
        ),
        classes=(
            "normal",
            "air_leak",
            "temperature_change",
            "volume_measurement_error",
        ),
        educational_goal=(
            "Научиться контролировать условия эксперимента, анализировать "
            "зависимость p(V) и проверять постоянство произведения pV."
        ),
        implementation_status="planned",
    ),

    "isochoric": LabConfig(
        lab_id="isochoric",
        title="Исследование изохорного процесса",
        short_title="Изохорный процесс",
        description=(
            "Исследование зависимости давления газа от абсолютной температуры "
            "при постоянном объёме."
        ),
        physics_model="p / T = const",
        input_columns=(
            "time_seconds",
            "temperature_c",
            "pressure_kpa",
            "volume_ml",
        ),
        classes=(
            "normal",
            "volume_instability",
            "temperature_sensor_lag",
            "wrong_temperature_scale",
        ),
        educational_goal=(
            "Научиться использовать абсолютную температуру, проверять "
            "условия изохорного процесса и интерпретировать зависимость p(T)."
        ),
        implementation_status="planned",
    ),

    "heat_balance": LabConfig(
        lab_id="heat_balance",
        title="Исследование теплового баланса",
        short_title="Тепловой баланс",
        description=(
            "Исследование теплообмена между телами и определение "
            "температуры теплового равновесия."
        ),
        physics_model="Q_lost = Q_received",
        input_columns=(
            "time_seconds",
            "temperature_c",
            "hot_mass_g",
            "cold_mass_g",
        ),
        classes=(
            "normal",
            "heat_loss",
            "mass_measurement_error",
            "insufficient_mixing",
        ),
        educational_goal=(
            "Научиться учитывать несколько физических параметров, "
            "анализировать теплопотери и аргументировать вывод."
        ),
        implementation_status="planned",
    ),
}


def get_lab_config(lab_id: str) -> LabConfig:
    if lab_id not in LAB_REGISTRY:
        available = ", ".join(LAB_REGISTRY)
        raise KeyError(
            f"Неизвестная лабораторная работа: {lab_id}. "
            f"Доступные значения: {available}"
        )

    return LAB_REGISTRY[lab_id]


def list_labs() -> list[LabConfig]:
    return list(LAB_REGISTRY.values())


def get_lab_titles() -> dict[str, str]:
    return {
        lab_id: config.short_title
        for lab_id, config in LAB_REGISTRY.items()
    }