from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import (
    extract_features_from_experiment,
    fit_cooling_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.joblib"

REQUIRED_COLUMNS = {
    "time_seconds",
    "measured_temperature",
}


CLASS_INFORMATION = {
    "normal": {
        "title": "Нормальный эксперимент",
        "description": (
            "Существенных отклонений от модели охлаждения "
            "не обнаружено."
        ),
        "recommendation": (
            "Сравните форму экспериментального графика с законом "
            "охлаждения Ньютона и сформулируйте физический вывод."
        ),
    },
    "single_outlier": {
        "title": "Единичный выброс",
        "description": (
            "Обнаружено отдельное измерение, резко отличающееся "
            "от общей динамики эксперимента."
        ),
        "recommendation": (
            "Проверьте контакт датчика с жидкостью, соединение "
            "оборудования и возможность случайного внешнего воздействия."
        ),
    },
    "sensor_drift": {
        "title": "Дрейф датчика",
        "description": (
            "Показания постепенно отклоняются от ожидаемой "
            "физической зависимости."
        ),
        "recommendation": (
            "Проверьте калибровку датчика и стабильность его положения "
            "в течение всего эксперимента."
        ),
    },
    "high_noise": {
        "title": "Повышенный шум",
        "description": (
            "В показаниях присутствуют частые случайные колебания "
            "повышенной амплитуды."
        ),
        "recommendation": (
            "Проверьте надежность подключения, неподвижность датчика "
            "и отсутствие внешних механических воздействий."
        ),
    },
}


def validate_experiment(data: pd.DataFrame) -> pd.DataFrame:
    """Проверяет и очищает загруженный временной ряд."""

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            "В файле отсутствуют обязательные столбцы: "
            + ", ".join(sorted(missing_columns))
        )

    experiment = data[
        [
            "time_seconds",
            "measured_temperature",
        ]
    ].copy()

    experiment["time_seconds"] = pd.to_numeric(
        experiment["time_seconds"],
        errors="coerce",
    )

    experiment["measured_temperature"] = pd.to_numeric(
        experiment["measured_temperature"],
        errors="coerce",
    )

    experiment = (
        experiment
        .dropna()
        .sort_values("time_seconds")
        .drop_duplicates(subset="time_seconds")
        .reset_index(drop=True)
    )

    if len(experiment) < 10:
        raise ValueError(
            "После очистки осталось менее 10 корректных измерений."
        )

    if not experiment["time_seconds"].is_monotonic_increasing:
        raise ValueError(
            "Значения времени должны располагаться по возрастанию."
        )

    return experiment


def get_model_classes(model) -> list[str]:
    """Возвращает порядок классов модели."""

    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")

        if classifier is not None and hasattr(classifier, "classes_"):
            return list(classifier.classes_)

    raise ValueError(
        "Не удалось определить список классов модели."
    )


def predict_experiment(data: pd.DataFrame) -> dict:
    """Анализирует эксперимент и возвращает результат диагностики."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Файл модели не найден. Сначала запустите train_model.py."
        )

    if not FEATURE_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Файл признаков модели не найден."
        )

    experiment = validate_experiment(data)

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    features = extract_features_from_experiment(
        experiment
    )

    feature_table = pd.DataFrame([features])

    x = feature_table[feature_names]

    predicted_class = str(
        model.predict(x)[0]
    )

    probabilities = model.predict_proba(x)[0]
    classes = get_model_classes(model)

    probability_by_class = {
        class_name: float(probability)
        for class_name, probability in zip(
            classes,
            probabilities,
        )
    }

    confidence = probability_by_class[predicted_class]

    time = experiment["time_seconds"].to_numpy(dtype=float)
    measured = experiment[
        "measured_temperature"
    ].to_numpy(dtype=float)

    fitted_temperature, fitted_parameters = fit_cooling_model(
        time=time,
        measured=measured,
    )

    class_information = CLASS_INFORMATION.get(
        predicted_class,
        {
            "title": predicted_class,
            "description": "Класс распознан моделью.",
            "recommendation": (
                "Рекомендуется дополнительная экспертная проверка."
            ),
        },
    )

    return {
        "experiment": experiment,
        "fitted_temperature": fitted_temperature,
        "fitted_parameters": fitted_parameters,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_by_class,
        "features": features,
        "title": class_information["title"],
        "description": class_information["description"],
        "recommendation": class_information["recommendation"],
    }