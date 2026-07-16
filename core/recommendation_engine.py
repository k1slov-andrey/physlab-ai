from core.schemas import ModelPrediction, ResearchFeedback


RECOMMENDATIONS: dict[str, dict[str, dict[str, object]]] = {
    "cooling": {
        "normal": {
            "title": "Данные соответствуют ожидаемому процессу",
            "explanation": (
                "Температура изменяется плавно, а экспериментальная кривая "
                "согласуется с моделью охлаждения."
            ),
            "student_question": (
                "Как форма графика связана с разностью температур жидкости "
                "и окружающей среды?"
            ),
            "recommended_action": (
                "Сформулируйте физический вывод и подтвердите его графиком."
            ),
            "teacher_note": "Серия пригодна для интерпретации.",
            "requires_repeat": False,
        },
        "single_outlier": {
            "title": "Обнаружено единичное отклонение",
            "explanation": (
                "Одна точка значительно отличается от соседних измерений "
                "и общей формы температурной кривой."
            ),
            "student_question": (
                "Какое событие во время опыта могло привести к резкому "
                "изменению одного измерения?"
            ),
            "recommended_action": (
                "Проверьте положение датчика и повторите измерение "
                "на подозрительном участке."
            ),
            "teacher_note": "Вероятен единичный сбой измерения.",
            "requires_repeat": True,
        },
        "sensor_drift": {
            "title": "Вероятен дрейф датчика",
            "explanation": (
                "Отклонение от физической модели постепенно увеличивается "
                "по мере выполнения эксперимента."
            ),
            "student_question": (
                "Почему систематическое отклонение может усиливаться со временем?"
            ),
            "recommended_action": (
                "Проверьте калибровку, контакт датчика со средой "
                "и стабильность условий опыта."
            ),
            "teacher_note": "Необходимо проверить калибровку датчика.",
            "requires_repeat": True,
        },
        "high_noise": {
            "title": "Повышенный случайный шум",
            "explanation": (
                "Измерения часто колеблются вокруг ожидаемой температурной кривой."
            ),
            "student_question": (
                "Какие внешние воздействия могли сделать показания нестабильными?"
            ),
            "recommended_action": (
                "Зафиксируйте датчик, исключите перемещения и повторите серию."
            ),
            "teacher_note": "Вероятны нестабильные условия измерения.",
            "requires_repeat": True,
        },
    },

    "boyle_mariotte": {
        "normal": {
            "title": "Закон Бойля — Мариотта подтверждается",
            "explanation": (
                "При изменении объёма произведение давления на объём "
                "остаётся приблизительно постоянным."
            ),
            "student_question": (
                "Как изменяется давление при уменьшении объёма газа?"
            ),
            "recommended_action": (
                "Постройте зависимость p(V) и сформулируйте вывод "
                "о произведении pV."
            ),
            "teacher_note": "Условия изотермического процесса соблюдены.",
            "requires_repeat": False,
        },
        "air_leak": {
            "title": "Возможна утечка воздуха",
            "explanation": (
                "Произведение pV систематически изменяется в одном направлении, "
                "что может указывать на нарушение герметичности."
            ),
            "student_question": (
                "Почему потеря части газа нарушает постоянство произведения pV?"
            ),
            "recommended_action": (
                "Проверьте соединения и герметичность установки, затем повторите серию."
            ),
            "teacher_note": "Вероятно негерметичное соединение.",
            "requires_repeat": True,
        },
        "temperature_change": {
            "title": "Температура газа могла измениться",
            "explanation": (
                "Изменения давления и объёма сопровождаются заметным "
                "изменением температуры."
            ),
            "student_question": (
                "Почему закон Бойля — Мариотта требует постоянной температуры?"
            ),
            "recommended_action": (
                "Изменяйте объём медленнее и ожидайте стабилизации температуры."
            ),
            "teacher_note": "Нарушено условие изотермичности.",
            "requires_repeat": True,
        },
        "volume_measurement_error": {
            "title": "Вероятна ошибка измерения объёма",
            "explanation": (
                "Отдельные значения объёма нарушают общую зависимость "
                "между давлением и объёмом."
            ),
            "student_question": (
                "Какие точки сильнее всего отклоняются от общей зависимости p(V)?"
            ),
            "recommended_action": (
                "Повторно считайте объём и проверьте единицы измерения."
            ),
            "teacher_note": "Следует проверить фиксацию и запись объёма.",
            "requires_repeat": True,
        },
    },

    "isochoric": {
        "normal": {
            "title": "Изохорная зависимость подтверждается",
            "explanation": (
                "Давление изменяется пропорционально абсолютной температуре."
            ),
            "student_question": (
                "Почему в расчётах необходимо использовать температуру в Кельвинах?"
            ),
            "recommended_action": (
                "Постройте график p(T) и объясните характер зависимости."
            ),
            "teacher_note": "Изохорный процесс воспроизведён корректно.",
            "requires_repeat": False,
        },
        "volume_instability": {
            "title": "Объём мог изменяться",
            "explanation": (
                "Зависимость давления от температуры не соответствует "
                "процессу при постоянном объёме."
            ),
            "student_question": (
                "Как изменение объёма влияет на давление газа?"
            ),
            "recommended_action": (
                "Проверьте жёсткость и герметичность сосуда."
            ),
            "teacher_note": "Возможно нарушение постоянства объёма.",
            "requires_repeat": True,
        },
        "temperature_sensor_lag": {
            "title": "Вероятно запаздывание датчика температуры",
            "explanation": (
                "Изменение давления происходит раньше, чем изменение "
                "зарегистрированной температуры."
            ),
            "student_question": (
                "Почему датчику требуется время для установления показаний?"
            ),
            "recommended_action": (
                "Увеличьте время ожидания между измерениями."
            ),
            "teacher_note": "Возможна тепловая инерция датчика.",
            "requires_repeat": True,
        },
        "wrong_temperature_scale": {
            "title": "Проверьте температурную шкалу",
            "explanation": (
                "Расчёт отношения p/T выполнен с температурой, "
                "которая может быть выражена не в абсолютной шкале."
            ),
            "student_question": (
                "Чем абсолютная температура отличается от температуры по Цельсию?"
            ),
            "recommended_action": (
                "Переведите температуру в Кельвины и повторите анализ."
            ),
            "teacher_note": "Вероятно использование градусов Цельсия вместо Кельвинов.",
            "requires_repeat": False,
        },
    },

    "heat_balance": {
        "normal": {
            "title": "Тепловой баланс достигнут",
            "explanation": (
                "Конечная температура согласуется с расчётной моделью теплообмена."
            ),
            "student_question": (
                "Почему конечная температура находится между начальными температурами?"
            ),
            "recommended_action": (
                "Сопоставьте экспериментальную и расчётную температуры "
                "и сформулируйте вывод."
            ),
            "teacher_note": "Данные пригодны для анализа теплового баланса.",
            "requires_repeat": False,
        },
        "heat_loss": {
            "title": "Обнаружены возможные теплопотери",
            "explanation": (
                "Экспериментальная температура равновесия отличается от расчётной "
                "в направлении, характерном для теплообмена с окружающей средой."
            ),
            "student_question": (
                "Куда могла перейти часть энергии системы?"
            ),
            "recommended_action": (
                "Уменьшите время переноса, используйте теплоизоляцию "
                "и повторите опыт."
            ),
            "teacher_note": "Следует учесть теплообмен с окружающей средой.",
            "requires_repeat": True,
        },
        "mass_measurement_error": {
            "title": "Вероятна ошибка определения массы",
            "explanation": (
                "Расчётная температура сильно зависит от введённых масс веществ."
            ),
            "student_question": (
                "Как изменение массы одного из тел влияет на температуру равновесия?"
            ),
            "recommended_action": (
                "Повторно проверьте массу и единицы измерения."
            ),
            "teacher_note": "Необходимо проверить исходные массы.",
            "requires_repeat": True,
        },
        "insufficient_mixing": {
            "title": "Возможно недостаточное перемешивание",
            "explanation": (
                "Температура остаётся неоднородной и не выходит "
                "на устойчивое значение."
            ),
            "student_question": (
                "Почему температура в разных частях смеси может отличаться?"
            ),
            "recommended_action": (
                "Перемешайте смесь и дождитесь стабилизации показаний."
            ),
            "teacher_note": "Система могла не достичь равновесия.",
            "requires_repeat": True,
        },
    },
}


def build_feedback(prediction: ModelPrediction) -> ResearchFeedback:
    lab_rules = RECOMMENDATIONS.get(prediction.lab_id)

    if lab_rules is None:
        raise KeyError(
            f"Для лабораторной работы {prediction.lab_id} "
            "не определены рекомендации."
        )

    rule = lab_rules.get(prediction.predicted_class)

    if rule is None:
        raise KeyError(
            f"Для класса {prediction.predicted_class} "
            f"лабораторной работы {prediction.lab_id} "
            "не определена рекомендация."
        )

    confidence_percent = round(prediction.confidence * 100, 1)

    evidence = [
        f"Уверенность модели: {confidence_percent}%",
    ]

    sorted_features = sorted(
        prediction.features.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    for feature_name, feature_value in sorted_features[:3]:
        evidence.append(
            f"{feature_name}: {feature_value:.4f}"
        )

    return ResearchFeedback(
        title=str(rule["title"]),
        explanation=str(rule["explanation"]),
        evidence=evidence,
        student_question=str(rule["student_question"]),
        recommended_action=str(rule["recommended_action"]),
        teacher_note=str(rule["teacher_note"]),
        requires_repeat=bool(rule["requires_repeat"]),
    )