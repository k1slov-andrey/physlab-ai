from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.lab_registry import LAB_REGISTRY
from core.recommendation_engine import build_feedback
from core.upload_pipeline import (
    assess_quality,
    detect_lab,
    extract_series_candidates,
    normalize_experiment,
    preferred_chart_columns,
    read_uploaded_tables,
)
from labs.boyle_mariotte.module import predict as predict_boyle
from labs.boyle_mariotte.module import simulate as simulate_boyle
from labs.cooling.module import predict as predict_cooling
from labs.cooling.module import simulate as simulate_cooling
from labs.heat_balance.module import predict as predict_heat_balance
from labs.heat_balance.module import simulate as simulate_heat_balance
from labs.isochoric.module import predict as predict_isochoric
from labs.isochoric.module import simulate as simulate_isochoric


ROOT = Path(__file__).resolve().parent

LAB_LABELS = {config.short_title: key for key, config in LAB_REGISTRY.items()}
LAB_TITLES = {key: config.short_title for key, config in LAB_REGISTRY.items()}

CLASS_LABELS = {
    "normal": "Корректный эксперимент",
    "single_outlier": "Единичный выброс",
    "sensor_drift": "Дрейф датчика",
    "high_noise": "Повышенный шум",
    "air_leak": "Утечка газа",
    "temperature_change": "Изменение температуры газа",
    "volume_measurement_error": "Ошибка измерения объёма",
    "volume_instability": "Нестабильный объём",
    "temperature_sensor_lag": "Запаздывание датчика температуры",
    "heat_loss": "Теплопотери",
    "mass_measurement_error": "Ошибка измерения массы",
    "insufficient_mixing": "Недостаточное перемешивание",
    "unknown": "Надёжная гипотеза не сформирована",
}

COLUMN_LABELS = {
    "time_seconds": "Время, с",
    "measurement_number": "Номер измерения",
    "measured_temperature": "Температура, °C",
    "temperature_c": "Температура, °C",
    "pressure_kpa": "Давление, кПа",
    "volume_ml": "Объём, мл",
    "ambient_temperature_c": "Температура среды, °C",
    "hot_mass_g": "Масса образца, г",
    "cold_mass_g": "Масса воды, г",
    "hot_initial_c": "Начальная температура образца, °C",
    "cold_initial_c": "Начальная температура воды, °C",
    "calorimeter_heat_capacity_j_k": "Теплоёмкость калориметра, Дж/К",
}

LAB_UI = {
    "cooling": {
        "title": "Исследование нагревания и охлаждения",
        "goal": (
            "Исследовать изменение температуры тела во времени, установить характер "
            "приближения к температуре окружающей среды и оценить влияние условий "
            "эксперимента на форму температурной зависимости."
        ),
        "formulas": [
            r"T(t)=T_{\mathrm{ср}}+\left(T_0-T_{\mathrm{ср}}\right)e^{-kt}",
        ],
        "model_name": "Закон охлаждения Ньютона",
    },
    "boyle_mariotte": {
        "title": "Закон Бойля — Мариотта",
        "goal": (
            "Исследовать зависимость давления газа от его объёма при постоянной "
            "температуре и экспериментально проверить закон Бойля — Мариотта."
        ),
        "formulas": [
            r"pV=\mathrm{const},\qquad T=\mathrm{const}",
            r"p(V)=\frac{C}{V}",
        ],
        "model_name": "Изотермический процесс",
    },
    "isochoric": {
        "title": "Изохорный процесс",
        "goal": (
            "Исследовать зависимость давления газа от абсолютной температуры при "
            "постоянном объёме и экспериментально проверить закон Гей-Люссака."
        ),
        "formulas": [
            r"\frac{p}{T}=\mathrm{const},\qquad V=\mathrm{const}",
            r"p(T)=kT",
        ],
        "model_name": "Закон Гей-Люссака",
    },
    "heat_balance": {
        "title": "Определение удельной теплоёмкости методом теплового баланса",
        "goal": (
            "Определить удельную теплоёмкость твёрдого тела методом теплового "
            "баланса и оценить влияние теплопотерь, перемешивания и качества "
            "измерений на результат эксперимента."
        ),
        "formulas": [
            r"Q_{\mathrm{отданное}}=Q_{\mathrm{полученное}}",
            (
                r"c_{\mathrm{тела}}="
                r"\frac{\left(m_{\mathrm{воды}}c_{\mathrm{воды}}+C_{\mathrm{кал}}\right)"
                r"\left(T_{\mathrm{равн}}-T_{\mathrm{воды}}\right)}"
                r"{m_{\mathrm{тела}}\left(T_{\mathrm{тела}}-T_{\mathrm{равн}}\right)}"
            ),
        ],
        "model_name": "Уравнение теплового баланса",
    },
}

DIAGNOSTIC_GUIDANCE = {
    "normal": {
        "status": "Данные согласуются с ожидаемой физической моделью",
        "tone": "success",
        "checks": [
            "Сопоставьте направление изменения величин с физическим законом.",
            "Объясните небольшие отклонения экспериментальных точек от модельной кривой.",
            "Сформулируйте вывод, опираясь на график и рассчитанные показатели.",
        ],
    },
    "single_outlier": {
        "status": "Обнаружена отдельная точка, не согласующаяся с общей зависимостью",
        "tone": "warning",
        "checks": [
            "Найдите точку с наибольшим отклонением от модельной кривой.",
            "Проверьте запись значения и условия измерения в этот момент.",
            "Повторите только подозрительное измерение, а не весь эксперимент.",
        ],
    },
    "sensor_drift": {
        "status": "Показания датчика постепенно смещаются относительно модели",
        "tone": "warning",
        "checks": [
            "Проверьте нулевое значение и калибровку датчика.",
            "Сравните начало и конец серии при близких физических условиях.",
            "После проверки датчика повторите контрольную часть серии.",
        ],
    },
    "high_noise": {
        "status": "Разброс измерений выше ожидаемого",
        "tone": "warning",
        "checks": [
            "Проверьте устойчивость установки и контакт датчика со средой.",
            "Убедитесь, что измерения проводились через одинаковые интервалы.",
            "Повторите несколько точек и сравните величину разброса.",
        ],
    },
    "air_leak": {
        "status": "Зависимость может быть нарушена из-за негерметичности установки",
        "tone": "error",
        "checks": [
            "Проверьте соединения, пробки, шланги и герметичность сосуда.",
            "Сравните давление в начале и конце контрольного интервала.",
            "Устраните утечку и повторите экспериментальную серию.",
        ],
    },
    "temperature_change": {
        "status": "Температура газа могла изменяться во время измерений",
        "tone": "warning",
        "checks": [
            "Оцените, достаточно ли времени давалось для установления температуры.",
            "Исключите быстрое сжатие или расширение газа.",
            "Повторите серию в более медленном и одинаковом темпе.",
        ],
    },
    "volume_measurement_error": {
        "status": "Вероятна ошибка определения объёма",
        "tone": "warning",
        "checks": [
            "Проверьте цену деления и положение поршня.",
            "Учитывайте дополнительный объём трубки и соединений.",
            "Повторите точки, где объём менялся наиболее резко.",
        ],
    },
    "volume_instability": {
        "status": "Условие постоянства объёма могло нарушаться",
        "tone": "warning",
        "checks": [
            "Проверьте фиксацию сосуда, поршня или соединительных элементов.",
            "Убедитесь, что объём не менялся при нагревании.",
            "Повторите серию после механической фиксации установки.",
        ],
    },
    "temperature_sensor_lag": {
        "status": "Датчик температуры мог не успевать за изменением среды",
        "tone": "warning",
        "checks": [
            "Дождитесь стабилизации показаний перед записью точки.",
            "Используйте одинаковое время выдержки для каждого измерения.",
            "Сравните температуру датчика и температуру внешней среды.",
        ],
    },
    "heat_loss": {
        "status": "На результат могли повлиять теплопотери",
        "tone": "warning",
        "checks": [
            "Сократите время переноса нагретого образца.",
            "Проверьте теплоизоляцию калориметра и наличие крышки.",
            "Используйте экстраполяцию к моменту смешивания при обработке данных.",
        ],
    },
    "mass_measurement_error": {
        "status": "Параметры массы могут быть заданы или измерены неверно",
        "tone": "warning",
        "checks": [
            "Проверьте единицы измерения массы.",
            "Повторно взвесьте образец и воду.",
            "Убедитесь, что образец был сухим и полностью помещён в воду.",
        ],
    },
    "insufficient_mixing": {
        "status": "Температура смеси ещё не достигла устойчивого значения",
        "tone": "warning",
        "checks": [
            "Перемешайте смесь одинаковыми движениями.",
            "Дождитесь устойчивого участка температурной кривой.",
            "Зафиксируйте температуру после стабилизации, а не сразу после смешивания.",
        ],
    },
    "unknown": {
        "status": (
            "Входные данные не позволяют надёжно выбрать одну диагностическую "
            "гипотезу."
        ),
        "tone": "warning",
        "checks": [
            "Проверьте названия столбцов и единицы измерения.",
            "Сопоставьте диапазоны значений с условиями лабораторной работы.",
            "Повторите анализ после проверки данных или обратитесь к преподавателю.",
        ],
    },
}


LAB_ASSESSMENT_QUESTIONS = {
    "cooling": [
        {
            "id": "hypothesis",
            "competency": "Гипотеза",
            "prompt": "Какая гипотеза соответствует исследованию охлаждения тела?",
            "options": [
                ("Чем больше разность температур тела и среды, тем быстрее изменяется температура тела.", 3, "Гипотеза связывает скорость изменения температуры с разностью температур."),
                ("Температура тела всегда уменьшается на одинаковое число градусов за минуту.", 1, "Линейное изменение возможно только на коротком участке, но не описывает весь процесс."),
                ("Температура тела не зависит от температуры окружающей среды.", 0, "Температура среды определяет предельное значение процесса."),
                ("Скорость охлаждения определяется только массой тела.", 1, "Масса влияет на процесс, но не является единственным определяющим фактором."),
            ],
        },
        {
            "id": "model",
            "competency": "Физическая модель",
            "prompt": "Какой вид должна иметь идеальная температурная зависимость?",
            "options": [
                ("Температура постепенно приближается к температуре среды по экспоненциальному закону.", 3, "Это соответствует закону охлаждения Ньютона."),
                ("Температура изменяется строго линейно до нуля.", 1, "Строгая линейность не описывает приближение к температуре среды."),
                ("Температура мгновенно становится равной температуре среды.", 0, "Теплообмен требует времени."),
                ("Температура периодически растёт и падает без внешнего воздействия.", 0, "Такая зависимость не соответствует рассматриваемой модели."),
            ],
        },
        {
            "id": "control",
            "competency": "Контроль условий",
            "prompt": "Какое условие важно сохранять при сравнении экспериментальных серий?",
            "options": [
                ("Одинаковые условия теплообмена: среду, положение датчика и способ размещения тела.", 3, "Контроль условий позволяет сравнивать серии корректно."),
                ("Только одинаковое количество точек на графике.", 1, "Число точек важно для обработки, но не заменяет контроль физических условий."),
                ("Только одинаковое начальное время записи.", 1, "Время запуска не является главным контролируемым условием."),
                ("Условия можно менять в ходе опыта без фиксации.", 0, "Изменение условий делает интерпретацию неоднозначной."),
            ],
        },
        {
            "id": "evidence",
            "competency": "Анализ данных",
            "prompt": "Что лучше всего подтверждает согласие данных с моделью?",
            "options": [
                ("Экспериментальные точки приближаются к модельной кривой, а отклонения не имеют систематического характера.", 3, "Оценивается и форма зависимости, и характер остаточных отклонений."),
                ("На графике есть много точек.", 1, "Количество точек само по себе не подтверждает модель."),
                ("Последняя температура меньше первой.", 2, "Это подтверждает направление процесса, но не всю форму зависимости."),
                ("Все значения температуры являются целыми числами.", 0, "Формат чисел не является физическим доказательством."),
            ],
        },
        {
            "id": "conclusion",
            "competency": "Интерпретация",
            "prompt": "Какой вывод корректно сформулировать по результатам опыта?",
            "options": [
                ("Температура тела стремится к температуре среды, а скорость изменения уменьшается по мере сближения температур.", 3, "Вывод отражает основную физическую закономерность."),
                ("Тело обязательно охладится до 0 °C.", 0, "Предельное значение определяется температурой среды."),
                ("Температура всегда меняется с постоянной скоростью.", 1, "Скорость обычно уменьшается по мере приближения к равновесию."),
                ("Любое отклонение точки означает поломку датчика.", 0, "Отклонения могут быть следствием нескольких причин."),
            ],
        },
    ],
    "boyle_mariotte": [
        {
            "id": "hypothesis",
            "competency": "Гипотеза",
            "prompt": "Какая гипотеза соответствует закону Бойля — Мариотта?",
            "options": [
                ("При уменьшении объёма газа его давление увеличивается, если температура постоянна.", 3, "Гипотеза отражает обратную зависимость давления от объёма."),
                ("Давление не зависит от объёма газа.", 0, "Это противоречит изотермическому закону."),
                ("Давление увеличивается вместе с объёмом.", 0, "Для изотермического процесса зависимость обратная."),
                ("Давление определяется только временем измерения.", 0, "Время не является основной переменной закона."),
            ],
        },
        {
            "id": "model",
            "competency": "Физическая модель",
            "prompt": "Какой показатель должен оставаться приблизительно постоянным?",
            "options": [
                ("Произведение давления на объём pV.", 3, "Для изотермического процесса pV ≈ const."),
                ("Сумма давления и объёма p + V.", 0, "Такой инвариант не следует из закона."),
                ("Отношение объёма к времени V/t.", 0, "Это не характеристика закона Бойля — Мариотта."),
                ("Разность давления и объёма p − V.", 0, "Величины имеют разные единицы и не образуют физический инвариант."),
            ],
        },
        {
            "id": "control",
            "competency": "Контроль условий",
            "prompt": "Какое условие обязательно для проверки закона?",
            "options": [
                ("Температура и масса газа должны оставаться постоянными.", 3, "Закон относится к данной массе газа при постоянной температуре."),
                ("Объём должен оставаться постоянным.", 0, "В опыте объём является изменяемой величиной."),
                ("Давление должно оставаться постоянным.", 0, "В опыте давление изменяется вместе с объёмом."),
                ("Температуру можно менять без учёта.", 0, "Изменение температуры нарушает условия изотермического процесса."),
            ],
        },
        {
            "id": "evidence",
            "competency": "Анализ данных",
            "prompt": "Что является наиболее сильным подтверждением закона?",
            "options": [
                ("Значения pV близки друг к другу, а график p(V) имеет вид гиперболы.", 3, "Используются и численный инвариант, и форма зависимости."),
                ("Давление в последней точке больше, чем в первой.", 2, "Направление зависимости учтено, но этого недостаточно для полной проверки."),
                ("Все объёмы записаны без пропусков.", 1, "Полнота данных важна, но не подтверждает закон."),
                ("На графике нет отрицательных значений.", 1, "Это физически ожидаемо, но не доказывает обратную зависимость."),
            ],
        },
        {
            "id": "conclusion",
            "competency": "Интерпретация",
            "prompt": "Какой вывод соответствует эксперименту?",
            "options": [
                ("При постоянной температуре давление газа обратно пропорционально объёму, а pV остаётся приблизительно постоянным.", 3, "Вывод полностью связывает условия и наблюдаемую зависимость."),
                ("Давление прямо пропорционально объёму.", 0, "Это противоположно экспериментальной модели."),
                ("Закон подтверждён только потому, что давление изменялось.", 1, "Нужно оценивать конкретный характер зависимости."),
                ("Любое несовпадение pV означает, что закон неверен.", 0, "Небольшие отклонения объясняются погрешностями эксперимента."),
            ],
        },
    ],
    "isochoric": [
        {
            "id": "hypothesis",
            "competency": "Гипотеза",
            "prompt": "Какая гипотеза соответствует изохорному процессу?",
            "options": [
                ("При увеличении абсолютной температуры давление газа увеличивается, если объём постоянен.", 3, "Гипотеза отражает прямую зависимость p от T."),
                ("При нагревании давление всегда уменьшается.", 0, "Это противоречит изохорному закону."),
                ("Давление не зависит от температуры.", 0, "Температура является основной изменяемой величиной."),
                ("Давление зависит только от времени нагревания.", 1, "Время может влиять косвенно, но закон связывает давление и температуру."),
            ],
        },
        {
            "id": "model",
            "competency": "Физическая модель",
            "prompt": "Какую температуру необходимо использовать в отношении p/T?",
            "options": [
                ("Абсолютную температуру в кельвинах.", 3, "Газовые законы используют абсолютную температурную шкалу."),
                ("Температуру в градусах Цельсия без преобразования.", 1, "Для графика можно использовать °C, но отношение p/T требует кельвинов."),
                ("Любую шкалу, если числа положительные.", 0, "Выбор шкалы принципиален для пропорциональности."),
                ("Только разность между соседними температурами.", 0, "Отношение закона строится по абсолютным значениям."),
            ],
        },
        {
            "id": "control",
            "competency": "Контроль условий",
            "prompt": "Какое условие должно сохраняться?",
            "options": [
                ("Объём и масса газа должны оставаться постоянными.", 3, "Это определение изохорного процесса для данной массы газа."),
                ("Давление должно оставаться постоянным.", 0, "Давление является измеряемой изменяющейся величиной."),
                ("Температура должна оставаться постоянной.", 0, "Температуру в опыте изменяют."),
                ("Объём можно менять без фиксации.", 0, "Изменение объёма нарушает изохорный процесс."),
            ],
        },
        {
            "id": "evidence",
            "competency": "Анализ данных",
            "prompt": "Как лучше проверить соответствие модели?",
            "options": [
                ("Построить p(T) и проверить линейность, а также сравнить значения p/T в кельвинах.", 3, "Два взаимодополняющих способа подтверждают закон."),
                ("Проверить только, что последняя температура выше первой.", 1, "Это не показывает связь давления с температурой."),
                ("Посчитать среднее давление без учёта температуры.", 0, "Среднее не характеризует зависимость."),
                ("Удалить все точки, не лежащие точно на прямой.", 0, "Такое удаление искажает эксперимент."),
            ],
        },
        {
            "id": "conclusion",
            "competency": "Интерпретация",
            "prompt": "Какой вывод корректен?",
            "options": [
                ("При постоянном объёме давление данной массы газа прямо пропорционально абсолютной температуре.", 3, "Вывод содержит условие и физическую зависимость."),
                ("Давление прямо пропорционально температуре в °C при любых условиях.", 1, "Не указана абсолютная шкала и постоянство объёма."),
                ("Нагревание не влияет на давление газа.", 0, "Это противоречит данным и модели."),
                ("Если график не идеален, закон нельзя проверять экспериментально.", 0, "Погрешности являются нормальной частью измерений."),
            ],
        },
    ],
    "heat_balance": [
        {
            "id": "hypothesis",
            "competency": "Гипотеза",
            "prompt": "Какая гипотеза соответствует методу теплового баланса?",
            "options": [
                ("Количество теплоты, отданное горячим телом, приблизительно равно теплоте, полученной водой и калориметром.", 3, "Гипотеза отражает закон сохранения энергии с учётом калориметра."),
                ("Горячее тело не передаёт тепло воде.", 0, "Это противоречит наблюдаемому теплообмену."),
                ("Температура смеси определяется только массой воды.", 1, "Масса воды важна, но результат зависит и от других параметров."),
                ("Количество теплоты всегда равно нулю.", 0, "При теплообмене энергия передаётся между частями системы."),
            ],
        },
        {
            "id": "model",
            "competency": "Физическая модель",
            "prompt": "Что необходимо включить в уравнение теплового баланса?",
            "options": [
                ("Теплоту, полученную водой, и теплоёмкость калориметра.", 3, "Учёт калориметра повышает физическую корректность модели."),
                ("Только конечную температуру смеси.", 1, "Одной температуры недостаточно для расчёта."),
                ("Только массу металлического образца.", 1, "Для баланса нужны массы, температуры и теплоёмкости."),
                ("Время суток проведения опыта.", 0, "Оно не входит в уравнение теплового баланса."),
            ],
        },
        {
            "id": "control",
            "competency": "Контроль условий",
            "prompt": "Как уменьшить систематическую ошибку опыта?",
            "options": [
                ("Сократить перенос образца, использовать теплоизоляцию и быстро начать перемешивание.", 3, "Эти действия уменьшают теплопотери и неоднородность температуры."),
                ("Дольше держать калориметр открытым.", 0, "Это увеличивает теплопотери."),
                ("Измерять температуру только сразу после помещения образца.", 1, "Нужно учитывать установление равновесия и перемешивание."),
                ("Не учитывать температуру горячего образца.", 0, "Она необходима для расчёта теплоёмкости."),
            ],
        },
        {
            "id": "evidence",
            "competency": "Анализ данных",
            "prompt": "Как определить температуру теплового равновесия?",
            "options": [
                ("Использовать устойчивый участок кривой или экстраполяцию к моменту смешивания с учётом теплопотерь.", 3, "Такой подход учитывает динамику и возможный теплообмен со средой."),
                ("Выбрать максимальную точку без анализа графика.", 1, "Максимальная точка может быть выбросом или кратковременным значением."),
                ("Взять любое значение из середины таблицы.", 0, "Выбор должен иметь физическое обоснование."),
                ("Сложить начальные температуры.", 0, "Температуры нельзя складывать для определения равновесия."),
            ],
        },
        {
            "id": "conclusion",
            "competency": "Интерпретация",
            "prompt": "Какой вывод является корректным?",
            "options": [
                ("Удельная теплоёмкость определяется из баланса энергии, а отклонение от справочного значения может быть связано с теплопотерями и измерениями.", 3, "Вывод связывает расчёт, модель и ограничения опыта."),
                ("Полученное число всегда должно точно совпадать со справочным.", 0, "Эксперимент содержит погрешности и систематические потери."),
                ("Материал определяется только по конечной температуре.", 1, "Нужен полный расчёт с массами и начальными температурами."),
                ("Теплопотери не влияют на результат.", 0, "Теплопотери являются одной из основных систематических ошибок."),
            ],
        },
    ],
}

DIAGNOSTIC_ACTIONS = {
    "normal": "Принять данные, сопоставить их с моделью и сформулировать обоснованный вывод.",
    "single_outlier": "Проверить и при необходимости повторить отдельное подозрительное измерение.",
    "sensor_drift": "Проверить калибровку датчика и выполнить контрольное измерение.",
    "high_noise": "Проверить устойчивость установки и повторить несколько точек.",
    "air_leak": "Проверить герметичность установки и затем повторить серию.",
    "temperature_change": "Стабилизировать температуру и повторить измерения в более медленном темпе.",
    "volume_measurement_error": "Проверить шкалу и способ определения объёма.",
    "volume_instability": "Зафиксировать объём установки и повторить серию.",
    "temperature_sensor_lag": "Увеличить время выдержки до стабилизации датчика.",
    "heat_loss": "Уменьшить теплопотери и повторить опыт с контролем времени переноса.",
    "mass_measurement_error": "Повторно измерить массы и проверить единицы.",
    "insufficient_mixing": "Перемешать смесь и дождаться устойчивого участка температуры.",
}

SIMULATORS = {
    "cooling": simulate_cooling,
    "boyle_mariotte": simulate_boyle,
    "isochoric": simulate_isochoric,
    "heat_balance": simulate_heat_balance,
}

PREDICTORS = {
    "cooling": predict_cooling,
    "boyle_mariotte": predict_boyle,
    "isochoric": predict_isochoric,
    "heat_balance": predict_heat_balance,
}

PLOT_CONFIG = {
    "scrollZoom": False,
    "displaylogo": False,
    "displayModeBar": False,
    "doubleClick": False,
    "responsive": True,
}


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fc;
        }
        .main .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .physlab-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: #163a63;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .physlab-subtitle {
            color: #5e6c84;
            font-size: 1.05rem;
            margin-bottom: 1.4rem;
        }
        .lab-card {
            background: linear-gradient(135deg, #edf5ff 0%, #f8fbff 100%);
            border: 1px solid #d4e4f6;
            border-radius: 18px;
            padding: 22px 24px;
            margin: 10px 0 14px 0;
            box-shadow: 0 6px 20px rgba(26, 65, 105, 0.06);
        }
        .lab-kicker {
            color: #2468a9;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .lab-title {
            color: #173b63;
            font-size: 1.65rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .lab-goal {
            color: #26384b;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .section-label {
            color: #183f68;
            font-weight: 800;
            font-size: 1.22rem;
            margin-top: 1.2rem;
            margin-bottom: 0.5rem;
        }
        .diagnostic-card {
            border-radius: 18px;
            padding: 20px 22px;
            margin-top: 10px;
            margin-bottom: 14px;
        }
        .diagnostic-success {
            background: #edf8f1;
            border: 1px solid #b9e2c7;
        }
        .diagnostic-warning {
            background: #fff8e8;
            border: 1px solid #f0d28a;
        }
        .diagnostic-error {
            background: #fff1f0;
            border: 1px solid #efb8b3;
        }
        .diagnostic-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #1f3448;
            margin-bottom: 8px;
        }
        .diagnostic-text {
            color: #35485b;
            line-height: 1.5;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e1e8f0;
            border-radius: 14px;
            padding: 12px 14px;
        }
        div[data-testid="stPlotlyChart"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_analysis_if_context_changed(context_key: str) -> None:
    if st.session_state.get("analysis_context") != context_key:
        st.session_state["analysis_context"] = context_key
        st.session_state.pop("prediction", None)
        st.session_state.pop("feedback", None)
        st.session_state.pop("competency_scores", None)
        st.session_state.pop("assessment_result", None)


def dataframe_fingerprint(dataframe: pd.DataFrame, lab_id: str) -> str:
    if dataframe.empty:
        return f"{lab_id}:empty"
    hashed = pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    return lab_id + hashlib.sha256(hashed).hexdigest()


def show_model_status() -> None:
    summary_path = ROOT / "evaluation" / "grid_search" / "all_labs_grid_search_summary.csv"
    if not summary_path.exists():
        summary_path = ROOT / "evaluation" / "all_labs_summary.csv"
    if not summary_path.exists():
        return

    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return

    with st.sidebar.expander("Состояние моделей"):
        if "lab_id" in summary.columns:
            for _, row in summary.iterrows():
                lab_id = str(row.get("lab_id", ""))
                title = LAB_TITLES.get(lab_id, lab_id)
                f1_value = row.get(
                    "current_holdout_macro_f1",
                    row.get("macro_f1", None),
                )
                if pd.notna(f1_value):
                    st.metric(title, f"Macro F1 {float(f1_value):.3f}")
        st.caption("Метрики рассчитаны на групповой синтетической holdout-выборке.")


def render_lab_header(lab_id: str) -> None:
    meta = LAB_UI[lab_id]

    st.markdown(
        f"""
        <div class="lab-card">
            <div class="lab-kicker">Лабораторная работа</div>
            <div class="lab-title">{meta["title"]}</div>
            <div class="lab-goal">
                <strong>Цель:</strong> {meta["goal"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            f"#### Физическая модель · {meta['model_name']}"
        )
        formula_columns = st.columns(len(meta["formulas"]))
        for column, formula in zip(formula_columns, meta["formulas"]):
            with column:
                st.latex(formula)


def metadata_controls(
    lab_id: str,
    frame: pd.DataFrame,
) -> tuple[dict[str, float | str], bool]:
    metadata: dict[str, float | str] = {}
    confirmed = True

    with st.expander(
        "Параметры установки и протокола",
        expanded=lab_id == "heat_balance",
    ):
        if lab_id == "boyle_mariotte":
            missing_temperature = not any(
                "temperature" in str(column) or "temp" in str(column)
                for column in frame.columns
            )
            missing_atmosphere = not any(
                "atmospheric" in str(column) or "p_atm" in str(column)
                for column in frame.columns
            )
            metadata["temperature_c"] = st.number_input(
                "Температура газа, °C",
                value=22.0,
                step=0.1,
                help="Используется, если температура отсутствует в файле.",
            )
            metadata["atmospheric_pressure_kpa"] = st.number_input(
                "Атмосферное давление, кПа",
                value=101.325,
                step=0.1,
                format="%.3f",
                help="Нужно для перевода избыточного давления в абсолютное.",
            )
            if missing_temperature or missing_atmosphere:
                st.caption(
                    "Часть параметров будет добавлена как постоянная для всей серии."
                )

        elif lab_id == "isochoric":
            metadata["volume_ml"] = st.number_input(
                "Постоянный объём сосуда, мл",
                min_value=1.0,
                value=250.0,
                step=10.0,
                help="Используется, если объём отсутствует в файле.",
            )

        elif lab_id == "heat_balance":
            metadata["hot_mass_g"] = st.number_input(
                "Масса горячего образца, г",
                min_value=0.1,
                value=50.0,
                step=1.0,
            )
            metadata["cold_mass_g"] = st.number_input(
                "Масса воды, г",
                min_value=0.1,
                value=150.0,
                step=1.0,
            )
            metadata["hot_initial_c"] = st.number_input(
                "Начальная температура образца, °C",
                value=95.0,
                step=0.5,
            )
            metadata["cold_initial_c"] = st.number_input(
                "Начальная температура воды, °C",
                value=22.0,
                step=0.5,
            )
            metadata["calorimeter_heat_capacity_j_k"] = st.number_input(
                "Теплоёмкость калориметра, Дж/К",
                min_value=0.0,
                value=70.0,
                step=5.0,
            )
            material_labels = {
                "steel": "Сталь",
                "aluminum": "Алюминий",
                "copper": "Медь",
                "iron": "Железо",
                "lead": "Свинец",
            }
            metadata["material"] = st.selectbox(
                "Материал образца",
                list(material_labels),
                format_func=material_labels.get,
            )
            confirmed = st.checkbox(
                "Подтверждаю, что параметры внесены по протоколу эксперимента",
                value=False,
            )
            st.caption(
                "Без подтверждения модель не запускается на параметрах, "
                "добавленных вручную."
            )

    return metadata, confirmed


def show_quality(report: Any, mapping: dict[str, str]) -> None:
    st.markdown('<div class="section-label">Проверка качества данных</div>', unsafe_allow_html=True)

    columns = st.columns(4)
    columns[0].metric("Строк", report.rows)
    columns[1].metric("Пригодных", report.usable_rows)
    columns[2].metric("Пропусков", report.missing_cells)
    columns[3].metric("Вне диапазона", report.out_of_range_cells)

    if report.ready_for_model:
        st.success("Данные готовы для анализа моделью.")
    else:
        st.error("Данные пока не готовы для анализа моделью.")

    for issue in report.issues:
        st.warning(issue)

    with st.expander("Паспорт преобразования"):
        if mapping:
            mapping_frame = pd.DataFrame(
                {
                    "Поле PhysLab AI": list(mapping.keys()),
                    "Исходный столбец": list(mapping.values()),
                }
            )
            st.dataframe(
                mapping_frame,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("Автоматическое сопоставление столбцов не выполнено.")

        st.dataframe(
            pd.DataFrame([report.as_dict()]),
            hide_index=True,
            use_container_width=True,
        )


def _fit_exponential_reference(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x0 = float(np.min(x))
    shifted_x = x - x0
    span = max(float(np.max(shifted_x)), 1.0)
    tail_count = max(3, len(y) // 8)
    equilibrium_guess = float(np.median(y[-tail_count:]))
    amplitude_guess = float(y[0] - equilibrium_guess)
    k_guess = 3.0 / span

    try:
        from scipy.optimize import curve_fit

        def model(
            values: np.ndarray,
            equilibrium: float,
            amplitude: float,
            k_value: float,
        ) -> np.ndarray:
            return equilibrium + amplitude * np.exp(-k_value * values)

        parameters, _ = curve_fit(
            model,
            shifted_x,
            y,
            p0=[equilibrium_guess, amplitude_guess, k_guess],
            bounds=(
                [-500.0, -1500.0, 1e-9],
                [1500.0, 1500.0, 20.0],
            ),
            maxfev=20_000,
        )
        return model(shifted_x, *parameters)
    except Exception:
        return (
            equilibrium_guess
            + amplitude_guess * np.exp(-k_guess * shifted_x)
        )


def build_reference_curve(
    lab_id: str,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        return y

    if lab_id in {"cooling", "heat_balance"}:
        return _fit_exponential_reference(x, y)

    if lab_id == "boyle_mariotte":
        safe_volume = np.where(np.abs(x) < 1e-12, np.nan, x)
        constants = safe_volume * y
        constant = float(np.nanmedian(constants))
        return constant / safe_volume

    if lab_id == "isochoric":
        temperature_kelvin = x + 273.15
        valid = np.abs(temperature_kelvin) > 1e-12
        coefficient = float(
            np.nanmedian(
                np.divide(
                    y,
                    temperature_kelvin,
                    out=np.full_like(y, np.nan, dtype=float),
                    where=valid,
                )
            )
        )
        return coefficient * temperature_kelvin

    return y.copy()


def calculate_reference_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(reference)

    if valid.sum() < 2:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "max_deviation": float("nan"),
        }

    actual = actual[valid]
    reference = reference[valid]
    residuals = actual - reference

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    max_deviation = float(np.max(np.abs(residuals)))

    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = (
        1.0 - float(np.sum(residuals**2)) / denominator
        if denominator > 1e-12
        else float("nan")
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_deviation": max_deviation,
    }


def show_experiment_chart(
    dataframe: pd.DataFrame,
    lab_id: str,
) -> None:
    x_column, y_columns = preferred_chart_columns(dataframe, lab_id)

    if x_column not in dataframe.columns or not y_columns:
        st.warning("Не удалось определить величины для построения графика.")
        return

    y_column = y_columns[0]
    chart_frame = dataframe[[x_column, y_column]].copy()
    chart_frame[x_column] = pd.to_numeric(
        chart_frame[x_column],
        errors="coerce",
    )
    chart_frame[y_column] = pd.to_numeric(
        chart_frame[y_column],
        errors="coerce",
    )
    chart_frame = (
        chart_frame
        .dropna()
        .sort_values(x_column)
        .drop_duplicates(subset=[x_column], keep="last")
    )

    if len(chart_frame) < 2:
        st.warning("Недостаточно точек для построения сравнительного графика.")
        return

    x = chart_frame[x_column].to_numpy(dtype=float)
    actual = chart_frame[y_column].to_numpy(dtype=float)
    reference = build_reference_curve(lab_id, x, actual)
    metrics = calculate_reference_metrics(actual, reference)

    st.markdown(
        '<div class="section-label">Сопоставление с физической моделью</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Модельная кривая показывает ожидаемую форму зависимости. "
        "Она служит ориентиром для анализа, а не заменяет интерпретацию эксперимента."
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x,
            y=reference,
            mode="lines",
            name="Физическая модель",
            line={
                "width": 4,
                "dash": "dash",
                "color": "#f28e2b",
            },
            hovertemplate=(
                f"{COLUMN_LABELS.get(x_column, x_column)}: %{{x:.3g}}"
                "<br>Модель: %{y:.3f}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines+markers",
            name="Эксперимент",
            line={
                "width": 3,
                "color": "#1769c2",
            },
            marker={
                "size": 7,
                "color": "#1769c2",
            },
            hovertemplate=(
                f"{COLUMN_LABELS.get(x_column, x_column)}: %{{x:.3g}}"
                "<br>Измерение: %{y:.3f}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        height=560,
        margin={"l": 30, "r": 25, "t": 55, "b": 125},
        title={
            "text": "Экспериментальная зависимость и модельная кривая",
            "x": 0.01,
            "xanchor": "left",
        },
        xaxis_title=COLUMN_LABELS.get(x_column, x_column),
        yaxis_title=COLUMN_LABELS.get(y_column, y_column),
        hovermode="x unified",
        dragmode=False,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13},
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    figure.update_xaxes(
        fixedrange=True,
        showgrid=True,
        gridcolor="#e8edf3",
        zeroline=False,
    )
    figure.update_yaxes(
        fixedrange=True,
        showgrid=True,
        gridcolor="#e8edf3",
        zeroline=False,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "RMSE",
        "—" if np.isnan(metrics["rmse"]) else f"{metrics['rmse']:.3f}",
        help="Среднеквадратичное отклонение измерений от модельной кривой.",
    )
    metric_columns[1].metric(
        "MAE",
        "—" if np.isnan(metrics["mae"]) else f"{metrics['mae']:.3f}",
        help="Среднее абсолютное отклонение от модельной кривой.",
    )
    metric_columns[2].metric(
        "R²",
        "—" if np.isnan(metrics["r2"]) else f"{metrics['r2']:.3f}",
        help="Доля вариации данных, объясняемая модельной зависимостью.",
    )
    metric_columns[3].metric(
        "Макс. отклонение",
        (
            "—"
            if np.isnan(metrics["max_deviation"])
            else f"{metrics['max_deviation']:.3f}"
        ),
        help="Наибольшее абсолютное отклонение экспериментальной точки.",
    )


def render_diagnostic_panel(
    prediction: Any,
    feedback: Any,
    source_mode: str,
) -> None:
    predicted_class = str(prediction.predicted_class)
    guidance = DIAGNOSTIC_GUIDANCE.get(
        predicted_class,
        {
            "status": "Результат требует дополнительной проверки",
            "tone": "warning",
            "checks": [
                "Проверьте исходные данные и единицы измерения.",
                "Сопоставьте эксперимент с физической моделью.",
                "Обсудите неоднозначный результат с преподавателем.",
            ],
        },
    )

    class_title = CLASS_LABELS.get(predicted_class, predicted_class)
    tone_class = f"diagnostic-{guidance['tone']}"

    st.markdown(
        f"""
        <div class="diagnostic-card {tone_class}">
            <div class="diagnostic-title">
                Диагностическая гипотеза: {class_title}
            </div>
            <div class="diagnostic-text">
                {guidance["status"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_mode == "Загрузить реальный файл":
        st.warning(
            "Это диагностическая гипотеза модели. Окончательное решение "
            "принимает учащийся совместно с преподавателем."
        )

    confidence_value = float(
        np.clip(float(prediction.confidence), 0.0, 1.0)
    )

    if confidence_value >= 0.85:
        confidence_label = "Высокая"
    elif confidence_value >= 0.65:
        confidence_label = "Средняя"
    else:
        confidence_label = "Низкая"

    if not getattr(prediction, "accepted", True):
        confidence_label = "Недостаточная"

    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Определённость гипотезы",
        confidence_label,
        help=(
            "Показывает, насколько явно модель выделяет одну гипотезу "
            "среди альтернатив. Это не точность модели и не вероятность "
            "правильного ответа."
        ),
    )
    metric_columns[1].metric(
        "Статус работы",
        (
            "Можно переходить к выводу"
            if predicted_class == "normal"
            else "Нужно проверить данные и эксперимент"
            if predicted_class == "unknown"
            else "Нужно проверить эксперимент"
        ),
    )

    reliability_warnings = getattr(prediction, "reliability_warnings", [])
    if reliability_warnings:
        for warning in reliability_warnings:
            st.warning(warning)

    st.caption(
        "В демонстрационном режиме сценарий задаётся заранее, поэтому "
        "его признаки могут быть выражены очень отчётливо. "
        "Оценка модели не заменяет проверку данных и решение учащегося."
    )

    st.write(feedback.explanation)

    if feedback.evidence:
        with st.expander("Какие признаки использовала модель"):
            for evidence in feedback.evidence:
                st.write(f"• {evidence}")

    question_column, action_column = st.columns(2)

    with question_column:
        st.info(
            f"**Исследовательский вопрос**\n\n{feedback.student_question}"
        )

    with action_column:
        st.success(
            f"**Рекомендуемое действие**\n\n{feedback.recommended_action}"
        )

    st.markdown("#### Что проверить самостоятельно")
    for check in guidance["checks"]:
        st.checkbox(
            check,
            key=f"check_{predicted_class}_{hash(check)}",
        )

    probabilities = getattr(prediction, "probabilities", {}) or {}
    if probabilities:
        probability_rows = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        with st.expander("Альтернативные гипотезы модели"):
            probability_table = pd.DataFrame(
                {
                    "Гипотеза": [
                        CLASS_LABELS.get(class_name, class_name)
                        for class_name, _ in probability_rows
                    ],
                    "Вероятность": [
                        f"{float(probability) * 100:.1f}%"
                        for _, probability in probability_rows
                    ],
                }
            )
            st.dataframe(
                probability_table,
                hide_index=True,
                use_container_width=True,
            )



def build_assessment_questions(
    lab_id: str,
    predicted_class: str,
) -> list[dict[str, Any]]:
    questions = [dict(question) for question in LAB_ASSESSMENT_QUESTIONS[lab_id]]

    questions.append(
        {
            "id": "planning",
            "competency": "Планирование",
            "prompt": "Какая последовательность лучше отражает исследовательскую работу?",
            "options": [
                ("Гипотеза → план → эксперимент → анализ данных → решение → вывод.", 3, "Последовательность сохраняет логику исследования."),
                ("Вывод → эксперимент → гипотеза → исправление данных.", 0, "Вывод не должен предшествовать сбору и анализу данных."),
                ("Эксперимент → готовый ответ → удаление неудобных точек.", 0, "Такой порядок подменяет исследование подтверждением ответа."),
                ("Сначала получить график, затем придумать подходящую гипотезу.", 1, "Гипотеза должна формулироваться до анализа результата."),
            ],
        }
    )

    correct_action = DIAGNOSTIC_ACTIONS.get(
        predicted_class,
        "Проверить данные и сопоставить их с физической моделью.",
    )
    questions.append(
        {
            "id": "correction",
            "competency": "Коррекция эксперимента",
            "prompt": "Какое следующее действие наиболее обосновано результатом анализа?",
            "options": [
                (correct_action, 3, "Действие связано с обнаруженной диагностической гипотезой."),
                ("Сразу удалить все точки, которые отличаются от модельной кривой.", 0, "Удаление данных требует отдельного обоснования."),
                ("Полностью игнорировать результат анализа и завершить работу.", 0, "Результат нужно проверить, а не принимать или отвергать автоматически."),
                ("Повторить весь эксперимент без проверки возможной причины.", 1, "Повтор может быть полезен, но сначала рациональнее проверить предполагаемую причину."),
            ],
        }
    )

    questions.append(
        {
            "id": "critical_ai",
            "competency": "Критическая работа с ИИ",
            "prompt": "Как следует относиться к диагностической гипотезе ИИ?",
            "options": [
                ("Проверить её по графику, исходным данным и условиям опыта, после чего принять собственное решение.", 3, "ИИ используется как инструмент проверки, а не как источник готового ответа."),
                ("Всегда принимать её, если определённость модели высокая.", 1, "Высокая определённость не гарантирует истинность гипотезы."),
                ("Всегда отклонять её, потому что ИИ может ошибаться.", 0, "Возможность ошибки не означает, что результат нельзя использовать для проверки."),
                ("Переписать рекомендацию ИИ в вывод без дополнительного анализа.", 0, "Вывод должен принадлежать учащемуся и опираться на данные."),
            ],
        }
    )

    return questions


def evaluate_assessment(
    questions: list[dict[str, Any]],
    answers: dict[str, str],
) -> dict[str, Any]:
    profile: dict[str, float] = {}
    details: list[dict[str, Any]] = []
    total_score = 0
    max_score = 0

    for question in questions:
        selected_text = answers.get(question["id"], "")
        selected_option = next(
            (
                option
                for option in question["options"]
                if option[0] == selected_text
            ),
            None,
        )

        if selected_option is None:
            score = 0
            feedback = "Ответ не выбран."
        else:
            _, score, feedback = selected_option

        competency = question["competency"]
        profile[competency] = float(score)
        total_score += int(score)
        max_score += 3

        details.append(
            {
                "Вопрос": question["prompt"],
                "Компетенция": competency,
                "Баллы": f"{score}/3",
                "Комментарий": feedback,
            }
        )

    if total_score >= max_score * 0.8:
        level = "Высокий уровень проявленных исследовательских действий"
    elif total_score >= max_score * 0.55:
        level = "Достаточный уровень, требующий точечной доработки"
    else:
        level = "Базовый уровень, необходим разбор исследовательской логики"

    return {
        "profile": profile,
        "details": details,
        "total_score": total_score,
        "max_score": max_score,
        "level": level,
    }


def render_competency_profile(
    result: dict[str, Any],
    hypothesis: str,
    conclusion: str,
) -> None:
    profile = result["profile"]
    labels = list(profile)
    score_values = list(profile.values())

    labels_closed = labels + [labels[0]]
    scores_closed = score_values + [score_values[0]]

    radar = go.Figure(
        go.Scatterpolar(
            r=scores_closed,
            theta=labels_closed,
            fill="toself",
            name="Текущая работа",
            line={"width": 3, "color": "#1769c2"},
            fillcolor="rgba(23, 105, 194, 0.20)",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        )
    )
    radar.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 3],
                "tickvals": [0, 1, 2, 3],
                "gridcolor": "#dfe7f0",
            },
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "gridcolor": "#e7edf4",
            },
        },
        showlegend=False,
        height=650,
        margin={"l": 110, "r": 110, "t": 60, "b": 60},
        dragmode=False,
        paper_bgcolor="white",
    )

    st.markdown(
        '<div class="section-label">Профиль исследовательских действий</div>',
        unsafe_allow_html=True,
    )

    score_column, level_column = st.columns([1, 2])
    score_column.metric(
        "Результат диагностики",
        f"{result['total_score']} из {result['max_score']}",
    )
    level_column.info(result["level"])

    st.plotly_chart(
        radar,
        use_container_width=True,
        config=PLOT_CONFIG,
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Исследовательское действие": labels,
                "Уровень (0–3)": score_values,
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Разбор ответов"):
        st.dataframe(
            pd.DataFrame(result["details"]),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Развёрнутые ответы ученика"):
        st.markdown("**Гипотеза**")
        st.write(hypothesis.strip() or "Ответ не заполнен.")
        st.markdown("**Вывод**")
        st.write(conclusion.strip() or "Ответ не заполнен.")
        st.caption(
            "Развёрнутые ответы сохраняются для анализа преподавателем, "
            "но в текущей версии не изменяют баллы и форму диаграммы."
        )

    st.caption(
        "Профиль строится только по восьми структурированным заданиям. "
        "Он отражает действия в конкретной лабораторной работе и не является "
        "общей оценкой способностей учащегося."
    )


def render_structured_assessment(
    lab_id: str,
    prediction: Any,
    hypothesis: str,
    context_key: str,
) -> None:
    questions = build_assessment_questions(
        lab_id,
        str(prediction.predicted_class),
    )

    st.markdown(
        '<div class="section-label">Итоговая диагностика исследовательских действий</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Диаграмма строится только по выбранным ответам. "
        "Текст гипотезы и вывода не повышает баллы автоматически."
    )

    answers: dict[str, str] = {}

    with st.form(
        key=f"assessment_form_{context_key}",
        clear_on_submit=False,
    ):
        for number, question in enumerate(questions, start=1):
            option_texts = [option[0] for option in question["options"]]
            answers[question["id"]] = st.selectbox(
                f"{number}. {question['prompt']}",
                ["— Выберите ответ —"] + option_texts,
                key=f"assessment_{context_key}_{question['id']}",
            )

        conclusion = st.text_area(
            "Сформулируйте собственный вывод",
            placeholder=(
                "Укажите наблюдаемую зависимость, доказательства "
                "и возможные ограничения эксперимента."
            ),
            key=f"assessment_conclusion_{context_key}",
        )

        submitted = st.form_submit_button(
            "Проверить ответы и построить профиль",
            use_container_width=True,
        )

    if submitted:
        unanswered = [
            question["prompt"]
            for question in questions
            if answers.get(question["id"]) == "— Выберите ответ —"
        ]

        if unanswered:
            st.error(
                f"Необходимо ответить на все задания. "
                f"Пропущено: {len(unanswered)}."
            )
            st.session_state.pop("assessment_result", None)
        else:
            result = evaluate_assessment(
                questions,
                answers,
            )
            st.session_state["assessment_result"] = {
                "context_key": context_key,
                "result": result,
                "hypothesis": hypothesis,
                "conclusion": conclusion,
            }

    stored = st.session_state.get("assessment_result")
    if stored and stored.get("context_key") == context_key:
        render_competency_profile(
            stored["result"],
            stored.get("hypothesis", ""),
            stored.get("conclusion", ""),
        )

st.set_page_config(
    page_title="PhysLab AI",
    page_icon="🔬",
    layout="wide",
)

apply_page_style()

st.markdown('<div class="physlab-title">PhysLab AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="physlab-subtitle">'
    "AI-платформа сопровождения учебного физического исследования"
    "</div>",
    unsafe_allow_html=True,
)
st.caption("Сборка интерфейса: 2026-07-19 v6 · структурированная диагностика")

show_model_status()

source_mode = st.radio(
    "Источник экспериментальных данных",
    [
        "Демонстрационный эксперимент",
        "Загрузить реальный файл",
    ],
    horizontal=True,
)

lab_id: str | None = None
dataframe: pd.DataFrame | None = None
quality_report = None
source_description = ""
can_analyze = False

if source_mode == "Демонстрационный эксперимент":
    lab_title = st.selectbox(
        "Лабораторная работа",
        list(LAB_LABELS),
    )
    lab_id = LAB_LABELS[lab_title]
    config = LAB_REGISTRY[lab_id]

    selected_class = st.selectbox(
        "Сценарий эксперимента",
        list(config.classes),
        format_func=lambda value: CLASS_LABELS.get(value, value),
    )

    dataframe = SIMULATORS[lab_id](
        selected_class,
        seed=123,
    )
    quality_report = assess_quality(dataframe, lab_id)
    source_description = (
        "Демонстрационный сценарий: "
        f"{CLASS_LABELS.get(selected_class, selected_class)}"
    )
    can_analyze = quality_report.ready_for_model

else:
    uploaded_file = st.file_uploader(
        "Загрузите CSV или Excel-файл",
        type=["csv", "txt", "xlsx", "xls"],
        help=(
            "Исходный файл не изменяется. "
            "Нормализация выполняется только в памяти приложения."
        ),
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()

        try:
            tables = read_uploaded_tables(
                uploaded_file.name,
                raw_bytes,
            )
        except Exception as error:
            st.error(f"Не удалось прочитать файл: {error}")
            tables = {}

        if tables:
            table_name = st.selectbox(
                "Лист или таблица",
                list(tables),
            )
            raw_table = tables[table_name]
            detection = detect_lab(
                raw_table,
                uploaded_file.name,
            )

            detection_table = (
                pd.DataFrame(
                    {
                        "Лабораторная": [
                            LAB_TITLES[key]
                            for key in detection.scores
                        ],
                        "Оценка соответствия": [
                            round(value, 1)
                            for value in detection.scores.values()
                        ],
                    }
                )
                .sort_values(
                    "Оценка соответствия",
                    ascending=False,
                )
            )

            detected_title = LAB_TITLES[detection.lab_id]

            if detection.confidence >= 0.65:
                st.success(
                    f"Автоматически определено: **{detected_title}** "
                    f"(уверенность {detection.confidence * 100:.0f}%)."
                )
            else:
                st.warning(
                    f"Предварительно определено: **{detected_title}**, "
                    "но требуется проверка пользователя."
                )

            with st.expander("Как определена лабораторная"):
                st.dataframe(
                    detection_table,
                    hide_index=True,
                    use_container_width=True,
                )
                if detection.reasons:
                    st.write("; ".join(detection.reasons))

            lab_options = list(LAB_REGISTRY)
            default_index = lab_options.index(detection.lab_id)

            lab_id = st.selectbox(
                "Подтвердите лабораторную работу",
                lab_options,
                index=default_index,
                format_func=lambda value: LAB_TITLES[value],
            )

            series = extract_series_candidates(
                raw_table,
                lab_id,
            )
            series_name = st.selectbox(
                "Экспериментальная серия",
                list(series),
            )
            selected_frame = series[series_name]

            metadata, metadata_confirmed = metadata_controls(
                lab_id,
                selected_frame,
            )

            dataframe, missing_columns, mapping = normalize_experiment(
                selected_frame,
                lab_id,
                metadata,
            )
            quality_report = assess_quality(
                dataframe,
                lab_id,
            )

            if missing_columns:
                quality_report.issues.insert(
                    0,
                    "После нормализации отсутствуют: "
                    + ", ".join(missing_columns),
                )
                quality_report.ready_for_model = False
                quality_report.status = "not_ready"

            if lab_id == "heat_balance" and not metadata_confirmed:
                quality_report.ready_for_model = False
                quality_report.status = "not_ready"
                quality_report.issues.append(
                    "Не подтверждены параметры теплового баланса."
                )

            source_description = (
                f"{uploaded_file.name} · {table_name} · {series_name}"
            )
            can_analyze = quality_report.ready_for_model

            show_quality(
                quality_report,
                mapping,
            )

            normalized_csv = (
                dataframe
                .to_csv(index=False)
                .encode("utf-8-sig")
            )
            st.download_button(
                "Скачать нормализованный CSV",
                data=normalized_csv,
                file_name=f"{lab_id}_normalized.csv",
                mime="text/csv",
                use_container_width=True,
            )

if lab_id is not None:
    render_lab_header(lab_id)

hypothesis = st.text_area(
    "1. Сформулируйте гипотезу",
    placeholder=(
        "Какую зависимость вы ожидаете увидеть? "
        "Почему она должна иметь именно такую форму?"
    ),
)

if dataframe is not None and lab_id is not None:
    reset_analysis_if_context_changed(
        dataframe_fingerprint(dataframe, lab_id)
    )

    st.markdown(
        '<div class="section-label">Экспериментальные данные</div>',
        unsafe_allow_html=True,
    )

    visible_columns = [
        column
        for column in dataframe.columns
        if not column.startswith("true_")
        and column
        not in {
            "class_name",
            "severity",
            "generation_group",
            "secondary_errors",
            "device_profile",
            "environment_profile",
            "dead_volume_ml",
        }
    ]

    with st.expander("Показать таблицу измерений"):
        st.dataframe(
            dataframe[visible_columns].head(500),
            use_container_width=True,
        )

    show_experiment_chart(
        dataframe,
        lab_id,
    )

    st.caption(f"Источник данных: {source_description}")

    analyze_button = st.button(
        "Проанализировать исследование",
        type="primary",
        use_container_width=True,
        disabled=not can_analyze,
    )

    if analyze_button:
        try:
            prediction = PREDICTORS[lab_id](dataframe)
            feedback = build_feedback(prediction)
            st.session_state["prediction"] = prediction
            st.session_state["feedback"] = feedback
        except Exception as error:
            st.error(f"Ошибка анализа: {error}")

if (
    lab_id is not None
    and "prediction" in st.session_state
    and st.session_state["prediction"].lab_id == lab_id
):
    prediction = st.session_state["prediction"]
    feedback = st.session_state["feedback"]

    st.markdown(
        '<div class="section-label">Результат AI-анализа</div>',
        unsafe_allow_html=True,
    )

    render_diagnostic_panel(
        prediction,
        feedback,
        source_mode,
    )

    assessment_context = dataframe_fingerprint(
        dataframe,
        lab_id,
    )
    render_structured_assessment(
        lab_id,
        prediction,
        hypothesis,
        assessment_context,
    )

with st.expander("О данных, методике и ограничениях"):
    st.write(
        "Модели обучены на физически обусловленных синтетических данных, "
        "откалиброванных по реальным измерениям датчиков. Проверка проводится "
        "на групповой отложенной выборке. Реальные серии без надёжных меток "
        "используются как внешний корпус проверки и не превращаются автоматически "
        "в обучающие примеры. Grid Search применяется как контроль "
        "гиперпараметров, но модель заменяется только при улучшении независимой "
        "holdout-метрики. Профиль исследовательских действий строится "
        "по структурированным заданиям с прозрачной шкалой 0–3. "
        "Развёрнутые ответы сохраняются для преподавателя и не оцениваются "
        "автоматически до отдельной валидации модуля анализа текста."
    )
