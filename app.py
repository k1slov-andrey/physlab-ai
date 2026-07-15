from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import predict_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"

DEMO_FILES = {
    "Корректный эксперимент": {
        "path": DATA_DIRECTORY / "normal_cooling_experiment.csv",
        "icon": "🟢",
        "description": "Эксперимент без существенных отклонений.",
    },
    "Единичный выброс": {
        "path": DATA_DIRECTORY / "single_outlier_experiment.csv",
        "icon": "🟡",
        "description": "Один ошибочный замер температуры.",
    },
    "Дрейф датчика": {
        "path": DATA_DIRECTORY / "sensor_drift_experiment.csv",
        "icon": "🟠",
        "description": "Постепенное смещение показаний датчика.",
    },
    "Повышенный шум": {
        "path": DATA_DIRECTORY / "high_noise_experiment.csv",
        "icon": "🔴",
        "description": "Частые случайные колебания повышенной амплитуды.",
    },
}

CLASS_STYLES = {
    "normal": {
        "icon": "🟢",
        "title": "Эксперимент выполнен корректно",
        "status": "success",
    },
    "single_outlier": {
        "icon": "🟡",
        "title": "Обнаружен единичный выброс",
        "status": "warning",
    },
    "sensor_drift": {
        "icon": "🟠",
        "title": "Обнаружен дрейф датчика",
        "status": "warning",
    },
    "high_noise": {
        "icon": "🔴",
        "title": "Обнаружен повышенный шум",
        "status": "error",
    },
}


st.set_page_config(
    page_title="PhysLab AI",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 PhysLab AI")

st.subheader(
    "AI-ассистент для диагностики лабораторных работ по физике"
)

st.markdown(
    """
PhysLab AI анализирует временной ряд охлаждения жидкости,
сопоставляет измерения с законом охлаждения Ньютона,
определяет вероятный тип ошибки и формирует рекомендацию преподавателю.
"""
)

st.info(
    """
**Что умеет система**

- анализировать CSV-файл с измерениями;
- проверять и очищать временной ряд;
- подбирать параметры физической модели;
- рассчитывать интерпретируемые признаки;
- определять тип ошибки;
- показывать уверенность модели;
- формировать педагогическую рекомендацию.
"""
)

st.warning(
    """
Текущая версия обучена на физически обоснованных синтетических данных.
Проверка на реальных данных цифровых лабораторий является следующим этапом проекта.
"""
)

st.divider()

input_mode = st.radio(
    "Выберите источник данных",
    [
        "Демонстрационный пример",
        "Загрузить свой CSV",
    ],
    horizontal=True,
)

data = None
source_name = None

if input_mode == "Демонстрационный пример":
    st.subheader("Демонстрационные сценарии")

    demo_names = list(DEMO_FILES.keys())
    columns = st.columns(4)

    for column, demo_name in zip(columns, demo_names):
        demo = DEMO_FILES[demo_name]

        with column:
            st.markdown(
                f"### {demo['icon']} {demo_name}"
            )
            st.write(demo["description"])

            if st.button(
                "Использовать",
                key=f"use_{demo_name}",
                use_container_width=True,
            ):
                st.session_state["selected_demo"] = demo_name

    selected_demo = st.session_state.get(
        "selected_demo",
        demo_names[0],
    )

    selected_demo_data = DEMO_FILES[selected_demo]
    demo_path = selected_demo_data["path"]

    st.caption(
        f"Выбран сценарий: "
        f"{selected_demo_data['icon']} {selected_demo}"
    )

    if demo_path.exists():
        data = pd.read_csv(demo_path)
        source_name = selected_demo
    else:
        st.error(
            f"Демонстрационный файл не найден: {demo_path.name}"
        )

else:
    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл",
        type=["csv"],
    )

    st.caption(
        "Обязательные столбцы: "
        "`time_seconds`, `measured_temperature`."
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            source_name = uploaded_file.name
        except Exception as error:
            st.error(
                f"Не удалось прочитать файл: {error}"
            )

if data is not None:
    st.divider()

    st.write(f"**Источник данных:** {source_name}")

    with st.expander("Посмотреть исходные данные"):
        st.dataframe(
            data.head(20),
            width="stretch",
        )

    if st.button(
        "Провести диагностику",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner(
                "Аппроксимация физической модели "
                "и классификация..."
            ):
                result = predict_experiment(data)

            experiment = result["experiment"]

            left_column, right_column = st.columns(
                [2, 1]
            )

            with left_column:
                figure = go.Figure()

                figure.add_trace(
                    go.Scatter(
                        x=experiment["time_seconds"],
                        y=experiment[
                            "measured_temperature"
                        ],
                        mode="lines+markers",
                        name="Измеренная температура",
                    )
                )

                figure.add_trace(
                    go.Scatter(
                        x=experiment["time_seconds"],
                        y=result["fitted_temperature"],
                        mode="lines",
                        name="Аппроксимированная модель",
                    )
                )

                figure.update_layout(
                    title="Температура жидкости во времени",
                    xaxis_title="Время, с",
                    yaxis_title="Температура, °C",
                    hovermode="x unified",
                )

                st.plotly_chart(
                    figure,
                    width="stretch",
                )

            with right_column:
                predicted_class = result["predicted_class"]

                style = CLASS_STYLES.get(
                    predicted_class,
                    {
                        "icon": "ℹ️",
                        "title": result["title"],
                        "status": "info",
                    },
                )

                st.metric(
                    "Уверенность модели",
                    f"{result['confidence']:.1%}",
                )

                message = (
                    f"{style['icon']} **{style['title']}**\n\n"
                    f"{result['description']}\n\n"
                    f"**Рекомендация:** "
                    f"{result['recommendation']}"
                )

                if style["status"] == "success":
                    st.success(message)
                elif style["status"] == "warning":
                    st.warning(message)
                elif style["status"] == "error":
                    st.error(message)
                else:
                    st.info(message)

            st.subheader("Вероятности классов")

            probability_table = pd.DataFrame(
                {
                    "Класс": list(
                        result["probabilities"].keys()
                    ),
                    "Вероятность": list(
                        result["probabilities"].values()
                    ),
                }
            ).sort_values(
                "Вероятность",
                ascending=False,
            )

            st.dataframe(
                probability_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Вероятность": (
                        st.column_config.ProgressColumn(
                            "Вероятность",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                        )
                    )
                },
            )

            st.subheader("Как модель получила результат")

            st.markdown(
                """
Система анализирует эксперимент в несколько этапов:

1. проверяет и очищает временной ряд;
2. подбирает параметры закона охлаждения Ньютона;
3. рассчитывает остатки между измерениями и моделью;
4. извлекает физические и статистические признаки;
5. передает признаки в классификатор;
6. выбирает наиболее вероятный тип ошибки.
"""
            )

            key_features = {
                "Максимальный скачок температуры": (
                    result["features"]["max_abs_jump"]
                ),
                "Стандартное отклонение остатков": (
                    result["features"]["std_residual"]
                ),
                "Максимальное отклонение от модели": (
                    result["features"]["max_abs_residual"]
                ),
                "Изменение среднего остатка": (
                    result["features"]["residual_mean_change"]
                ),
                "RMSE физической модели": (
                    result["features"]["fit_rmse"]
                ),
                "R² физической модели": (
                    result["features"]["fit_r2"]
                ),
            }

            feature_table = pd.DataFrame(
                {
                    "Признак": list(key_features.keys()),
                    "Значение": [
                        round(value, 5)
                        for value in key_features.values()
                    ],
                }
            )

            st.dataframe(
                feature_table,
                width="stretch",
                hide_index=True,
            )

            with st.expander(
                "Параметры подобранной физической модели"
            ):
                parameters = result[
                    "fitted_parameters"
                ]

                st.write(
                    {
                        "Начальная температура, °C": round(
                            parameters[
                                "fitted_initial_temperature"
                            ],
                            3,
                        ),
                        "Температура среды, °C": round(
                            parameters[
                                "fitted_room_temperature"
                            ],
                            3,
                        ),
                        "Коэффициент охлаждения": round(
                            parameters[
                                "fitted_cooling_coefficient"
                            ],
                            6,
                        ),
                    }
                )

            st.caption(
                "Результат модели является поддержкой принятия "
                "решения и не заменяет экспертную оценку преподавателя."
            )

        except Exception as error:
            st.error(
                f"Диагностика не выполнена: {error}"
            )