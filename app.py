from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import predict_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"

DEMO_FILES = {
    "Нормальный эксперимент": (
        DATA_DIRECTORY / "normal_cooling_experiment.csv"
    ),
    "Единичный выброс": (
        DATA_DIRECTORY / "single_outlier_experiment.csv"
    ),
    "Дрейф датчика": (
        DATA_DIRECTORY / "sensor_drift_experiment.csv"
    ),
    "Повышенный шум": (
        DATA_DIRECTORY / "high_noise_experiment.csv"
    ),
}


st.set_page_config(
    page_title="PhysLab AI",
    page_icon="🌡️",
    layout="wide",
)

st.title("PhysLab AI")

st.subheader(
    "Интеллектуальная диагностика ошибок "
    "цифрового физического эксперимента"
)

st.write(
    "Система анализирует временной ряд охлаждения жидкости, "
    "сопоставляет измерения с физической моделью и определяет "
    "вероятный тип ошибки."
)

st.warning(
    "Текущая версия обучена на физически обоснованных "
    "синтетических данных. Проверка на реальных данных "
    "цифровых лабораторий является следующим этапом проекта."
)

input_mode = st.radio(
    "Источник данных",
    [
        "Демонстрационный эксперимент",
        "Загрузить CSV",
    ],
    horizontal=True,
)

data = None
source_name = None

if input_mode == "Демонстрационный эксперимент":
    demo_name = st.selectbox(
        "Выберите пример",
        list(DEMO_FILES.keys()),
    )

    demo_path = DEMO_FILES[demo_name]

    if demo_path.exists():
        data = pd.read_csv(demo_path)
        source_name = demo_name
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

    st.write(f"**Источник:** {source_name}")

    with st.expander("Посмотреть исходные данные"):
        st.dataframe(
            data.head(20),
            use_container_width=True,
        )

    if st.button(
        "Провести диагностику",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Аппроксимация физической модели "
                "и классификация..."
            ):
                result = predict_experiment(data)

            left_column, right_column = st.columns(
                [2, 1]
            )

            with left_column:
                experiment = result["experiment"]

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
                    use_container_width=True,
                )

            with right_column:
                st.metric(
                    "Результат",
                    result["title"],
                )

                st.metric(
                    "Уверенность модели",
                    f"{result['confidence']:.1%}",
                )

                st.write(
                    f"**Интерпретация:** "
                    f"{result['description']}"
                )

                st.info(
                    result["recommendation"]
                )

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
                use_container_width=True,
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

        except Exception as error:
            st.error(
                f"Диагностика не выполнена: {error}"
            )