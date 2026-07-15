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
        "description": "Эксперимент без существенных отклонений.",
    },
    "Единичный выброс": {
        "path": DATA_DIRECTORY / "single_outlier_experiment.csv",
        "description": (
            "Одно измерение резко отличается "
            "от соседних значений."
        ),
    },
    "Дрейф датчика": {
        "path": DATA_DIRECTORY / "sensor_drift_experiment.csv",
        "description": (
            "Показания постепенно отклоняются "
            "от ожидаемой зависимости."
        ),
    },
    "Повышенный шум": {
        "path": DATA_DIRECTORY / "high_noise_experiment.csv",
        "description": (
            "В измерениях присутствуют частые "
            "случайные колебания."
        ),
    },
}

CLASS_LABELS = {
    "normal": "Нормальный эксперимент",
    "single_outlier": "Единичный выброс",
    "sensor_drift": "Дрейф датчика",
    "high_noise": "Повышенный шум",
}


st.set_page_config(
    page_title="PhysLab",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        html,
        body,
        [data-testid="stAppViewContainer"],
        .stApp {
            background: #ffffff;
        }

        .block-container {
            max-width: 860px;
            padding-top: 1.4rem;
            padding-bottom: 4rem;
        }

        [data-testid="stHeader"] {
            height: 0;
            visibility: hidden;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none;
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        h1,
        h2,
        h3,
        p,
        li,
        label,
        div {
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Arial,
                sans-serif;
        }

        h1 {
            color: #202124;
            font-size: 2rem !important;
            font-weight: 650 !important;
            margin-bottom: 0.35rem !important;
        }

        h2 {
            color: #202124;
            font-size: 1.35rem !important;
            font-weight: 650 !important;
            margin-top: 2.2rem !important;
        }

        h3 {
            color: #202124;
            font-size: 1.05rem !important;
            font-weight: 620 !important;
        }

        p,
        li {
            color: #4a4f55;
            line-height: 1.65;
        }

        .top-navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 0.9rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid #e4e6e8;
        }

        .site-name {
            color: #202124;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .site-section {
            color: #7a7f85;
            font-size: 0.82rem;
        }

        .intro {
            max-width: 720px;
            margin-bottom: 1.6rem;
            color: #5f6368;
            font-size: 0.98rem;
            line-height: 1.7;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #dfe1e5 !important;
            border-radius: 6px !important;
            box-shadow: none !important;
        }

        .calculator-title {
            margin-bottom: 0.2rem;
            color: #202124;
            font-size: 1.08rem;
            font-weight: 650;
        }

        .calculator-description {
            margin-bottom: 1rem;
            color: #777c82;
            font-size: 0.88rem;
        }

        .stButton > button {
            min-height: 42px;
            border-radius: 4px;
            font-weight: 620;
        }

        .stButton > button[kind="primary"] {
            background: #2f65bd !important;
            border-color: #2f65bd !important;
            color: #ffffff !important;
        }

        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span {
            color: #ffffff !important;
        }

        .stButton > button[kind="primary"]:hover {
            background: #2757a4 !important;
            border-color: #2757a4 !important;
        }

        .result-box {
            margin-top: 1rem;
            padding: 1rem 1.1rem;
            background: #f8f9fa;
            border: 1px solid #dfe1e5;
            border-radius: 5px;
        }

        .result-label {
            margin-bottom: 0.25rem;
            color: #73777c;
            font-size: 0.78rem;
        }

        .result-value {
            margin-bottom: 0.7rem;
            color: #202124;
            font-size: 1.2rem;
            font-weight: 650;
        }

        .result-text {
            color: #4f545a;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .result-subtitle {
            margin-top: 0.8rem;
            margin-bottom: 0.25rem;
            color: #30343a;
            font-size: 0.85rem;
            font-weight: 650;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dfe1e5;
            border-radius: 5px;
            padding: 0.9rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #dfe1e5;
            border-radius: 5px;
            overflow: hidden;
        }

        .info-section {
            padding: 0.2rem 0;
        }

        .formula-box {
            margin: 1rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border: 1px solid #e1e3e6;
            border-radius: 5px;
        }

        .limitations {
            padding: 0.9rem 1rem;
            background: #fffaf0;
            border: 1px solid #ead9b4;
            border-radius: 5px;
            color: #545960;
            line-height: 1.6;
        }

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e4e6e8;
            color: #858a90;
            font-size: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="top-navigation">
        <div class="site-name">PhysLab</div>
        <div class="site-section">
            Анализ физических экспериментов
        </div>
    </div>

    <h1>Диагностика эксперимента охлаждения</h1>

    <div class="intro">
        Выберите демонстрационный пример или загрузите CSV-файл.
        Система сопоставит измерения с законом охлаждения Ньютона
        и определит вероятный характер отклонения.
    </div>
    """,
    unsafe_allow_html=True,
)


data = None
source_name = None
source_signature = None

with st.container(border=True):
    st.markdown(
        """
        <div class="calculator-title">
            Данные эксперимента
        </div>
        <div class="calculator-description">
            Укажите источник данных и выполните расчет.
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Источник данных",
        options=[
            "Готовый пример",
            "Загрузить CSV",
        ],
        horizontal=True,
    )

    if input_mode == "Готовый пример":
        selected_demo = st.selectbox(
            "Тип эксперимента",
            options=list(DEMO_FILES.keys()),
            format_func=lambda name: (
                f"{name} — "
                f"{DEMO_FILES[name]['description']}"
            ),
        )

        demo_path = DEMO_FILES[selected_demo]["path"]

        if demo_path.exists():
            data = pd.read_csv(demo_path)
            source_name = selected_demo
            source_signature = f"demo:{selected_demo}"
        else:
            st.error(
                f"Файл не найден: {demo_path.name}"
            )

    else:
        uploaded_file = st.file_uploader(
            "CSV-файл",
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
                source_signature = (
                    f"upload:{uploaded_file.name}:"
                    f"{uploaded_file.size}"
                )
            except Exception as error:
                st.error(
                    f"Не удалось прочитать файл: {error}"
                )

    if data is not None:
        with st.expander(
            "Предварительный просмотр данных"
        ):
            st.dataframe(
                data.head(20),
                width="stretch",
                hide_index=True,
            )

        calculate = st.button(
            "Рассчитать",
            type="primary",
            width="stretch",
        )

        reset = st.button(
            "Сбросить",
            width="stretch",
        )

        if reset:
            st.session_state.pop(
                "analysis_result",
                None,
            )
            st.session_state.pop(
                "analysis_signature",
                None,
            )

        if calculate:
            try:
                with st.spinner(
                    "Выполняется расчет..."
                ):
                    analysis_result = predict_experiment(
                        data
                    )

                st.session_state["analysis_result"] = (
                    analysis_result
                )
                st.session_state["analysis_signature"] = (
                    source_signature
                )

            except Exception as error:
                st.error(
                    f"Не удалось выполнить расчет: {error}"
                )


saved_result = st.session_state.get(
    "analysis_result"
)

saved_signature = st.session_state.get(
    "analysis_signature"
)

if (
    saved_result is not None
    and saved_signature == source_signature
):
    result = saved_result
    experiment = result["experiment"]
    predicted_class = result["predicted_class"]

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">
                Результат диагностики
            </div>

            <div class="result-value">
                {
                    CLASS_LABELS.get(
                        predicted_class,
                        predicted_class,
                    )
                }
            </div>

            <div class="result-text">
                {result['description']}
            </div>

            <div class="result-subtitle">
                Рекомендация
            </div>

            <div class="result-text">
                {result['recommendation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric(
        "Уверенность модели",
        f"{result['confidence']:.1%}",
    )

    st.subheader("График эксперимента")

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=experiment["time_seconds"],
            y=experiment["measured_temperature"],
            mode="lines+markers",
            name="Измеренная температура",
            line={
                "color": "#2f65bd",
                "width": 2,
            },
            marker={
                "size": 4,
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=experiment["time_seconds"],
            y=result["fitted_temperature"],
            mode="lines",
            name="Расчетная модель",
            line={
                "color": "#c27b2b",
                "width": 2.5,
            },
        )
    )

    figure.update_layout(
        height=450,
        xaxis_title="Время, с",
        yaxis_title="Температура, °C",
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin={
            "l": 30,
            "r": 20,
            "t": 30,
            "b": 30,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    figure.update_xaxes(
        showgrid=True,
        gridcolor="#eceff1",
        zeroline=False,
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor="#eceff1",
        zeroline=False,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Вероятности классов")

        probability_table = pd.DataFrame(
            {
                "Класс": [
                    CLASS_LABELS.get(
                        class_name,
                        class_name,
                    )
                    for class_name
                    in result["probabilities"].keys()
                ],
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

    with right_column:
        st.subheader("Параметры модели")

        parameters = result["fitted_parameters"]

        parameter_table = pd.DataFrame(
            {
                "Параметр": [
                    "Начальная температура, °C",
                    "Температура среды, °C",
                    "Коэффициент охлаждения",
                ],
                "Значение": [
                    round(
                        parameters[
                            "fitted_initial_temperature"
                        ],
                        3,
                    ),
                    round(
                        parameters[
                            "fitted_room_temperature"
                        ],
                        3,
                    ),
                    round(
                        parameters[
                            "fitted_cooling_coefficient"
                        ],
                        6,
                    ),
                ],
            }
        )

        st.dataframe(
            parameter_table,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Ключевые показатели")

    features = result["features"]

    feature_table = pd.DataFrame(
        {
            "Показатель": [
                "Максимальный скачок температуры",
                "Стандартное отклонение остатков",
                "Максимальное отклонение от модели",
                "Изменение среднего остатка",
                "RMSE физической модели",
                "R² физической модели",
            ],
            "Значение": [
                round(
                    features["max_abs_jump"],
                    5,
                ),
                round(
                    features["std_residual"],
                    5,
                ),
                round(
                    features["max_abs_residual"],
                    5,
                ),
                round(
                    features["residual_mean_change"],
                    5,
                ),
                round(
                    features["fit_rmse"],
                    5,
                ),
                round(
                    features["fit_r2"],
                    5,
                ),
            ],
        }
    )

    st.dataframe(
        feature_table,
        width="stretch",
        hide_index=True,
    )


st.header("Как выполняется расчет")

st.markdown(
    """
    <div class="info-section">
        Измеренный временной ряд очищается и сортируется по времени.
        Затем по данным подбираются параметры закона охлаждения Ньютона.
        Отклонения от расчетной кривой преобразуются в физические
        и статистические признаки, которые анализирует
        классификационная модель.
    </div>
    """,
    unsafe_allow_html=True,
)


st.header("Формула расчета")

st.markdown(
    """
    <div class="formula-box">
        Для описания процесса используется закон охлаждения Ньютона.
        Начальная температура, температура среды и коэффициент охлаждения
        оцениваются непосредственно по измерениям.
    </div>
    """,
    unsafe_allow_html=True,
)

st.latex(
    r"T(t)=T_{env}+(T_0-T_{env})e^{-kt}"
)


st.header("Поддерживаемые состояния")

supported_classes = pd.DataFrame(
    {
        "Состояние": [
            "Нормальный эксперимент",
            "Единичный выброс",
            "Дрейф датчика",
            "Повышенный шум",
        ],
        "Описание": [
            (
                "Существенных отклонений "
                "от физической модели не обнаружено."
            ),
            (
                "Одно измерение резко отличается "
                "от соседних значений."
            ),
            (
                "Отклонение постепенно увеличивается "
                "по мере проведения эксперимента."
            ),
            (
                "В ряду присутствуют частые "
                "случайные колебания."
            ),
        ],
    }
)

st.dataframe(
    supported_classes,
    width="stretch",
    hide_index=True,
)


st.header("Ограничения")

st.markdown(
    """
    <div class="limitations">
        Текущая версия обучена на физически обоснованных
        синтетических данных и предназначена для анализа
        эксперимента охлаждения жидкости. Результат является
        инструментом первичной диагностики и не заменяет
        экспертную оценку преподавателя.
    </div>
    """,
    unsafe_allow_html=True,
)


st.header("Часто задаваемые вопросы")

with st.expander(
    "Какие столбцы необходимы в CSV-файле?"
):
    st.write(
        "Файл должен содержать столбцы "
        "`time_seconds` и `measured_temperature`."
    )

with st.expander(
    "Используется ли заранее известная идеальная температура?"
):
    st.write(
        "Нет. Расчетная кривая строится "
        "по загруженным измерениям."
    )

with st.expander(
    "Почему применяется классическое машинное обучение?"
):
    st.write(
        "При небольшом количестве интерпретируемых "
        "физических признаков классические модели "
        "дают устойчивый результат и позволяют "
        "объяснить логику диагностики."
    )

with st.expander(
    "Можно ли анализировать другие лабораторные работы?"
):
    st.write(
        "Текущая версия предназначена для эксперимента "
        "охлаждения жидкости. Расширение на другие "
        "физические процессы предусмотрено в развитии проекта."
    )


st.markdown(
    """
    <div class="footer">
        PhysLab · Анализ цифровых физических экспериментов · 2026
    </div>
    """,
    unsafe_allow_html=True,
)