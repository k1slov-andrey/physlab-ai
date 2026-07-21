from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DOCUMENTS = (
    "README.md",
    "PRODUCT.md",
    "TECHNICAL_APPENDIX.md",
)
METRIC_DOCUMENTS = (
    "README.md",
    "TECHNICAL_APPENDIX.md",
)
RETIRED_DOCUMENTS = (
    "PROJECT_DESCRIPTION.md",
    "DATA_SCIENCE_REPORT.md",
    "AI_USAGE.md",
    "FIELD_VALIDATION_PROTOCOL.md",
    "CHANGELOG_V3.md",
)


def _read(name: str) -> str:
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def _combined() -> str:
    return "\n".join(_read(name) for name in CORE_DOCUMENTS)


def _ru(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def _local_markdown_targets(text: str) -> list[str]:
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    return [
        target.split("#", 1)[0]
        for target in targets
        if target
        and not target.startswith(("http://", "https://", "mailto:", "#"))
    ]


def test_deployed_metrics_are_documented() -> None:
    summary = pd.read_csv(PROJECT_ROOT / "evaluation" / "final_model_summary.csv")
    texts = {name: _read(name) for name in METRIC_DOCUMENTS}

    for row in summary.itertuples(index=False):
        for metric in (row.accuracy, row.balanced_accuracy, row.macro_f1):
            value = _ru(float(metric))
            assert all(value in text for text in texts.values()), value


def test_stale_metrics_and_claims_are_absent() -> None:
    combined = _combined()
    stale_fragments = (
        "| Нагревание и охлаждение | 0,783 |",
        "| Закон Бойля — Мариотта | 0,956 |",
        "| Изохорный процесс | 0,981 |",
        "| Тепловой баланс | 0,893 |",
        "16 автоматических тестов",
        "35 серий, пригодных",
        "независимой групповой синтетической holdout-выборке",
        "доказанная эффективность",
        "уникальная модель развития",
    )
    for fragment in stale_fragments:
        assert fragment not in combined


def test_real_data_limit_is_stated() -> None:
    for name in METRIC_DOCUMENTS:
        text = _read(name)
        assert "real_data_raw" in text
        assert "не включ" in text or "не публику" in text


def test_product_hypotheses_are_exactly_three() -> None:
    product = _read("PRODUCT.md")
    assert product.count("### H1.") == 1
    assert product.count("### H2.") == 1
    assert product.count("### H3.") == 1
    assert "### H4." not in product


def test_readme_has_fast_demo_and_resource_profile() -> None:
    readme = _read("README.md")
    appendix = _read("TECHNICAL_APPENDIX.md")
    assert "## Демо за 60 секунд" in readme
    assert "demo_sensor_drift.csv" in readme
    assert "## Инженерный паспорт" in readme
    assert "### 13.2. Референсный ресурсный профиль" in appendix
    assert "random_state=42" in appendix


def test_only_three_root_markdown_documents_exist() -> None:
    actual = {path.name for path in PROJECT_ROOT.glob("*.md")}
    assert actual == set(CORE_DOCUMENTS)


def test_readme_navigation_points_only_to_core_documents() -> None:
    targets = {
        target
        for target in _local_markdown_targets(_read("README.md"))
        if target.endswith(".md")
    }
    assert targets == {"PRODUCT.md", "TECHNICAL_APPENDIX.md"}


def test_all_local_links_in_core_documents_resolve() -> None:
    failures: list[str] = []
    for name in CORE_DOCUMENTS:
        for target in _local_markdown_targets(_read(name)):
            path = (PROJECT_ROOT / target).resolve()
            if not path.exists():
                failures.append(f"{name} -> {target}")
    assert not failures, failures


def test_retired_document_names_are_absent() -> None:
    combined = _combined()
    for name in RETIRED_DOCUMENTS:
        assert name not in combined
        assert not (PROJECT_ROOT / name).exists()


def test_forbidden_editorial_phrases_are_absent() -> None:
    combined = _combined().lower()
    forbidden = (
        "не подтверждено",
        "эти ограничения — часть продуктовой архитектуры",
        "однако продукт не считается успешным",
        "таким образом, ml-метрика",
        "методические элементы лабораторных работ ранее применялись",
        "эти продукты задают высокий стандарт",
        "плохо масштабируется",
        "модель не должна заставлять эксперта",
        "смешивать их нельзя",
        "следующий этап не должен начинаться автоматически",
        "бизнес-модель не считается проверенной",
        "конкурентный обзор и внешняя доказательная база актуализированы",
        "северная звезда",
        "закрывает критерий",
    )
    for phrase in forbidden:
        assert phrase not in combined


def test_no_placeholder_future_document_language() -> None:
    combined = _combined().lower()
    placeholders = (
        "отдельный документ",
        "документ будет создан",
        "будет добавлен в документ",
        "будет содержать",
        "появится позже",
        "создадим файл",
    )
    for phrase in placeholders:
        assert phrase not in combined


def test_ai_usage_is_documented_in_readme_and_appendix() -> None:
    readme = _read("README.md")
    appendix = _read("TECHNICAL_APPENDIX.md")
    assert "## Применение генеративного ИИ" in readme
    assert "## 11. Использование генеративного ИИ при разработке" in appendix
    assert "ChatGPT" in appendix
    assert "отклон" in appendix.lower()


def test_russian_competitors_precede_international_benchmarks() -> None:
    product = _read("PRODUCT.md")
    russian = product.index("### 9.1. Российские цифровые лаборатории")
    international = product.index(
        "### 9.4. Международные решения для работы с экспериментальными данными"
    )
    assert russian < international


def test_field_validation_is_documented_consistently() -> None:
    readme = _read("README.md")
    product = _read("PRODUCT.md")
    appendix = _read("TECHNICAL_APPENDIX.md")
    for text in (readme, product, appendix):
        assert "Каир" in text
        assert "2026" in text
    assert "shadow mode" in product.lower()
    assert "shadow mode" in appendix.lower()


def test_personal_contribution_and_responsibility_are_visible() -> None:
    readme = _read("README.md")
    appendix = _read("TECHNICAL_APPENDIX.md")
    assert "## Личный вклад" in readme
    assert "Проект выполнен автором самостоятельно" in readme
    assert "### 11.5. Ответственность и конфиденциальность" in appendix
