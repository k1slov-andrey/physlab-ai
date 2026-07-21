from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "README.md",
    "PROJECT_DESCRIPTION.md",
    "TECHNICAL_APPENDIX.md",
    "PRODUCT.md",
)
METRIC_DOCUMENTS = (
    "README.md",
    "PROJECT_DESCRIPTION.md",
    "TECHNICAL_APPENDIX.md",
)


def _read(name: str) -> str:
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def _ru(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def test_deployed_metrics_are_documented() -> None:
    summary = pd.read_csv(PROJECT_ROOT / "evaluation" / "final_model_summary.csv")
    texts = {name: _read(name) for name in METRIC_DOCUMENTS}

    for row in summary.itertuples(index=False):
        for metric in (row.accuracy, row.balanced_accuracy, row.macro_f1):
            value = _ru(float(metric))
            assert all(value in text for text in texts.values()), value


def test_stale_metrics_and_claims_are_absent() -> None:
    combined = "\n".join(_read(name) for name in DOCUMENTS)
    stale_fragments = (
        "0,783",
        "0,956",
        "0,981",
        "0,893",
        "16 автоматических тестов",
        "35 серий, пригодных",
        "независимой групповой синтетической holdout-выборке",
        "доказанная эффективность",
        "уникальная модель развития",
    )
    for fragment in stale_fragments:
        assert fragment not in combined


def test_real_data_limit_is_stated() -> None:
    readme = _read("README.md")
    project = _read("PROJECT_DESCRIPTION.md")
    appendix = _read("TECHNICAL_APPENDIX.md")

    for text in (readme, project, appendix):
        assert "real_data_raw" in text
        assert ("не включ" in text or "не публику" in text)


def test_product_hypotheses_are_exactly_three() -> None:
    product = _read("PRODUCT.md")
    assert product.count("### H1.") == 1
    assert product.count("### H2.") == 1
    assert product.count("### H3.") == 1
    assert "### H4." not in product
