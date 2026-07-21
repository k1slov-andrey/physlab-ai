from __future__ import annotations

from importlib import metadata
from pathlib import Path

import environment_check


def test_read_exact_requirements_ignores_comments_and_ranges(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "# comment\n"
        "numpy==2.5.1\n"
        "pytest>=8,<10\n"
        "\n"
        "scikit-learn==1.8.0\n",
        encoding="utf-8",
    )

    assert environment_check.read_exact_requirements(requirements) == {
        "numpy": "2.5.1",
        "scikit-learn": "1.8.0",
    }


def test_environment_check_accepts_matching_stack(monkeypatch) -> None:
    monkeypatch.setattr(environment_check, "REQUIRED_PYTHON", (3, 13))
    monkeypatch.setattr(
        environment_check,
        "CORE_DISTRIBUTIONS",
        ("numpy", "scikit-learn"),
    )
    monkeypatch.setattr(
        environment_check,
        "read_exact_requirements",
        lambda _: {"numpy": "2.5.1", "scikit-learn": "1.8.0"},
    )
    versions = {"numpy": "2.5.1", "scikit-learn": "1.8.0"}
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])

    assert environment_check.inspect_environment() == []


def test_environment_check_reports_missing_or_mismatched_packages(monkeypatch) -> None:
    monkeypatch.setattr(environment_check, "REQUIRED_PYTHON", (3, 13))
    monkeypatch.setattr(
        environment_check,
        "CORE_DISTRIBUTIONS",
        ("numpy", "streamlit"),
    )
    monkeypatch.setattr(
        environment_check,
        "read_exact_requirements",
        lambda _: {"numpy": "2.5.1", "streamlit": "1.59.2"},
    )

    def fake_version(name: str) -> str:
        if name == "streamlit":
            raise metadata.PackageNotFoundError(name)
        return "2.4.0"

    monkeypatch.setattr(metadata, "version", fake_version)

    issues = environment_check.inspect_environment()
    assert [(issue.component, issue.actual) for issue in issues] == [
        ("numpy", "2.4.0"),
        ("streamlit", "not installed"),
    ]
