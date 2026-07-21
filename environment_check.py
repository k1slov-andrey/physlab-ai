from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
REQUIRED_PYTHON = (3, 13)
CORE_DISTRIBUTIONS = (
    "joblib",
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "plotly",
    "scikit-learn",
    "scipy",
    "streamlit",
)


@dataclass(frozen=True)
class EnvironmentIssue:
    component: str
    expected: str
    actual: str


def read_exact_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        requirements[name.strip().lower()] = version.strip()
    return requirements


def inspect_environment() -> list[EnvironmentIssue]:
    issues: list[EnvironmentIssue] = []

    actual_python = sys.version_info[:2]
    if actual_python != REQUIRED_PYTHON:
        issues.append(
            EnvironmentIssue(
                component="python",
                expected=".".join(map(str, REQUIRED_PYTHON)),
                actual=".".join(map(str, actual_python)),
            )
        )

    pinned = read_exact_requirements(REQUIREMENTS_PATH)
    for distribution in CORE_DISTRIBUTIONS:
        expected = pinned.get(distribution.lower())
        if expected is None:
            issues.append(
                EnvironmentIssue(
                    component=distribution,
                    expected="exact pin in requirements.txt",
                    actual="not pinned",
                )
            )
            continue

        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            issues.append(
                EnvironmentIssue(
                    component=distribution,
                    expected=expected,
                    actual="not installed",
                )
            )
            continue

        if actual != expected:
            issues.append(
                EnvironmentIssue(
                    component=distribution,
                    expected=expected,
                    actual=actual,
                )
            )

    return issues


def print_environment_summary(issues: list[EnvironmentIssue]) -> None:
    print(
        "Python "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    pinned = read_exact_requirements(REQUIREMENTS_PATH)
    for distribution in CORE_DISTRIBUTIONS:
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            actual = "not installed"
        print(
            f"{distribution}: installed={actual}; "
            f"required={pinned.get(distribution.lower(), 'not pinned')}"
        )

    if not issues:
        print("Environment check passed")
        return

    print("Environment check found incompatible components:")
    for issue in issues:
        print(
            f"- {issue.component}: expected {issue.expected}; actual {issue.actual}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the Python and ML stack used by PhysLab AI."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print incompatibilities without returning a non-zero exit code.",
    )
    args = parser.parse_args()

    issues = inspect_environment()
    print_environment_summary(issues)
    if issues and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
