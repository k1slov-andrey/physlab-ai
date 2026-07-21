from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from check_all_models import CASES
from evaluate_deployed_models import build_summary, check_outputs
from core.lab_registry import list_labs


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_ROOT_FILES = (
    "README.md",
    "PROJECT_DESCRIPTION.md",
    "TECHNICAL_APPENDIX.md",
    "requirements.txt",
    "requirements-dev.txt",
    "Dockerfile",
    "app.py",
    ".github/workflows/ci.yml",
    "evaluation/final_model_summary.csv",
)
REQUIRED_DATA_FILES = ("dataset.csv", "features.csv")
REQUIRED_MODEL_FILES = (
    "best_model.joblib",
    "feature_names.joblib",
    "inference_profile.json",
)
REQUIRED_EVALUATION_FILES = (
    "summary.csv",
    "model_metrics.csv",
    "split_manifest.csv",
    "evaluation_protocol.json",
    "deployed_model_metrics.json",
    "deployed_model_predictions.csv",
)
EXPECTED_DATASET_ROLES = {"train", "validation", "test"}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _check_required_files(paths: Iterable[Path], name: str) -> CheckResult:
    missing = [_relative(path) for path in paths if not path.is_file()]
    if missing:
        return CheckResult(name, False, "missing: " + ", ".join(missing))
    return CheckResult(name, True, "all required files are present")


def check_root_files() -> CheckResult:
    return _check_required_files(
        (PROJECT_ROOT / relative for relative in REQUIRED_ROOT_FILES),
        "root files",
    )


def check_lab_artifacts(lab_id: str) -> list[CheckResult]:
    data_dir = PROJECT_ROOT / "data" / lab_id
    model_dir = PROJECT_ROOT / "models" / lab_id
    evaluation_dir = PROJECT_ROOT / "evaluation" / lab_id

    results = [
        _check_required_files(
            (data_dir / filename for filename in REQUIRED_DATA_FILES),
            f"{lab_id}: data artifacts",
        ),
        _check_required_files(
            (model_dir / filename for filename in REQUIRED_MODEL_FILES),
            f"{lab_id}: model artifacts",
        ),
        _check_required_files(
            (evaluation_dir / filename for filename in REQUIRED_EVALUATION_FILES),
            f"{lab_id}: evaluation artifacts",
        ),
    ]

    if not all(result.passed for result in results):
        return results

    feature_names = [
        str(value) for value in joblib.load(model_dir / "feature_names.joblib")
    ]
    with (model_dir / "inference_profile.json").open(encoding="utf-8") as source:
        profile = json.load(source)

    profile_features = [str(value) for value in profile.get("feature_names", [])]
    profile_samples = int(profile.get("n_samples", 0))
    profile_roles = set(profile.get("dataset_roles", []))

    profile_errors: list[str] = []
    if feature_names != profile_features:
        profile_errors.append("feature list does not match feature_names.joblib")
    if profile_samples <= 0:
        profile_errors.append("n_samples must be positive")
    if profile_roles and profile_roles != {"train", "validation"}:
        profile_errors.append("profile must be built from train and validation only")

    results.append(
        CheckResult(
            f"{lab_id}: inference profile",
            not profile_errors,
            "; ".join(profile_errors) if profile_errors else f"{profile_samples} samples",
        )
    )

    model = joblib.load(model_dir / "best_model.joblib")
    model_errors: list[str] = []
    if not callable(getattr(model, "predict", None)):
        model_errors.append("model has no predict method")
    if not callable(getattr(model, "predict_proba", None)):
        model_errors.append("model has no predict_proba method")
    results.append(
        CheckResult(
            f"{lab_id}: model interface",
            not model_errors,
            "; ".join(model_errors) if model_errors else type(model).__name__,
        )
    )

    manifest = pd.read_csv(evaluation_dir / "split_manifest.csv")
    required_columns = {"row_index", "generation_group", "dataset_role"}
    manifest_errors: list[str] = []
    if not required_columns.issubset(manifest.columns):
        missing = sorted(required_columns.difference(manifest.columns))
        manifest_errors.append("missing columns: " + ", ".join(missing))
    else:
        roles = set(manifest["dataset_role"].astype(str))
        if roles != EXPECTED_DATASET_ROLES:
            manifest_errors.append(f"unexpected dataset roles: {sorted(roles)}")
        role_groups = {
            role: set(
                manifest.loc[
                    manifest["dataset_role"].astype(str) == role,
                    "generation_group",
                ].astype(str)
            )
            for role in EXPECTED_DATASET_ROLES
        }
        if role_groups["train"] & role_groups["validation"]:
            manifest_errors.append("train and validation groups overlap")
        if role_groups["train"] & role_groups["test"]:
            manifest_errors.append("train and test groups overlap")
        if role_groups["validation"] & role_groups["test"]:
            manifest_errors.append("validation and test groups overlap")

    results.append(
        CheckResult(
            f"{lab_id}: split manifest",
            not manifest_errors,
            "; ".join(manifest_errors)
            if manifest_errors
            else f"{len(manifest)} rows with disjoint groups",
        )
    )
    return results



def check_deployed_model_evaluation() -> CheckResult:
    try:
        summary, predictions = build_summary()
        check_outputs(summary, predictions)
    except (Exception, SystemExit) as error:
        return CheckResult(
            "deployed model evaluation",
            False,
            str(error),
        )
    return CheckResult(
        "deployed model evaluation",
        True,
        f"{len(summary)} deployed models match the saved test results",
    )


def check_control_scenarios() -> CheckResult:
    failures: list[str] = []
    checked = 0
    for lab_id, simulator, predictor, scenarios in CASES:
        for class_name, seed in scenarios.items():
            checked += 1
            result = predictor(simulator(class_name, seed=seed))
            if result.predicted_class != class_name:
                failures.append(
                    f"{lab_id}/{class_name}: predicted {result.predicted_class}"
                )
    return CheckResult(
        "control scenarios",
        not failures,
        "; ".join(failures) if failures else f"{checked} scenarios passed",
    )


def run_checks() -> list[CheckResult]:
    results = [check_root_files()]
    for lab in list_labs():
        results.extend(check_lab_artifacts(lab.lab_id))
    results.append(check_deployed_model_evaluation())
    results.append(check_control_scenarios())
    return results


def main() -> None:
    results = run_checks()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.details}")

    failed = [result for result in results if not result.passed]
    if failed:
        raise SystemExit(f"Project verification failed: {len(failed)} check(s)")
    print(f"Project verification passed: {len(results)} checks")


if __name__ == "__main__":
    main()
