from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from labs.common.splitting import DatasetSplit, split_train_validation_test


PROJECT_ROOT = Path(__file__).resolve().parent
LAB_IDS = ("cooling", "boyle_mariotte", "isochoric", "heat_balance")
NON_FEATURE_COLUMNS = {
    "class_name",
    "experiment_id",
    "generation_group",
    "device_profile",
    "environment_profile",
    "secondary_errors",
    "severity",
}
REPORT_ROOT = PROJECT_ROOT / "evaluation" / "grid_search"
BACKUP_ROOT = PROJECT_ROOT / "backups" / "pre_grid_search"


@dataclass(frozen=True)
class CheckResult:
    check: str
    status: str
    details: str


@dataclass(frozen=True)
class SearchResult:
    lab_id: str
    baseline_model: str
    tuned_model: str
    baseline_cv_macro_f1_mean: float
    baseline_cv_macro_f1_std: float
    tuned_cv_macro_f1_mean: float
    tuned_cv_macro_f1_std: float
    cv_macro_f1_change: float
    baseline_cv_accuracy_mean: float
    tuned_cv_accuracy_mean: float
    baseline_cv_balanced_accuracy_mean: float
    tuned_cv_balanced_accuracy_mean: float
    tuned_train_macro_f1_mean: float
    tuned_generalization_gap: float
    final_test_accuracy: float
    final_test_balanced_accuracy: float
    final_test_macro_f1: float
    selected_configuration: str
    model_file_action: str
    best_params: str
    combinations_tested: int
    split_strategy: str
    inner_cv_strategy: str
    development_samples: int
    test_samples: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _model_name(model: Any) -> str:
    if isinstance(model, Pipeline):
        classifier = model.named_steps.get("classifier")
        return classifier.__class__.__name__ if classifier is not None else "Pipeline"
    return model.__class__.__name__


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        short_key = key.replace("classifier__", "")
        if hasattr(value, "get_params"):
            cleaned[short_key] = value.__class__.__name__
        else:
            cleaned[short_key] = _json_safe(value)
    return cleaned


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _score_model(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "macro_f1": float(f1_score(target, predictions, average="macro", zero_division=0)),
    }


def _select_configuration(
    baseline_cv_macro_f1: float,
    tuned_cv_macro_f1: float,
    min_cv_improvement: float,
) -> str:
    """Select a configuration using development cross-validation only."""
    improvement = tuned_cv_macro_f1 - baseline_cv_macro_f1
    return "tuned" if improvement >= min_cv_improvement else "baseline"


def _load_split(lab_id: str, frame: pd.DataFrame, random_state: int) -> DatasetSplit:
    manifest_path = PROJECT_ROOT / "evaluation" / lab_id / "split_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        required = {"row_index", "dataset_role"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"Invalid split manifest: {manifest_path}")
        if len(manifest) != len(frame):
            raise ValueError(
                f"Split manifest has {len(manifest)} rows, features.csv has {len(frame)} rows"
            )
        if sorted(manifest["row_index"].astype(int).tolist()) != list(range(len(frame))):
            raise ValueError("split_manifest.csv does not cover each row exactly once")

        role = manifest.set_index("row_index")["dataset_role"]
        split = DatasetSplit(
            train_index=np.sort(role[role == "train"].index.to_numpy(dtype=int)),
            validation_index=np.sort(role[role == "validation"].index.to_numpy(dtype=int)),
            test_index=np.sort(role[role == "test"].index.to_numpy(dtype=int)),
            strategy="saved_train_validation_test_manifest",
        )
        group_column = "generation_group" if "generation_group" in frame.columns else None
        split.validate(frame, "class_name", group_column)
        return split

    return split_train_validation_test(
        frame=frame,
        target_column="class_name",
        group_column="generation_group",
        random_state=random_state,
        test_fraction=0.25,
        validation_fraction=0.25,
    )


def _development_cv_splits(
    target: pd.Series,
    groups: pd.Series | None,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    placeholder = np.zeros((len(target), 1), dtype=np.float32)
    if groups is not None and groups.nunique() >= 4:
        n_splits = min(4, int(groups.nunique()))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(splitter.split(placeholder, target, groups))
        return splits, f"StratifiedGroupKFold(n_splits={n_splits})"

    splitter = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(placeholder, target))
    return splits, "StratifiedKFold(n_splits=4)"


def _search_space(current_model: Any, mode: str, random_state: int):
    classifier = (
        current_model.named_steps.get("classifier")
        if isinstance(current_model, Pipeline)
        else current_model
    )

    if isinstance(classifier, RandomForestClassifier):
        estimator = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        if mode == "quick":
            grid = {
                "n_estimators": [220, 350],
                "max_depth": [None],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt"],
            }
        elif mode == "full":
            grid = {
                "n_estimators": [220, 350, 500, 700],
                "max_depth": [None, 8, 12, 18],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", 0.6, 0.85],
                "class_weight": ["balanced", "balanced_subsample"],
            }
        else:
            grid = {
                "n_estimators": [250, 450],
                "max_depth": [None, 10, 16],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", 0.7],
            }
        return estimator, grid

    if isinstance(classifier, GradientBoostingClassifier):
        estimator = GradientBoostingClassifier(random_state=random_state)
        if mode == "quick":
            grid = {
                "n_estimators": [100, 160],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2],
                "subsample": [0.9],
            }
        elif mode == "full":
            grid = {
                "n_estimators": [100, 160, 240, 320],
                "learning_rate": [0.03, 0.05, 0.08, 0.12],
                "max_depth": [1, 2, 3],
                "subsample": [0.75, 0.9, 1.0],
                "min_samples_leaf": [1, 3, 5],
            }
        else:
            grid = {
                "n_estimators": [120, 220],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
                "subsample": [0.85, 1.0],
            }
        return estimator, grid

    if isinstance(classifier, LogisticRegression):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        values = {
            "quick": [0.25, 1.0, 4.0],
            "balanced": [0.1, 0.35, 1.0, 3.0, 10.0],
            "full": [0.03, 0.07, 0.15, 0.35, 0.75, 1.5, 3.0, 7.0, 15.0],
        }[mode]
        return estimator, {"classifier__C": values}

    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    return estimator, {"classifier__C": [0.1, 0.35, 1.0, 3.0, 10.0]}


def _count_grid_combinations(grid: dict[str, Any] | list[dict[str, Any]]) -> int:
    blocks = grid if isinstance(grid, list) else [grid]
    total = 0
    for block in blocks:
        combinations = 1
        for values in block.values():
            combinations *= len(values)
        total += combinations
    return total


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def _backup_models() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_ROOT / stamp
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PROJECT_ROOT / "models", destination / "models")
    return destination


def _restore_models(backup_dir: Path) -> None:
    source = backup_dir / "models"
    destination = PROJECT_ROOT / "models"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _compile_project() -> CheckResult:
    roots = [PROJECT_ROOT / "core", PROJECT_ROOT / "labs", PROJECT_ROOT / "tests"]
    root_files = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "build_all.py",
        PROJECT_ROOT / "check_all_models.py",
        PROJECT_ROOT / "validate_and_gridsearch.py",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.py"))
    files.extend(path for path in root_files if path.exists())

    failures: list[str] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {error.msg}")

    if failures:
        return CheckResult("python_compile", "FAIL", " | ".join(failures[:5]))
    return CheckResult("python_compile", "PASS", f"files={len(files)}")


def _check_required_structure() -> CheckResult:
    required = [
        "app.py",
        "requirements.txt",
        "core/lab_registry.py",
        "labs/common/pipeline.py",
        "labs/common/splitting.py",
        "data",
        "models",
    ]
    missing = [item for item in required if not (PROJECT_ROOT / item).exists()]
    if missing:
        return CheckResult("required_structure", "FAIL", ", ".join(missing))
    return CheckResult("required_structure", "PASS", "required files found")


def _run_subprocess(command: list[str], timeout: int = 240) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as error:
        return False, str(error)
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output


def _check_control_scenarios(label: str) -> CheckResult:
    ok, output = _run_subprocess([sys.executable, "check_all_models.py"])
    tail = " | ".join(output.splitlines()[-8:])
    return CheckResult(label, "PASS" if ok else "FAIL", tail)


def _streamlit_smoke(seconds: int = 8) -> CheckResult:
    app_path = PROJECT_ROOT / "app.py"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    log_path = REPORT_ROOT / "streamlit_smoke.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.headless=true",
                f"--server.port={port}",
                "--browser.gatherUsageStats=false",
            ],
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start = time.time()
        page_loaded = False
        try:
            while time.time() - start < seconds:
                if process.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}", timeout=1
                    ) as response:
                        page_loaded = response.status == 200
                        if page_loaded:
                            break
                except Exception:
                    time.sleep(0.5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    logs = log_path.read_text(encoding="utf-8", errors="replace")
    fatal = any(
        token in logs
        for token in ("Traceback", "ModuleNotFoundError", "ImportError", "SyntaxError")
    )
    if page_loaded and not fatal:
        return CheckResult("streamlit_smoke", "PASS", "HTTP 200")
    return CheckResult(
        "streamlit_smoke",
        "FAIL",
        " | ".join(logs.splitlines()[-10:]) or "no response",
    )


def _save_test_report(
    lab_dir: Path,
    target: pd.Series,
    predictions: np.ndarray,
) -> None:
    labels = sorted(pd.Series(target).unique())
    matrix = confusion_matrix(target, predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        lab_dir / "final_test_confusion_matrix.csv"
    )
    pd.DataFrame(
        classification_report(target, predictions, output_dict=True, zero_division=0)
    ).T.to_csv(lab_dir / "final_test_classification_report.csv")


def _run_grid_search(
    lab_id: str,
    mode: str,
    random_state: int,
    min_cv_improvement: float,
    apply_selected: bool,
    jobs: int,
) -> SearchResult:
    feature_path = PROJECT_ROOT / "data" / lab_id / "features.csv"
    model_path = PROJECT_ROOT / "models" / lab_id / "best_model.joblib"
    names_path = PROJECT_ROOT / "models" / lab_id / "feature_names.joblib"

    frame = pd.read_csv(feature_path)
    features = _feature_columns(frame)
    split = _load_split(lab_id, frame, random_state)

    development_index = np.sort(
        np.concatenate([split.train_index, split.validation_index])
    )
    test_index = split.test_index

    x_development = frame.iloc[development_index][features]
    y_development = frame.iloc[development_index]["class_name"]
    x_test = frame.iloc[test_index][features]
    y_test = frame.iloc[test_index]["class_name"]

    development_groups = None
    if "generation_group" in frame.columns:
        development_groups = frame.iloc[development_index]["generation_group"]

    cv_splits, inner_cv_strategy = _development_cv_splits(
        target=y_development,
        groups=development_groups,
        random_state=random_state + 31,
    )
    scoring = {
        "macro_f1": "f1_macro",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }

    baseline_model = joblib.load(model_path)
    baseline_cv = cross_validate(
        estimator=clone(baseline_model),
        X=x_development,
        y=y_development,
        cv=cv_splits,
        scoring=scoring,
        n_jobs=jobs,
        return_train_score=True,
        error_score="raise",
    )
    baseline_macro_mean, baseline_macro_std = _mean_std(
        baseline_cv["test_macro_f1"]
    )

    estimator, grid = _search_space(baseline_model, mode, random_state)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=scoring,
        refit="macro_f1",
        cv=cv_splits,
        n_jobs=jobs,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(x_development, y_development)

    best_index = int(search.best_index_)
    cv_results = search.cv_results_
    tuned_macro_mean = float(cv_results["mean_test_macro_f1"][best_index])
    tuned_macro_std = float(cv_results["std_test_macro_f1"][best_index])
    tuned_train_macro = float(cv_results["mean_train_macro_f1"][best_index])
    selected_configuration = _select_configuration(
        baseline_cv_macro_f1=baseline_macro_mean,
        tuned_cv_macro_f1=tuned_macro_mean,
        min_cv_improvement=min_cv_improvement,
    )

    selected_template = (
        search.best_estimator_ if selected_configuration == "tuned" else baseline_model
    )
    final_model = clone(selected_template)
    final_model.fit(x_development, y_development)
    final_scores = _score_model(final_model, x_test, y_test)
    final_predictions = final_model.predict(x_test)

    lab_dir = REPORT_ROOT / lab_id
    lab_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, lab_dir / "candidate_best_model.joblib")
    joblib.dump(final_model, lab_dir / "selected_model.joblib")
    joblib.dump(features, lab_dir / "selected_feature_names.joblib")

    model_file_action = "not_updated"
    if apply_selected:
        joblib.dump(final_model, model_path)
        joblib.dump(features, names_path)
        model_file_action = "updated"

    cv_frame = pd.DataFrame(cv_results).copy()
    columns = [
        column
        for column in cv_frame.columns
        if column.startswith("mean_")
        or column.startswith("std_")
        or column.startswith("rank_")
        or column == "params"
    ]
    cv_frame = cv_frame[columns]
    cv_frame["params"] = cv_frame["params"].map(
        lambda value: json.dumps(_clean_params(value), ensure_ascii=False)
    )
    cv_frame.sort_values("rank_test_macro_f1").to_csv(
        lab_dir / "grid_search_cv_results.csv", index=False
    )

    test_frame = pd.DataFrame(
        {
            "row_index": test_index,
            "true_class": y_test.to_numpy(),
            "predicted_class": final_predictions,
        }
    )
    if "generation_group" in frame.columns:
        test_frame["generation_group"] = frame.iloc[test_index][
            "generation_group"
        ].to_numpy()
    if hasattr(final_model, "predict_proba"):
        test_frame["confidence"] = np.max(
            final_model.predict_proba(x_test), axis=1
        )
    test_frame.to_csv(lab_dir / "final_test_predictions.csv", index=False)
    _save_test_report(lab_dir, y_test, final_predictions)

    result = SearchResult(
        lab_id=lab_id,
        baseline_model=_model_name(baseline_model),
        tuned_model=_model_name(search.best_estimator_),
        baseline_cv_macro_f1_mean=baseline_macro_mean,
        baseline_cv_macro_f1_std=baseline_macro_std,
        tuned_cv_macro_f1_mean=tuned_macro_mean,
        tuned_cv_macro_f1_std=tuned_macro_std,
        cv_macro_f1_change=tuned_macro_mean - baseline_macro_mean,
        baseline_cv_accuracy_mean=float(np.mean(baseline_cv["test_accuracy"])),
        tuned_cv_accuracy_mean=float(cv_results["mean_test_accuracy"][best_index]),
        baseline_cv_balanced_accuracy_mean=float(
            np.mean(baseline_cv["test_balanced_accuracy"])
        ),
        tuned_cv_balanced_accuracy_mean=float(
            cv_results["mean_test_balanced_accuracy"][best_index]
        ),
        tuned_train_macro_f1_mean=tuned_train_macro,
        tuned_generalization_gap=tuned_train_macro - tuned_macro_mean,
        final_test_accuracy=final_scores["accuracy"],
        final_test_balanced_accuracy=final_scores["balanced_accuracy"],
        final_test_macro_f1=final_scores["macro_f1"],
        selected_configuration=selected_configuration,
        model_file_action=model_file_action,
        best_params=json.dumps(
            _clean_params(search.best_params_),
            ensure_ascii=False,
            sort_keys=True,
        ),
        combinations_tested=_count_grid_combinations(grid),
        split_strategy=split.strategy,
        inner_cv_strategy=inner_cv_strategy,
        development_samples=len(development_index),
        test_samples=len(test_index),
    )
    pd.DataFrame([asdict(result)]).to_csv(
        lab_dir / "grid_search_summary.csv", index=False
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-validated hyperparameter search. The test partition is excluded "
            "from model selection and used only for the final report."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "balanced", "full"),
        default="balanced",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the selected model to models/<lab_id>/best_model.joblib.",
    )
    parser.add_argument(
        "--min-cv-improvement",
        type=float,
        default=0.002,
        help="Minimum development CV Macro F1 improvement required to select tuned parameters.",
    )
    parser.add_argument(
        "--labs",
        nargs="+",
        choices=LAB_IDS,
        default=list(LAB_IDS),
        help="Labs to process. By default all four labs are processed.",
    )
    parser.add_argument("--skip-streamlit", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    checks = [
        _check_required_structure(),
        _compile_project(),
        _check_control_scenarios("control_scenarios_before_search"),
    ]
    checks_frame = pd.DataFrame([asdict(item) for item in checks])
    checks_frame.to_csv(REPORT_ROOT / "project_health_before.csv", index=False)
    if (checks_frame["status"] == "FAIL").any():
        raise SystemExit("Pre-search validation failed. See project_health_before.csv")

    backup_dir: Path | None = _backup_models() if args.apply else None
    results: list[SearchResult] = []

    for lab_id in args.labs:
        print(f"\n[{lab_id}]", flush=True)
        result = _run_grid_search(
            lab_id=lab_id,
            mode=args.mode,
            random_state=args.random_state,
            min_cv_improvement=args.min_cv_improvement,
            apply_selected=args.apply,
            jobs=args.jobs,
        )
        results.append(result)
        print(
            f"baseline CV={result.baseline_cv_macro_f1_mean:.4f}; "
            f"tuned CV={result.tuned_cv_macro_f1_mean:.4f}; "
            f"selected={result.selected_configuration}; "
            f"test={result.final_test_macro_f1:.4f}"
        )

    results_frame = pd.DataFrame([asdict(item) for item in results])
    results_frame.to_csv(REPORT_ROOT / "all_labs_grid_search_summary.csv", index=False)

    if args.apply:
        after_check = _check_control_scenarios("control_scenarios_after_search")
        checks.append(after_check)
        if after_check.status == "FAIL" and backup_dir is not None:
            _restore_models(backup_dir)
            checks.append(
                CheckResult(
                    "model_rollback",
                    "PASS",
                    str(backup_dir.relative_to(PROJECT_ROOT)),
                )
            )

    if not args.skip_streamlit:
        checks.append(_streamlit_smoke())

    checks_frame = pd.DataFrame([asdict(item) for item in checks])
    checks_frame.to_csv(REPORT_ROOT / "project_health.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "models_written": args.apply,
        "selection_metric": "development cross-validation Macro F1",
        "test_use": "final evaluation only; never used in configuration selection",
        "minimum_cv_improvement": args.min_cv_improvement,
        "backup_dir": (
            str(backup_dir.relative_to(PROJECT_ROOT)) if backup_dir else None
        ),
        "checks_failed": int((checks_frame["status"] == "FAIL").sum()),
    }
    (REPORT_ROOT / "validation_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSummary")
    print(
        results_frame[
            [
                "lab_id",
                "baseline_cv_macro_f1_mean",
                "tuned_cv_macro_f1_mean",
                "cv_macro_f1_change",
                "selected_configuration",
                "final_test_macro_f1",
                "model_file_action",
            ]
        ].to_string(index=False)
    )

    if (checks_frame["status"] == "FAIL").any():
        raise SystemExit("Validation completed with failures. See project_health.csv")


if __name__ == "__main__":
    main()
