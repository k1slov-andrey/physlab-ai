from __future__ import annotations

import argparse
import json
import math
import os
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
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


@dataclass
class CheckResult:
    check: str
    status: str
    details: str


@dataclass
class SearchResult:
    lab_id: str
    current_model: str
    tuned_model: str
    current_holdout_accuracy: float
    current_holdout_balanced_accuracy: float
    current_holdout_macro_f1: float
    tuned_holdout_accuracy: float
    tuned_holdout_balanced_accuracy: float
    tuned_holdout_macro_f1: float
    macro_f1_change: float
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    cv_accuracy_mean: float
    cv_balanced_accuracy_mean: float
    train_macro_f1_mean: float
    generalization_gap: float
    best_params: str
    combinations_tested: int
    validation_strategy: str
    decision: str
    production_refit_samples: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
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
        short_key = key.replace("classifier__", "").replace("scaler", "preprocessing")
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


def _outer_split(
    frame: pd.DataFrame,
    features: list[str],
    random_state: int,
):
    x = frame[features]
    y = frame["class_name"]

    if "generation_group" in frame.columns and frame["generation_group"].nunique() >= 4:
        groups = frame["generation_group"]
        splitter = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state,
        )
        train_index, test_index = next(splitter.split(x, y, groups))
        return (
            x.iloc[train_index],
            x.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
            groups.iloc[train_index],
            groups.iloc[test_index],
            "nested_stratified_group_holdout",
        )

    train_index, test_index = train_test_split(
        np.arange(len(frame)),
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )
    return (
        x.iloc[train_index],
        x.iloc[test_index],
        y.iloc[train_index],
        y.iloc[test_index],
        None,
        None,
        "nested_stratified_random_holdout",
    )


def _inner_cv(y_train: pd.Series, train_groups: pd.Series | None, random_state: int):
    if train_groups is not None and train_groups.nunique() >= 3:
        n_splits = min(3, int(train_groups.nunique()))
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + 17,
        )
    return StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state + 17,
    )


def _search_space(current_model: Any, mode: str, random_state: int):
    """Return a targeted GridSearchCV estimator and grid for the current best model family."""
    model = current_model
    if isinstance(model, Pipeline):
        classifier = model.named_steps.get("classifier")
    else:
        classifier = model

    if isinstance(classifier, RandomForestClassifier):
        estimator = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        if mode == "quick":
            grid = {
                "n_estimators": [250, 450],
                "max_depth": [None, 12],
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
                "n_estimators": [120, 220],
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
        if mode == "quick":
            values = [0.25, 1.0, 4.0]
        elif mode == "full":
            values = [0.03, 0.07, 0.15, 0.35, 0.75, 1.5, 3.0, 7.0, 15.0]
        else:
            values = [0.1, 0.35, 1.0, 3.0, 10.0]
        return estimator, {"classifier__C": values}

    # Conservative fallback for an unknown estimator family.
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


def _score_model(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
    }


def _backup_models() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_ROOT / stamp
    destination.mkdir(parents=True, exist_ok=False)
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        shutil.copytree(models_dir, destination / "models")
    return destination


def _restore_models(backup_dir: Path) -> None:
    source = backup_dir / "models"
    destination = PROJECT_ROOT / "models"
    if not source.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _compile_project(checks: list[CheckResult]) -> None:
    roots = [PROJECT_ROOT / "core", PROJECT_ROOT / "labs"]
    root_files = [
        PROJECT_ROOT / "app_backup.py",
        PROJECT_ROOT / "build_all.py",
        PROJECT_ROOT / "check_all_models.py",
        PROJECT_ROOT / "prepare_real_data.py",
        PROJECT_ROOT / "apply_real_calibration.py",
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
        checks.append(CheckResult("python_compile", "FAIL", " | ".join(failures[:5])))
    else:
        checks.append(CheckResult("python_compile", "PASS", f"Проверено файлов: {len(files)}"))


def _check_required_structure(checks: list[CheckResult]) -> None:
    required = [
        "app_backup.py",
        "requirements.txt",
        "core/lab_registry.py",
        "core/recommendation_engine.py",
        "core/competency_engine.py",
        "labs/common/pipeline.py",
        "labs/common/realism.py",
        "data/real_normalized",
        "evaluation/real_data_integration",
        "models",
    ]
    missing = [item for item in required if not (PROJECT_ROOT / item).exists()]
    if missing:
        checks.append(CheckResult("required_structure", "FAIL", "Не найдено: " + ", ".join(missing)))
    else:
        checks.append(CheckResult("required_structure", "PASS", "Обязательная структура проекта присутствует"))


def _check_lab_data(lab_id: str, checks: list[CheckResult]) -> dict[str, Any]:
    data_dir = PROJECT_ROOT / "data" / lab_id
    feature_path = data_dir / "features.csv"
    raw_path = data_dir / "dataset.csv"
    model_path = PROJECT_ROOT / "models" / lab_id / "best_model.joblib"
    names_path = PROJECT_ROOT / "models" / lab_id / "feature_names.joblib"

    for path in (feature_path, raw_path, model_path, names_path):
        if not path.exists():
            checks.append(CheckResult(f"{lab_id}_files", "FAIL", f"Не найден: {path.relative_to(PROJECT_ROOT)}"))
            return {}

    features = pd.read_csv(feature_path)
    raw = pd.read_csv(raw_path)
    feature_columns = _feature_columns(features)

    numeric = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    missing_cells = int(numeric.isna().sum().sum())
    duplicate_ids = int(features["experiment_id"].duplicated().sum()) if "experiment_id" in features else -1
    class_counts = features["class_name"].value_counts().to_dict()
    groups = int(features["generation_group"].nunique()) if "generation_group" in features else 0
    raw_experiments = int(raw["experiment_id"].nunique()) if "experiment_id" in raw else 0

    issues: list[str] = []
    if missing_cells:
        issues.append(f"NaN/inf в признаках: {missing_cells}")
    if duplicate_ids:
        issues.append(f"дубли experiment_id: {duplicate_ids}")
    if raw_experiments != len(features):
        issues.append(f"raw experiments={raw_experiments}, feature rows={len(features)}")
    if len(class_counts) < 2:
        issues.append("менее двух классов")
    if groups and groups < 4:
        issues.append(f"мало generation_group: {groups}")

    model = joblib.load(model_path)
    saved_names = list(joblib.load(names_path))
    if saved_names != feature_columns:
        issues.append("feature_names.joblib не совпадает с features.csv")
    try:
        _ = model.predict(features[saved_names].head(2))
    except Exception as error:
        issues.append(f"модель не предсказывает: {error}")

    status = "PASS" if not issues else "FAIL"
    details = (
        f"rows={len(features)}, features={len(feature_columns)}, classes={class_counts}, "
        f"groups={groups}, model={_model_name(model)}"
    )
    if issues:
        details += "; " + "; ".join(issues)
    checks.append(CheckResult(f"{lab_id}_data_model", status, details))

    return {
        "lab_id": lab_id,
        "rows": len(features),
        "features": len(feature_columns),
        "classes": len(class_counts),
        "groups": groups,
        "missing_cells": missing_cells,
        "duplicate_experiment_ids": duplicate_ids,
        "raw_experiments": raw_experiments,
        "class_balance_ratio": float(min(class_counts.values()) / max(class_counts.values())),
        "current_model": _model_name(model),
    }


def _run_subprocess(command: list[str], timeout: int = 180) -> tuple[bool, str]:
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


def _check_control_scenarios(checks: list[CheckResult], label: str) -> bool:
    path = PROJECT_ROOT / "check_all_models.py"
    if not path.exists():
        checks.append(CheckResult(label, "FAIL", "check_all_models.py не найден"))
        return False
    ok, output = _run_subprocess([sys.executable, str(path)], timeout=240)
    tail = " | ".join(output.splitlines()[-8:])
    checks.append(CheckResult(label, "PASS" if ok else "FAIL", tail))
    return ok


def _check_real_data(checks: list[CheckResult]) -> dict[str, Any]:
    quality_path = PROJECT_ROOT / "evaluation" / "real_data_integration" / "quality_report.csv"
    predictions_path = PROJECT_ROOT / "evaluation" / "real_data_integration" / "real_data_model_predictions.csv"
    normalized_root = PROJECT_ROOT / "data" / "real_normalized"

    if not quality_path.exists() or not predictions_path.exists() or not normalized_root.exists():
        checks.append(CheckResult("real_data", "FAIL", "Не найдены нормализованные данные или отчёты интеграции"))
        return {}

    quality = pd.read_csv(quality_path)
    predictions = pd.read_csv(predictions_path)
    successful = predictions[predictions["prediction_status"] == "success"] if "prediction_status" in predictions else pd.DataFrame()
    if "ready_for_model" in quality.columns:
        ready_values = quality["ready_for_model"].astype(str).str.lower().isin({"true", "1", "yes"})
        review_count = int((~ready_values).sum())
    elif "quality_status" in quality.columns:
        review_count = int((quality["quality_status"] != "ready").sum())
    else:
        review_count = 0
    low_confidence = int((successful["confidence"] < 0.60).sum()) if not successful.empty and "confidence" in successful else 0
    mean_confidence = float(successful["confidence"].mean()) if not successful.empty and "confidence" in successful else math.nan

    details = (
        f"quality_rows={len(quality)}, predictions={len(predictions)}, "
        f"successful={len(successful)}, manual_review={review_count}, "
        f"mean_confidence={mean_confidence:.3f}, low_confidence={low_confidence}"
    )
    checks.append(CheckResult("real_data", "PASS", details))
    return {
        "quality_rows": len(quality),
        "predictions": len(predictions),
        "successful_predictions": len(successful),
        "manual_review": review_count,
        "mean_confidence": mean_confidence,
        "low_confidence_below_0_60": low_confidence,
    }


def _streamlit_smoke(checks: list[CheckResult], seconds: int = 8) -> None:
    app_path = PROJECT_ROOT / "app_backup.py"
    if not app_path.exists():
        checks.append(CheckResult("streamlit_smoke", "FAIL", "app_backup.py не найден"))
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    log_path = REPORT_ROOT / "streamlit_smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=1) as response:
                        page_loaded = response.status == 200
                        if page_loaded:
                            time.sleep(2)
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

    logs = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    fatal_tokens = ("Traceback", "ModuleNotFoundError", "ImportError", "SyntaxError")
    fatal = any(token in logs for token in fatal_tokens)
    if page_loaded and not fatal:
        checks.append(CheckResult("streamlit_smoke", "PASS", f"HTTP 200 на локальном порту {port}"))
    else:
        tail = " | ".join(logs.splitlines()[-10:])
        checks.append(CheckResult("streamlit_smoke", "FAIL", tail or "Streamlit не ответил"))


def _save_confusion_and_report(
    lab_dir: Path,
    y_true: pd.Series,
    predictions: np.ndarray,
) -> None:
    labels = sorted(pd.Series(y_true).unique())
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(lab_dir / "holdout_confusion_matrix.csv")
    pd.DataFrame(
        classification_report(y_true, predictions, output_dict=True, zero_division=0)
    ).T.to_csv(lab_dir / "holdout_classification_report.csv")


def _run_grid_search(
    lab_id: str,
    mode: str,
    random_state: int,
    min_improvement: float,
    apply_best: bool,
    jobs: int,
) -> SearchResult:
    feature_path = PROJECT_ROOT / "data" / lab_id / "features.csv"
    model_path = PROJECT_ROOT / "models" / lab_id / "best_model.joblib"
    names_path = PROJECT_ROOT / "models" / lab_id / "feature_names.joblib"
    frame = pd.read_csv(feature_path)
    features = _feature_columns(frame)
    x_all = frame[features]
    y_all = frame["class_name"]

    (
        x_train,
        x_holdout,
        y_train,
        y_holdout,
        train_groups,
        _holdout_groups,
        validation_strategy,
    ) = _outer_split(frame, features, random_state)

    current_model = joblib.load(model_path)
    current_scores = _score_model(current_model, x_holdout, y_holdout)

    estimator, grid = _search_space(current_model, mode, random_state)
    combinations = _count_grid_combinations(grid)
    cv = _inner_cv(y_train, train_groups, random_state)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring={
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit="macro_f1",
        cv=cv,
        n_jobs=jobs,
        return_train_score=True,
        error_score="raise",
        verbose=0,
    )
    fit_kwargs: dict[str, Any] = {}
    if train_groups is not None:
        fit_kwargs["groups"] = train_groups
    search.fit(x_train, y_train, **fit_kwargs)

    tuned_model = search.best_estimator_
    tuned_scores = _score_model(tuned_model, x_holdout, y_holdout)
    holdout_predictions = tuned_model.predict(x_holdout)

    best_index = int(search.best_index_)
    cv_results = search.cv_results_
    cv_macro_mean = float(cv_results["mean_test_macro_f1"][best_index])
    cv_macro_std = float(cv_results["std_test_macro_f1"][best_index])
    cv_accuracy = float(cv_results["mean_test_accuracy"][best_index])
    cv_balanced = float(cv_results["mean_test_balanced_accuracy"][best_index])
    train_macro = float(cv_results["mean_train_macro_f1"][best_index])
    generalization_gap = train_macro - cv_macro_mean

    improvement = tuned_scores["macro_f1"] - current_scores["macro_f1"]
    should_apply = apply_best and (
        improvement >= min_improvement
        or (
            improvement >= -1e-9
            and tuned_scores["accuracy"] > current_scores["accuracy"] + 0.005
        )
    )

    production_model = clone(search.best_estimator_)
    production_model.fit(x_all, y_all)

    lab_dir = REPORT_ROOT / lab_id
    lab_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_model, lab_dir / "candidate_best_model.joblib")
    joblib.dump(features, lab_dir / "candidate_feature_names.joblib")

    cv_frame = pd.DataFrame(cv_results).copy()
    columns_to_keep = [
        column
        for column in cv_frame.columns
        if column.startswith("mean_")
        or column.startswith("std_")
        or column.startswith("rank_")
        or column == "params"
    ]
    cv_frame = cv_frame[columns_to_keep]
    cv_frame["params"] = cv_frame["params"].map(lambda value: json.dumps(_clean_params(value), ensure_ascii=False))
    cv_frame.sort_values("rank_test_macro_f1").to_csv(lab_dir / "grid_search_cv_results.csv", index=False)

    holdout_frame = pd.DataFrame(
        {
            "row_index": x_holdout.index,
            "true_class": y_holdout.to_numpy(),
            "current_prediction": current_model.predict(x_holdout),
            "tuned_prediction": holdout_predictions,
        }
    )
    if hasattr(tuned_model, "predict_proba"):
        holdout_frame["tuned_confidence"] = np.max(tuned_model.predict_proba(x_holdout), axis=1)
    holdout_frame.to_csv(lab_dir / "holdout_predictions.csv", index=False)
    _save_confusion_and_report(lab_dir, y_holdout, holdout_predictions)

    decision = "applied" if should_apply else "kept_current"
    if should_apply:
        joblib.dump(production_model, model_path)
        joblib.dump(features, names_path)

    params_text = json.dumps(_clean_params(search.best_params_), ensure_ascii=False, sort_keys=True)
    result = SearchResult(
        lab_id=lab_id,
        current_model=_model_name(current_model),
        tuned_model=_model_name(tuned_model),
        current_holdout_accuracy=current_scores["accuracy"],
        current_holdout_balanced_accuracy=current_scores["balanced_accuracy"],
        current_holdout_macro_f1=current_scores["macro_f1"],
        tuned_holdout_accuracy=tuned_scores["accuracy"],
        tuned_holdout_balanced_accuracy=tuned_scores["balanced_accuracy"],
        tuned_holdout_macro_f1=tuned_scores["macro_f1"],
        macro_f1_change=improvement,
        cv_macro_f1_mean=cv_macro_mean,
        cv_macro_f1_std=cv_macro_std,
        cv_accuracy_mean=cv_accuracy,
        cv_balanced_accuracy_mean=cv_balanced,
        train_macro_f1_mean=train_macro,
        generalization_gap=generalization_gap,
        best_params=params_text,
        combinations_tested=combinations,
        validation_strategy=validation_strategy,
        decision=decision,
        production_refit_samples=len(frame),
    )
    pd.DataFrame([asdict(result)]).to_csv(lab_dir / "grid_search_summary.csv", index=False)
    return result


def _update_all_labs_summary(results: list[SearchResult]) -> None:
    rows = []
    for result in results:
        row = asdict(result)
        row["grid_search_used"] = True
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORT_ROOT / "all_labs_grid_search_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Полная проверка PhysLab AI и group-aware GridSearchCV для четырех моделей."
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "balanced", "full"),
        default="balanced",
        help="Размер сетки: quick, balanced или full.",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Не заменять производственные модели, только построить отчёт.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.002,
        help="Минимальный прирост holdout Macro F1 для замены модели.",
    )
    parser.add_argument(
        "--skip-streamlit",
        action="store_true",
        help="Пропустить локальную проверку запуска Streamlit.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2, help="Параллельные процессы GridSearchCV (по умолчанию 2).")
    args = parser.parse_args()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    checks: list[CheckResult] = []
    data_rows: list[dict[str, Any]] = []

    print("=== 1. ПРОВЕРКА ТЕКУЩЕГО ПРОЕКТА ===")
    _check_required_structure(checks)
    _compile_project(checks)
    for lab_id in LAB_IDS:
        row = _check_lab_data(lab_id, checks)
        if row:
            data_rows.append(row)
    real_summary_before = _check_real_data(checks)
    baseline_controls_ok = _check_control_scenarios(checks, "control_scenarios_before_grid")

    if not baseline_controls_ok or any(item.status == "FAIL" for item in checks):
        pd.DataFrame([asdict(item) for item in checks]).to_csv(REPORT_ROOT / "project_health.csv", index=False)
        raise SystemExit("Базовая проверка не пройдена. Grid Search не запущен; см. evaluation/grid_search/project_health.csv")

    backup_dir = _backup_models()
    print(f"Резервная копия моделей: {backup_dir.relative_to(PROJECT_ROOT)}")

    print("\n=== 2. GROUP-AWARE GRID SEARCH ===")
    results: list[SearchResult] = []
    for lab_id in LAB_IDS:
        print(f"\n--- {lab_id} ---")
        result = _run_grid_search(
            lab_id=lab_id,
            mode=args.mode,
            random_state=args.random_state,
            min_improvement=args.min_improvement,
            apply_best=not args.no_apply,
            jobs=args.jobs,
        )
        results.append(result)
        print(
            f"current F1={result.current_holdout_macro_f1:.4f}; "
            f"tuned F1={result.tuned_holdout_macro_f1:.4f}; "
            f"CV={result.cv_macro_f1_mean:.4f}±{result.cv_macro_f1_std:.4f}; "
            f"decision={result.decision}"
        )

    _update_all_labs_summary(results)

    print("\n=== 3. СКВОЗНАЯ ПРОВЕРКА ПОСЛЕ GRID SEARCH ===")
    controls_ok = _check_control_scenarios(checks, "control_scenarios_after_grid")
    if not controls_ok:
        _restore_models(backup_dir)
        checks.append(CheckResult("grid_search_rollback", "PASS", "Модели восстановлены из резервной копии"))
        _check_control_scenarios(checks, "control_scenarios_after_rollback")
        for result in results:
            if result.decision == "applied":
                result.decision = "rolled_back_after_smoke_failure"
        _update_all_labs_summary(results)

    prepare_path = PROJECT_ROOT / "prepare_real_data.py"
    if prepare_path.exists():
        ok, output = _run_subprocess([sys.executable, str(prepare_path)], timeout=600)
        checks.append(CheckResult("real_data_reprediction", "PASS" if ok else "FAIL", " | ".join(output.splitlines()[-8:])))
    real_summary_after = _check_real_data(checks)

    if not args.skip_streamlit:
        _streamlit_smoke(checks)

    pd.DataFrame(data_rows).to_csv(REPORT_ROOT / "data_health.csv", index=False)
    checks_df = pd.DataFrame([asdict(item) for item in checks])
    checks_df.to_csv(REPORT_ROOT / "project_health.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "grid_mode": args.mode,
        "models_applied": int(sum(result.decision == "applied" for result in results)),
        "models_kept": int(sum(result.decision == "kept_current" for result in results)),
        "project_checks_passed": int((checks_df["status"] == "PASS").sum()),
        "project_checks_failed": int((checks_df["status"] == "FAIL").sum()),
        "real_data_before": real_summary_before,
        "real_data_after": real_summary_after,
        "backup_dir": str(backup_dir.relative_to(PROJECT_ROOT)),
        "note": "GridSearchCV is a hyperparameter search procedure; primary selection metric is Macro F1.",
    }
    (REPORT_ROOT / "validation_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== ИТОГ GRID SEARCH ===")
    result_frame = pd.DataFrame([asdict(result) for result in results])
    print(
        result_frame[
            [
                "lab_id",
                "current_holdout_macro_f1",
                "tuned_holdout_macro_f1",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "generalization_gap",
                "decision",
            ]
        ].to_string(index=False)
    )
    print("\n=== СОСТОЯНИЕ ПРОЕКТА ===")
    print(checks_df[["check", "status"]].to_string(index=False))
    print(f"\nОтчёты: {REPORT_ROOT.relative_to(PROJECT_ROOT)}")

    if (checks_df["status"] == "FAIL").any():
        raise SystemExit("Проверка завершена с ошибками. См. project_health.csv")
    print("\nPhysLab AI: данные, модели, реальные файлы и интерфейс проверены успешно.")


if __name__ == "__main__":
    main()
