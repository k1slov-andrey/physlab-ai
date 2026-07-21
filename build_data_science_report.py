from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from core.lab_registry import list_labs
from labs.common.reliability import assess_reliability


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "data_science"
MANIFEST_PATH = OUTPUT_DIR / "report_manifest.json"
FLOAT_TOLERANCE = 1e-10
BOOTSTRAP_REPEATS = 1000
ROBUSTNESS_SEEDS = (7, 19, 42, 73, 101)

LAB_TITLES = {
    "cooling": "Нагревание и охлаждение",
    "boyle_mariotte": "Закон Бойля — Мариотта",
    "isochoric": "Изохорный процесс",
    "heat_balance": "Тепловой баланс",
}


@dataclass(frozen=True)
class LabBundle:
    lab_id: str
    model: object
    feature_names: list[str]
    features: pd.DataFrame
    dataset: pd.DataFrame
    manifest: pd.DataFrame
    profile: dict[str, object]
    x_development: pd.DataFrame
    y_development: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    test_manifest: pd.DataFrame
    predictions: np.ndarray
    probabilities: np.ndarray


def _load_bundle(lab_id: str) -> LabBundle:
    features = pd.read_csv(PROJECT_ROOT / "data" / lab_id / "features.csv")
    dataset = pd.read_csv(PROJECT_ROOT / "data" / lab_id / "dataset.csv")
    manifest = pd.read_csv(
        PROJECT_ROOT / "evaluation" / lab_id / "split_manifest.csv"
    )
    feature_names = [
        str(value)
        for value in joblib.load(
            PROJECT_ROOT / "models" / lab_id / "feature_names.joblib"
        )
    ]
    model = joblib.load(PROJECT_ROOT / "models" / lab_id / "best_model.joblib")
    profile = json.loads(
        (PROJECT_ROOT / "models" / lab_id / "inference_profile.json").read_text(
            encoding="utf-8"
        )
    )

    development_manifest = manifest.loc[
        manifest["dataset_role"].isin(["train", "validation"])
    ].copy()
    test_manifest = manifest.loc[manifest["dataset_role"] == "test"].copy()

    development_rows = development_manifest["row_index"].astype(int).to_numpy()
    test_rows = test_manifest["row_index"].astype(int).to_numpy()

    x_development = features.iloc[development_rows][feature_names].reset_index(drop=True)
    y_development = development_manifest["target"].astype(str).reset_index(drop=True)
    x_test = features.iloc[test_rows][feature_names].reset_index(drop=True)
    y_test = test_manifest["target"].astype(str).reset_index(drop=True)
    test_manifest = test_manifest.reset_index(drop=True)

    predictions = np.asarray(model.predict(x_test), dtype=str)
    probabilities = np.asarray(model.predict_proba(x_test), dtype=float)

    return LabBundle(
        lab_id=lab_id,
        model=model,
        feature_names=feature_names,
        features=features,
        dataset=dataset,
        manifest=manifest,
        profile=profile,
        x_development=x_development,
        y_development=y_development,
        x_test=x_test,
        y_test=y_test,
        test_manifest=test_manifest,
        predictions=predictions,
        probabilities=probabilities,
    )


def _safe_macro_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def build_eda_overview(bundles: list[LabBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        point_counts = bundle.dataset.groupby("experiment_id", sort=False).size()
        class_counts = bundle.features["class_name"].astype(str).value_counts()
        model_values = bundle.features[bundle.feature_names]
        rows.append(
            {
                "lab_id": bundle.lab_id,
                "lab_title": LAB_TITLES[bundle.lab_id],
                "raw_measurements": int(len(bundle.dataset)),
                "experiments": int(bundle.dataset["experiment_id"].nunique()),
                "median_points_per_experiment": float(point_counts.median()),
                "min_points_per_experiment": int(point_counts.min()),
                "max_points_per_experiment": int(point_counts.max()),
                "feature_rows": int(len(bundle.features)),
                "model_features": int(len(bundle.feature_names)),
                "generation_groups": int(
                    bundle.features["generation_group"].nunique()
                ),
                "classes": int(bundle.features["class_name"].nunique()),
                "min_class_samples": int(class_counts.min()),
                "max_class_samples": int(class_counts.max()),
                "missing_model_values": int(model_values.isna().sum().sum()),
                "duplicate_experiment_ids": int(
                    bundle.features["experiment_id"].duplicated().sum()
                ),
                "duplicate_model_vectors": int(model_values.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("lab_id").reset_index(drop=True)


def build_class_balance(bundles: list[LabBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        merged = bundle.manifest[["row_index", "dataset_role", "target"]].copy()
        for (role, target), group in merged.groupby(["dataset_role", "target"]):
            rows.append(
                {
                    "lab_id": bundle.lab_id,
                    "dataset_role": str(role),
                    "class_name": str(target),
                    "samples": int(len(group)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["lab_id", "dataset_role", "class_name"]
    ).reset_index(drop=True)


def build_candidate_comparison(bundles: list[LabBundle]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        frame = pd.read_csv(
            PROJECT_ROOT / "evaluation" / bundle.lab_id / "model_metrics.csv"
        )
        frame.insert(0, "lab_id", bundle.lab_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["lab_id", "validation_macro_f1"], ascending=[True, False]
    ).reset_index(drop=True)


def build_model_selection_summary(bundles: list[LabBundle]) -> pd.DataFrame:
    final_metrics = pd.read_csv(PROJECT_ROOT / "evaluation" / "final_model_summary.csv")
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        grid = pd.read_csv(
            PROJECT_ROOT
            / "evaluation"
            / "grid_search"
            / bundle.lab_id
            / "grid_search_summary.csv"
        ).iloc[0]
        family = pd.read_csv(
            PROJECT_ROOT / "evaluation" / bundle.lab_id / "model_metrics.csv"
        )
        selected_family = family.loc[family["selected"].astype(bool)].iloc[0]
        final = final_metrics.loc[final_metrics["lab_id"] == bundle.lab_id].iloc[0]
        rows.append(
            {
                "lab_id": bundle.lab_id,
                "lab_title": LAB_TITLES[bundle.lab_id],
                "dummy_validation_macro_f1": float(
                    family.loc[family["model"] == "dummy", "validation_macro_f1"].iloc[0]
                ),
                "selected_family": str(selected_family["model"]),
                "selected_family_validation_macro_f1": float(
                    selected_family["validation_macro_f1"]
                ),
                "baseline_cv_macro_f1_mean": float(grid["baseline_cv_macro_f1_mean"]),
                "baseline_cv_macro_f1_std": float(grid["baseline_cv_macro_f1_std"]),
                "tuned_cv_macro_f1_mean": float(grid["tuned_cv_macro_f1_mean"]),
                "tuned_cv_macro_f1_std": float(grid["tuned_cv_macro_f1_std"]),
                "cv_macro_f1_change": float(grid["cv_macro_f1_change"]),
                "selected_configuration": str(grid["selected_configuration"]),
                "deployed_model_class": str(final["model_class"]),
                "final_test_accuracy": float(final["accuracy"]),
                "final_test_balanced_accuracy": float(final["balanced_accuracy"]),
                "final_test_macro_f1": float(final["macro_f1"]),
                "best_params": str(grid["best_params"]),
                "development_samples": int(grid["development_samples"]),
                "test_samples": int(grid["test_samples"]),
            }
        )
    return pd.DataFrame(rows).sort_values("lab_id").reset_index(drop=True)


def build_per_class_metrics(bundles: list[LabBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        report = classification_report(
            bundle.y_test,
            bundle.predictions,
            output_dict=True,
            zero_division=0,
        )
        for class_name in sorted(bundle.y_test.unique()):
            values = report[str(class_name)]
            rows.append(
                {
                    "lab_id": bundle.lab_id,
                    "class_name": str(class_name),
                    "precision": float(values["precision"]),
                    "recall": float(values["recall"]),
                    "f1_score": float(values["f1-score"]),
                    "support": int(values["support"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["lab_id", "class_name"]
    ).reset_index(drop=True)


def build_confusion_matrices(bundles: list[LabBundle]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for bundle in bundles:
        labels = sorted(bundle.y_test.unique())
        matrix = confusion_matrix(bundle.y_test, bundle.predictions, labels=labels)
        frame = pd.DataFrame(matrix, index=labels, columns=labels)
        frame.index.name = "actual_class"
        outputs[bundle.lab_id] = frame.reset_index()
    return outputs


def build_feature_importance(bundles: list[LabBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        values = getattr(bundle.model, "feature_importances_", None)
        if values is None:
            raise ValueError(
                f"{bundle.lab_id}: deployed model has no native feature importance"
            )
        order = np.argsort(np.asarray(values, dtype=float))[::-1]
        for rank, index in enumerate(order, start=1):
            rows.append(
                {
                    "lab_id": bundle.lab_id,
                    "rank": rank,
                    "feature": bundle.feature_names[int(index)],
                    "importance": float(values[int(index)]),
                    "evaluation_note": "native_importance_from_deployed_tree_model",
                }
            )
    return pd.DataFrame(rows).sort_values(["lab_id", "rank"]).reset_index(drop=True)

def _bootstrap_macro_f1(bundle: LabBundle) -> np.ndarray:
    rng = np.random.default_rng(42)
    groups = bundle.test_manifest["generation_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    y_true = bundle.y_test.to_numpy(dtype=str)
    y_pred = bundle.predictions
    scores = np.empty(BOOTSTRAP_REPEATS, dtype=float)

    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    for iteration in range(BOOTSTRAP_REPEATS):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        scores[iteration] = _safe_macro_f1(
            y_true[sampled_indices], y_pred[sampled_indices]
        )
    return scores


def build_uncertainty_summary(bundles: list[LabBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        scores = _bootstrap_macro_f1(bundle)
        rows.append(
            {
                "lab_id": bundle.lab_id,
                "test_groups": int(
                    bundle.test_manifest["generation_group"].nunique()
                ),
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "macro_f1": _safe_macro_f1(bundle.y_test, bundle.predictions),
                "macro_f1_ci95_low": float(np.quantile(scores, 0.025)),
                "macro_f1_ci95_high": float(np.quantile(scores, 0.975)),
                "bootstrap_mean": float(scores.mean()),
                "bootstrap_std": float(scores.std(ddof=1)),
                "resampling_unit": "generation_group",
            }
        )
    return pd.DataFrame(rows).sort_values("lab_id").reset_index(drop=True)


def build_seed_robustness(bundles: list[LabBundle]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        for seed in ROBUSTNESS_SEEDS:
            model = clone(bundle.model)
            params = model.get_params(deep=False)
            if "random_state" in params:
                model.set_params(random_state=seed)
            model.fit(bundle.x_development, bundle.y_development)
            prediction = model.predict(bundle.x_test)
            rows.append(
                {
                    "lab_id": bundle.lab_id,
                    "random_state": seed,
                    "accuracy": float(accuracy_score(bundle.y_test, prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(bundle.y_test, prediction)
                    ),
                    "macro_f1": _safe_macro_f1(bundle.y_test, prediction),
                    "protocol": "same_locked_split_refit_selected_configuration",
                }
            )
    detail = pd.DataFrame(rows).sort_values(
        ["lab_id", "random_state"]
    ).reset_index(drop=True)
    summary = (
        detail.groupby("lab_id", as_index=False)
        .agg(
            seeds=("random_state", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
        )
        .sort_values("lab_id")
        .reset_index(drop=True)
    )
    return detail, summary


def _selective_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accepted: np.ndarray,
) -> float:
    if not accepted.any():
        return float("nan")
    return _safe_macro_f1(y_true[accepted], y_pred[accepted])


def _stress_rejection_rates(bundle: LabBundle) -> tuple[float, float]:
    profile_features = bundle.profile["features"]
    allowed_outside = max(1, int(ceil(len(bundle.feature_names) * 0.15)))
    mild_shift_count = min(len(bundle.feature_names), allowed_outside + 1)

    severe = bundle.x_test.copy()
    first_name = bundle.feature_names[0]
    first_profile = profile_features[first_name]
    severe[first_name] = float(first_profile["upper_bound"]) + 5.0 * float(
        first_profile["scale"]
    )
    severe_probabilities = bundle.model.predict_proba(severe)
    severe_rejected = sum(
        not assess_reliability(
            row.to_dict(), severe_probabilities[index], bundle.profile
        ).accepted
        for index, (_, row) in enumerate(severe.iterrows())
    )

    multivariate = bundle.x_test.copy()
    for feature_name in bundle.feature_names[:mild_shift_count]:
        parameters = profile_features[feature_name]
        multivariate[feature_name] = float(parameters["upper_bound"]) + 0.5 * float(
            parameters["scale"]
        )
    multivariate_probabilities = bundle.model.predict_proba(multivariate)
    multivariate_rejected = sum(
        not assess_reliability(
            row.to_dict(), multivariate_probabilities[index], bundle.profile
        ).accepted
        for index, (_, row) in enumerate(multivariate.iterrows())
    )

    sample_count = len(bundle.x_test)
    return severe_rejected / sample_count, multivariate_rejected / sample_count

def build_reliability_summary(bundles: list[LabBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        assessments = [
            assess_reliability(row.to_dict(), bundle.probabilities[index], bundle.profile)
            for index, (_, row) in enumerate(bundle.x_test.iterrows())
        ]
        accepted = np.asarray([item.accepted for item in assessments], dtype=bool)
        statuses = pd.Series([item.status for item in assessments]).value_counts()
        y_true = bundle.y_test.to_numpy(dtype=str)
        y_pred = bundle.predictions
        correct = y_true == y_pred
        errors = ~correct
        rejected = ~accepted
        severe_rate, multivariate_rate = _stress_rejection_rates(bundle)

        rows.append(
            {
                "lab_id": bundle.lab_id,
                "test_samples": int(len(y_true)),
                "accepted_samples": int(accepted.sum()),
                "rejected_samples": int(rejected.sum()),
                "coverage": float(accepted.mean()),
                "abstention_rate": float(rejected.mean()),
                "overall_accuracy": float(correct.mean()),
                "overall_macro_f1": _safe_macro_f1(y_true, y_pred),
                "selective_accuracy": float(correct[accepted].mean())
                if accepted.any()
                else float("nan"),
                "selective_macro_f1": _selective_macro_f1(y_true, y_pred, accepted),
                "errors_captured_by_abstention": int((errors & rejected).sum()),
                "total_errors": int(errors.sum()),
                "error_capture_rate": float((errors & rejected).sum() / errors.sum())
                if errors.any()
                else 0.0,
                "correct_predictions_rejected": int((correct & rejected).sum()),
                "accepted_status": int(statuses.get("accepted", 0)),
                "ambiguous_status": int(statuses.get("ambiguous", 0)),
                "out_of_distribution_status": int(
                    statuses.get("out_of_distribution", 0)
                ),
                "single_severe_shift_rejection_rate": float(severe_rate),
                "multifeature_shift_rejection_rate": float(multivariate_rate),
                "stress_test_note": "deterministic_feature_range_stress_not_external_validation",
            }
        )
    return pd.DataFrame(rows).sort_values("lab_id").reset_index(drop=True)


def build_all_outputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    bundles = [_load_bundle(lab.lab_id) for lab in list_labs()]
    seed_detail, seed_summary = build_seed_robustness(bundles)
    tables = {
        "eda_overview.csv": build_eda_overview(bundles),
        "class_balance.csv": build_class_balance(bundles),
        "candidate_model_comparison.csv": build_candidate_comparison(bundles),
        "model_selection_summary.csv": build_model_selection_summary(bundles),
        "per_class_metrics.csv": build_per_class_metrics(bundles),
        "feature_importance.csv": build_feature_importance(bundles),
        "uncertainty_summary.csv": build_uncertainty_summary(bundles),
        "seed_robustness.csv": seed_detail,
        "seed_robustness_summary.csv": seed_summary,
        "reliability_summary.csv": build_reliability_summary(bundles),
    }
    matrices = build_confusion_matrices(bundles)
    return tables, matrices


def _format_decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(tables: dict[str, pd.DataFrame]) -> str:
    eda = tables["eda_overview.csv"]
    selection = tables["model_selection_summary.csv"]
    per_class = tables["per_class_metrics.csv"]
    uncertainty = tables["uncertainty_summary.csv"]
    seed = tables["seed_robustness_summary.csv"]
    reliability = tables["reliability_summary.csv"]
    importance = tables["feature_importance.csv"]

    eda_rows = []
    for row in eda.itertuples(index=False):
        eda_rows.append(
            [
                row.lab_title,
                str(row.feature_rows),
                str(row.generation_groups),
                str(row.model_features),
                f"{row.min_class_samples} / {row.max_class_samples}",
                str(row.missing_model_values),
            ]
        )

    selection_rows = []
    for row in selection.itertuples(index=False):
        change = _format_decimal(float(row.cv_macro_f1_change), 3)
        selection_rows.append(
            [
                row.lab_title,
                _format_decimal(float(row.dummy_validation_macro_f1)),
                row.selected_family,
                f"{_format_decimal(float(row.baseline_cv_macro_f1_mean))} ± {_format_decimal(float(row.baseline_cv_macro_f1_std))}",
                f"{_format_decimal(float(row.tuned_cv_macro_f1_mean))} ± {_format_decimal(float(row.tuned_cv_macro_f1_std))}",
                f"{row.selected_configuration} ({change})",
                _format_decimal(float(row.final_test_macro_f1)),
            ]
        )

    per_class_rows = []
    for lab_id, group in per_class.groupby("lab_id"):
        weakest = group.sort_values("f1_score").iloc[0]
        strongest = group.sort_values("f1_score", ascending=False).iloc[0]
        per_class_rows.append(
            [
                LAB_TITLES[lab_id],
                f"{weakest.class_name}: {_format_decimal(float(weakest.f1_score))}",
                f"{strongest.class_name}: {_format_decimal(float(strongest.f1_score))}",
                f"[`CSV`](evaluation/data_science/confusion_{lab_id}.csv)",
            ]
        )

    robustness_rows = []
    for row in uncertainty.merge(seed, on="lab_id").itertuples(index=False):
        robustness_rows.append(
            [
                LAB_TITLES[row.lab_id],
                f"[{_format_decimal(float(row.macro_f1_ci95_low))}; {_format_decimal(float(row.macro_f1_ci95_high))}]",
                f"{_format_decimal(float(row.macro_f1_mean))} ± {_format_decimal(float(row.macro_f1_std))}",
                f"{_format_decimal(float(row.macro_f1_min))}–{_format_decimal(float(row.macro_f1_max))}",
            ]
        )

    reliability_rows = []
    for row in reliability.itertuples(index=False):
        reliability_rows.append(
            [
                LAB_TITLES[row.lab_id],
                _format_decimal(float(row.coverage)),
                _format_decimal(float(row.selective_accuracy)),
                _format_decimal(float(row.selective_macro_f1)),
                f"{row.errors_captured_by_abstention}/{row.total_errors}",
                _format_decimal(float(row.single_severe_shift_rejection_rate)),
                _format_decimal(float(row.multifeature_shift_rejection_rate)),
            ]
        )

    importance_rows = []
    for lab_id, group in importance.groupby("lab_id"):
        top = group.head(3)
        features = ", ".join(
            f"`{row.feature}` ({_format_decimal(float(row.importance))})"
            for row in top.itertuples(index=False)
        )
        importance_rows.append([LAB_TITLES[lab_id], features])

    return f"""# Data Science PhysLab AI

> **Цель модели — не угадать ярлык любой ценой, а отделить воспроизводимую диагностическую гипотезу от случая, который нужно вернуть учащемуся и преподавателю на проверку.**

Этот отчёт собирает доказательства Data Science-контура из сохранённых данных, моделей и манифестов разбиения. Все таблицы воспроизводятся командой `python build_data_science_report.py`; актуальность проверяется в `quality_check.py`.

## 1. Исследовательская постановка

Каждый модуль решает многоклассовую задачу классификации полной экспериментальной серии. Класс описывает не личность учащегося, а наблюдаемый сценарий проведения опыта: нормальный ход, единичный выброс, дрейф датчика, утечка, изменение температуры, ошибка массы и другие предметные причины.

Главный риск такой постановки — получить высокую метрику за счёт утечки между почти одинаковыми вариантами одного опыта. Поэтому единицей разбиения служит не отдельная строка, а **экспериментальное семейство**: общий набор условий, профиль оборудования и базовый план измерений, из которого построены четыре контрфактических класса.

## 2. Разведочный анализ и качество данных

{_markdown_table(
        ["Модуль", "Серий", "Семейств", "Признаков", "Мин./макс. в классе", "Пропусков"],
        eda_rows,
    )}

Что установлено до обучения:

- во всех модулях по 640 серий и 160 независимых семейств;
- каждый класс представлен 160 сериями, а в каждой части разбиения — одинаковым числом примеров;
- в модельных признаках нет пропусков;
- одна строка `features.csv` соответствует одному эксперименту;
- число измерений внутри серии различается и сохраняется в исходном `dataset.csv`;
- признаки имеют физический смысл: отклонение от закона, форма остатка, устойчивость параметра, локальный шум, монотонность, лаг и баланс энергии.

Полные числовые результаты: [`eda_overview.csv`](evaluation/data_science/eda_overview.csv) и [`class_balance.csv`](evaluation/data_science/class_balance.csv).

## 3. Разбиение без утечки

Для каждого модуля сохранён один групповой протокол:

| Часть | Серий | Семейств | Назначение |
|---|---:|---:|---|
| `train` | 320 | 80 | обучение кандидатов |
| `validation` | 160 | 40 | выбор семейства модели |
| `test` | 160 | 40 | однократная финальная оценка |

`generation_group` не пересекается между частями. Grid Search выполняется только на объединении `train + validation` с `StratifiedGroupKFold(n_splits=4)`. Тестовая часть не участвует ни в выборе семейства, ни в выборе гиперпараметров.

## 4. Baseline, выбор модели и настройка

{_markdown_table(
        ["Модуль", "Dummy", "Семейство", "Baseline CV", "Tuned CV", "Выбор и Δ", "Test Macro F1"],
        selection_rows,
    )}

Логика выбора прозрачна:

1. `DummyClassifier` задаёт нижнюю границу Macro F1 = 0,100.
2. На `validation` сравниваются Logistic Regression, Random Forest и Gradient Boosting.
3. Победившее семейство проходит компактный Grid Search на групповой CV.
4. Настроенная конфигурация принимается только при улучшении CV; иначе сохраняется baseline.
5. Выбранная конфигурация переобучается на `train + validation` и один раз оценивается на `test`.

Так, настройка улучшила CV для Бойля — Мариотта и теплового баланса, но не дала выигрыша для охлаждения и изохорного процесса. Это зафиксировано как результат эксперимента, а не скрыто за финальной метрикой.

Полные кандидаты и параметры: [`candidate_model_comparison.csv`](evaluation/data_science/candidate_model_comparison.csv) и [`model_selection_summary.csv`](evaluation/data_science/model_selection_summary.csv).

## 5. Ошибки по классам

{_markdown_table(
        ["Модуль", "Наиболее сложный класс", "Наиболее устойчивый класс", "Матрица ошибок"],
        per_class_rows,
    )}

Агрегированная Macro F1 не скрывает слабые места. Для охлаждения сложнее всего разделить нормальную серию и постепенный дрейф; для теплового баланса — нормальную серию, теплопотери и ошибку массы. Именно эти пары должны стать приоритетом при сборе размеченных реальных данных.

Полный отчёт Precision/Recall/F1: [`per_class_metrics.csv`](evaluation/data_science/per_class_metrics.csv).

## 6. Какие признаки использует модель

{_markdown_table(["Модуль", "Три ведущих признака"], importance_rows)}

Важность извлечена из уже установленной древесной модели и используется только для интерпретации, а не для выбора конфигурации. Это относительный показатель вклада признаков внутри конкретной модели; он не доказывает причинную связь и может распределяться между коррелирующими признаками.

Полный список: [`feature_importance.csv`](evaluation/data_science/feature_importance.csv).

## 7. Неопределённость результата

{_markdown_table(
        ["Модуль", "Групповой bootstrap 95% CI", "5 seed: mean ± std", "Диапазон seed"],
        robustness_rows,
    )}

Доверительные интервалы получены групповым bootstrap по 40 тестовым семействам, поэтому четыре связанных класса одного семейства всегда ресемплируются вместе. Проверка пяти `random_state` переобучает уже выбранную конфигурацию на том же зафиксированном `train + validation`; она оценивает чувствительность к случайности алгоритма, но не заменяет внешнюю валидацию на новом оборудовании.

Артефакты: [`uncertainty_summary.csv`](evaluation/data_science/uncertainty_summary.csv), [`seed_robustness.csv`](evaluation/data_science/seed_robustness.csv).

## 8. Контур отказа `unknown`

{_markdown_table(
        ["Модуль", "Coverage", "Selective accuracy", "Selective Macro F1", "Ошибок перехвачено", "Severe shift reject", "Multi-shift reject"],
        reliability_rows,
    )}

`Coverage` — доля тестовых серий, для которых система разрешила итоговую классификацию. `Selective`-метрики рассчитаны только по принятым прогнозам. Отказ срабатывает при низкой уверенности, малом разрыве между двумя классами или выходе признаков за профиль `train + validation`.

Два стресс-теста проверяют сам механизм защиты:

- один признак выводится далеко за обучающий диапазон;
- число умеренно сдвинутых признаков превышает допустимую долю.

Это **детерминированные тесты диапазона**, а не доказательство качества на реальном OOD-корпусе. Их задача — подтвердить, что система не принуждает модель к известному классу при явно некорректном входе.

Полный отчёт: [`reliability_summary.csv`](evaluation/data_science/reliability_summary.csv).

## 9. Что результат доказывает — и чего не доказывает

Подтверждено:

- модели распознают четыре сценария генератора на новых экспериментальных семействах;
- выбор конфигурации отделён от финального теста;
- класс-баланс не создаёт преимуществ отдельной категории;
- метрики устойчивы к случайности обучения в проверенном диапазоне seed;
- механизм `unknown` повышает точность среди принятых прогнозов в трёх из четырёх модулей и отклоняет искусственно выведенные за диапазон входы.

Не подтверждено:

- качество на реальных учащихся, других датчиках и иных методиках опыта;
- оптимальность порогов `unknown` для школьного пилота;
- педагогический эффект рекомендаций;
- переносимость на ошибки, отсутствующие в генераторе.

Следующий исследовательский этап — размеченный полевой корпус, предварительно зарегистрированный протокол пилота, калибровка порогов по паре `coverage / selective risk` и анализ ошибок вместе с преподавателями физики.
"""


def _assert_frames_equal(expected: pd.DataFrame, actual: pd.DataFrame, source: str) -> None:
    if list(expected.columns) != list(actual.columns):
        raise SystemExit(f"{source}: columns are stale")
    if len(expected) != len(actual):
        raise SystemExit(f"{source}: row count is stale")
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left):
            left_values = left.astype(float).to_numpy()
            right_values = right.astype(float).to_numpy()
            if not np.allclose(
                left_values,
                right_values,
                rtol=0.0,
                atol=FLOAT_TOLERANCE,
                equal_nan=True,
            ):
                raise SystemExit(f"{source}: numeric column {column!r} is stale")
        elif not left.astype(str).equals(right.astype(str)):
            raise SystemExit(f"{source}: column {column!r} is stale")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT / "build_data_science_report.py",
        PROJECT_ROOT / "evaluation" / "final_model_summary.csv",
    ]
    for lab in list_labs():
        lab_id = lab.lab_id
        paths.extend(
            [
                PROJECT_ROOT / "data" / lab_id / "dataset.csv",
                PROJECT_ROOT / "data" / lab_id / "features.csv",
                PROJECT_ROOT / "models" / lab_id / "best_model.joblib",
                PROJECT_ROOT / "models" / lab_id / "feature_names.joblib",
                PROJECT_ROOT / "models" / lab_id / "inference_profile.json",
                PROJECT_ROOT / "evaluation" / lab_id / "split_manifest.csv",
                PROJECT_ROOT / "evaluation" / lab_id / "model_metrics.csv",
                PROJECT_ROOT / "evaluation" / "grid_search" / lab_id / "grid_search_summary.csv",
            ]
        )
    return paths


def build_manifest() -> dict[str, object]:
    return {
        "version": 2,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in _source_paths()
        },
    }


def check_committed_outputs() -> None:
    required = [
        "eda_overview.csv",
        "class_balance.csv",
        "candidate_model_comparison.csv",
        "model_selection_summary.csv",
        "per_class_metrics.csv",
        "feature_importance.csv",
        "uncertainty_summary.csv",
        "seed_robustness.csv",
        "seed_robustness_summary.csv",
        "reliability_summary.csv",
    ]
    required.extend(f"confusion_{lab.lab_id}.csv" for lab in list_labs())
    missing = [name for name in required if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise SystemExit("Missing Data Science artifacts: " + ", ".join(missing))
    if not MANIFEST_PATH.is_file():
        raise SystemExit("Missing evaluation/data_science/report_manifest.json")
    stored = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current = build_manifest()
    if stored != current:
        raise SystemExit(
            "Data Science evidence sources changed. Run: python build_data_science_report.py"
        )
    print("Data Science evidence is current")


def write_outputs(
    tables: dict[str, pd.DataFrame],
    matrices: dict[str, pd.DataFrame],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)
    for lab_id, frame in matrices.items():
        frame.to_csv(OUTPUT_DIR / f"confusion_{lab_id}.csv", index=False)
    MANIFEST_PATH.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def check_outputs(
    tables: dict[str, pd.DataFrame],
    matrices: dict[str, pd.DataFrame],
) -> None:
    for filename, expected in tables.items():
        path = OUTPUT_DIR / filename
        if not path.is_file():
            raise SystemExit(f"Missing {path.relative_to(PROJECT_ROOT)}")
        _assert_frames_equal(
            expected,
            pd.read_csv(path),
            str(path.relative_to(PROJECT_ROOT)),
        )
    for lab_id, expected in matrices.items():
        path = OUTPUT_DIR / f"confusion_{lab_id}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path.relative_to(PROJECT_ROOT)}")
        _assert_frames_equal(
            expected,
            pd.read_csv(path),
            str(path.relative_to(PROJECT_ROOT)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible Data Science evidence artifacts."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that committed Data Science artifacts are current",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check_committed_outputs()
        return
    tables, matrices = build_all_outputs()
    write_outputs(tables, matrices)
    print(f"Saved {len(tables) + len(matrices)} Data Science evidence tables")


if __name__ == "__main__":
    main()
