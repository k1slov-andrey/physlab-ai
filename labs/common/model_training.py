from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelResult:
    name: str
    model: BaseEstimator
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    training_time_seconds: float


def build_candidate_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=220,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def score_model(
    model: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "macro_f1": float(
            f1_score(target, predictions, average="macro", zero_division=0)
        ),
    }


def evaluate_models(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    random_state: int = 42,
) -> tuple[list[ModelResult], ModelResult]:
    results: list[ModelResult] = []

    for name, base_model in build_candidate_models(random_state).items():
        model = clone(base_model)
        started_at = perf_counter()
        model.fit(x_train, y_train)
        training_time = perf_counter() - started_at
        scores = score_model(model, x_validation, y_validation)
        results.append(
            ModelResult(
                name=name,
                model=model,
                accuracy=scores["accuracy"],
                balanced_accuracy=scores["balanced_accuracy"],
                macro_f1=scores["macro_f1"],
                training_time_seconds=float(training_time),
            )
        )

    best = max(
        results,
        key=lambda result: (
            result.macro_f1,
            result.balanced_accuracy,
            result.accuracy,
        ),
    )
    return results, best


def fit_selected_model(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> BaseEstimator:
    candidates = build_candidate_models(random_state)
    if model_name not in candidates:
        raise KeyError(f"Unknown model family: {model_name}")
    model = clone(candidates[model_name])
    model.fit(features, target)
    return model


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    rows = [
        {
            "model": result.name,
            "validation_accuracy": result.accuracy,
            "validation_balanced_accuracy": result.balanced_accuracy,
            "validation_macro_f1": result.macro_f1,
            "training_time_seconds": result.training_time_seconds,
        }
        for result in results
    ]
    return pd.DataFrame(rows).sort_values(
        [
            "validation_macro_f1",
            "validation_balanced_accuracy",
            "validation_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)
