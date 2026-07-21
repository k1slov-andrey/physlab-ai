from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray
    strategy: str

    def validate(
        self,
        frame: pd.DataFrame,
        target_column: str,
        group_column: str | None,
    ) -> None:
        partitions = {
            "train": set(self.train_index.tolist()),
            "validation": set(self.validation_index.tolist()),
            "test": set(self.test_index.tolist()),
        }

        if any(not indices for indices in partitions.values()):
            raise ValueError("Train, validation and test partitions must be non-empty")

        if partitions["train"] & partitions["validation"]:
            raise ValueError("Train and validation partitions overlap")
        if partitions["train"] & partitions["test"]:
            raise ValueError("Train and test partitions overlap")
        if partitions["validation"] & partitions["test"]:
            raise ValueError("Validation and test partitions overlap")

        assigned = set().union(*partitions.values())
        expected = set(range(len(frame)))
        if assigned != expected:
            raise ValueError("Dataset split does not cover every row exactly once")

        expected_classes = set(frame[target_column].unique())
        for role, indices in partitions.items():
            actual_classes = set(frame.iloc[sorted(indices)][target_column].unique())
            if actual_classes != expected_classes:
                missing = sorted(expected_classes - actual_classes)
                raise ValueError(f"Partition '{role}' is missing classes: {missing}")

        if group_column is None or group_column not in frame.columns:
            return

        group_sets = {
            role: set(frame.iloc[sorted(indices)][group_column].unique())
            for role, indices in partitions.items()
        }
        if group_sets["train"] & group_sets["validation"]:
            raise ValueError("Train and validation groups overlap")
        if group_sets["train"] & group_sets["test"]:
            raise ValueError("Train and test groups overlap")
        if group_sets["validation"] & group_sets["test"]:
            raise ValueError("Validation and test groups overlap")


def _distribution_distance(reference: pd.Series, candidate: pd.Series) -> float:
    labels = sorted(reference.unique())
    reference_share = reference.value_counts(normalize=True).reindex(labels, fill_value=0.0)
    candidate_share = candidate.value_counts(normalize=True).reindex(labels, fill_value=0.0)
    return float(np.abs(reference_share - candidate_share).sum())


def _best_group_fold(
    frame: pd.DataFrame,
    indices: np.ndarray,
    target_column: str,
    group_column: str,
    n_splits: int,
    random_state: int,
    target_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    subset = frame.iloc[indices]
    y = subset[target_column]
    groups = subset[group_column]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    placeholder = np.zeros((len(subset), 1), dtype=np.float32)

    for development_local, held_out_local in splitter.split(placeholder, y, groups):
        held_out_fraction = len(held_out_local) / len(subset)
        size_penalty = abs(held_out_fraction - target_fraction)
        distribution_penalty = _distribution_distance(y, y.iloc[held_out_local])
        score = size_penalty + distribution_penalty

        development_index = indices[development_local]
        held_out_index = indices[held_out_local]
        if best is None or score < best[0]:
            best = (score, development_index, held_out_index)

    if best is None:
        raise RuntimeError("Unable to construct a group-aware split")
    return best[1], best[2]


def split_train_validation_test(
    frame: pd.DataFrame,
    target_column: str = "class_name",
    group_column: str = "generation_group",
    random_state: int = 42,
    test_fraction: float = 0.25,
    validation_fraction: float = 0.25,
) -> DatasetSplit:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("test_fraction + validation_fraction must be below 1")

    all_indices = np.arange(len(frame))
    use_groups = (
        group_column in frame.columns
        and frame[group_column].notna().all()
        and frame[group_column].nunique() >= 6
    )

    if use_groups:
        outer_splits = max(2, round(1.0 / test_fraction))
        development_index, test_index = _best_group_fold(
            frame=frame,
            indices=all_indices,
            target_column=target_column,
            group_column=group_column,
            n_splits=outer_splits,
            random_state=random_state,
            target_fraction=test_fraction,
        )

        validation_share_of_development = validation_fraction / (1.0 - test_fraction)
        development_groups = frame.iloc[development_index][group_column].nunique()
        inner_splits = min(
            development_groups,
            max(2, round(1.0 / validation_share_of_development)),
        )
        train_index, validation_index = _best_group_fold(
            frame=frame,
            indices=development_index,
            target_column=target_column,
            group_column=group_column,
            n_splits=inner_splits,
            random_state=random_state + 17,
            target_fraction=validation_share_of_development,
        )
        strategy = "stratified_group_train_validation_test"
        effective_group_column: str | None = group_column
    else:
        development_index, test_index = train_test_split(
            all_indices,
            test_size=test_fraction,
            stratify=frame[target_column],
            random_state=random_state,
        )
        validation_share_of_development = validation_fraction / (1.0 - test_fraction)
        train_index, validation_index = train_test_split(
            development_index,
            test_size=validation_share_of_development,
            stratify=frame.iloc[development_index][target_column],
            random_state=random_state + 17,
        )
        strategy = "stratified_random_train_validation_test"
        effective_group_column = None

    split = DatasetSplit(
        train_index=np.sort(train_index),
        validation_index=np.sort(validation_index),
        test_index=np.sort(test_index),
        strategy=strategy,
    )
    split.validate(frame, target_column, effective_group_column)
    return split


def build_split_manifest(
    frame: pd.DataFrame,
    split: DatasetSplit,
    target_column: str = "class_name",
    group_column: str = "generation_group",
) -> pd.DataFrame:
    role_by_index: dict[int, str] = {}
    for index in split.train_index:
        role_by_index[int(index)] = "train"
    for index in split.validation_index:
        role_by_index[int(index)] = "validation"
    for index in split.test_index:
        role_by_index[int(index)] = "test"

    manifest = pd.DataFrame(
        {
            "row_index": np.arange(len(frame)),
            "dataset_role": [role_by_index[index] for index in range(len(frame))],
            "target": frame[target_column].astype(str).to_numpy(),
        }
    )
    if group_column in frame.columns:
        manifest["generation_group"] = frame[group_column].astype(str).to_numpy()
    if "experiment_id" in frame.columns:
        manifest["experiment_id"] = frame["experiment_id"].astype(str).to_numpy()
    return manifest
