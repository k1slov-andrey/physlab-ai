from __future__ import annotations

import pandas as pd

from labs.common.splitting import build_split_manifest, split_train_validation_test


def _balanced_group_frame() -> pd.DataFrame:
    rows = []
    classes = ("normal", "error_a", "error_b", "error_c")
    for group_index in range(8):
        for class_name in classes:
            for repetition in range(5):
                rows.append(
                    {
                        "class_name": class_name,
                        "generation_group": f"group_{group_index:02d}",
                        "experiment_id": (
                            f"{class_name}_{group_index:02d}_{repetition:02d}"
                        ),
                        "feature": float(group_index + repetition),
                    }
                )
    return pd.DataFrame(rows)


def test_group_split_is_disjoint_and_complete() -> None:
    frame = _balanced_group_frame()
    split = split_train_validation_test(frame, random_state=42)

    train_groups = set(frame.iloc[split.train_index]["generation_group"])
    validation_groups = set(frame.iloc[split.validation_index]["generation_group"])
    test_groups = set(frame.iloc[split.test_index]["generation_group"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert len(split.train_index) + len(split.validation_index) + len(split.test_index) == len(frame)


def test_each_partition_contains_every_class() -> None:
    frame = _balanced_group_frame()
    split = split_train_validation_test(frame, random_state=42)
    expected = set(frame["class_name"])

    assert set(frame.iloc[split.train_index]["class_name"]) == expected
    assert set(frame.iloc[split.validation_index]["class_name"]) == expected
    assert set(frame.iloc[split.test_index]["class_name"]) == expected


def test_split_manifest_assigns_one_role_per_row() -> None:
    frame = _balanced_group_frame()
    split = split_train_validation_test(frame, random_state=42)
    manifest = build_split_manifest(frame, split)

    assert len(manifest) == len(frame)
    assert set(manifest["dataset_role"]) == {"train", "validation", "test"}
    assert manifest["row_index"].is_unique
