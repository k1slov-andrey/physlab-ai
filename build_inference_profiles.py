from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from labs.common.reliability import (
    PROFILE_FILENAME,
    build_feature_profile,
    save_feature_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parent
LAB_IDS = ("cooling", "boyle_mariotte", "isochoric", "heat_balance")
DEVELOPMENT_ROLES = {"train", "validation"}


def build_profile_for_lab(lab_id: str) -> Path:
    features_path = PROJECT_ROOT / "data" / lab_id / "features.csv"
    manifest_path = PROJECT_ROOT / "evaluation" / lab_id / "split_manifest.csv"
    feature_names_path = PROJECT_ROOT / "models" / lab_id / "feature_names.joblib"

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (features_path, manifest_path, feature_names_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot build the inference profile for '{lab_id}'. Missing: "
            + ", ".join(missing)
        )

    frame = pd.read_csv(features_path)
    manifest = pd.read_csv(manifest_path)
    feature_names = [str(name) for name in joblib.load(feature_names_path)]

    if len(frame) != len(manifest):
        raise ValueError(
            f"Row count mismatch for '{lab_id}': "
            f"features={len(frame)}, manifest={len(manifest)}"
        )
    if sorted(manifest["row_index"].astype(int).tolist()) != list(range(len(frame))):
        raise ValueError(f"Invalid split manifest for '{lab_id}'")

    roles = manifest.sort_values("row_index")["dataset_role"].astype(str)
    development_mask = roles.isin(DEVELOPMENT_ROLES).to_numpy()
    development = frame.loc[development_mask, feature_names]
    if development.empty:
        raise ValueError(f"Development partition is empty for '{lab_id}'")

    profile = build_feature_profile(development)
    profile["lab_id"] = lab_id
    profile["dataset_roles"] = sorted(DEVELOPMENT_ROLES)

    output_path = PROJECT_ROOT / "models" / lab_id / PROFILE_FILENAME
    save_feature_profile(profile, output_path)
    return output_path


def main() -> None:
    for lab_id in LAB_IDS:
        output_path = build_profile_for_lab(lab_id)
        relative = output_path.relative_to(PROJECT_ROOT)
        print(f"{lab_id}: {relative}")


if __name__ == "__main__":
    main()
