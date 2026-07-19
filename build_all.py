from __future__ import annotations

import pandas as pd

from labs.common.artifacts import get_lab_data_dir
from labs.common.pipeline import train_and_save
from labs.cooling.module import generate_dataset as generate_cooling
from labs.boyle_mariotte.module import generate_dataset as generate_boyle
from labs.isochoric.module import generate_dataset as generate_isochoric
from labs.heat_balance.module import generate_dataset as generate_heat_balance


LABS = {
    "cooling": generate_cooling,
    "boyle_mariotte": generate_boyle,
    "isochoric": generate_isochoric,
    "heat_balance": generate_heat_balance,
}


def main(n_per_class: int = 120) -> None:
    summaries: list[dict] = []

    for lab_id, generator in LABS.items():
        print(f"\n=== {lab_id} ===")
        raw, features = generator(
            n_per_class=n_per_class,
            seed=42,
        )
        data_dir = get_lab_data_dir(lab_id)
        raw.to_csv(data_dir / "dataset.csv", index=False)

        summary = train_and_save(lab_id, features)
        summaries.append(summary)
        print(summary)

        for class_name in sorted(raw["class_name"].unique()):
            experiment_id = raw.loc[
                raw["class_name"] == class_name,
                "experiment_id",
            ].iloc[0]
            raw.loc[raw["experiment_id"] == experiment_id].to_csv(
                data_dir / f"demo_{class_name}.csv",
                index=False,
            )

    pd.DataFrame(summaries).to_csv(
        "evaluation/all_labs_summary.csv",
        index=False,
    )
    print("\nВсе четыре модели обучены и сохранены.")


if __name__ == "__main__":
    main()
