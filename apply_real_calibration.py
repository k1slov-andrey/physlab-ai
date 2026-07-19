from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REALISM_PATH = PROJECT_ROOT / "labs" / "common" / "realism.py"
CALIBRATION_MODULE_PATH = PROJECT_ROOT / "labs" / "common" / "empirical_calibration.py"
PROFILE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "real_data_integration"
    / "empirical_realism_profile.json"
)
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
BACKUP_ROOT = PROJECT_ROOT / "backups" / "pre_real_calibration"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "real_calibration"
LAB_IDS = ("cooling", "boyle_mariotte", "isochoric", "heat_balance")

CALIBRATION_MODULE = '''from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "real_data_integration"
    / "empirical_realism_profile.json"
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def load_empirical_profile() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def temperature_calibration() -> dict[str, float] | None:
    profile = load_empirical_profile()
    sensor = profile.get("temperature_sensor", {})

    resolution = sensor.get("temperature_resolution_c_median")
    noise_median = sensor.get("temperature_noise_sigma_c_median")
    noise_q90 = sensor.get("temperature_noise_sigma_c_q90")

    numeric_values = (resolution, noise_median, noise_q90)
    if any(value is None for value in numeric_values):
        return None

    try:
        return {
            "resolution_c": _clamp(float(resolution), 0.05, 0.5),
            "noise_median_c": _clamp(float(noise_median), 0.02, 0.6),
            "noise_q90_c": _clamp(float(noise_q90), 0.03, 0.9),
        }
    except (TypeError, ValueError):
        return None


def calibrate_device_profile(profile):
    calibration = temperature_calibration()
    if calibration is None:
        return profile

    profile_multiplier = {
        "school_digital": 0.75,
        "school_basic": 1.00,
        "low_cost_sensor": 1.35,
    }.get(profile.name, 1.0)

    empirical_noise = calibration["noise_median_c"] * profile_multiplier
    if profile.name == "low_cost_sensor":
        empirical_noise = max(
            empirical_noise,
            calibration["noise_q90_c"],
        )

    calibrated_noise = 0.45 * profile.temperature_noise_c + 0.55 * empirical_noise
    calibrated_resolution = max(
        profile.temperature_resolution_c,
        calibration["resolution_c"],
    )

    return replace(
        profile,
        temperature_noise_c=_clamp(calibrated_noise, 0.03, 0.8),
        temperature_resolution_c=_clamp(calibrated_resolution, 0.05, 0.5),
    )
'''


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Не найден обязательный файл: {path}")


def backup_project_state() -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = BACKUP_ROOT / stamp
    destination.mkdir(parents=True, exist_ok=False)

    shutil.copy2(REALISM_PATH, destination / "realism.py")
    if REQUIREMENTS_PATH.exists():
        shutil.copy2(REQUIREMENTS_PATH, destination / "requirements.txt")

    for folder_name in ("models", "evaluation"):
        source = PROJECT_ROOT / folder_name
        if source.exists():
            shutil.copytree(source, destination / folder_name)

    return destination


def patch_realism() -> bool:
    text = REALISM_PATH.read_text(encoding="utf-8")
    changed = False

    import_line = "from labs.common.empirical_calibration import calibrate_device_profile\n"
    if import_line not in text:
        marker = "import numpy as np\n"
        if marker not in text:
            raise RuntimeError("Не найдена точка вставки импорта в realism.py")
        text = text.replace(marker, marker + "\n" + import_line, 1)
        changed = True

    old_return = "    return DEVICE_PROFILES[index]\n"
    new_return = "    return calibrate_device_profile(DEVICE_PROFILES[index])\n"
    if new_return not in text:
        if old_return not in text:
            raise RuntimeError("Не найдена функция choose_device_profile в realism.py")
        text = text.replace(old_return, new_return, 1)
        changed = True

    if changed:
        REALISM_PATH.write_text(text, encoding="utf-8")
    return changed


def update_requirements() -> None:
    lines = []
    if REQUIREMENTS_PATH.exists():
        lines = REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()

    result: list[str] = []
    seen_sklearn = False
    seen_openpyxl = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("scikit-learn=="):
            result.append("scikit-learn==1.8.0")
            seen_sklearn = True
        elif lower.startswith("openpyxl=="):
            result.append("openpyxl==3.1.5")
            seen_openpyxl = True
        else:
            result.append(line)

    if not seen_sklearn:
        result.append("scikit-learn==1.8.0")
    if not seen_openpyxl:
        result.append("openpyxl==3.1.5")

    REQUIREMENTS_PATH.write_text("\n".join(result).rstrip() + "\n", encoding="utf-8")


def read_lab_summaries() -> list[dict]:
    rows: list[dict] = []
    import pandas as pd

    for lab_id in LAB_IDS:
        path = PROJECT_ROOT / "evaluation" / lab_id / "summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["lab_id"] = lab_id
        rows.append(row)
    return rows


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def rebuild_models(n_per_class: int) -> None:
    code = (
        "import build_all; "
        f"build_all.main(n_per_class={int(n_per_class)})"
    )
    run_command([sys.executable, "-c", code])
    run_command([sys.executable, "prepare_real_data.py"])


def write_report(before: list[dict], after: list[dict], backup_dir: Path) -> None:
    import pandas as pd

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    before_df = pd.DataFrame(before)
    after_df = pd.DataFrame(after)

    before_df.to_csv(REPORT_DIR / "metrics_before.csv", index=False)
    after_df.to_csv(REPORT_DIR / "metrics_after.csv", index=False)

    comparison_rows: list[dict] = []
    before_map = {str(row.get("lab_id")): row for row in before}
    after_map = {str(row.get("lab_id")): row for row in after}

    for lab_id in LAB_IDS:
        old = before_map.get(lab_id, {})
        new = after_map.get(lab_id, {})
        old_f1 = float(old.get("macro_f1", float("nan")))
        new_f1 = float(new.get("macro_f1", float("nan")))
        old_acc = float(old.get("accuracy", float("nan")))
        new_acc = float(new.get("accuracy", float("nan")))
        comparison_rows.append(
            {
                "lab_id": lab_id,
                "accuracy_before": old_acc,
                "accuracy_after": new_acc,
                "accuracy_change": new_acc - old_acc,
                "macro_f1_before": old_f1,
                "macro_f1_after": new_f1,
                "macro_f1_change": new_f1 - old_f1,
                "validation_strategy": new.get("validation_strategy", ""),
                "n_samples_after": new.get("n_samples", ""),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(REPORT_DIR / "metrics_comparison.csv", index=False)

    prediction_path = (
        PROJECT_ROOT
        / "evaluation"
        / "real_data_integration"
        / "real_data_model_predictions.csv"
    )
    real_summary: dict[str, object] = {}
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        successful = predictions[predictions["prediction_status"] == "success"].copy()
        real_summary = {
            "real_experiments_total": int(len(predictions)),
            "real_predictions_successful": int(len(successful)),
            "mean_confidence": (
                float(successful["confidence"].mean()) if not successful.empty else None
            ),
            "low_confidence_below_0_60": (
                int((successful["confidence"] < 0.60).sum())
                if not successful.empty
                else 0
            ),
            "predicted_class_counts": (
                successful.groupby(["lab_id", "predicted_class"])
                .size()
                .rename("count")
                .reset_index()
                .to_dict(orient="records")
                if not successful.empty
                else []
            ),
        }

    calibration = {
        "calibration_version": "1.0",
        "method": "real sensor noise calibrates shared synthetic device profiles",
        "supervised_real_labels_used": False,
        "reason": (
            "Real experiments are retained as an external validation corpus because "
            "reliable ground-truth error labels are not available."
        ),
        "profile_source": str(PROFILE_PATH.relative_to(PROJECT_ROOT)),
        "backup": str(backup_dir.relative_to(PROJECT_ROOT)),
        "real_validation": real_summary,
    }
    (REPORT_DIR / "calibration_summary.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== СРАВНЕНИЕ МЕТРИК ===")
    print(comparison_df.to_string(index=False))
    print(f"\nОтчёт сохранён: {REPORT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Калибровка синтетических данных по реальному шуму и переобучение моделей."
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=160,
        help="Число синтетических экспериментов на класс (по умолчанию 160).",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Только применить патч, не переобучать модели.",
    )
    args = parser.parse_args()

    require_file(REALISM_PATH)
    require_file(PROFILE_PATH)
    require_file(PROJECT_ROOT / "build_all.py")
    require_file(PROJECT_ROOT / "prepare_real_data.py")

    backup_dir = backup_project_state()
    CALIBRATION_MODULE_PATH.write_text(CALIBRATION_MODULE, encoding="utf-8")
    changed = patch_realism()
    update_requirements()

    print("PhysLab AI — эмпирическая калибровка применена")
    print(f"Резервная копия: {backup_dir}")
    print(f"realism.py изменён: {'да' if changed else 'уже был настроен'}")

    before = read_lab_summaries()
    if args.no_rebuild:
        print("Переобучение пропущено по параметру --no-rebuild.")
        return

    rebuild_models(args.n_per_class)
    after = read_lab_summaries()
    write_report(before, after, backup_dir)

    print("\nГотово:")
    print("- реальные измерения откалибровали общий слой шума;")
    print("- четыре модели переобучены;")
    print("- реальные данные повторно использованы как внешний корпус проверки;")
    print("- исходные реальные файлы не изменены;")
    print("- реальные данные без надёжных меток не использованы как псевдоразмеченные.")


if __name__ == "__main__":
    main()
