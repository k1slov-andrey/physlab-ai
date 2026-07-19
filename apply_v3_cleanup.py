from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

FILES_TO_REMOVE = [
    ROOT / "data" / "isochoric" / "demo_wrong_temperature_scale.csv",
    ROOT / "models" / "best_model.joblib",
    ROOT / "models" / "feature_names.joblib",
]

for path in FILES_TO_REMOVE:
    if path.exists():
        path.unlink()
        print(f"Removed obsolete file: {path.relative_to(ROOT)}")


for lab_id in ("cooling", "boyle_mariotte", "isochoric", "heat_balance"):
    dataset_path = ROOT / "data" / lab_id / "dataset.csv"
    if dataset_path.exists():
        dataset_path.unlink()
        print(f"Removed outdated generated dataset: {dataset_path.relative_to(ROOT)}")

for cache in ROOT.rglob("__pycache__"):
    shutil.rmtree(cache, ignore_errors=True)

print("PhysLab AI v3 cleanup completed.")
