from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

def ensure_project_directories():
    for p in (DATA_DIR, MODELS_DIR, EVALUATION_DIR):
        p.mkdir(parents=True, exist_ok=True)

def get_lab_data_dir(lab_id):
    p = DATA_DIR / lab_id; p.mkdir(parents=True, exist_ok=True); return p

def get_lab_model_dir(lab_id):
    p = MODELS_DIR / lab_id; p.mkdir(parents=True, exist_ok=True); return p

def get_lab_evaluation_dir(lab_id):
    p = EVALUATION_DIR / lab_id; p.mkdir(parents=True, exist_ok=True); return p
