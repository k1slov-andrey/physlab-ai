from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist() -> None:
    required = [
        ROOT / "core",
        ROOT / "labs",
        ROOT / "data",
        ROOT / "models",
        ROOT / "evaluation",
    ]

    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_dir()]
    assert not missing, f"Отсутствуют обязательные каталоги: {missing}"


def test_required_root_files_exist() -> None:
    required = [
        ROOT / "app.py",
        ROOT / "requirements.txt",
        ROOT / "README.md",
        ROOT / "PROJECT_DESCRIPTION.md",
        ROOT / "DATA_CARD.md",
        ROOT / "MODEL_CARD.md",
        ROOT / "AI_USAGE.md",
        ROOT / "EVALUATION_REPORT.md",
        ROOT / "Dockerfile",
        ROOT / ".dockerignore",
    ]

    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"Отсутствуют обязательные файлы: {missing}"
