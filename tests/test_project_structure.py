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
        ROOT / "requirements-dev.txt",
        ROOT / "README.md",
        ROOT / "PRODUCT.md",
        ROOT / "TECHNICAL_APPENDIX.md",
        ROOT / "Dockerfile",
        ROOT / ".dockerignore",
        ROOT / "environment_check.py",
        ROOT / "evaluate_deployed_models.py",
        ROOT / "build_data_science_report.py",
        ROOT / "quality_check.py",
        ROOT / "verify_project.py",
    ]

    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"Отсутствуют обязательные файлы: {missing}"
