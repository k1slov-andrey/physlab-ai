import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_has_valid_python_syntax() -> None:
    app_path = ROOT / "app.py"
    source = app_path.read_text(encoding="utf-8")
    ast.parse(source)
