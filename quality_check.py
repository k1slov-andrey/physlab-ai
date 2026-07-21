from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(name: str, command: list[str]) -> None:
    print(f"\n== {name} ==")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")


def main() -> None:
    python = sys.executable
    run_step("Check Python environment", [python, "environment_check.py"])
    run_step(
        "Compile Python sources",
        [
            python,
            "-m",
            "compileall",
            "-q",
            "app.py",
            "core",
            "labs",
            "tests",
            "environment_check.py",
        ],
    )
    run_step("Run automated tests", [python, "-m", "pytest"])
    run_step(
        "Check deployed model evaluation",
        [python, "evaluate_deployed_models.py", "--check"],
    )
    run_step("Verify project artifacts", [python, "verify_project.py"])
    print("\nQuality check passed")


if __name__ == "__main__":
    main()
