import importlib

import pytest


MODULES = [
    "core.schemas",
    "core.lab_registry",
    "core.recommendation_engine",
    "core.competency_engine",
    "core.upload_pipeline",
    "labs.cooling.module",
    "labs.boyle_mariotte.module",
    "labs.isochoric.module",
    "labs.heat_balance.module",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_project_module_imports(module_name: str) -> None:
    imported = importlib.import_module(module_name)
    assert imported is not None
