from __future__ import annotations

from verify_project import check_control_scenarios, check_lab_artifacts, check_root_files


def test_required_root_files_are_present() -> None:
    assert check_root_files().passed


def test_all_lab_artifacts_are_consistent() -> None:
    for lab_id in ("cooling", "boyle_mariotte", "isochoric", "heat_balance"):
        failures = [result for result in check_lab_artifacts(lab_id) if not result.passed]
        assert failures == []


def test_control_scenarios_pass_project_verification() -> None:
    assert check_control_scenarios().passed
