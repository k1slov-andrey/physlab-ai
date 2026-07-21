from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import prepare_real_data
from inspect_real_data import detect_concepts


def _cooling_frame(rows: int = 30) -> pd.DataFrame:
    time = np.arange(rows, dtype=float) * 10.0
    return pd.DataFrame(
        {
            "Time": time,
            "Flowrate": np.linspace(590.0, 610.0, rows),
            "Pressure": np.linspace(101.0, 103.0, rows),
            "T_out": np.linspace(55.0, 31.0, rows),
            "T1": np.linspace(54.0, 32.0, rows),
            "T_ambient": np.linspace(22.0, 23.0, rows),
        }
    )


def test_temperature_selector_rejects_other_physical_channels() -> None:
    frame = _cooling_frame()
    decisions = prepare_real_data.temperature_column_decisions(frame, "Time")
    accepted = {decision.column_name for decision in decisions if decision.accepted}
    rejected = {decision.column_name for decision in decisions if not decision.accepted}

    assert accepted == {"T_out", "T1", "T_ambient"}
    assert {"Time", "Flowrate", "Pressure"}.issubset(rejected)


def test_temperature_unit_conversion_is_explicit() -> None:
    frame = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "temperature_k": [293.15, 294.15, 295.15, 296.15, 297.15],
        }
    )
    decision = prepare_real_data.classify_temperature_column(
        frame,
        "temperature_k",
        "time",
    )
    converted = prepare_real_data.convert_temperature_to_celsius(
        frame["temperature_k"],
        decision.inferred_unit,
    )

    assert decision.accepted
    assert decision.inferred_unit == "kelvin"
    assert np.allclose(converted.to_numpy(), [20.0, 21.0, 22.0, 23.0, 24.0])


def test_normalize_cooling_uses_only_temperature_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "real_data_raw"
    cooling_dir = raw_dir / "cooling"
    cooling_dir.mkdir(parents=True)
    source = cooling_dir / "cooling_test.xlsx"
    _cooling_frame().to_excel(source, index=False, sheet_name="Data")

    monkeypatch.setattr(prepare_real_data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prepare_real_data, "RAW_DIR", raw_dir)
    report: list[dict[str, object]] = []
    normalized = prepare_real_data.normalize_cooling(report)

    assert set(normalized["sensor_id"].unique()) == {"t_out", "t1", "t_ambient"}
    assert normalized["experiment_id"].nunique() == 3
    assert not normalized["sensor_id"].isin({"flowrate", "pressure"}).any()

    accepted = {
        row["column_name"]
        for row in report
        if bool(row["accepted"])
    }
    assert accepted == {"t_out", "t1", "t_ambient"}


def test_audit_recognizes_compact_temperature_sensor_names() -> None:
    concepts = detect_concepts(
        ["Time", "Flowrate", "Pressure", "T_out", "T1", "T_ambient"]
    )

    assert "time" in concepts
    assert "pressure" in concepts
    assert "temperature" in concepts


def test_temperature_name_does_not_create_false_time_concept() -> None:
    concepts = detect_concepts(["Temperature"])

    assert concepts == {"temperature"}
