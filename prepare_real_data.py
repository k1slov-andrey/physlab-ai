from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "real_data_raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_normalized"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "real_data_integration"

LAB_OUTPUTS = {
    "boyle_mariotte": OUTPUT_DIR / "boyle_mariotte",
    "isochoric": OUTPUT_DIR / "isochoric",
    "cooling": OUTPUT_DIR / "cooling",
    "heat_balance": OUTPUT_DIR / "heat_balance",
    "sensor_noise": OUTPUT_DIR / "sensor_noise",
}

MODEL_MIN_POINTS = {
    "boyle_mariotte": 8,
    "isochoric": 10,
    "cooling": 20,
    "heat_balance": 20,
}

NUMERIC_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "boyle_mariotte": {
        "volume_ml": (1.0, 5000.0),
        "pressure_kpa": (1.0, 2000.0),
        "temperature_c": (-80.0, 250.0),
    },
    "isochoric": {
        "temperature_c": (-80.0, 500.0),
        "pressure_kpa": (1.0, 2000.0),
        "volume_ml": (1.0, 10000.0),
    },
    "cooling": {
        "time_seconds": (0.0, 10_000_000.0),
        "measured_temperature": (-100.0, 500.0),
    },
    "heat_balance": {
        "time_seconds": (0.0, 100_000.0),
        "temperature_c": (-50.0, 200.0),
        "hot_initial_c": (-50.0, 500.0),
        "cold_initial_c": (-50.0, 200.0),
    },
}


@dataclass
class QualityRecord:
    lab_id: str
    experiment_id: str
    source_file: str
    rows: int
    required_rows: int
    missing_required_columns: str
    missing_cells: int
    duplicate_rows: int
    out_of_range_cells: int
    monotonic_time: bool
    ready_for_model: bool
    status: str
    notes: str


@dataclass(frozen=True)
class ColumnDecision:
    column_name: str
    normalized_name: str
    accepted: bool
    inferred_unit: str
    numeric_values: int
    unique_values: int
    in_range_fraction: float
    dynamic_range_c: float
    reason: str


def slugify(value: Any) -> str:
    text = str(value).strip().lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", "_", text, flags=re.IGNORECASE)
    return text.strip("_") or "unknown"


def unique_names(columns: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: dict[str, int] = {}
    for column in columns:
        base = slugify(column)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "latin1")
    separators = (";", ",", "\t", "|")
    best: pd.DataFrame | None = None
    best_score = -1
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
        except Exception as error:
            last_error = error
            continue

        for separator in separators:
            try:
                frame = pd.read_csv(
                    path,
                    sep=separator,
                    encoding=encoding,
                    engine="python",
                    dtype=str,
                    keep_default_na=True,
                )
            except Exception as error:
                last_error = error
                continue

            score = frame.shape[1] * 100 + min(frame.shape[0], 100)
            if score > best_score:
                best = frame
                best_score = score

    if best is None:
        raise RuntimeError(f"CSV не прочитан: {path}. Последняя ошибка: {last_error}")

    best.columns = unique_names(best.columns)
    return best


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    temp = OUTPUT_DIR / "_temporary_sensor_file.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp.write_bytes(raw)
    try:
        return read_csv_auto(temp)
    finally:
        temp.unlink(missing_ok=True)


def best_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    best: pd.DataFrame | None = None
    best_score = -1
    for header in range(0, 12):
        try:
            frame = pd.read_excel(path, sheet_name=sheet_name, header=header)
        except Exception:
            continue
        frame.columns = unique_names(frame.columns)
        useful = sum(
            1
            for column in frame.columns
            if not column.startswith("unnamed") and column != "nan"
        )
        keyword_score = sum(
            10
            for column in frame.columns
            if any(token in column for token in ("time", "temp", "temperature", "t_", "sec", "min"))
        )
        numeric_score = sum(
            1
            for column in frame.columns
            if numeric(frame[column]).notna().sum() >= max(3, len(frame) // 4)
        )
        score = useful + keyword_score + numeric_score
        if score > best_score:
            best = frame
            best_score = score
    if best is None:
        raise RuntimeError(f"Не удалось прочитать лист {sheet_name} файла {path}")
    return best


def infer_time_seconds(series: pd.Series, column_name: str) -> pd.Series:
    name = slugify(column_name)
    values = numeric(series)

    if values.notna().sum() >= max(3, len(series) // 3):
        multiplier = 1.0
        if any(token in name for token in ("hour", "hours", "_h", "час")):
            multiplier = 3600.0
        elif any(token in name for token in ("minute", "minutes", "_min", "мин")):
            multiplier = 60.0
        elif any(token in name for token in ("millisecond", "_ms", "мс")):
            multiplier = 0.001
        numeric_time = values * multiplier
        valid = numeric_time.dropna()
        if not valid.empty:
            numeric_time = numeric_time - float(valid.iloc[0])
        return numeric_time

    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.notna().sum() >= 3:
        origin = parsed.dropna().iloc[0]
        return (parsed - origin).dt.total_seconds()

    timedeltas = pd.to_timedelta(series, errors="coerce")
    if timedeltas.notna().sum() >= 3:
        seconds = timedeltas.dt.total_seconds()
        return seconds - seconds.dropna().iloc[0]

    return pd.Series(np.arange(len(series), dtype=float), index=series.index)


def save_experiments(lab_id: str, frame: pd.DataFrame) -> tuple[Path, int]:
    output_dir = LAB_OUTPUTS[lab_id]
    experiment_dir = output_dir / "experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "real_experiments.csv"
    frame.to_csv(combined_path, index=False, encoding="utf-8-sig")

    count = 0
    if "experiment_id" in frame.columns:
        for experiment_id, part in frame.groupby("experiment_id", sort=True):
            part.to_csv(
                experiment_dir / f"{slugify(experiment_id)}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            count += 1
    return combined_path, count


def normalize_boyle() -> pd.DataFrame:
    candidates = sorted((RAW_DIR / "boyle_mariotte").rglob("*.csv"))
    parts: list[pd.DataFrame] = []

    for path in candidates:
        raw = read_csv_auto(path)
        columns = list(raw.columns)
        run_numbers: set[int] = set()
        for column in columns:
            match = re.search(r"run_?(\d+)", column)
            if match:
                run_numbers.add(int(match.group(1)))

        if not run_numbers:
            run_numbers = {1}

        for run_number in sorted(run_numbers):
            def find_column(tokens: tuple[str, ...]) -> str | None:
                run_suffixes = (f"run_{run_number}", f"run{run_number}")
                for column in columns:
                    has_measure = any(token in column for token in tokens)
                    has_run = (
                        len(run_numbers) == 1
                        or any(suffix in column for suffix in run_suffixes)
                    )
                    if has_measure and has_run:
                        return column
                return None

            time_col = find_column(("время_сек", "time_sec", "seconds", "time"))
            pressure_col = find_column(("давление", "pressure", "кпа", "kpa"))
            volume_col = find_column(("volume", "объем", "обьем", "мл", "ml"))

            if pressure_col is None or volume_col is None:
                continue

            pressure = numeric(raw[pressure_col])
            volume = numeric(raw[volume_col])
            if time_col is not None:
                time_seconds = infer_time_seconds(raw[time_col], time_col)
            else:
                time_seconds = pd.Series(np.arange(len(raw), dtype=float) * 10.0)

            part = pd.DataFrame(
                {
                    "experiment_id": f"pasco_boyle_run_{run_number:02d}",
                    "measurement_number": np.arange(1, len(raw) + 1),
                    "time_seconds": time_seconds,
                    "volume_ml": volume,
                    "pressure_kpa": pressure,
                    "temperature_c": 22.0,
                    "atmospheric_pressure_kpa": 101.325,
                    "pressure_type": "absolute",
                    "source_file": str(path.relative_to(PROJECT_ROOT)),
                    "data_origin": "real",
                }
            )
            part = part.dropna(subset=["volume_ml", "pressure_kpa"]).reset_index(drop=True)
            part["measurement_number"] = np.arange(1, len(part) + 1)
            if len(part) >= 3:
                parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def normalize_isochoric() -> pd.DataFrame:
    candidates = sorted((RAW_DIR / "isochoric").rglob("*.csv"))
    parts: list[pd.DataFrame] = []

    for path in candidates:
        raw = read_csv_auto(path)
        if "pressure_absolute_kpa" in raw.columns:
            pressure = numeric(raw["pressure_absolute_kpa"])
        elif "pressure_kpa" in raw.columns:
            pressure = numeric(raw["pressure_kpa"])
        else:
            continue

        if "temperature_measured_c" in raw.columns:
            temperature_c = numeric(raw["temperature_measured_c"])
        elif "temperature_c" in raw.columns:
            temperature_c = numeric(raw["temperature_c"])
        elif "temperature_measured_k" in raw.columns:
            temperature_c = numeric(raw["temperature_measured_k"]) - 273.15
        else:
            continue

        if "time_seconds" in raw.columns:
            time_seconds = infer_time_seconds(raw["time_seconds"], "time_seconds")
        else:
            # В источнике нет временных отметок. Используем только порядок измерений.
            time_seconds = pd.Series(np.arange(len(raw), dtype=float), index=raw.index)

        if "experiment_id" in raw.columns:
            experiment_ids = raw["experiment_id"].fillna(path.stem).astype(str)
        else:
            experiment_ids = pd.Series([path.stem] * len(raw), index=raw.index)

        part = pd.DataFrame(
            {
                "experiment_id": experiment_ids.map(slugify),
                "measurement_number": np.arange(1, len(raw) + 1),
                "time_seconds": time_seconds,
                "temperature_c": temperature_c,
                "pressure_kpa": pressure,
                "volume_ml": 1000.0,
                "pressure_type": "absolute",
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "data_origin": "real",
            }
        )
        part = part.dropna(subset=["temperature_c", "pressure_kpa"]).reset_index(drop=True)
        part["measurement_number"] = np.arange(1, len(part) + 1)
        if len(part) >= 3:
            parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def normalize_heat_balance() -> pd.DataFrame:
    candidates = sorted((RAW_DIR / "heat_balance").rglob("*.csv"))
    parts: list[pd.DataFrame] = []

    for path in candidates:
        raw = read_csv_auto(path)
        temperature_column = next(
            (
                column
                for column in (
                    "temperature_water_c",
                    "temperature_c",
                    "measured_temperature_c",
                    "t_c",
                )
                if column in raw.columns
            ),
            None,
        )
        if temperature_column is None:
            continue

        if "time_s" in raw.columns:
            time_seconds = infer_time_seconds(raw["time_s"], "time_s")
        elif "time_seconds" in raw.columns:
            time_seconds = infer_time_seconds(raw["time_seconds"], "time_seconds")
        else:
            time_seconds = pd.Series(np.arange(len(raw), dtype=float), index=raw.index)

        initial_water = (
            numeric(raw["initial_water_temperature_c"])
            if "initial_water_temperature_c" in raw.columns
            else pd.Series([np.nan] * len(raw), index=raw.index)
        )
        initial_hot = (
            numeric(raw["initial_metal_temperature_c"])
            if "initial_metal_temperature_c" in raw.columns
            else pd.Series([np.nan] * len(raw), index=raw.index)
        )

        experiment_ids = (
            raw["experiment_id"].fillna(path.stem).astype(str).map(slugify)
            if "experiment_id" in raw.columns
            else pd.Series([slugify(path.stem)] * len(raw), index=raw.index)
        )

        part = pd.DataFrame(
            {
                "experiment_id": experiment_ids,
                "measurement_number": np.arange(1, len(raw) + 1),
                "time_seconds": time_seconds,
                "temperature_c": numeric(raw[temperature_column]),
                "hot_mass_g": np.nan,
                "cold_mass_g": np.nan,
                "hot_initial_c": initial_hot,
                "cold_initial_c": initial_water,
                "calorimeter_heat_capacity_j_k": 70.0,
                "material": "unknown",
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "data_origin": "real",
            }
        )
        part = part.dropna(subset=["time_seconds", "temperature_c"]).reset_index(drop=True)
        part["measurement_number"] = np.arange(1, len(part) + 1)
        if len(part) >= 3:
            parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def select_time_column(frame: pd.DataFrame) -> str | None:
    priority_tokens = (
        "time_seconds",
        "elapsed_time",
        "timestamp",
        "datetime",
        "time",
        "время",
        "second",
        "sec",
        "minute",
        "min",
        "hour",
    )
    for token in priority_tokens:
        for column in frame.columns:
            if token in slugify(column):
                return str(column)
    return None


TEMPERATURE_EXCLUSION_TOKENS = (
    "time",
    "date",
    "pressure",
    "press",
    "flow",
    "flowrate",
    "rate",
    "volume",
    "humidity",
    "mass",
    "voltage",
    "current",
    "power",
    "speed",
    "расход",
    "давление",
    "объем",
    "объём",
    "влажность",
    "масса",
    "напряжение",
)

TEMPERATURE_NAME_PATTERNS = (
    re.compile(r"(?:^|_)(?:temperature|temp|thermo|температура|темп)(?:_|$)"),
    re.compile(r"^t_?(?:in|out|ambient|pcm|water|air|sample|body|hot|cold)$"),
    re.compile(r"^t_?\d{1,3}$"),
    re.compile(r"^sensor_?t_?\d*$"),
)


def infer_temperature_unit(column_name: str) -> str:
    name = slugify(column_name)
    if name.endswith("_k") or any(token in name for token in ("kelvin", "temp_k", "temperature_k")):
        return "kelvin"
    if name.endswith("_f") or any(token in name for token in ("fahrenheit", "temp_f", "temperature_f")):
        return "fahrenheit"
    return "celsius"


def convert_temperature_to_celsius(series: pd.Series, unit: str) -> pd.Series:
    values = numeric(series)
    if unit == "kelvin":
        return values - 273.15
    if unit == "fahrenheit":
        return (values - 32.0) * 5.0 / 9.0
    return values


def classify_temperature_column(
    frame: pd.DataFrame,
    column: str,
    time_column: str | None,
) -> ColumnDecision:
    name = slugify(column)
    if column == time_column or name.startswith("unnamed"):
        return ColumnDecision(column, name, False, "unknown", 0, 0, 0.0, 0.0, "служебная или временная колонка")

    if any(token in name for token in TEMPERATURE_EXCLUSION_TOKENS):
        return ColumnDecision(column, name, False, "unknown", 0, 0, 0.0, 0.0, "название указывает на другую физическую величину")

    name_matches = any(pattern.search(name) for pattern in TEMPERATURE_NAME_PATTERNS)
    if not name_matches:
        return ColumnDecision(column, name, False, "unknown", 0, 0, 0.0, 0.0, "нет явного признака температурного канала")

    unit = infer_temperature_unit(name)
    values = convert_temperature_to_celsius(frame[column], unit)
    valid = values.dropna()
    minimum_values = max(5, len(frame) // 5)
    if len(valid) < minimum_values:
        return ColumnDecision(column, name, False, unit, len(valid), int(valid.nunique()), 0.0, 0.0, "недостаточно числовых значений")

    in_range_fraction = float(((valid >= -100.0) & (valid <= 200.0)).mean())
    unique_values = int(valid.nunique())
    dynamic_range = float(valid.max() - valid.min()) if len(valid) else 0.0

    if in_range_fraction < 0.95:
        reason = "значительная часть значений вне допустимого температурного диапазона"
        accepted = False
    elif unique_values < 4:
        reason = "канал почти не изменяется"
        accepted = False
    else:
        reason = "явный температурный канал с корректным числовым диапазоном"
        accepted = True

    return ColumnDecision(
        column_name=column,
        normalized_name=name,
        accepted=accepted,
        inferred_unit=unit,
        numeric_values=int(len(valid)),
        unique_values=unique_values,
        in_range_fraction=in_range_fraction,
        dynamic_range_c=dynamic_range,
        reason=reason,
    )


def temperature_column_decisions(
    frame: pd.DataFrame,
    time_column: str | None,
) -> list[ColumnDecision]:
    return [
        classify_temperature_column(frame, str(column), time_column)
        for column in frame.columns
    ]


def plausible_temperature_columns(frame: pd.DataFrame, time_column: str | None) -> list[str]:
    return [
        decision.column_name
        for decision in temperature_column_decisions(frame, time_column)
        if decision.accepted
    ]


def normalize_cooling(
    column_report: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    candidates = sorted((RAW_DIR / "cooling").rglob("*.xlsx")) + sorted(
        (RAW_DIR / "cooling").rglob("*.xls")
    )
    parts: list[pd.DataFrame] = []

    for path in candidates:
        excel = pd.ExcelFile(path)
        for sheet_name in excel.sheet_names:
            frame = best_excel_sheet(path, sheet_name)
            time_column = select_time_column(frame)
            if time_column is None:
                time_seconds = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
            else:
                time_seconds = infer_time_seconds(frame[time_column], time_column)

            decisions = temperature_column_decisions(frame, time_column)
            if column_report is not None:
                for decision in decisions:
                    column_report.append(
                        {
                            "source_file": str(path.relative_to(PROJECT_ROOT)),
                            "sheet_name": str(sheet_name),
                            **asdict(decision),
                        }
                    )

            for decision in decisions:
                if not decision.accepted:
                    continue
                temperature = convert_temperature_to_celsius(
                    frame[decision.column_name],
                    decision.inferred_unit,
                )
                sensor_id = slugify(decision.column_name)
                experiment_id = slugify(f"{path.stem}_{sheet_name}_{sensor_id}")
                condition = "with_pcm" if "with_pcm" in slugify(path.stem) else "without_pcm"

                part = pd.DataFrame(
                    {
                        "experiment_id": experiment_id,
                        "measurement_number": np.arange(1, len(frame) + 1),
                        "time_seconds": time_seconds,
                        "measured_temperature": temperature,
                        "sensor_id": sensor_id,
                        "source_temperature_column": decision.column_name,
                        "source_temperature_unit": decision.inferred_unit,
                        "condition": condition,
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "sheet_name": str(sheet_name),
                        "data_origin": "real",
                    }
                )
                part = part.dropna(subset=["time_seconds", "measured_temperature"])
                part = part.sort_values("time_seconds").drop_duplicates("time_seconds")
                part = part.reset_index(drop=True)
                part["measurement_number"] = np.arange(1, len(part) + 1)
                if len(part) >= 5:
                    parts.append(part)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def robust_noise_metrics(values: pd.Series) -> dict[str, float]:
    clean = numeric(values).dropna().to_numpy(dtype=float)
    if len(clean) < 20:
        return {
            "n_values": float(len(clean)),
            "resolution_c": math.nan,
            "noise_sigma_c": math.nan,
            "outlier_fraction": math.nan,
        }

    diffs = np.diff(clean)
    nonzero = np.abs(diffs[np.abs(diffs) > 1e-12])
    resolution = float(np.quantile(nonzero, 0.1)) if len(nonzero) else 0.0

    series = pd.Series(clean)
    trend = series.rolling(window=25, center=True, min_periods=5).median()
    trend = trend.bfill().ffill()
    residuals = series - trend
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    sigma = float(1.4826 * mad)
    if sigma <= 1e-12:
        sigma = float(np.std(residuals))
    threshold = max(0.2, 4.0 * sigma)
    outlier_fraction = float(np.mean(np.abs(residuals - median) > threshold))

    return {
        "n_values": float(len(clean)),
        "resolution_c": resolution,
        "noise_sigma_c": sigma,
        "outlier_fraction": outlier_fraction,
    }


def sensor_frames_from_zip(path: Path) -> list[tuple[str, pd.DataFrame]]:
    results: list[tuple[str, pd.DataFrame]] = []
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith("/") or Path(member).suffix.lower() != ".csv":
                continue
            try:
                frame = read_csv_bytes(archive.read(member))
            except Exception:
                continue
            results.append((member, frame))
    return results


def find_temperature_column(frame: pd.DataFrame) -> str | None:
    decisions = temperature_column_decisions(frame, select_time_column(frame))
    accepted = [decision.column_name for decision in decisions if decision.accepted]
    return accepted[0] if len(accepted) == 1 else None


def build_sensor_noise_profile() -> tuple[pd.DataFrame, dict[str, Any]]:
    zip_candidates = sorted((RAW_DIR / "sensor_noise").rglob("Preprocessed.zip"))
    if not zip_candidates:
        zip_candidates = sorted((RAW_DIR / "sensor_noise").rglob("*.zip"))

    rows: list[dict[str, Any]] = []
    for zip_path in zip_candidates:
        for member_name, frame in sensor_frames_from_zip(zip_path):
            temperature_column = find_temperature_column(frame)
            if temperature_column is None:
                continue
            metrics = robust_noise_metrics(frame[temperature_column])
            rows.append(
                {
                    "source_archive": str(zip_path.relative_to(PROJECT_ROOT)),
                    "sensor_file": member_name,
                    "temperature_column": temperature_column,
                    **metrics,
                }
            )

    summary = pd.DataFrame(rows)
    profile: dict[str, Any] = {
        "source": "real DHT22 measurements",
        "sensor_count": int(len(summary)),
        "temperature_resolution_c_median": None,
        "temperature_noise_sigma_c_median": None,
        "temperature_noise_sigma_c_q90": None,
        "outlier_fraction_median": None,
    }

    if not summary.empty:
        for column in ("resolution_c", "noise_sigma_c", "outlier_fraction"):
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
        profile.update(
            {
                "temperature_resolution_c_median": float(summary["resolution_c"].median()),
                "temperature_noise_sigma_c_median": float(summary["noise_sigma_c"].median()),
                "temperature_noise_sigma_c_q90": float(summary["noise_sigma_c"].quantile(0.9)),
                "outlier_fraction_median": float(summary["outlier_fraction"].median()),
            }
        )

    return summary, profile


def required_columns(lab_id: str) -> tuple[str, ...]:
    return {
        "boyle_mariotte": ("volume_ml", "pressure_kpa"),
        "isochoric": ("temperature_c", "pressure_kpa"),
        "cooling": ("time_seconds", "measured_temperature"),
        "heat_balance": ("time_seconds", "temperature_c"),
    }[lab_id]


def validate_experiment(lab_id: str, experiment_id: str, frame: pd.DataFrame) -> QualityRecord:
    required = required_columns(lab_id)
    missing_columns = [column for column in required if column not in frame.columns]
    missing_cells = int(frame[list(set(required) & set(frame.columns))].isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    out_of_range_cells = 0

    for column, (lower, upper) in NUMERIC_LIMITS.get(lab_id, {}).items():
        if column not in frame.columns:
            continue
        values = numeric(frame[column])
        out_of_range_cells += int(((values < lower) | (values > upper)).fillna(False).sum())

    monotonic_time = True
    if "time_seconds" in frame.columns:
        time_values = numeric(frame["time_seconds"]).dropna().to_numpy(dtype=float)
        monotonic_time = bool(len(time_values) < 2 or np.all(np.diff(time_values) >= 0))

    required_rows = MODEL_MIN_POINTS[lab_id]
    ready = (
        not missing_columns
        and len(frame) >= required_rows
        and missing_cells == 0
        and out_of_range_cells == 0
        and monotonic_time
    )

    notes: list[str] = []
    if len(frame) < required_rows:
        notes.append(f"нужно минимум {required_rows} точек")
    if missing_columns:
        notes.append("нет колонок: " + ", ".join(missing_columns))
    if missing_cells:
        notes.append(f"пропусков в обязательных полях: {missing_cells}")
    if out_of_range_cells:
        notes.append(f"значений вне физического диапазона: {out_of_range_cells}")
    if not monotonic_time:
        notes.append("время не монотонно")

    source_file = ""
    if "source_file" in frame.columns and not frame.empty:
        source_file = str(frame["source_file"].iloc[0])

    return QualityRecord(
        lab_id=lab_id,
        experiment_id=str(experiment_id),
        source_file=source_file,
        rows=int(len(frame)),
        required_rows=required_rows,
        missing_required_columns=";".join(missing_columns),
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        out_of_range_cells=out_of_range_cells,
        monotonic_time=monotonic_time,
        ready_for_model=ready,
        status="ready" if ready else "needs_review",
        notes="; ".join(notes) if notes else "ok",
    )


def predict_real_experiments(
    normalized: dict[str, pd.DataFrame],
    quality: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    module_map = {
        "boyle_mariotte": "labs.boyle_mariotte.module",
        "isochoric": "labs.isochoric.module",
        "cooling": "labs.cooling.module",
        "heat_balance": "labs.heat_balance.module",
    }

    import importlib

    ready_lookup = {
        (row.lab_id, row.experiment_id): bool(row.ready_for_model)
        for row in quality.itertuples(index=False)
    }

    for lab_id, frame in normalized.items():
        if frame.empty or lab_id not in module_map:
            continue
        module = importlib.import_module(module_map[lab_id])
        for experiment_id, part in frame.groupby("experiment_id", sort=True):
            record: dict[str, Any] = {
                "lab_id": lab_id,
                "experiment_id": experiment_id,
                "ready_for_model": ready_lookup.get((lab_id, experiment_id), False),
                "predicted_class": "",
                "confidence": np.nan,
                "probabilities_json": "",
                "prediction_status": "skipped",
                "error": "",
            }
            if not record["ready_for_model"]:
                rows.append(record)
                continue
            try:
                prediction = module.predict(part.copy())
                record.update(
                    {
                        "predicted_class": prediction.predicted_class,
                        "confidence": prediction.confidence,
                        "probabilities_json": json.dumps(
                            prediction.probabilities,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "prediction_status": "success",
                    }
                )
            except Exception as error:
                record["prediction_status"] = "error"
                record["error"] = str(error)
            rows.append(record)

    return pd.DataFrame(rows)


def clean_output_directories() -> None:
    for directory in (OUTPUT_DIR, REPORT_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка {RAW_DIR}")

    clean_output_directories()
    for directory in LAB_OUTPUTS.values():
        directory.mkdir(parents=True, exist_ok=True)

    cooling_column_report: list[dict[str, Any]] = []
    normalized = {
        "boyle_mariotte": normalize_boyle(),
        "isochoric": normalize_isochoric(),
        "cooling": normalize_cooling(cooling_column_report),
        "heat_balance": normalize_heat_balance(),
    }

    manifest_rows: list[dict[str, Any]] = []
    quality_records: list[QualityRecord] = []

    for lab_id, frame in normalized.items():
        if frame.empty:
            manifest_rows.append(
                {
                    "lab_id": lab_id,
                    "rows": 0,
                    "experiments": 0,
                    "output_file": "",
                    "status": "no_data",
                }
            )
            continue

        combined_path, experiment_count = save_experiments(lab_id, frame)
        manifest_rows.append(
            {
                "lab_id": lab_id,
                "rows": len(frame),
                "experiments": experiment_count,
                "output_file": str(combined_path.relative_to(PROJECT_ROOT)),
                "status": "normalized",
            }
        )

        for experiment_id, part in frame.groupby("experiment_id", sort=True):
            quality_records.append(validate_experiment(lab_id, experiment_id, part))

    sensor_summary, sensor_profile = build_sensor_noise_profile()
    sensor_summary_path = LAB_OUTPUTS["sensor_noise"] / "sensor_noise_summary.csv"
    sensor_profile_path = LAB_OUTPUTS["sensor_noise"] / "sensor_noise_profile.json"
    sensor_summary.to_csv(sensor_summary_path, index=False, encoding="utf-8-sig")
    sensor_profile_path.write_text(
        json.dumps(sensor_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_df = pd.DataFrame(manifest_rows)
    quality_df = pd.DataFrame([asdict(record) for record in quality_records])

    manifest_path = REPORT_DIR / "normalization_manifest.csv"
    quality_path = REPORT_DIR / "quality_report.csv"
    predictions_path = REPORT_DIR / "real_data_model_predictions.csv"
    cooling_columns_path = REPORT_DIR / "cooling_column_selection.csv"
    calibration_path = REPORT_DIR / "empirical_realism_profile.json"
    summary_path = REPORT_DIR / "integration_summary.txt"

    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    quality_df.to_csv(quality_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(cooling_column_report).to_csv(
        cooling_columns_path,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df = predict_real_experiments(normalized, quality_df)
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    calibration = {
        "version": "1.0",
        "purpose": "Empirical calibration values derived from real measurements",
        "temperature_sensor": sensor_profile,
        "normalized_experiments": {
            row["lab_id"]: {
                "rows": int(row["rows"]),
                "experiments": int(row["experiments"]),
                "status": row["status"],
            }
            for row in manifest_rows
        },
    }
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ready_count = int(quality_df["ready_for_model"].sum()) if not quality_df.empty else 0
    review_count = int(len(quality_df) - ready_count)
    successful_predictions = (
        int((predictions_df["prediction_status"] == "success").sum())
        if not predictions_df.empty
        else 0
    )

    accepted_cooling_columns = sum(
        bool(row["accepted"]) for row in cooling_column_report
    )
    rejected_cooling_columns = len(cooling_column_report) - accepted_cooling_columns

    lines = [
        "PhysLab AI — подготовка реальных данных завершена",
        "",
        f"Нормализовано лабораторных наборов: {(manifest_df['status'] == 'normalized').sum()}",
        f"Экспериментов сформировано: {len(quality_df)}",
        f"Температурных каналов охлаждения принято: {accepted_cooling_columns}",
        f"Колонок охлаждения отклонено: {rejected_cooling_columns}",
        f"Готово для текущих моделей: {ready_count}",
        f"Требует ручной проверки: {review_count}",
        f"Предсказаний текущих моделей выполнено: {successful_predictions}",
        f"Профилей температурных датчиков: {sensor_profile['sensor_count']}",
        "",
        f"Нормализованные данные: {OUTPUT_DIR}",
        f"Манифест: {manifest_path}",
        f"Отчёт качества: {quality_path}",
        f"Предсказания: {predictions_path}",
        f"Отчёт выбора температурных каналов: {cooling_columns_path}",
        f"Профиль реалистичности: {calibration_path}",
        "",
        "Исходные файлы real_data_raw не изменены.",
        "Реальные данные пока не добавлены в обучение: они сохранены как внешний корпус проверки.",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
