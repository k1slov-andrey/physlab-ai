from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.lab_registry import LAB_REGISTRY


LAB_MIN_POINTS = {
    "cooling": 20,
    "boyle_mariotte": 8,
    "isochoric": 10,
    "heat_balance": 20,
}

LAB_REQUIRED = {
    "cooling": ("time_seconds", "measured_temperature"),
    "boyle_mariotte": ("volume_ml", "pressure_kpa"),
    "isochoric": ("temperature_c", "pressure_kpa"),
    "heat_balance": (
        "time_seconds",
        "temperature_c",
        "hot_mass_g",
        "cold_mass_g",
        "hot_initial_c",
        "cold_initial_c",
    ),
}

NUMERIC_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "cooling": {
        "time_seconds": (0.0, 10_000_000.0),
        "measured_temperature": (-100.0, 500.0),
    },
    "boyle_mariotte": {
        "volume_ml": (1.0, 5000.0),
        "pressure_kpa": (1.0, 2000.0),
        "temperature_c": (-100.0, 500.0),
    },
    "isochoric": {
        "temperature_c": (-100.0, 700.0),
        "pressure_kpa": (1.0, 2000.0),
        "volume_ml": (1.0, 20_000.0),
    },
    "heat_balance": {
        "time_seconds": (0.0, 1_000_000.0),
        "temperature_c": (-100.0, 500.0),
        "hot_mass_g": (0.01, 100_000.0),
        "cold_mass_g": (0.01, 100_000.0),
        "hot_initial_c": (-100.0, 1000.0),
        "cold_initial_c": (-100.0, 500.0),
    },
}

ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "cooling": {
        "time_seconds": (
            "time_seconds", "time_s", "t_sec", "seconds", "second", "time",
            "elapsed_time", "elapsed_seconds", "время", "сек", "секунды",
        ),
        "measured_temperature": (
            "measured_temperature", "temperature_c", "temp_c", "t_c",
            "temperature", "temp", "water_temperature", "temperature_water_c",
            "температура", "температура_c",
        ),
    },
    "boyle_mariotte": {
        "measurement_number": (
            "measurement_number", "measurement", "trial", "point", "n", "номер",
        ),
        "time_seconds": (
            "time_seconds", "time_s", "t_sec", "seconds", "time", "время", "сек",
        ),
        "volume_ml": (
            "volume_ml", "v_ml", "volume_cm3", "volume_cc", "volume", "vol",
            "объем", "объём", "объем_мл", "объём_мл",
        ),
        "pressure_kpa": (
            "pressure_kpa", "p_kpa", "pressure", "press", "p", "давление",
        ),
        "temperature_c": (
            "temperature_c", "temp_c", "t_c", "temperature", "temp", "температура",
        ),
        "atmospheric_pressure_kpa": (
            "atmospheric_pressure_kpa", "atmospheric_kpa", "p_atm_kpa", "patm_kpa",
        ),
    },
    "isochoric": {
        "time_seconds": (
            "time_seconds", "time_s", "t_sec", "seconds", "time", "время", "сек",
        ),
        "temperature_c": (
            "temperature_c", "temperature_measured_c", "temperature_measured_k",
            "temperature_k", "temp_c", "temp_k", "t_c", "t_k",
            "temperature", "temp", "температура",
        ),
        "pressure_kpa": (
            "pressure_absolute_kpa", "pressure_kpa", "p_kpa", "pressure", "press",
            "p", "давление",
        ),
        "volume_ml": (
            "volume_ml", "v_ml", "volume_cm3", "volume_cc", "volume", "объем", "объём",
        ),
    },
    "heat_balance": {
        "time_seconds": (
            "time_seconds", "time_s", "t_sec", "seconds", "time", "время", "сек",
        ),
        "temperature_c": (
            "temperature_c", "temperature_water_c", "temp_c", "t_c", "temperature",
            "temp", "температура",
        ),
        "hot_mass_g": (
            "hot_mass_g", "sample_mass_g", "body_mass_g", "m_sample_g", "m_body_g",
            "metal_mass_g", "масса_образца",
        ),
        "cold_mass_g": (
            "cold_mass_g", "water_mass_g", "m_water_g", "масса_воды",
        ),
        "hot_initial_c": (
            "hot_initial_c", "initial_metal_temperature_c", "sample_initial_c",
            "body_initial_c", "t_hot_c", "t_sample_c", "температура_образца",
        ),
        "cold_initial_c": (
            "cold_initial_c", "initial_water_temperature_c", "water_initial_c",
            "t_cold_c", "t_water_c", "температура_воды",
        ),
        "calorimeter_heat_capacity_j_k": (
            "calorimeter_heat_capacity_j_k", "calorimeter_constant_j_k", "c_cal_j_k",
        ),
        "material": ("material", "substance", "металл", "материал"),
    },
}

FILE_HINTS = {
    "cooling": ("cool", "heating", "temperature_time", "охлаж", "нагрев"),
    "boyle_mariotte": ("boyle", "mariotte", "pressure_volume", "изотерм"),
    "isochoric": ("isochor", "gay_lussac", "pressure_temperature", "constant_volume"),
    "heat_balance": ("calor", "heat_balance", "specific_heat", "теплов", "теплоем"),
}


@dataclass
class DetectionResult:
    lab_id: str
    confidence: float
    scores: dict[str, float]
    reasons: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    lab_id: str
    rows: int
    usable_rows: int
    min_points: int
    missing_columns: list[str]
    missing_cells: int
    duplicate_rows: int
    out_of_range_cells: int
    monotonic_axis: bool
    dynamic_range_ok: bool
    ready_for_model: bool
    status: str
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "Лабораторная": LAB_REGISTRY[self.lab_id].short_title,
            "Строк": self.rows,
            "Пригодных строк": self.usable_rows,
            "Минимум для модели": self.min_points,
            "Пропущенных ячеек": self.missing_cells,
            "Дубликатов строк": self.duplicate_rows,
            "Значений вне диапазона": self.out_of_range_cells,
            "Ось монотонна": self.monotonic_axis,
            "Динамический диапазон достаточен": self.dynamic_range_ok,
            "Готово для модели": self.ready_for_model,
            "Статус": self.status,
        }


def normalize_name(value: Any) -> str:
    text = str(value).strip().lower().replace("ё", "е")
    text = text.replace("°", "")
    text = re.sub(r"[^a-zа-я0-9]+", "_", text, flags=re.IGNORECASE)
    return text.strip("_")


def unique_columns(columns: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        base = normalize_name(column) or "column"
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


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "latin1")
    separators = (";", ",", "\t", "|")
    best: pd.DataFrame | None = None
    best_score = -1
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except Exception as error:
            last_error = error
            continue

        for separator in separators:
            try:
                frame = pd.read_csv(
                    io.StringIO(text),
                    sep=separator,
                    engine="python",
                    dtype=str,
                    keep_default_na=True,
                )
            except Exception as error:
                last_error = error
                continue
            score = frame.shape[1] * 1000 + min(frame.shape[0], 500)
            if score > best_score:
                best = frame
                best_score = score

    if best is None:
        raise ValueError(f"CSV не удалось прочитать: {last_error}")

    best.columns = unique_columns(list(best.columns))
    return best


def _excel_header_score(frame: pd.DataFrame) -> float:
    names = unique_columns(list(frame.columns))
    useful = sum(1 for name in names if not name.startswith("unnamed") and name != "nan")
    tokens = (
        "time", "temp", "temperature", "pressure", "volume", "mass",
        "время", "температура", "давление", "объем", "объём", "масса",
    )
    keywords = sum(8 for name in names if any(token in name for token in tokens))
    numeric_count = 0
    for column in frame.columns:
        try:
            if numeric(frame[column]).notna().sum() >= max(3, len(frame) // 4):
                numeric_count += 1
        except Exception:
            continue
    return useful + keywords + numeric_count


def read_uploaded_tables(filename: str, raw: bytes) -> dict[str, pd.DataFrame]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv" or suffix == ".txt":
        return {"CSV": _read_csv_bytes(raw)}

    if suffix not in {".xlsx", ".xls"}:
        raise ValueError("Поддерживаются CSV, XLSX и XLS.")

    book = pd.ExcelFile(io.BytesIO(raw))
    result: dict[str, pd.DataFrame] = {}
    for sheet_name in book.sheet_names:
        best: pd.DataFrame | None = None
        best_score = -1.0
        for header in range(0, 12):
            try:
                frame = pd.read_excel(
                    io.BytesIO(raw),
                    sheet_name=sheet_name,
                    header=header,
                )
            except Exception:
                continue
            score = _excel_header_score(frame)
            if score > best_score:
                best = frame
                best_score = score
        if best is not None:
            best.columns = unique_columns(list(best.columns))
            result[str(sheet_name)] = best

    if not result:
        raise ValueError("В Excel-файле не найдено читаемых листов.")
    return result


def _alias_matches(frame: pd.DataFrame, lab_id: str) -> dict[str, str]:
    columns = {normalize_name(column): str(column) for column in frame.columns}
    matches: dict[str, str] = {}
    for canonical, aliases in ALIASES[lab_id].items():
        for alias in aliases:
            normalized_alias = normalize_name(alias)
            if normalized_alias in columns:
                matches[canonical] = columns[normalized_alias]
                break
    return matches


def detect_lab(frame: pd.DataFrame, filename: str = "") -> DetectionResult:
    scores: dict[str, float] = {}
    reasons_by_lab: dict[str, list[str]] = {}
    normalized_filename = normalize_name(filename)

    for lab_id in LAB_REGISTRY:
        matches = _alias_matches(frame, lab_id)
        required = set(LAB_REQUIRED[lab_id])
        required_matches = required.intersection(matches)
        optional_matches = set(matches) - required
        score = len(required_matches) * 12.0 + len(optional_matches) * 2.0

        reasons: list[str] = []
        if required_matches:
            reasons.append("распознаны: " + ", ".join(sorted(required_matches)))
        for hint in FILE_HINTS[lab_id]:
            if normalize_name(hint) in normalized_filename:
                score += 8.0
                reasons.append(f"подсказка в имени файла: {hint}")
                break

        if required and required.issubset(matches):
            score += 10.0
        scores[lab_id] = score
        reasons_by_lab[lab_id] = reasons

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_lab, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    coverage = len(set(_alias_matches(frame, best_lab)).intersection(LAB_REQUIRED[best_lab])) / max(
        len(LAB_REQUIRED[best_lab]), 1
    )
    margin = max(best_score - second_score, 0.0) / max(best_score, 1.0)
    confidence = float(np.clip(0.55 * coverage + 0.45 * margin, 0.0, 1.0))
    return DetectionResult(
        lab_id=best_lab,
        confidence=confidence,
        scores=scores,
        reasons=reasons_by_lab[best_lab],
    )


def _run_key(name: str) -> str:
    normalized = normalize_name(name)
    match = re.search(r"(?:run|серия)_?(\d+)", normalized)
    if match:
        return match.group(1)
    trailing = re.search(r"_(\d+)$", normalized)
    return trailing.group(1) if trailing else "1"


def _find_columns_by_tokens(frame: pd.DataFrame, tokens: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        name = normalize_name(column)
        if any(token in name for token in tokens):
            result.append(str(column))
    return result


def extract_series_candidates(
    frame: pd.DataFrame,
    lab_id: str,
) -> dict[str, pd.DataFrame]:
    cleaned = frame.copy()
    cleaned = cleaned.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if cleaned.empty:
        return {"Основная таблица": cleaned}

    if "experiment_id" in cleaned.columns and cleaned["experiment_id"].nunique(dropna=True) > 1:
        return {
            f"Эксперимент {experiment_id}": part.reset_index(drop=True)
            for experiment_id, part in cleaned.groupby("experiment_id", sort=True)
        }

    candidates: dict[str, pd.DataFrame] = {}

    if lab_id == "boyle_mariotte":
        pressure_columns = _find_columns_by_tokens(cleaned, ("pressure", "давление", "p_kpa"))
        volume_columns = _find_columns_by_tokens(cleaned, ("volume", "объем", "объём", "v_ml"))
        time_columns = _find_columns_by_tokens(cleaned, ("time", "время", "сек"))
        calibrated_pressure = [column for column in pressure_columns if "voltage" not in normalize_name(column)]
        calibrated_volume = [column for column in volume_columns if "voltage" not in normalize_name(column)]
        if calibrated_pressure:
            pressure_columns = calibrated_pressure
        if calibrated_volume:
            volume_columns = calibrated_volume
        for pressure_column in pressure_columns:
            key = _run_key(pressure_column)
            volume_column = next((column for column in volume_columns if _run_key(column) == key), None)
            if volume_column is None:
                continue
            data = pd.DataFrame(
                {
                    pressure_column: cleaned[pressure_column],
                    volume_column: cleaned[volume_column],
                }
            )
            time_column = next((column for column in time_columns if _run_key(column) == key), None)
            if time_column is not None:
                data[time_column] = cleaned[time_column]
            candidates[f"Серия {key}"] = data

    elif lab_id in {"cooling", "heat_balance"}:
        time_columns = _find_columns_by_tokens(cleaned, ("time", "время", "сек", "мин", "date"))
        temperature_columns = _find_columns_by_tokens(
            cleaned,
            ("temperature", "temp", "температура", "t_c"),
        )
        time_column = time_columns[0] if time_columns else None
        for index, temperature_column in enumerate(temperature_columns, start=1):
            data = pd.DataFrame({temperature_column: cleaned[temperature_column]})
            if time_column is not None:
                data[time_column] = cleaned[time_column]
            candidates[f"Температурный канал {index}: {temperature_column}"] = data

    elif lab_id == "isochoric":
        pressure_columns = _find_columns_by_tokens(cleaned, ("pressure", "давление", "p_kpa"))
        temperature_columns = _find_columns_by_tokens(
            cleaned,
            ("temperature", "temp", "температура", "t_c", "t_k"),
        )
        time_columns = _find_columns_by_tokens(cleaned, ("time", "время", "сек"))
        calibrated_pressure = [column for column in pressure_columns if "voltage" not in normalize_name(column)]
        measured_temperature = [
            column
            for column in temperature_columns
            if "voltage" not in normalize_name(column) and "ideal" not in normalize_name(column)
        ]
        if calibrated_pressure:
            pressure_columns = calibrated_pressure
        if measured_temperature:
            temperature_columns = measured_temperature
        if pressure_columns and temperature_columns:
            for index, (pressure_column, temperature_column) in enumerate(
                zip(pressure_columns, temperature_columns), start=1
            ):
                data = pd.DataFrame(
                    {
                        pressure_column: cleaned[pressure_column],
                        temperature_column: cleaned[temperature_column],
                    }
                )
                if time_columns:
                    selected_time = time_columns[min(index - 1, len(time_columns) - 1)]
                    data[selected_time] = cleaned[selected_time]
                candidates[f"Серия {index}"] = data

    if not candidates:
        candidates["Основная таблица"] = cleaned.reset_index(drop=True)
    return candidates


def _find_source_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {normalize_name(column): str(column) for column in frame.columns}
    for alias in aliases:
        candidate = normalize_name(alias)
        if candidate in normalized:
            return normalized[candidate]
    return None


def _convert_time(series: pd.Series, source_name: str) -> pd.Series:
    name = normalize_name(source_name)
    values = numeric(series)
    if values.notna().sum() >= max(3, len(series) // 3):
        multiplier = 1.0
        if any(token in name for token in ("hour", "hours", "_h", "час")):
            multiplier = 3600.0
        elif any(token in name for token in ("minute", "minutes", "_min", "мин")):
            multiplier = 60.0
        elif any(token in name for token in ("millisecond", "_ms", "мс")):
            multiplier = 0.001
        result = values * multiplier
        valid = result.dropna()
        return result - float(valid.iloc[0]) if not valid.empty else result

    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.notna().sum() >= 3:
        origin = parsed.dropna().iloc[0]
        return (parsed - origin).dt.total_seconds()

    durations = pd.to_timedelta(series, errors="coerce")
    if durations.notna().sum() >= 3:
        seconds = durations.dt.total_seconds()
        return seconds - float(seconds.dropna().iloc[0])

    return pd.Series(np.arange(len(series), dtype=float), index=series.index)


def _convert_pressure(series: pd.Series, source_name: str) -> pd.Series:
    values = numeric(series)
    name = normalize_name(source_name)
    if "mpa" in name:
        return values * 1000.0
    if "hpa" in name or "mbar" in name:
        return values * 0.1
    if "bar" in name and "mbar" not in name:
        return values * 100.0
    if "psi" in name:
        return values * 6.894757
    if re.search(r"(?:^|_)pa(?:_|$)", name) and "kpa" not in name:
        return values / 1000.0
    return values


def _convert_volume(series: pd.Series, source_name: str) -> pd.Series:
    values = numeric(series)
    name = normalize_name(source_name)
    if "m3" in name or "m_3" in name:
        return values * 1_000_000.0
    if any(token in name for token in ("liter", "litre", "_l", "литр")) and not any(
        token in name for token in ("ml", "мл")
    ):
        return values * 1000.0
    return values


def _convert_temperature(series: pd.Series, source_name: str) -> pd.Series:
    values = numeric(series)
    name = normalize_name(source_name)
    if name.endswith("_k") or any(
        token in name for token in ("kelvin", "temp_k", "temperature_k", "t_k")
    ):
        return values - 273.15
    if name.endswith("_f") or any(
        token in name for token in ("fahrenheit", "temp_f", "temperature_f", "t_f")
    ):
        return (values - 32.0) * 5.0 / 9.0
    return values


def normalize_experiment(
    frame: pd.DataFrame,
    lab_id: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    metadata = metadata or {}
    source = frame.copy().dropna(axis=1, how="all").dropna(axis=0, how="all")
    source.columns = unique_columns(list(source.columns))
    result = pd.DataFrame(index=source.index)
    mapping: dict[str, str] = {}

    for canonical, aliases in ALIASES[lab_id].items():
        source_column = _find_source_column(source, aliases)
        if source_column is None:
            continue
        mapping[canonical] = source_column
        if canonical == "time_seconds":
            result[canonical] = _convert_time(source[source_column], source_column)
        elif canonical == "pressure_kpa":
            result[canonical] = _convert_pressure(source[source_column], source_column)
        elif canonical == "volume_ml":
            result[canonical] = _convert_volume(source[source_column], source_column)
        elif canonical in {"temperature_c", "measured_temperature", "hot_initial_c", "cold_initial_c"}:
            result[canonical] = _convert_temperature(source[source_column], source_column)
        elif canonical == "material":
            result[canonical] = source[source_column].astype(str)
        else:
            result[canonical] = numeric(source[source_column])

    if lab_id == "cooling":
        if "temperature_c" in result.columns and "measured_temperature" not in result.columns:
            result = result.rename(columns={"temperature_c": "measured_temperature"})
        if "time_seconds" not in result.columns:
            result["time_seconds"] = np.arange(len(result), dtype=float)

    if lab_id == "boyle_mariotte":
        if "measurement_number" not in result.columns:
            result["measurement_number"] = np.arange(1, len(result) + 1)
        if "time_seconds" not in result.columns:
            result["time_seconds"] = np.arange(len(result), dtype=float) * 10.0
        if "temperature_c" not in result.columns:
            result["temperature_c"] = float(metadata.get("temperature_c", 22.0))
        if "atmospheric_pressure_kpa" not in result.columns:
            result["atmospheric_pressure_kpa"] = float(
                metadata.get("atmospheric_pressure_kpa", 101.325)
            )

    if lab_id == "isochoric":
        if "time_seconds" not in result.columns:
            result["time_seconds"] = np.arange(len(result), dtype=float) * 10.0
        if "volume_ml" not in result.columns:
            result["volume_ml"] = float(metadata.get("volume_ml", 250.0))

    if lab_id == "heat_balance":
        defaults = {
            "hot_mass_g": 50.0,
            "cold_mass_g": 150.0,
            "hot_initial_c": 95.0,
            "cold_initial_c": 22.0,
            "calorimeter_heat_capacity_j_k": 70.0,
            "material": "steel",
        }
        for column, default in defaults.items():
            if column not in result.columns:
                result[column] = metadata.get(column, default)
        if "time_seconds" not in result.columns:
            result["time_seconds"] = np.arange(len(result), dtype=float) * 5.0

    required = list(LAB_REQUIRED[lab_id])
    missing = [column for column in required if column not in result.columns]

    numeric_columns = [column for column in result.columns if column != "material"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.replace([np.inf, -np.inf], np.nan)
    subset = [column for column in required if column in result.columns]
    if subset:
        result = result.dropna(subset=subset)

    axis = "time_seconds"
    if lab_id == "boyle_mariotte":
        axis = "measurement_number"
    if axis in result.columns:
        result = result.sort_values(axis)

    result = result.drop_duplicates().reset_index(drop=True)
    return result, missing, mapping


def assess_quality(frame: pd.DataFrame, lab_id: str) -> QualityReport:
    required = list(LAB_REQUIRED[lab_id])
    missing = [column for column in required if column not in frame.columns]
    min_points = LAB_MIN_POINTS[lab_id]
    usable_rows = len(frame.dropna(subset=[column for column in required if column in frame.columns]))
    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum()) if len(frame) else 0
    out_of_range = 0

    for column, (minimum, maximum) in NUMERIC_LIMITS[lab_id].items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        out_of_range += int(((values < minimum) | (values > maximum)).sum())

    axis = "time_seconds" if lab_id != "boyle_mariotte" else "measurement_number"
    monotonic = True
    if axis in frame.columns and len(frame) > 1:
        axis_values = pd.to_numeric(frame[axis], errors="coerce").dropna()
        monotonic = bool(axis_values.is_monotonic_increasing)

    dynamic_range_ok = True
    if lab_id == "cooling" and "measured_temperature" in frame.columns:
        dynamic_range_ok = float(frame["measured_temperature"].max() - frame["measured_temperature"].min()) >= 1.0
    elif lab_id == "boyle_mariotte" and {"volume_ml", "pressure_kpa"}.issubset(frame.columns):
        dynamic_range_ok = (
            float(frame["volume_ml"].max() - frame["volume_ml"].min()) >= 2.0
            and float(frame["pressure_kpa"].max() - frame["pressure_kpa"].min()) >= 2.0
        )
    elif lab_id == "isochoric" and {"temperature_c", "pressure_kpa"}.issubset(frame.columns):
        dynamic_range_ok = (
            float(frame["temperature_c"].max() - frame["temperature_c"].min()) >= 3.0
            and float(frame["pressure_kpa"].max() - frame["pressure_kpa"].min()) >= 1.0
        )
    elif lab_id == "heat_balance" and "temperature_c" in frame.columns:
        dynamic_range_ok = float(frame["temperature_c"].max() - frame["temperature_c"].min()) >= 0.5

    issues: list[str] = []
    if missing:
        issues.append("Не хватает столбцов: " + ", ".join(missing))
    if usable_rows < min_points:
        issues.append(f"Недостаточно измерений: {usable_rows}, требуется минимум {min_points}")
    if out_of_range:
        issues.append(f"Значений вне физических диапазонов: {out_of_range}")
    if not monotonic:
        issues.append("Независимая переменная не упорядочена")
    if not dynamic_range_ok:
        issues.append("Изменение измеряемой величины слишком мало для устойчивого анализа")

    ready = not missing and usable_rows >= min_points and out_of_range == 0 and dynamic_range_ok
    if ready and missing_cells == 0:
        status = "ready"
    elif ready:
        status = "ready_with_warnings"
    else:
        status = "not_ready"

    return QualityReport(
        lab_id=lab_id,
        rows=len(frame),
        usable_rows=usable_rows,
        min_points=min_points,
        missing_columns=missing,
        missing_cells=missing_cells,
        duplicate_rows=duplicate_rows,
        out_of_range_cells=out_of_range,
        monotonic_axis=monotonic,
        dynamic_range_ok=dynamic_range_ok,
        ready_for_model=ready,
        status=status,
        issues=issues,
    )


def preferred_chart_columns(frame: pd.DataFrame, lab_id: str) -> tuple[str, list[str]]:
    if lab_id == "cooling":
        return "time_seconds", [column for column in ("measured_temperature",) if column in frame.columns]
    if lab_id == "boyle_mariotte":
        return "volume_ml", [column for column in ("pressure_kpa",) if column in frame.columns]
    if lab_id == "isochoric":
        return "temperature_c", [column for column in ("pressure_kpa",) if column in frame.columns]
    return "time_seconds", [column for column in ("temperature_c",) if column in frame.columns]
