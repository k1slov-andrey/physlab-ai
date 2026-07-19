from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "real_data_raw"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "real_data_audit"

INVENTORY_PATH = REPORT_DIR / "file_inventory.csv"
TABLE_REPORT_PATH = REPORT_DIR / "table_report.csv"
SUMMARY_PATH = REPORT_DIR / "audit_summary.txt"


EXPECTED_CONCEPTS: dict[str, set[str]] = {
    "boyle_mariotte": {"pressure", "volume"},
    "isochoric": {"pressure", "temperature"},
    "cooling": {"time", "temperature"},
    "heat_balance": {"time", "temperature"},
    "sensor_noise": {"time", "temperature"},
}


CONCEPT_TOKENS: dict[str, tuple[str, ...]] = {
    "pressure": (
        "pressure",
        "p_kpa",
        "press",
        "давление",
        "кпа",
        "kpa",
    ),
    "volume": (
        "volume",
        "v_ml",
        "vol",
        "объем",
        "объём",
        "мл",
        "ml",
    ),
    "temperature": (
        "temperature",
        "temp",
        "t_c",
        "t_k",
        "температура",
        "градус",
        "celsius",
        "kelvin",
    ),
    "time": (
        "time",
        "timestamp",
        "datetime",
        "date",
        "seconds",
        "minutes",
        "t_sec",
        "t_min",
        "время",
        "дата",
        "сек",
        "мин",
    ),
    "mass": (
        "mass",
        "m_g",
        "m_kg",
        "масса",
    ),
    "humidity": (
        "humidity",
        "влажность",
    ),
}


def normalize_text(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("ё", "е")
        .replace("\n", " ")
    )


def detect_lab(path: Path) -> str:
    parts = {normalize_text(part) for part in path.parts}

    for lab_name in EXPECTED_CONCEPTS:
        if lab_name in parts:
            return lab_name

    return "unknown"


def detect_concepts(columns: list[str]) -> set[str]:
    text = " | ".join(normalize_text(column) for column in columns)
    found: set[str] = set()

    for concept, tokens in CONCEPT_TOKENS.items():
        if any(normalize_text(token) in text for token in tokens):
            found.add(concept)

    return found


def assess_suitability(
    lab: str,
    rows: int,
    found_concepts: set[str],
) -> tuple[str, str]:
    if lab not in EXPECTED_CONCEPTS:
        return "unknown", "Не удалось определить лабораторную работу."

    required = EXPECTED_CONCEPTS[lab]
    found_required = required.intersection(found_concepts)

    if rows < 3:
        return "not_ready", "Недостаточно строк для анализа."

    if required.issubset(found_concepts):
        return "suitable", "Найдены все основные физические величины."

    if found_required:
        missing = sorted(required - found_concepts)
        return (
            "partial",
            f"Найдена только часть величин. Не обнаружено: {', '.join(missing)}.",
        )

    return (
        "not_ready",
        "Не найдены обязательные физические величины.",
    )


def dataframe_report(
    df: pd.DataFrame,
    source_file: str,
    lab: str,
    data_format: str,
    table_name: str = "",
) -> dict[str, Any]:
    columns = [str(column) for column in df.columns]
    concepts = detect_concepts(columns)
    required = EXPECTED_CONCEPTS.get(lab, set())

    rows = int(len(df))
    columns_count = int(len(columns))
    total_cells = rows * columns_count

    missing_cells = int(df.isna().sum().sum())
    missing_percent = (
        round(missing_cells / total_cells * 100, 3)
        if total_cells
        else 0.0
    )

    duplicate_rows = int(df.duplicated().sum()) if rows else 0
    numeric_columns = int(
        len(df.select_dtypes(include="number").columns)
    )

    suitability, notes = assess_suitability(
        lab=lab,
        rows=rows,
        found_concepts=concepts,
    )

    return {
        "source_file": source_file,
        "table_name": table_name,
        "lab": lab,
        "format": data_format,
        "rows": rows,
        "columns": columns_count,
        "numeric_columns": numeric_columns,
        "missing_cells": missing_cells,
        "missing_percent": missing_percent,
        "duplicate_rows": duplicate_rows,
        "column_names": " | ".join(columns),
        "concepts_found": " | ".join(sorted(concepts)),
        "required_concepts": " | ".join(sorted(required)),
        "suitability": suitability,
        "notes": notes,
    }


def try_read_csv_text(text: str) -> pd.DataFrame:
    sample = text[:10000]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|",
        )
        separator = dialect.delimiter
    except csv.Error:
        separator = None

    if separator:
        return pd.read_csv(
            io.StringIO(text),
            sep=separator,
            decimal="," if separator == ";" else ".",
            engine="python",
        )

    return pd.read_csv(
        io.StringIO(text),
        sep=None,
        engine="python",
    )


def read_csv_file(path: Path) -> pd.DataFrame:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1251",
        "windows-1251",
        "latin1",
    )

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            return try_read_csv_text(text)
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"CSV не прочитан: {last_error}"
    )


def score_excel_header(df: pd.DataFrame) -> int:
    columns = [normalize_text(column) for column in df.columns]

    useful_columns = sum(
        1
        for column in columns
        if column
        and not column.startswith("unnamed")
        and column != "nan"
    )

    detected = len(detect_concepts(columns))

    return useful_columns + detected * 10


def read_excel_sheet(
    path: Path,
    sheet_name: str,
) -> tuple[pd.DataFrame, int]:
    best_df: pd.DataFrame | None = None
    best_header = 0
    best_score = -1

    for header_row in range(0, 6):
        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row,
            )
        except Exception:
            continue

        score = score_excel_header(df)

        if score > best_score:
            best_df = df
            best_header = header_row
            best_score = score

    if best_df is None:
        raise RuntimeError(
            f"Не удалось прочитать лист {sheet_name}"
        )

    return best_df, best_header


def inspect_zip_csv(
    zip_path: Path,
    member_name: str,
) -> pd.DataFrame:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1251",
        "latin1",
    )

    with zipfile.ZipFile(zip_path) as archive:
        raw_bytes = archive.read(member_name)

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            text = raw_bytes.decode(encoding)
            return try_read_csv_text(text)
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"CSV внутри ZIP не прочитан: {last_error}"
    )


def main() -> None:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Папка не найдена: {RAW_DATA_DIR}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    files = sorted(
        path
        for path in RAW_DATA_DIR.rglob("*")
        if path.is_file()
    )

    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        extension = path.suffix.lower()
        lab = detect_lab(path)

        inventory_entry: dict[str, Any] = {
            "file": str(relative_path),
            "lab": lab,
            "extension": extension,
            "size_kb": round(path.stat().st_size / 1024, 2),
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "status": "not_processed",
            "tables_found": 0,
            "error": "",
        }

        try:
            if extension == ".csv":
                df = read_csv_file(path)

                table_rows.append(
                    dataframe_report(
                        df=df,
                        source_file=str(relative_path),
                        lab=lab,
                        data_format="csv",
                    )
                )

                inventory_entry["status"] = "read_successfully"
                inventory_entry["tables_found"] = 1

            elif extension in {".xlsx", ".xls"}:
                excel_file = pd.ExcelFile(path)
                tables_found = 0

                for sheet_name in excel_file.sheet_names:
                    df, header_row = read_excel_sheet(
                        path=path,
                        sheet_name=sheet_name,
                    )

                    report = dataframe_report(
                        df=df,
                        source_file=str(relative_path),
                        lab=lab,
                        data_format="excel",
                        table_name=str(sheet_name),
                    )

                    report["notes"] = (
                        f"{report['notes']} "
                        f"Заголовок определён по строке {header_row + 1}."
                    )

                    table_rows.append(report)
                    tables_found += 1

                inventory_entry["status"] = "read_successfully"
                inventory_entry["tables_found"] = tables_found

            elif extension == ".zip":
                tables_found = 0

                with zipfile.ZipFile(path) as archive:
                    members = [
                        name
                        for name in archive.namelist()
                        if not name.endswith("/")
                    ]

                for member_name in members:
                    if Path(member_name).suffix.lower() != ".csv":
                        continue

                    try:
                        df = inspect_zip_csv(
                            zip_path=path,
                            member_name=member_name,
                        )

                        table_rows.append(
                            dataframe_report(
                                df=df,
                                source_file=str(relative_path),
                                lab=lab,
                                data_format="zip_csv",
                                table_name=member_name,
                            )
                        )

                        tables_found += 1

                    except Exception as error:
                        table_rows.append(
                            {
                                "source_file": str(relative_path),
                                "table_name": member_name,
                                "lab": lab,
                                "format": "zip_csv",
                                "rows": 0,
                                "columns": 0,
                                "numeric_columns": 0,
                                "missing_cells": 0,
                                "missing_percent": 0,
                                "duplicate_rows": 0,
                                "column_names": "",
                                "concepts_found": "",
                                "required_concepts": " | ".join(
                                    sorted(
                                        EXPECTED_CONCEPTS.get(
                                            lab,
                                            set(),
                                        )
                                    )
                                ),
                                "suitability": "read_error",
                                "notes": str(error),
                            }
                        )

                inventory_entry["status"] = "archive_inspected"
                inventory_entry["tables_found"] = tables_found

            elif extension == ".txt":
                line_count = len(
                    path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()
                )

                inventory_entry["status"] = (
                    f"text_reference_{line_count}_lines"
                )

            elif extension == ".pdf":
                inventory_entry["status"] = "reference_document"

            elif extension == ".spklab":
                inventory_entry["status"] = (
                    "proprietary_source_file"
                )

            else:
                inventory_entry["status"] = "unsupported_format"

        except Exception as error:
            inventory_entry["status"] = "read_error"
            inventory_entry["error"] = str(error)

        inventory_rows.append(inventory_entry)

    inventory_df = pd.DataFrame(inventory_rows)
    table_df = pd.DataFrame(table_rows)

    inventory_df.to_csv(
        INVENTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    table_df.to_csv(
        TABLE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    suitability_counts: dict[str, int] = {}

    if not table_df.empty and "suitability" in table_df.columns:
        suitability_counts = (
            table_df["suitability"]
            .value_counts()
            .to_dict()
        )

    summary_lines = [
        "PHysLab AI — аудит реальных данных",
        "",
        f"Файлов найдено: {len(inventory_df)}",
        f"Таблиц найдено: {len(table_df)}",
        "",
        "Оценка пригодности:",
    ]

    if suitability_counts:
        for status, count in suitability_counts.items():
            summary_lines.append(
                f"- {status}: {count}"
            )
    else:
        summary_lines.append("- Табличные данные не найдены.")

    summary_lines.extend(
        [
            "",
            f"Инвентаризация: {INVENTORY_PATH}",
            f"Отчёт по таблицам: {TABLE_REPORT_PATH}",
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()