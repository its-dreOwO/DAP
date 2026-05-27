"""
DAP391m Project 8 - Ingestion & Cleaning
========================================

Loads the active modeling dataset, validates the expected schema, applies
light cleaning, and writes Data/filtered/clean_data.csv.

Important: the cleaned file intentionally preserves `risk_probability` and
`port_delay_days` for EDA/leakage checks. Modeling code must drop both columns.

Run:
    .venv/bin/python3 src-code/01_ingestion_cleaning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "supply_chain_risk_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered"
OUTPUT_CSV = OUTPUT_DIR / "clean_data.csv"
REPORT_TXT = OUTPUT_DIR / "cleaning_report.txt"

TARGET_COL = "risk_label"
VALID_LABELS = {"Low", "Medium", "High"}

REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "temperature_C",
    "vibration_level",
    "machine_runtime_hours",
    "inventory_level_units",
    "pending_orders",
    "supplier_id",
    "supplier_lead_time_days",
    "supplier_quality_score",
    "supplier_reliability_index",
    "fuel_price_index",
    "port_delay_days",
    "market_demand_index",
    "weather_disruption_score",
    "risk_probability",
    TARGET_COL,
]

NUMERIC_COLUMNS = [
    "temperature_C",
    "vibration_level",
    "machine_runtime_hours",
    "inventory_level_units",
    "pending_orders",
    "supplier_lead_time_days",
    "supplier_quality_score",
    "supplier_reliability_index",
    "fuel_price_index",
    "port_delay_days",
    "market_demand_index",
    "weather_disruption_score",
    "risk_probability",
]


def load_raw(path: Path = INPUT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_schema(df)
    cleaned = df.copy()

    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    invalid_timestamps = int(cleaned["timestamp"].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"Invalid timestamp values found: {invalid_timestamps}")

    for col in ["machine_id", "supplier_id", TARGET_COL]:
        cleaned[col] = cleaned[col].astype("string").str.strip()

    missing_target = int(cleaned[TARGET_COL].isna().sum())
    if missing_target:
        raise ValueError(f"Missing target values found: {missing_target}")

    unknown_labels = sorted(set(cleaned[TARGET_COL].dropna()) - VALID_LABELS)
    if unknown_labels:
        raise ValueError(f"Unexpected risk_label values: {unknown_labels}")

    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    numeric_nulls = cleaned[NUMERIC_COLUMNS].isna().sum()
    numeric_nulls = numeric_nulls[numeric_nulls > 0]
    if not numeric_nulls.empty:
        raise ValueError(
            "Missing or non-numeric values found in numeric columns: "
            f"{numeric_nulls.to_dict()}"
        )

    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def write_report(df: pd.DataFrame, path: Path) -> None:
    label_counts = (
        df[TARGET_COL].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    )
    lines = [
        "DAP391m ingestion and cleaning report",
        "=" * 40,
        f"Input: {INPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Output: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "Risk label counts:",
        label_counts.to_string(),
        "",
        "Leakage warning:",
        "`risk_probability` and `port_delay_days` are preserved here for EDA,",
        "but must be dropped before model fitting.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    cleaned = clean_data(raw)
    cleaned.to_csv(OUTPUT_CSV, index=False)
    write_report(cleaned, REPORT_TXT)

    print(f"Loaded {INPUT_CSV.relative_to(PROJECT_ROOT)}: {raw.shape}")
    print(f"Saved {OUTPUT_CSV.relative_to(PROJECT_ROOT)}: {cleaned.shape}")
    print(
        "Leakage columns preserved for EDA only: " "risk_probability, port_delay_days"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
