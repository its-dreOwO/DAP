"""
DAP391m Project 8 - Ingestion & Cleaning
========================================

Loads the active modeling dataset (consumer credit-default risk), validates the
expected schema, applies light cleaning (impossible-value removal, duplicate
drop, type-fix), and writes Data/filtered/clean_data.csv.

Unlike the previous supply-chain dataset, this credit dataset has NO target
leakage column to preserve: `loan_status` is a real observed outcome, and the
remaining columns are all decision-time loan/applicant attributes.

Missing values in `person_emp_length` and `loan_int_rate` are reported but
intentionally preserved here; they are imputed inside the modeling pipeline.

Run:
    .venv/bin/python3 src-code/01_ingestion_cleaning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "credit_risk_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered"
OUTPUT_CSV = OUTPUT_DIR / "clean_data.csv"
REPORT_TXT = OUTPUT_DIR / "cleaning_report.txt"

TARGET_COL = "loan_status"
VALID_TARGETS = {0, 1}

# Clearly impossible data-entry errors in this dataset (e.g. age=144, emp=123y).
MAX_PLAUSIBLE_AGE = 100
MAX_PLAUSIBLE_EMP_LENGTH = 60

REQUIRED_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

NUMERIC_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def load_raw(path: Path = INPUT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    validate_schema(df)
    cleaned = df.copy()
    stats: dict[str, int] = {"input_rows": len(cleaned)}

    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        cleaned[col] = cleaned[col].astype("string").str.strip()

    # Target must be present and binary {0, 1}.
    cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
    missing_target = int(cleaned[TARGET_COL].isna().sum())
    if missing_target:
        raise ValueError(f"Missing target values found: {missing_target}")
    cleaned[TARGET_COL] = cleaned[TARGET_COL].astype(int)
    unknown = sorted(set(cleaned[TARGET_COL].unique()) - VALID_TARGETS)
    if unknown:
        raise ValueError(f"Unexpected loan_status values: {unknown}")

    # Remove impossible data-entry errors (not statistical outliers).
    before = len(cleaned)
    cleaned = cleaned[cleaned["person_age"] <= MAX_PLAUSIBLE_AGE]
    cleaned = cleaned[
        cleaned["person_emp_length"].isna()
        | (cleaned["person_emp_length"] <= MAX_PLAUSIBLE_EMP_LENGTH)
    ]
    stats["impossible_rows_removed"] = before - len(cleaned)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    stats["duplicate_rows_removed"] = before - len(cleaned)

    stats["null_emp_length"] = int(cleaned["person_emp_length"].isna().sum())
    stats["null_loan_int_rate"] = int(cleaned["loan_int_rate"].isna().sum())
    stats["output_rows"] = len(cleaned)
    return cleaned, stats


def write_report(df: pd.DataFrame, stats: dict[str, int], path: Path) -> None:
    target_counts = df[TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
    default_rate = 100.0 * target_counts.get(1, 0) / max(len(df), 1)
    lines = [
        "DAP391m ingestion and cleaning report",
        "=" * 40,
        f"Input: {INPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Output: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Input rows: {stats['input_rows']}",
        f"Impossible-value rows removed: {stats['impossible_rows_removed']}",
        f"Duplicate rows removed: {stats['duplicate_rows_removed']}",
        f"Output rows: {stats['output_rows']}",
        f"Columns: {len(df.columns)}",
        "",
        "Target (loan_status) counts:",
        f"  0 (repaid / non-default): {int(target_counts.get(0, 0))}",
        f"  1 (default):              {int(target_counts.get(1, 0))}",
        f"  default rate:             {default_rate:.1f}%",
        "",
        "Missing values preserved for pipeline imputation:",
        f"  person_emp_length nulls: {stats['null_emp_length']}",
        f"  loan_int_rate nulls:     {stats['null_loan_int_rate']}",
        "",
        "Leakage note:",
        "`loan_status` is a real observed outcome. All remaining columns are",
        "decision-time loan/applicant attributes (no synthetic target leakage).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    cleaned, stats = clean_data(raw)
    cleaned.to_csv(OUTPUT_CSV, index=False)
    write_report(cleaned, stats, REPORT_TXT)

    print(f"Loaded {INPUT_CSV.relative_to(PROJECT_ROOT)}: {raw.shape}")
    print(
        f"Removed {stats['impossible_rows_removed']} impossible-value rows, "
        f"{stats['duplicate_rows_removed']} duplicates"
    )
    print(f"Saved {OUTPUT_CSV.relative_to(PROJECT_ROOT)}: {cleaned.shape}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
