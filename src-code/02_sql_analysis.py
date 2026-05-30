"""
DAP391m Project 8 - SQL Analysis
================================

Builds an in-memory SQLite database from the cleaned credit dataset
(Data/filtered/clean_data.csv), executes the six SQL queries in
sql/analysis.sql, and saves each result under Data/filtered/sql_outputs/.

The credit dataset is a single flat table, so it is loaded as one SQLite
table named `credit`. The queries cover default-rate analysis by grade,
home ownership, intent, income band, loan-amount band, and a grade
penetration index.

Run:
    .venv/bin/python3 src-code/02_sql_analysis.py
    (run src-code/01_ingestion_cleaning.py first)
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
SQL_FILE = PROJECT_ROOT / "sql" / "analysis.sql"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "sql_outputs"

EXPECTED_QUERIES = 6


def load_credit() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"{CLEAN_CSV} not found. Run src-code/01_ingestion_cleaning.py first."
        )
    return pd.read_csv(CLEAN_CSV)


def split_sql_queries(sql_text: str) -> list[tuple[str, str]]:
    chunks = [chunk.strip() for chunk in sql_text.split(";") if chunk.strip()]
    queries: list[tuple[str, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        match = re.search(r"--\s*\d+\.\s*(.+)", chunk)
        title = match.group(1).strip() if match else f"query_{index}"
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        queries.append((f"{index:02d}_{slug}", chunk))
    return queries


def execute_queries(credit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(":memory:")
    try:
        credit.to_sql("credit", conn, index=False, if_exists="replace")
        queries = split_sql_queries(SQL_FILE.read_text(encoding="utf-8"))
        if len(queries) != EXPECTED_QUERIES:
            raise ValueError(
                f"Expected {EXPECTED_QUERIES} SQL queries, found {len(queries)}"
            )
        return {name: pd.read_sql_query(sql, conn) for name, sql in queries}
    finally:
        conn.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    credit = load_credit()
    outputs = execute_queries(credit)
    for name, result in outputs.items():
        out_path = OUTPUT_DIR / f"{name}.csv"
        result.to_csv(out_path, index=False)
        print(f"Saved {out_path.relative_to(PROJECT_ROOT)} ({len(result)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
