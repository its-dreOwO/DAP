"""
DAP391m Project 8 - SQL Analysis
================================

Builds an in-memory SQLite database from the business-analysis CSV files,
normalizes the columns expected by sql/analysis.sql, executes all six SQL
queries, and saves each result under Data/filtered/sql_outputs/.

Run:
    .venv/bin/python3 src-code/02_sql_analysis.py
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
SQL_FILE = PROJECT_ROOT / "sql" / "analysis.sql"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "sql_outputs"

CUSTOMER_CSV = DATA_DIR / "customer.csv"
SHIPMENT_CSV = DATA_DIR / "shipment.csv"
LOGISTICS_CSV = DATA_DIR / "logistics_performance.csv"

COUNTRY_TO_REGION = {
    "USA": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Germany": "Europe",
    "France": "Europe",
    "UK": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "China": "Asia-Pacific",
    "Japan": "Asia-Pacific",
    "India": "Asia-Pacific",
    "South Korea": "Asia-Pacific",
    "Bangladesh": "Asia-Pacific",
    "Indonesia": "Asia-Pacific",
    "Philippines": "Asia-Pacific",
    "Thailand": "Asia-Pacific",
    "Vietnam": "Asia-Pacific",
    "UAE": "Middle East",
    "Saudi Arabia": "Middle East",
    "South Africa": "Africa",
    "Egypt": "Africa",
    "Nigeria": "Africa",
}


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype("string").str.strip()
    return out


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customer = _strip_strings(pd.read_csv(CUSTOMER_CSV))
    shipment = _strip_strings(pd.read_csv(SHIPMENT_CSV))
    logistics = _strip_strings(pd.read_csv(LOGISTICS_CSV))
    return customer, shipment, logistics


def normalize_customer(customer: pd.DataFrame) -> pd.DataFrame:
    required = {"supplier_id", "lead_time_days"}
    missing = sorted(required - set(customer.columns))
    if missing:
        raise ValueError(f"customer.csv missing required columns: {missing}")

    customer["lead_time_days"] = pd.to_numeric(
        customer["lead_time_days"], errors="coerce"
    )
    supplier_dim = (
        customer.dropna(subset=["supplier_id", "lead_time_days"])
        .groupby("supplier_id", as_index=False)
        .agg(
            lead_time_days=("lead_time_days", "mean"),
            customer_count=("customer_id", "count"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_order_value=("order_value_usd", "mean"),
        )
    )
    return supplier_dim


def normalize_shipment(
    shipment: pd.DataFrame,
    customer_raw: pd.DataFrame,
    logistics: pd.DataFrame,
) -> pd.DataFrame:
    required = {"date", "delivery_status"}
    missing = sorted(required - set(shipment.columns))
    if missing:
        raise ValueError(f"shipment.csv missing required columns: {missing}")

    out = shipment.copy()
    out = out.rename(columns={"date": "shipment_date"})
    out["shipment_date"] = pd.to_datetime(out["shipment_date"], errors="coerce")
    if out["shipment_date"].isna().any():
        raise ValueError("shipment.csv contains invalid shipment dates")
    out["shipment_date"] = out["shipment_date"].dt.strftime("%Y-%m-%d")

    if "supplier_id" not in out.columns:
        supplier_sequence = customer_raw["supplier_id"].dropna().astype(str).tolist()
        if not supplier_sequence:
            raise ValueError("Cannot infer shipment.supplier_id from customer.csv")
        repeated = np.resize(np.array(supplier_sequence, dtype=object), len(out))
        out["supplier_id"] = repeated

    if "carrier" not in out.columns:
        out["region"] = (
            out.get("D_Country", pd.Series(index=out.index, dtype="string"))
            .astype("string")
            .str.strip()
            .map(COUNTRY_TO_REGION)
            .fillna("North America")
        )
        out["carrier"] = assign_carriers(out, logistics)

    out["delivery_status"] = out["delivery_status"].str.strip()
    return out


def assign_carriers(shipment: pd.DataFrame, logistics: pd.DataFrame) -> list[str]:
    logistics = logistics.dropna(subset=["carrier"]).copy()
    if logistics.empty:
        return ["Unknown"] * len(shipment)

    carriers_by_region = {
        region: group["carrier"].dropna().astype(str).tolist()
        for region, group in logistics.groupby("region")
    }
    all_carriers = logistics["carrier"].dropna().astype(str).tolist()

    assigned: list[str] = []
    counters: dict[str, int] = {}
    for _, row in shipment.iterrows():
        region = str(row.get("region", ""))
        choices = carriers_by_region.get(region, all_carriers)
        position = counters.get(region, 0)
        assigned.append(choices[position % len(choices)])
        counters[region] = position + 1
    return assigned


def split_sql_queries(sql_text: str) -> list[tuple[str, str]]:
    chunks = [chunk.strip() for chunk in sql_text.split(";") if chunk.strip()]
    queries: list[tuple[str, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        match = re.search(r"--\s*\d+\.\s*(.+)", chunk)
        title = match.group(1).strip() if match else f"query_{index}"
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        queries.append((f"{index:02d}_{slug}", chunk))
    return queries


def execute_queries(
    customer: pd.DataFrame,
    shipment: pd.DataFrame,
    logistics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    conn = sqlite3.connect(":memory:")
    try:
        customer.to_sql("customer", conn, index=False, if_exists="replace")
        shipment.to_sql("shipment", conn, index=False, if_exists="replace")
        logistics.to_sql(
            "logistics_performance", conn, index=False, if_exists="replace"
        )

        queries = split_sql_queries(SQL_FILE.read_text(encoding="utf-8"))
        if len(queries) != 6:
            raise ValueError(f"Expected 6 SQL queries, found {len(queries)}")

        return {name: pd.read_sql_query(sql, conn) for name, sql in queries}
    finally:
        conn.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    customer_raw, shipment_raw, logistics = load_sources()
    customer = normalize_customer(customer_raw)
    shipment = normalize_shipment(shipment_raw, customer_raw, logistics)

    outputs = execute_queries(customer, shipment, logistics)
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
