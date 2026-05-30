"""
DAP391m Project 8 - Ingestion & Cleaning
========================================

Loads the active modeling dataset (DataCo Smart Supply Chain) and prepares it
for late-delivery-risk prediction. The dataset is at order-ITEM grain
(~2.75 items per order); the target `Late_delivery_risk` is a real observed
outcome (the shipment was late or not).

The decisive cleaning step is dropping columns that would leak the outcome or
expose PII, with each drop documented by reason:

  * LEAKAGE (post-delivery, unknown at order time):
      Days for shipping (real), Delivery Status, shipping date (DateOrders),
      Order Status, and the realized-profit fields.
      `Late_delivery_risk` is, by construction, `Delivery Status == "Late
      delivery"` (exactly) and `real days > scheduled days` (~97.5%), so these
      columns reconstruct the label and MUST be removed.

  * PII: customer email, password, name, street, zipcode.

  * EMPTY / REDUNDANT: all-null or near-empty columns, image URLs, constant
      flags, and numeric ID twins of the kept *Name columns.

Everything else is a decision-time (order-time) attribute and is kept. The
identifier columns `Order Id` (group key) and `Order Item Id` are preserved so
downstream splits can be GROUP-AWARE on `Order Id` (items from one order share
the same outcome and must never straddle the train/test boundary).

Run:
    .venv/bin/python3 src-code/01_ingestion_cleaning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "dataco_raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered"
OUTPUT_CSV = OUTPUT_DIR / "clean_data.csv"
REPORT_TXT = OUTPUT_DIR / "cleaning_report.txt"

TARGET_COL = "Late_delivery_risk"
VALID_TARGETS = {0, 1}
GROUP_KEY = "Order Id"

# Columns that encode the outcome (known only AFTER delivery). Dropping these is
# the central integrity decision for this dataset.
LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "shipping date (DateOrders)",
    "Order Status",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
]

# Personally identifiable information — removed for privacy, not predictive.
PII_COLUMNS = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
]

# All-null / near-empty / non-informative columns and numeric ID twins of the
# kept *Name columns (Category Name, Department Name, Product Name).
EMPTY_OR_REDUNDANT_COLUMNS = [
    "Product Description",  # 100% null
    "Order Zipcode",  # ~86% null
    "Product Image",  # image URL, not a feature
    "Product Status",  # constant
    "Category Id",
    "Department Id",
    "Product Card Id",
    "Product Category Id",
    "Order Item Cardprod Id",
    "Customer Id",
    "Order Customer Id",
]

# A subset that must exist in the raw file before we trust / drop anything.
REQUIRED_COLUMNS = [
    TARGET_COL,
    GROUP_KEY,
    "Order Item Id",
    "Type",
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Customer Segment",
    "Market",
    "Order Region",
    "Department Name",
    "Category Name",
    "order date (DateOrders)",
]

NUMERIC_COLUMNS = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Product Price",
    "Sales",
    "Order Item Total",
    "Sales per customer",
]


def load_raw(path: Path = INPUT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {path}\n"
            "Download it with:\n"
            "  kaggle datasets download -d "
            "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis "
            "-p Data/dataco_raw --unzip"
        )
    # The DataCo CSV uses latin-1 (accented customer/city names).
    return pd.read_csv(path, encoding="latin-1")


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    validate_schema(df)
    cleaned = df.copy()
    stats: dict[str, object] = {
        "input_rows": len(cleaned),
        "input_cols": cleaned.shape[1],
    }

    # --- Drop leakage / PII / empty columns (only those actually present). ---
    leak_present = [c for c in LEAKAGE_COLUMNS if c in cleaned.columns]
    pii_present = [c for c in PII_COLUMNS if c in cleaned.columns]
    junk_present = [c for c in EMPTY_OR_REDUNDANT_COLUMNS if c in cleaned.columns]
    cleaned = cleaned.drop(columns=leak_present + pii_present + junk_present)
    stats["dropped_leakage"] = leak_present
    stats["dropped_pii"] = pii_present
    stats["dropped_redundant"] = junk_present

    # --- Type-fix. ---
    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Parse order date to ISO so downstream feature work is unambiguous.
    cleaned["order date (DateOrders)"] = pd.to_datetime(
        cleaned["order date (DateOrders)"], errors="coerce"
    )

    # --- Target must be present and binary {0, 1}. ---
    cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
    missing_target = int(cleaned[TARGET_COL].isna().sum())
    if missing_target:
        raise ValueError(f"Missing target values found: {missing_target}")
    cleaned[TARGET_COL] = cleaned[TARGET_COL].astype(int)
    unknown = sorted(set(cleaned[TARGET_COL].unique()) - VALID_TARGETS)
    if unknown:
        raise ValueError(f"Unexpected {TARGET_COL} values: {unknown}")

    # --- Remove impossible data-entry errors (not statistical outliers). ---
    before = len(cleaned)
    cleaned = cleaned[cleaned["Order Item Quantity"] > 0]
    cleaned = cleaned[cleaned["Sales"] >= 0]
    cleaned = cleaned[cleaned["Days for shipment (scheduled)"] >= 0]
    stats["impossible_rows_removed"] = before - len(cleaned)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    stats["duplicate_rows_removed"] = before - len(cleaned)

    stats["output_rows"] = len(cleaned)
    stats["output_cols"] = cleaned.shape[1]
    stats["unique_orders"] = int(cleaned[GROUP_KEY].nunique())
    return cleaned, stats


def write_report(df: pd.DataFrame, stats: dict[str, object], path: Path) -> None:
    target_counts = df[TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
    late_rate = 100.0 * target_counts.get(1, 0) / max(len(df), 1)
    items_per_order = stats["output_rows"] / max(int(stats["unique_orders"]), 1)
    lines = [
        "DAP391m ingestion and cleaning report",
        "=" * 40,
        f"Input:  {INPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Output: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}",
        f"Input rows x cols:  {stats['input_rows']} x {stats['input_cols']}",
        f"Output rows x cols: {stats['output_rows']} x {stats['output_cols']}",
        f"Impossible-value rows removed: {stats['impossible_rows_removed']}",
        f"Duplicate rows removed:        {stats['duplicate_rows_removed']}",
        "",
        "Grain (IMPORTANT for splitting):",
        f"  rows (order items): {stats['output_rows']}",
        f"  unique orders:      {stats['unique_orders']}",
        f"  items per order:    {items_per_order:.2f}",
        "  -> use a GROUP-AWARE split on `Order Id`; items from one order",
        "     share one outcome and must not straddle train/test.",
        "",
        f"Target ({TARGET_COL}) counts:",
        f"  0 (on-time / early): {int(target_counts.get(0, 0))}",
        f"  1 (late delivery):   {int(target_counts.get(1, 0))}",
        f"  late-delivery rate:  {late_rate:.1f}%",
        "",
        "Columns dropped to prevent OUTCOME LEAKAGE (post-delivery):",
        *(f"  - {c}" for c in stats["dropped_leakage"]),
        "",
        "Columns dropped as PII:",
        *(f"  - {c}" for c in stats["dropped_pii"]),
        "",
        "Columns dropped as empty / redundant:",
        *(f"  - {c}" for c in stats["dropped_redundant"]),
        "",
        "Leakage note:",
        f"`{TARGET_COL}` equals (Delivery Status == 'Late delivery') exactly and",
        "(real shipping days > scheduled days) ~97.5%. Those columns reconstruct",
        "the label and are removed. Kept columns are all known at order time",
        "(scheduled days, shipping mode, geography, product, customer segment).",
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
        f"Dropped {len(stats['dropped_leakage'])} leakage, "
        f"{len(stats['dropped_pii'])} PII, "
        f"{len(stats['dropped_redundant'])} empty/redundant columns"
    )
    print(
        f"Removed {stats['impossible_rows_removed']} impossible-value rows, "
        f"{stats['duplicate_rows_removed']} duplicates"
    )
    print(
        f"Saved {OUTPUT_CSV.relative_to(PROJECT_ROOT)}: {cleaned.shape} "
        f"({stats['unique_orders']} unique orders)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
