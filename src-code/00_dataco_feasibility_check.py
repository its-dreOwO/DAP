"""
DAP391m Project 8 - DataCo dataset feasibility check
====================================================

Read-only assessment of the DataCo Smart Supply Chain raw CSV before
changing the modeling pipeline. Does not modify raw data or existing outputs.

Run:
    .venv/Scripts/python.exe src-code/00_dataco_feasibility_check.py
    # or: .venv/bin/python3 src-code/00_dataco_feasibility_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "dataco_raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "dataco_feasibility_outputs"

PRIMARY_TARGET = "Late_delivery_risk"
ALT_TARGET_CANDIDATES = [
    "Late_delivery_risk",
    "Delivery Status",
    "late_delivery_risk",
]
SHIPPING_MODE_COL = "Shipping Mode"
GROUP_KEY = "Order Id"

# Same leakage / PII lists as 01_ingestion_cleaning.py (documented drops).
KNOWN_LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "shipping date (DateOrders)",
    "Order Status",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
]
KNOWN_PII_COLUMNS = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
]

LEAKAGE_NAME_PATTERNS = re.compile(
    r"(late|delay|delivery\s*status|shipping\s*\(real\)|real\s*days|"
    r"order\s*status|profit|benefit|shipped|delivered|actual)",
    re.IGNORECASE,
)


def load_dataco(path: Path = INPUT_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"DataCo dataset not found: {path}\n"
            "Place the Kaggle file at:\n"
            f"  {path.relative_to(PROJECT_ROOT)}\n"
            "Download:\n"
            "  kaggle datasets download -d "
            "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis "
            "-p Data/dataco_raw --unzip"
        )
    return pd.read_csv(path, encoding="latin-1")


def find_target_column(df: pd.DataFrame) -> str | None:
    for name in ALT_TARGET_CANDIDATES:
        if name in df.columns:
            return name
    return None


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null": non_null,
                "null_count": int(series.isna().sum()),
                "null_pct": round(100.0 * series.isna().mean(), 2),
                "nunique": int(series.nunique(dropna=True)),
                "sample_value": str(series.dropna().iloc[0])[:80] if non_null else "",
            }
        )
    return pd.DataFrame(rows)


def target_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    counts = df[target_col].value_counts(dropna=False).sort_index()
    total = len(df)
    rows = []
    for value, count in counts.items():
        rows.append(
            {
                "target_column": target_col,
                "class_value": value,
                "count": int(count),
                "pct": round(100.0 * count / total, 2),
            }
        )
    rows.append(
        {
            "target_column": target_col,
            "class_value": "TOTAL",
            "count": total,
            "pct": 100.0,
        }
    )
    return pd.DataFrame(rows)


def shipping_mode_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if SHIPPING_MODE_COL not in df.columns:
        return pd.DataFrame(columns=["shipping_mode", "count", "pct", "late_rate_pct"])
    total = len(df)
    counts = df[SHIPPING_MODE_COL].value_counts(dropna=False)
    late_col = PRIMARY_TARGET if PRIMARY_TARGET in df.columns else None
    rows = []
    for mode, count in counts.items():
        row = {
            "shipping_mode": mode,
            "count": int(count),
            "pct": round(100.0 * count / total, 2),
        }
        if late_col:
            subset = df[df[SHIPPING_MODE_COL] == mode]
            row["late_rate_pct"] = round(100.0 * subset[late_col].mean(), 1)
        else:
            row["late_rate_pct"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def heuristic_leakage_candidates(df: pd.DataFrame, target_col: str | None) -> list[str]:
    found: list[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        if col in KNOWN_LEAKAGE_COLUMNS:
            found.append(col)
            continue
        if LEAKAGE_NAME_PATTERNS.search(col):
            found.append(col)
    return sorted(set(found))


def leakage_verification_notes(df: pd.DataFrame, target_col: str) -> list[str]:
    notes: list[str] = []
    if target_col not in df.columns:
        return ["Target column not found; skip cross-checks."]

    target = pd.to_numeric(df[target_col], errors="coerce")
    if "Delivery Status" in df.columns:
        status_late = df["Delivery Status"].astype(str).str.strip().eq("Late delivery")
        match_pct = 100.0 * (target.eq(1) == status_late).mean()
        notes.append(
            f"Late_delivery_risk == (Delivery Status == 'Late delivery'): "
            f"{match_pct:.2f}% agreement"
        )

    real_col = "Days for shipping (real)"
    sched_col = "Days for shipment (scheduled)"
    if real_col in df.columns and sched_col in df.columns:
        real = pd.to_numeric(df[real_col], errors="coerce")
        sched = pd.to_numeric(df[sched_col], errors="coerce")
        implied_late = real > sched
        valid = target.notna() & implied_late.notna()
        if valid.any():
            agree = 100.0 * (target[valid].eq(1) == implied_late[valid]).mean()
            notes.append(
                f"Late_delivery_risk == (real days > scheduled days): "
                f"{agree:.2f}% agreement (where comparable)"
            )

    if SHIPPING_MODE_COL in df.columns and sched_col in df.columns:
        mode_sched = (
            df.groupby(SHIPPING_MODE_COL)[sched_col].nunique(dropna=True).to_dict()
        )
        notes.append(
            f"Unique scheduled-day values per shipping mode: {mode_sched} "
            "(1 per mode => perfect collinearity)"
        )

    return notes


def write_leakage_report(
    path: Path,
    df_columns: set[str],
    present_known: list[str],
    heuristic: list[str],
    verification: list[str],
) -> None:
    pii_present = [c for c in KNOWN_PII_COLUMNS if c in df_columns]
    extra_heuristic = [c for c in heuristic if c not in present_known]
    lines = [
        "DataCo leakage candidate report",
        "=" * 40,
        "",
        "KNOWN post-outcome columns (drop if target is Late_delivery_risk):",
        *(f"  - {c}" for c in present_known),
        "",
        "Heuristic name-based candidates (review; may overlap above):",
        *(f"  - {c}" for c in extra_heuristic),
        "",
        "Empirical cross-checks:",
        *(f"  - {n}" for n in verification),
        "",
        "PII columns (drop for privacy):",
        *(f"  - {c}" for c in pii_present),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_recommendation(df: pd.DataFrame, target_col: str | None) -> None:
    print("\n" + "=" * 60)
    print("FEASIBILITY RECOMMENDATION")
    print("=" * 60)

    has_target = target_col == PRIMARY_TARGET and target_col in df.columns
    has_shipping = SHIPPING_MODE_COL in df.columns
    n_orders = int(df[GROUP_KEY].nunique()) if GROUP_KEY in df.columns else None

    suitable_late = has_target and len(df) > 1000
    print(f"Late delivery prediction suitable: {'YES' if suitable_late else 'NO'}")
    if suitable_late:
        late_rate = 100.0 * df[PRIMARY_TARGET].mean()
        print(
            f"  - Binary target `{PRIMARY_TARGET}` present; "
            f"late rate ~{late_rate:.1f}% (near-balanced)."
        )
        print(
            "  - Use GroupShuffleSplit on `Order Id` "
            f"({n_orders:,} orders, {len(df):,} line-items)."
        )
        print(
            "  - Expect weak ROC lift beyond shipping-mode base rates "
            "(see report/dataco_dataset_assessment.md)."
        )

    print(f"Shipping Mode testing suitable: " f"{'YES' if has_shipping else 'NO'}")
    if has_shipping:
        print(
            f"  - Column `{SHIPPING_MODE_COL}` present with "
            f"{df[SHIPPING_MODE_COL].nunique()} modes; "
            "strong predictor but collinear with scheduled days."
        )

    print(
        "Supplier / procurement framing: PARTIAL — retail order data "
        "(departments, categories, markets); no supplier master table."
    )

    print(
        f"Four-model training (LR, DT, RF, XGBoost): "
        f"{'YES' if suitable_late else 'NO'} — needs binary target and "
        "leakage columns removed before modeling."
    )

    print(
        f"\nRecommended target: `{PRIMARY_TARGET}`"
        if has_target
        else "\nRecommended target: TBD"
    )
    print("\nColumns to drop as leakage (if target is Late_delivery_risk):")
    for col in KNOWN_LEAKAGE_COLUMNS:
        mark = " [present]" if col in df.columns else " [absent]"
        print(f"  - {col}{mark}")
    print("\nAlso drop PII columns listed in 01_ingestion_cleaning.py.")
    print("Disclose that DataCo is a simulated teaching dataset in the report.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {INPUT_CSV.relative_to(PROJECT_ROOT)}")
    df = load_dataco()
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns ({len(df.columns)}):")
    for name in df.columns:
        print(f"  - {name}")

    target_col = find_target_column(df)
    print(f"\nLikely target column: {target_col or 'NOT FOUND'}")
    if "Delivery Status" in df.columns:
        print(
            "Alternate / label source: Delivery Status (categorical — drop as leakage)"
        )
    if SHIPPING_MODE_COL in df.columns:
        print(
            f"Shipping Mode column: {SHIPPING_MODE_COL} "
            f"({df[SHIPPING_MODE_COL].nunique()} values)"
        )
    else:
        print(f"Shipping Mode column: NOT FOUND (expected `{SHIPPING_MODE_COL}`)")

    if target_col:
        print(f"\nClass distribution for `{target_col}`:")
        print(df[target_col].value_counts(dropna=False).to_string())
    else:
        print("\nClass distribution: skipped (no target column)")

    if SHIPPING_MODE_COL in df.columns:
        print("\nShipping Mode distribution:")
        print(df[SHIPPING_MODE_COL].value_counts(dropna=False).to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nColumns with missing values: {len(missing)}")
    if len(missing):
        print(missing.head(15).to_string())

    present_known = [c for c in KNOWN_LEAKAGE_COLUMNS if c in df.columns]
    heuristic = heuristic_leakage_candidates(df, target_col)
    verification = (
        leakage_verification_notes(df, PRIMARY_TARGET)
        if PRIMARY_TARGET in df.columns
        else []
    )

    print("\nLeakage candidates (known + heuristic):")
    for col in sorted(set(present_known) | set(heuristic)):
        print(f"  - {col}")

    # --- Save outputs (new directory only; does not overwrite other artifacts). ---
    column_summary(df).to_csv(OUTPUT_DIR / "dataco_column_summary.csv", index=False)
    miss_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (100.0 * df.isna().mean()).round(2).values,
        }
    )
    miss_df = miss_df[miss_df["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    miss_df.to_csv(OUTPUT_DIR / "dataco_missing_values.csv", index=False)

    if target_col:
        target_distribution(df, target_col).to_csv(
            OUTPUT_DIR / "dataco_target_distribution.csv", index=False
        )
    shipping_mode_distribution(df).to_csv(
        OUTPUT_DIR / "dataco_shipping_mode_distribution.csv", index=False
    )

    write_leakage_report(
        OUTPUT_DIR / "dataco_leakage_candidates.txt",
        set(df.columns),
        present_known,
        heuristic,
        verification,
    )

    print(f"\nSaved outputs under: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")
    print_recommendation(df, target_col)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
