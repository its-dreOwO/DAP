"""
DAP391m Project 8 - Exploratory Data Analysis
=============================================

Reads Data/filtered/clean_data.csv and saves focused EDA tables/figures under
Data/filtered/eda_outputs/.

Run:
    .venv/bin/python3 src-code/03_eda.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "eda_outputs"

TARGET_COL = "risk_label"
LABEL_ORDER = ["Low", "Medium", "High"]


def load_clean_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run src-code/01_ingestion_cleaning.py first."
        )
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in {INPUT_CSV.name}")
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    return df


def save_class_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts().reindex(LABEL_ORDER, fill_value=0)
    balance = pd.DataFrame(
        {
            TARGET_COL: counts.index,
            "count": counts.values,
            "percent": (counts.values / len(df) * 100).round(2),
        }
    )
    balance.to_csv(OUTPUT_DIR / "class_balance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=balance, x=TARGET_COL, y="count", order=LABEL_ORDER, ax=ax)
    ax.set_title("Risk Label Class Balance")
    ax.set_xlabel("Risk label")
    ax.set_ylabel("Rows")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def save_numeric_distributions(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    numeric.describe().T.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    cols = list(numeric.columns)
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, nrows * 3)))
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, col in zip(axes_flat, cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    for ax in axes_flat[len(cols) :]:
        ax.axis("off")

    fig.suptitle("Numeric Feature Distributions", y=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=150)
    plt.close(fig)


def save_leakage_heatmap(df: pd.DataFrame) -> None:
    columns = [
        "risk_probability",
        "port_delay_days",
        "supplier_lead_time_days",
        "supplier_quality_score",
        "supplier_reliability_index",
        "weather_disruption_score",
        "market_demand_index",
        "fuel_price_index",
    ]
    available = [col for col in columns if col in df.columns]
    corr = df[available].corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / "leakage_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Leakage and Supplier-Signal Correlation Check")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "leakage_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def save_supplier_outputs(df: pd.DataFrame) -> None:
    supplier = (
        df.groupby("supplier_id", as_index=False)
        .agg(
            shipments=("supplier_id", "size"),
            avg_lead_time_days=("supplier_lead_time_days", "mean"),
            std_lead_time_days=("supplier_lead_time_days", "std"),
            avg_quality_score=("supplier_quality_score", "mean"),
            avg_reliability_index=("supplier_reliability_index", "mean"),
            high_risk_rate=(
                TARGET_COL,
                lambda s: (s == "High").mean(),
            ),
        )
        .sort_values(["high_risk_rate", "avg_lead_time_days"], ascending=False)
    )
    supplier.to_csv(OUTPUT_DIR / "supplier_risk_summary.csv", index=False)

    top_suppliers = supplier.head(20)["supplier_id"]
    plot_df = df[df["supplier_id"].isin(top_suppliers)]
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=plot_df,
        x="supplier_id",
        y="supplier_lead_time_days",
        hue=TARGET_COL,
        hue_order=LABEL_ORDER,
        ax=ax,
    )
    ax.set_title("Supplier Lead-Time Spread (Top 20 Risk Suppliers)")
    ax.set_xlabel("Supplier")
    ax.set_ylabel("Lead time days")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "supplier_lead_time_spread.png", dpi=150)
    plt.close(fig)


def save_risk_label_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(TARGET_COL)
        .agg(
            rows=(TARGET_COL, "size"),
            avg_lead_time_days=("supplier_lead_time_days", "mean"),
            avg_port_delay_days=("port_delay_days", "mean"),
            avg_risk_probability=("risk_probability", "mean"),
            avg_quality_score=("supplier_quality_score", "mean"),
            avg_reliability_index=("supplier_reliability_index", "mean"),
            avg_weather_disruption=("weather_disruption_score", "mean"),
        )
        .reindex(LABEL_ORDER)
        .round(4)
    )
    summary.to_csv(OUTPUT_DIR / "risk_label_summary.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()

    save_class_balance(df)
    save_numeric_distributions(df)
    save_leakage_heatmap(df)
    save_supplier_outputs(df)
    save_risk_label_summary(df)

    print(f"EDA outputs saved to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
