"""
DAP391m Project 8 - Exploratory Data Analysis
=============================================

Reads Data/filtered/clean_data.csv (consumer credit-default risk) and saves
focused EDA tables/figures under Data/filtered/eda_outputs/.

Unlike the previous dataset, there is genuine predictive signal here, so the
correlation check is used to confirm signal (not to hunt for leakage).

Run:
    .venv/bin/python3 src-code/03_eda.py
    (run src-code/01_ingestion_cleaning.py first)
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

TARGET_COL = "loan_status"
TARGET_LABELS = {0: "Non-default", 1: "Default"}
CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def load_clean_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run src-code/01_ingestion_cleaning.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in {INPUT_CSV.name}")
    return df


def save_class_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
    balance = pd.DataFrame(
        {
            "loan_status": [TARGET_LABELS[i] for i in counts.index],
            "count": counts.values,
            "percent": (counts.values / len(df) * 100).round(2),
        }
    )
    balance.to_csv(OUTPUT_DIR / "class_balance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=balance, x="loan_status", y="count", ax=ax)
    ax.set_title("Loan Status Class Balance")
    ax.set_xlabel("Loan status")
    ax.set_ylabel("Rows")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def save_numeric_distributions(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number").drop(columns=[TARGET_COL])
    numeric.describe().T.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    cols = list(numeric.columns)
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, nrows * 3)))
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, col in zip(axes_flat, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    for ax in axes_flat[len(cols) :]:
        ax.axis("off")

    fig.suptitle("Numeric Feature Distributions", y=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=150)
    plt.close(fig)


def save_signal_heatmap(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)
    corr.to_csv(OUTPUT_DIR / "feature_correlation_matrix.csv")

    # Correlation of each numeric feature with the target (signal confirmation).
    target_corr = (
        corr[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    )
    target_corr.to_frame("corr_with_loan_status").to_csv(
        OUTPUT_DIR / "feature_target_correlation.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix (incl. loan_status)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def save_default_rate_by_category(df: pd.DataFrame) -> None:
    frames = []
    for col in CATEGORICAL_COLS:
        grp = (
            df.groupby(col)[TARGET_COL]
            .agg(total="size", defaults="sum")
            .assign(default_rate=lambda g: (g["defaults"] / g["total"]).round(4))
            .reset_index()
            .rename(columns={col: "category_value"})
        )
        grp.insert(0, "feature", col)
        frames.append(grp)
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "default_rate_by_category.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), CATEGORICAL_COLS):
        rate = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
        sns.barplot(x=rate.values, y=rate.index, ax=ax)
        ax.set_title(f"Default rate by {col}")
        ax.set_xlabel("Default rate")
        ax.set_ylabel("")
    fig.suptitle("Default Rate by Categorical Feature", y=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "default_rate_by_category.png", dpi=150)
    plt.close(fig)


def save_target_numeric_summary(df: pd.DataFrame) -> None:
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != TARGET_COL
    ]
    summary = (
        df.groupby(TARGET_COL)[numeric_cols].mean().round(3).rename(index=TARGET_LABELS)
    )
    summary.to_csv(OUTPUT_DIR / "numeric_means_by_loan_status.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()

    save_class_balance(df)
    save_numeric_distributions(df)
    save_signal_heatmap(df)
    save_default_rate_by_category(df)
    save_target_numeric_summary(df)

    print(f"EDA outputs saved to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
