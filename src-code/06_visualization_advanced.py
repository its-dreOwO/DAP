"""
DAP391m Project 8 - Advanced Visualization Exports
==================================================

Creates Plotly HTML dashboards and CSV scorecards from the cleaned credit
dataset plus model outputs. Files are saved under
Data/filtered/visualization_outputs/.

Run:
    .venv/bin/python3 src-code/06_visualization_advanced.py
    (run 01_ingestion_cleaning.py and 05_modeling.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "filtered"
INPUT_CSV = DATA_DIR / "clean_data.csv"
MODEL_OUTPUT_DIR = DATA_DIR / "model_outputs"
OUTPUT_DIR = DATA_DIR / "visualization_outputs"

GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


def load_clean_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run src-code/01_ingestion_cleaning.py first."
        )
    return pd.read_csv(INPUT_CSV)


def build_grade_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    scorecard = (
        df.groupby("loan_grade", as_index=False)
        .agg(
            total_loans=("loan_status", "size"),
            defaults=("loan_status", "sum"),
            default_rate=("loan_status", "mean"),
            avg_int_rate=("loan_int_rate", "mean"),
            avg_loan_amnt=("loan_amnt", "mean"),
            avg_loan_pct_income=("loan_percent_income", "mean"),
        )
        .round(4)
    )
    scorecard["risk_band"] = pd.cut(
        scorecard["default_rate"],
        bins=[-0.01, 0.20, 0.50, 1.0],
        labels=["Monitor", "Elevated", "Critical"],
    )
    return scorecard.sort_values("default_rate", ascending=False)


def save_grade_ranking(scorecard: pd.DataFrame) -> None:
    scorecard.to_csv(OUTPUT_DIR / "grade_risk_scorecard.csv", index=False)
    fig = px.bar(
        scorecard.sort_values("default_rate"),
        x="default_rate",
        y="loan_grade",
        color="risk_band",
        orientation="h",
        hover_data=["total_loans", "defaults", "avg_int_rate", "avg_loan_amnt"],
        title="Default Rate by Loan Grade",
        labels={"default_rate": "Default rate", "loan_grade": "Loan grade"},
    )
    fig.update_xaxes(tickformat=".0%")
    fig.write_html(OUTPUT_DIR / "grade_risk_ranking.html", include_plotlyjs="cdn")


def save_intent_default_rate(df: pd.DataFrame) -> None:
    intent = (
        df.groupby("loan_intent", as_index=False)["loan_status"]
        .mean()
        .rename(columns={"loan_status": "default_rate"})
        .sort_values("default_rate", ascending=False)
    )
    intent.to_csv(OUTPUT_DIR / "intent_default_rate.csv", index=False)
    fig = px.bar(
        intent,
        x="loan_intent",
        y="default_rate",
        color="default_rate",
        color_continuous_scale="Reds",
        title="Default Rate by Loan Intent",
        labels={"loan_intent": "Loan intent", "default_rate": "Default rate"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.write_html(OUTPUT_DIR / "intent_default_rate.html", include_plotlyjs="cdn")


def save_income_default_heatmap(df: pd.DataFrame) -> None:
    tmp = df.copy()
    tmp["income_band"] = pd.cut(
        tmp["person_income"],
        bins=[-1, 30000, 60000, 100000, float("inf")],
        labels=["<30k", "30k-60k", "60k-100k", "100k+"],
    )
    pivot = (
        tmp.pivot_table(
            index="income_band",
            columns="loan_grade",
            values="loan_status",
            aggfunc="mean",
            observed=False,
        )
        .reindex(columns=GRADE_ORDER)
        .round(3)
    )
    pivot.to_csv(OUTPUT_DIR / "income_grade_default_matrix.csv")
    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Reds",
        title="Default Rate by Income Band × Loan Grade",
        labels={"color": "Default rate"},
    )
    fig.write_html(
        OUTPUT_DIR / "income_grade_default_matrix.html", include_plotlyjs="cdn"
    )


def save_loan_amount_distribution(df: pd.DataFrame) -> None:
    plot_df = df.assign(status=df["loan_status"].map({0: "Non-default", 1: "Default"}))
    fig = px.box(
        plot_df,
        x="status",
        y="loan_percent_income",
        color="status",
        points="outliers",
        title="Loan-to-Income Ratio by Loan Status",
        labels={"status": "Loan status", "loan_percent_income": "Loan / income"},
    )
    fig.write_html(
        OUTPUT_DIR / "loan_pct_income_distribution.html", include_plotlyjs="cdn"
    )


def save_model_summary_export() -> None:
    comparison_path = MODEL_OUTPUT_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    comparison.to_csv(OUTPUT_DIR / "model_comparison_for_dashboard.csv", index=False)

    metric_cols = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    available = [col for col in metric_cols if col in comparison.columns]
    long_df = comparison.melt(
        id_vars="model", value_vars=available, var_name="metric", value_name="score"
    )
    fig = px.bar(
        long_df,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Model Comparison for Reporting",
        labels={"model": "Model", "score": "Score"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.write_html(OUTPUT_DIR / "model_comparison.html", include_plotlyjs="cdn")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()

    scorecard = build_grade_scorecard(df)
    save_grade_ranking(scorecard)
    save_intent_default_rate(df)
    save_income_default_heatmap(df)
    save_loan_amount_distribution(df)
    save_model_summary_export()

    print(f"Visualization outputs saved to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
