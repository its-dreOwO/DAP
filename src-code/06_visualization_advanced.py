"""
DAP391m Project 8 - Advanced Visualization Exports
==================================================

Creates Plotly HTML dashboards and CSV scorecards from cleaned data plus model
outputs. Files are saved under Data/filtered/visualization_outputs/.

Run:
    .venv/bin/python3 src-code/06_visualization_advanced.py
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

LABEL_ORDER = ["Low", "Medium", "High"]
RISK_SCORE = {"Low": 0, "Medium": 1, "High": 2}


def load_clean_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run src-code/01_ingestion_cleaning.py first."
        )
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
    df["risk_label"] = df["risk_label"].astype(str).str.strip()
    df["risk_score"] = df["risk_label"].map(RISK_SCORE)
    return df


def build_supplier_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    scorecard = (
        df.groupby("supplier_id", as_index=False)
        .agg(
            shipments=("supplier_id", "size"),
            high_risk_shipments=("risk_label", lambda s: int((s == "High").sum())),
            medium_risk_shipments=(
                "risk_label",
                lambda s: int((s == "Medium").sum()),
            ),
            high_risk_rate=("risk_label", lambda s: float((s == "High").mean())),
            avg_risk_score=("risk_score", "mean"),
            avg_risk_probability=("risk_probability", "mean"),
            avg_lead_time_days=("supplier_lead_time_days", "mean"),
            std_lead_time_days=("supplier_lead_time_days", "std"),
            avg_port_delay_days=("port_delay_days", "mean"),
            avg_quality_score=("supplier_quality_score", "mean"),
            avg_reliability_index=("supplier_reliability_index", "mean"),
            avg_weather_disruption=("weather_disruption_score", "mean"),
        )
        .round(4)
    )
    scorecard["risk_band"] = pd.cut(
        scorecard["high_risk_rate"],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["Monitor", "Elevated", "Critical"],
    )
    return scorecard.sort_values(
        ["high_risk_rate", "avg_risk_probability", "avg_lead_time_days"],
        ascending=False,
    )


def save_supplier_ranking(scorecard: pd.DataFrame) -> None:
    ranking = scorecard.head(25).copy()
    ranking.to_csv(OUTPUT_DIR / "supplier_risk_ranking.csv", index=False)

    fig = px.bar(
        ranking.sort_values("high_risk_rate"),
        x="high_risk_rate",
        y="supplier_id",
        color="risk_band",
        orientation="h",
        hover_data=[
            "shipments",
            "avg_risk_probability",
            "avg_lead_time_days",
            "avg_quality_score",
            "avg_reliability_index",
        ],
        title="Top Supplier Risk Ranking",
        labels={
            "high_risk_rate": "High-risk shipment rate",
            "supplier_id": "Supplier",
            "risk_band": "Risk band",
        },
    )
    fig.update_layout(height=800)
    fig.write_html(OUTPUT_DIR / "supplier_risk_ranking.html", include_plotlyjs="cdn")


def save_risk_trend(df: pd.DataFrame) -> None:
    trend = (
        df.assign(month=df["timestamp"].dt.to_period("M").astype(str))
        .groupby(["month", "risk_label"], as_index=False)
        .size()
    )
    totals = trend.groupby("month")["size"].transform("sum")
    trend["share"] = trend["size"] / totals
    trend.to_csv(OUTPUT_DIR / "monthly_risk_trend.csv", index=False)

    fig = px.line(
        trend,
        x="month",
        y="share",
        color="risk_label",
        category_orders={"risk_label": LABEL_ORDER},
        markers=True,
        title="Monthly Risk Label Trend",
        labels={"share": "Share of shipments", "month": "Month"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.write_html(OUTPUT_DIR / "risk_trend.html", include_plotlyjs="cdn")


def save_lead_time_distribution(df: pd.DataFrame) -> None:
    fig = px.box(
        df,
        x="risk_label",
        y="supplier_lead_time_days",
        color="risk_label",
        category_orders={"risk_label": LABEL_ORDER},
        points="outliers",
        title="Lead-Time Distribution by Risk Label",
        labels={
            "risk_label": "Risk label",
            "supplier_lead_time_days": "Supplier lead time days",
        },
    )
    fig.write_html(OUTPUT_DIR / "lead_time_distribution.html", include_plotlyjs="cdn")


def save_model_summary_export() -> None:
    comparison_path = MODEL_OUTPUT_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        return
    comparison = pd.read_csv(comparison_path)
    comparison.to_csv(OUTPUT_DIR / "model_comparison_for_dashboard.csv", index=False)

    metric_cols = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "macro_pr_auc",
        "roc_auc_ovr_macro",
    ]
    available = [col for col in metric_cols if col in comparison.columns]
    long_df = comparison.melt(
        id_vars="model",
        value_vars=available,
        var_name="metric",
        value_name="score",
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
    scorecard = build_supplier_scorecard(df)
    scorecard.to_csv(OUTPUT_DIR / "supplier_scorecard.csv", index=False)

    save_supplier_ranking(scorecard)
    save_risk_trend(df)
    save_lead_time_distribution(df)
    save_model_summary_export()

    print(f"Visualization outputs saved to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
