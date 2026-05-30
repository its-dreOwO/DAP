"""Interactive data-quality dashboard — before vs after cleaning.

Run from project root:
    .venv/bin/python3 -m streamlit run app/data_quality_dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "Data"
CLEAN_DIR = PROJECT_ROOT / "Data" / "filtered"

DATASETS = {
    "credit_risk": ("credit_risk_dataset.csv", "clean_data.csv"),
}

st.set_page_config(page_title="DAP391m — Data Quality Dashboard", layout="wide")
st.title("Data Quality Dashboard — Before vs After Cleaning")
st.caption(
    "DAP391m Project 8 · Group 8 · compares raw `Data/*.csv` against "
    "`Data/filtered/*_clean.csv` with IQR, Z-score, range, and distribution diffs."
)


@st.cache_data(show_spinner=False)
def load_pair(raw_name: str, clean_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_DIR / raw_name)
    clean = pd.read_csv(CLEAN_DIR / clean_name)
    return raw, clean


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column numeric summary: count, nulls, mean, std, min, Q1, median, Q3,
    max, range, IQR, IQR-outliers, |Z|>3 count, skew, kurtosis."""
    rows = []
    for col in numeric_columns(df):
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_outliers = int(((s < lo_fence) | (s > hi_fence)).sum())
        std = s.std(ddof=0)
        z = (s - s.mean()) / std if std > 0 else pd.Series(0, index=s.index)
        z_outliers = int((z.abs() > 3).sum())
        rows.append(
            {
                "column": col,
                "count": int(s.size),
                "nulls": int(df[col].isna().sum()),
                "mean": s.mean(),
                "std": std,
                "min": s.min(),
                "Q1": q1,
                "median": s.median(),
                "Q3": q3,
                "max": s.max(),
                "range": s.max() - s.min(),
                "IQR": iqr,
                "IQR_outliers": iqr_outliers,
                "|Z|>3": z_outliers,
                "skew": s.skew(),
                "kurtosis": s.kurtosis(),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def overview_metrics(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows (raw)", f"{len(raw):,}")
    c2.metric("Rows (clean)", f"{len(clean):,}", f"{len(clean) - len(raw):+,}")
    c3.metric("Columns (raw)", raw.shape[1])
    c4.metric("Columns (clean)", clean.shape[1])
    null_raw = int(raw.isna().sum().sum())
    null_clean = int(clean.isna().sum().sum())
    c5.metric("Total nulls", f"{null_clean:,}", f"{null_clean - null_raw:+,}")


def dtype_diff(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(raw.columns) | set(clean.columns))
    return pd.DataFrame(
        {
            "raw_dtype": [str(raw[c].dtype) if c in raw else "—" for c in cols],
            "clean_dtype": [str(clean[c].dtype) if c in clean else "—" for c in cols],
            "raw_nulls": [int(raw[c].isna().sum()) if c in raw else 0 for c in cols],
            "clean_nulls": [
                int(clean[c].isna().sum()) if c in clean else 0 for c in cols
            ],
        },
        index=cols,
    )


def stats_diff(raw_stats: pd.DataFrame, clean_stats: pd.DataFrame) -> pd.DataFrame:
    """Aligned numeric diff (clean − raw) on shared columns."""
    common = raw_stats.index.intersection(clean_stats.index)
    diff = clean_stats.loc[common] - raw_stats.loc[common]
    diff.columns = [f"Δ {c}" for c in diff.columns]
    return diff


def distribution_figure(raw: pd.DataFrame, clean: pd.DataFrame, col: str) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"{col} — histogram", f"{col} — box (IQR fences)"),
        column_widths=[0.6, 0.4],
    )
    fig.add_trace(
        go.Histogram(
            x=raw[col].dropna(),
            name="raw",
            opacity=0.55,
            marker_color="#d62728",
            nbinsx=40,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=clean[col].dropna(),
            name="clean",
            opacity=0.55,
            marker_color="#2ca02c",
            nbinsx=40,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(y=raw[col].dropna(), name="raw", marker_color="#d62728"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Box(y=clean[col].dropna(), name="clean", marker_color="#2ca02c"),
        row=1,
        col=2,
    )
    fig.update_layout(barmode="overlay", height=420, legend_title_text="")
    return fig


def zscore_scatter(series: pd.Series, name: str) -> go.Figure:
    s = series.dropna().reset_index(drop=True)
    std = s.std(ddof=0)
    z = (s - s.mean()) / std if std > 0 else pd.Series(0, index=s.index)
    colors = np.where(z.abs() > 3, "#d62728", "#1f77b4")
    fig = go.Figure(
        go.Scatter(
            x=s.index,
            y=z,
            mode="markers",
            marker=dict(color=colors, size=6),
            hovertemplate="idx=%{x}<br>z=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=3, line_dash="dot", line_color="#d62728")
    fig.add_hline(y=-3, line_dash="dot", line_color="#d62728")
    fig.update_layout(
        title=f"Z-score — {name}",
        xaxis_title="row index",
        yaxis_title="z",
        height=320,
    )
    return fig


def categorical_diff(raw: pd.DataFrame, clean: pd.DataFrame, col: str) -> pd.DataFrame:
    r = raw[col].astype(str).value_counts(dropna=False).rename("raw")
    c = clean[col].astype(str).value_counts(dropna=False).rename("clean")
    out = pd.concat([r, c], axis=1).fillna(0).astype(int)
    out["Δ"] = out["clean"] - out["raw"]
    return out.sort_values("raw", ascending=False)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Dataset")
    ds_name = st.selectbox("Select source", list(DATASETS.keys()))
    raw_file, clean_file = DATASETS[ds_name]
    st.caption(f"raw: `Data/{raw_file}`\n\nclean: `Data/filtered/{clean_file}`")

try:
    raw_df, clean_df = load_pair(raw_file, clean_file)
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# ---------------- overview ----------------
st.subheader(f"`{ds_name}` — overview")
overview_metrics(raw_df, clean_df)

with st.expander("Schema & null diff", expanded=False):
    st.dataframe(dtype_diff(raw_df, clean_df), width="stretch")

# ---------------- descriptive stats ----------------
raw_stats = descriptive_table(raw_df)
clean_stats = descriptive_table(clean_df)

tab_stats, tab_diff, tab_dist, tab_z, tab_cat, tab_preview = st.tabs(
    [
        "Descriptive stats",
        "Δ (clean − raw)",
        "Distributions",
        "Z-score",
        "Categorical",
        "Row preview",
    ]
)

with tab_stats:
    left, right = st.columns(2)
    with left:
        st.markdown("**Raw**")
        st.dataframe(raw_stats.round(3), width="stretch")
    with right:
        st.markdown("**Cleaned**")
        st.dataframe(clean_stats.round(3), width="stretch")

with tab_diff:
    diff = stats_diff(raw_stats, clean_stats)
    if diff.empty:
        st.info("No shared numeric columns to diff.")
    else:
        st.dataframe(
            diff.round(3).style.background_gradient(cmap="RdYlGn", axis=0),
            width="stretch",
        )
        st.caption(
            "Positive Δ means the cleaned dataset has a larger value than the raw "
            "dataset for that statistic."
        )

with tab_dist:
    num_cols = sorted(set(numeric_columns(raw_df)) & set(numeric_columns(clean_df)))
    if not num_cols:
        st.info("No shared numeric columns.")
    else:
        col = st.selectbox("Numeric column", num_cols, key="dist_col")
        st.plotly_chart(distribution_figure(raw_df, clean_df, col), width="stretch")
        rs, cs = raw_df[col].dropna(), clean_df[col].dropna()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("range raw", f"{rs.max() - rs.min():.2f}")
        m2.metric(
            "range clean",
            f"{cs.max() - cs.min():.2f}",
            f"{(cs.max() - cs.min()) - (rs.max() - rs.min()):+.2f}",
        )
        raw_iqr = rs.quantile(0.75) - rs.quantile(0.25)
        clean_iqr = cs.quantile(0.75) - cs.quantile(0.25)
        m3.metric("IQR raw", f"{raw_iqr:.2f}")
        m4.metric("IQR clean", f"{clean_iqr:.2f}", f"{clean_iqr - raw_iqr:+.2f}")

with tab_z:
    num_cols = sorted(set(numeric_columns(raw_df)) & set(numeric_columns(clean_df)))
    if not num_cols:
        st.info("No shared numeric columns.")
    else:
        col = st.selectbox("Numeric column", num_cols, key="z_col")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(zscore_scatter(raw_df[col], "raw"), width="stretch")
        with right:
            st.plotly_chart(zscore_scatter(clean_df[col], "clean"), width="stretch")
        st.caption("Red points have |z| > 3 — classical outlier flag.")

with tab_cat:
    cat_cols = sorted(
        set(raw_df.select_dtypes(exclude=[np.number]).columns)
        & set(clean_df.select_dtypes(exclude=[np.number]).columns)
    )
    if not cat_cols:
        st.info("No shared categorical columns.")
    else:
        col = st.selectbox("Categorical column", cat_cols, key="cat_col")
        cd = categorical_diff(raw_df, clean_df, col)
        st.dataframe(cd, width="stretch")
        top = cd.head(20).reset_index().rename(columns={"index": col})
        fig = px.bar(
            top.melt(id_vars=col, value_vars=["raw", "clean"]),
            x=col,
            y="value",
            color="variable",
            barmode="group",
            title=f"Top 20 categories — {col}",
        )
        st.plotly_chart(fig, width="stretch")

with tab_preview:
    n = st.slider("Rows to show", 5, 50, 10)
    left, right = st.columns(2)
    with left:
        st.markdown("**Raw**")
        st.dataframe(raw_df.head(n), width="stretch")
    with right:
        st.markdown("**Cleaned**")
        st.dataframe(clean_df.head(n), width="stretch")
