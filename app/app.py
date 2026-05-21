"""
DAP391m — Supplier Lead-Time Risk Predictor (Streamlit).

Loads trained binary delay classifiers and processed tree features.
Run from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"
FEATURE_TRAIN_PATH = PROCESSED_DIR / "X_train_tree.csv"

MODEL_CANDIDATES: list[tuple[str, str]] = [
    ("best_model.pkl", "Best model (PR-AUC winner)"),
    ("xgb_model.pkl", "XGBoost"),
    ("random_forest_model.pkl", "Random Forest"),
]

PIPELINE_HINT = (
    "Run the pipeline first:\n"
    "- `python src-code/04_feature_engineering.py`\n"
    "- `python src-code/05_modeling.py`"
)

# Sidebar fields (skipped if column missing from training matrix)
EDITABLE_SPECS: dict[str, dict[str, Any]] = {
    "avg_lead_time_days": {
        "label": "Avg lead time (days)",
        "min_value": 1.0,
        "max_value": 120.0,
        "step": 0.5,
    },
    "customs_clearance_time_days": {
        "label": "Customs clearance (days)",
        "min_value": 0.0,
        "max_value": 30.0,
        "step": 0.1,
    },
    "freight_cost": {
        "label": "Freight cost (USD)",
        "min_value": 0.0,
        "max_value": 100_000.0,
        "step": 50.0,
    },
    "freight_ratio": {
        "label": "Freight ratio",
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.01,
    },
    "avg_delay_hours": {
        "label": "Avg carrier delay (hours)",
        "min_value": 0.0,
        "max_value": 200.0,
        "step": 0.1,
    },
    "avg_warehouse_util": {
        "label": "Avg warehouse utilisation (%)",
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 1.0,
    },
    "avg_satisfaction": {
        "label": "Avg supplier satisfaction (1–5)",
        "min_value": 1.0,
        "max_value": 5.0,
        "step": 0.1,
    },
    "avg_support_tickets": {
        "label": "Avg support tickets",
        "min_value": 0.0,
        "max_value": 50.0,
        "step": 0.1,
    },
    "value": {
        "label": "Shipment value (USD)",
        "min_value": 0.0,
        "max_value": 1_000_000.0,
        "step": 100.0,
    },
}

ARTIFACT_IMAGES: list[tuple[str, str]] = [
    ("shap_summary.png", "SHAP summary (test sample)"),
    ("feature_importance.png", "Feature importance (best model)"),
    ("precision_recall_curve.png", "Precision–recall curves (test set)"),
    ("roc_curve.png", "ROC curves (test set)"),
]


def _float_default(series: pd.Series) -> float:
    med = series.median()
    if pd.isna(med):
        return 0.0
    return float(med)


@st.cache_data(show_spinner=False)
def load_training_reference(path_str: str) -> tuple[list[str], dict[str, float]]:
    path = Path(path_str)
    df = pd.read_csv(path)
    columns = list(df.columns)
    defaults = {col: _float_default(df[col]) for col in columns}
    return columns, defaults


def resolve_model() -> tuple[Any | None, str | None]:
    for filename, label in MODEL_CANDIDATES:
        path = OUTPUT_DIR / filename
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                model = pickle.load(handle)
            return model, f"{label} (`{path.name}`)"
        except Exception as exc:
            st.warning(f"Skipping `{path.name}`: {exc}")
    return None, None


def risk_band(delay_prob: float) -> tuple[str, str]:
    if delay_prob < 0.30:
        return "LOW", "green"
    if delay_prob < 0.60:
        return "MEDIUM", "orange"
    return "HIGH", "red"


def procurement_note(delay_prob: float, risk: str) -> str:
    if risk == "LOW":
        return (
            "Delay risk is relatively low. Standard procurement monitoring "
            "is appropriate; no immediate expediting required."
        )
    if risk == "MEDIUM":
        return (
            "Moderate delay risk. Consider buffer stock, alternate carriers, "
            "or earlier PO placement for critical SKUs."
        )
    return (
        "High delay risk. Escalate with the supplier, review lead-time "
        "commitments, and prioritize contingency sourcing for this lane."
    )


def build_input_row(
    feature_columns: list[str],
    defaults: dict[str, float],
    overrides: dict[str, float],
) -> pd.DataFrame:
    row = {col: defaults.get(col, 0.0) for col in feature_columns}
    for key, value in overrides.items():
        if key in row:
            row[key] = value
    return pd.DataFrame([row], columns=feature_columns)


def predict_delay(
    model: Any, features: pd.DataFrame
) -> tuple[int, str, float | None]:
    pred = int(model.predict(features)[0])
    label = "Delayed" if pred == 1 else "On-Time"
    prob: float | None = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        classes = getattr(model, "classes_", [0, 1])
        if 1 in classes:
            idx = list(classes).index(1)
            prob = float(proba[idx])
        elif len(proba) > 1:
            prob = float(proba[1])
    return pred, label, prob


def show_image_if_exists(filename: str, caption: str) -> None:
    path = OUTPUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"`{filename}` not found in model outputs.")


def render_sidebar_inputs(
    available: set[str],
    defaults: dict[str, float],
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    st.header("Shipment parameters")
    for col, spec in EDITABLE_SPECS.items():
        if col not in available:
            continue
        default = defaults.get(col, _float_default(pd.Series([0.0])))
        overrides[col] = st.number_input(
            spec["label"],
            min_value=float(spec["min_value"]),
            max_value=float(spec["max_value"]),
            value=float(default),
            step=float(spec["step"]),
            key=f"input_{col}",
        )
    return overrides


# ── page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Supplier Delay Risk", layout="wide")
st.title("Supplier Lead-Time Risk Predictor")
st.caption("DAP391m — Group 8, FPT University HCMC · Binary delay classification")

missing_core: list[str] = []
if not FEATURE_TRAIN_PATH.exists():
    missing_core.append(str(FEATURE_TRAIN_PATH.relative_to(PROJECT_ROOT)))

model, model_label = resolve_model()
if model is None:
    missing_core.append("trained model (.pkl in model_outputs/)")

if missing_core:
    st.error("Required artifacts are missing.")
    for item in missing_core:
        st.markdown(f"- `{item}`")
    st.markdown(PIPELINE_HINT)
    st.stop()

feature_columns, median_defaults = load_training_reference(str(FEATURE_TRAIN_PATH))
available_cols = set(feature_columns)

with st.sidebar:
    user_overrides = render_sidebar_inputs(available_cols, median_defaults)
    run_predict = st.button("Predict delay risk", type="primary")

# ── prediction ─────────────────────────────────────────────────────────────────
st.header("Prediction")
if not run_predict:
    st.info("Adjust parameters in the sidebar, then click **Predict delay risk**.")
else:
    features = build_input_row(feature_columns, median_defaults, user_overrides)
    _, class_label, delay_prob = predict_delay(model, features)

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_label or "Unknown")
    with col2:
        st.metric("Predicted class", class_label)
    with col3:
        if delay_prob is not None:
            risk, color = risk_band(delay_prob)
            st.metric("P(Delayed)", f"{delay_prob:.1%}")
            st.markdown(f"**Risk level:** :{color}[{risk}]")
        else:
            st.metric("P(Delayed)", "N/A")
            st.warning("Model does not expose `predict_proba`; risk band unavailable.")

    if delay_prob is not None:
        st.progress(min(max(delay_prob, 0.0), 1.0))
        risk, _ = risk_band(delay_prob)
        st.markdown("**Procurement note**")
        st.write(procurement_note(delay_prob, risk))

    with st.expander("Feature vector used for this prediction"):
        st.dataframe(features, use_container_width=True, hide_index=True)

# ── model performance ──────────────────────────────────────────────────────────
st.header("Model performance")
comparison_path = OUTPUT_DIR / "model_comparison.csv"
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
else:
    st.warning(f"`model_comparison.csv` not found. {PIPELINE_HINT}")

st.subheader("Evaluation plots")
plot_cols = st.columns(2)
for idx, (filename, caption) in enumerate(ARTIFACT_IMAGES[2:]):
    with plot_cols[idx % 2]:
        show_image_if_exists(filename, caption)

# ── explainability ─────────────────────────────────────────────────────────────
st.header("Explainability")
explain_cols = st.columns(2)
for idx, (filename, caption) in enumerate(ARTIFACT_IMAGES[:2]):
    with explain_cols[idx % 2]:
        show_image_if_exists(filename, caption)

# ── limitations ────────────────────────────────────────────────────────────────
st.header("Limitations")
st.markdown(
    """
- Predictions use **median defaults** for features not edited in the sidebar; \
tune key supplier and lane fields for scenario analysis.
- The model is trained on historical shipment data; **new suppliers or routes** \
may behave differently.
- Risk bands (LOW / MEDIUM / HIGH) are heuristic thresholds on P(Delayed), \
not contractual SLAs.
- Country and category encodings (`O_Country`, `D_Country`, etc.) stay at \
training medians unless the pipeline exposes UI controls.
- For production use, validate with procurement stakeholders and monitor \
drift against `model_comparison.csv` metrics.
"""
)
