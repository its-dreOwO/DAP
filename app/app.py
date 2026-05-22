"""
DAP391m — Supplier Lead-Time Risk Predictor (Streamlit).

Loads the primary XGBoost 3-class supplier risk classifier and builds
inputs from supply_chain_risk_dataset.csv (same feature prep as training).

Run from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = PROJECT_ROOT / "Data" / "supply_chain_risk_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

TARGET_COL = "risk_label"
CLASS_LABELS = ["Low", "Medium", "High"]
INT_TO_LABEL: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}
LABEL_TO_INT: dict[str, int] = {label: i for i, label in enumerate(CLASS_LABELS)}

LEAKAGE_COLS = ["risk_probability", "port_delay_days"]
DROP_FEATURE_COLS = ["machine_id"]

MODEL_CANDIDATES: list[tuple[str, str, bool]] = [
    ("primary_model.pkl", "XGBoost (primary deployment model)", False),
    ("xgb_model.pkl", "XGBoost", False),
    (
        "best_model.pkl",
        "Best model by Macro PR-AUC (fallback — may not be XGBoost)",
        True,
    ),
]

PIPELINE_HINT = "Run the modeling step first:\n" "- `python src-code/05_modeling.py`"

# Sidebar fields (skipped if column missing from feature matrix)
EDITABLE_SPECS: dict[str, dict[str, Any]] = {
    "supplier_lead_time_days": {
        "label": "Supplier lead time (days)",
        "min_value": 0.5,
        "max_value": 30.0,
        "step": 0.1,
    },
    "supplier_quality_score": {
        "label": "Supplier quality score",
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 0.5,
    },
    "supplier_reliability_index": {
        "label": "Supplier reliability index",
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.01,
    },
    "inventory_level_units": {
        "label": "Inventory level (units)",
        "min_value": 0.0,
        "max_value": 5000.0,
        "step": 1.0,
    },
    "pending_orders": {
        "label": "Pending orders",
        "min_value": 0.0,
        "max_value": 2000.0,
        "step": 1.0,
    },
    "temperature_C": {
        "label": "Temperature (°C)",
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 0.1,
    },
    "vibration_level": {
        "label": "Vibration level",
        "min_value": 0.0,
        "max_value": 100.0,
        "step": 0.1,
    },
    "machine_runtime_hours": {
        "label": "Machine runtime (hours)",
        "min_value": 0.0,
        "max_value": 500.0,
        "step": 0.5,
    },
    "fuel_price_index": {
        "label": "Fuel price index",
        "min_value": 0.0,
        "max_value": 3.0,
        "step": 0.01,
    },
    "market_demand_index": {
        "label": "Market demand index",
        "min_value": 0.0,
        "max_value": 2.0,
        "step": 0.01,
    },
    "weather_disruption_score": {
        "label": "Weather disruption score",
        "min_value": 0.0,
        "max_value": 15.0,
        "step": 0.1,
    },
    "order_month": {
        "label": "Order month",
        "min_value": 1.0,
        "max_value": 12.0,
        "step": 1.0,
    },
    "order_day_of_week": {
        "label": "Order day of week (0=Mon)",
        "min_value": 0.0,
        "max_value": 6.0,
        "step": 1.0,
    },
    "order_hour": {
        "label": "Order hour",
        "min_value": 0.0,
        "max_value": 23.0,
        "step": 1.0,
    },
}

CATEGORICAL_UI_COLS = ["supplier_id"]

ARTIFACT_IMAGES: list[tuple[str, str]] = [
    ("shap_summary.png", "SHAP summary — XGBoost (test sample)"),
    ("feature_importance.png", "Feature importance — XGBoost (primary)"),
    ("model_comparison.png", "Model comparison — 3-class metrics"),
    ("confusion_matrix_xgboost.png", "Confusion matrix — XGBoost (primary)"),
]


def _float_default(series: pd.Series) -> float:
    med = series.median()
    if pd.isna(med):
        return 0.0
    return float(med)


def _default_for_column(series: pd.Series) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        return _float_default(series)
    mode = series.mode()
    if mode.empty:
        return ""
    return mode.iloc[0]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror src-code/05_modeling.py feature preparation (without target)."""
    out = df.copy()
    for col in LEAKAGE_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])
    for col in DROP_FEATURE_COLS:
        if col in out.columns:
            out = out.drop(columns=[col])
    if TARGET_COL in out.columns:
        out = out.drop(columns=[TARGET_COL])
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out["order_month"] = out["timestamp"].dt.month
        out["order_day_of_week"] = out["timestamp"].dt.dayofweek
        out["order_hour"] = out["timestamp"].dt.hour
        out = out.drop(columns=["timestamp"])
    return out


@st.cache_data(show_spinner=False)
def load_feature_reference(path_str: str) -> tuple[list[str], dict[str, Any]]:
    path = Path(path_str)
    raw = pd.read_csv(path)
    features = prepare_features(raw)
    columns = list(features.columns)
    defaults = {col: _default_for_column(features[col]) for col in columns}
    return columns, defaults


def get_model_classes(model: Any) -> list[Any]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    named_steps = getattr(model, "named_steps", None)
    if named_steps is not None:
        final = named_steps.get("model")
        if final is not None and hasattr(final, "classes_"):
            return list(final.classes_)
    return [0, 1, 2]


def class_value_to_label(value: Any, classes: list[Any] | None = None) -> str:
    if isinstance(value, str) and value in CLASS_LABELS:
        return value
    idx = int(value)
    classes = classes or [0, 1, 2]
    if len(classes) == len(CLASS_LABELS) and all(
        isinstance(c, (int, np.integer, float, np.floating)) for c in classes
    ):
        return INT_TO_LABEL.get(idx, str(idx))
    if 0 <= idx < len(classes):
        cls = classes[idx]
        if isinstance(cls, str):
            return cls
        return INT_TO_LABEL.get(int(cls), str(cls))
    return INT_TO_LABEL.get(idx, str(idx))


def probability_by_label(model: Any, proba: np.ndarray) -> dict[str, float]:
    classes = get_model_classes(model)
    probs: dict[str, float] = {label: 0.0 for label in CLASS_LABELS}
    for idx, cls in enumerate(classes):
        if idx >= len(proba):
            break
        label = class_value_to_label(cls, classes)
        if label not in probs and label in CLASS_LABELS:
            probs[label] = float(proba[idx])
        elif label in probs:
            probs[label] = float(proba[idx])
        else:
            mapped = INT_TO_LABEL.get(int(cls), None)
            if mapped:
                probs[mapped] = float(proba[idx])
    return probs


def predict_risk(
    model: Any, features: pd.DataFrame
) -> tuple[int, str, np.ndarray | None]:
    pred = int(model.predict(features)[0])
    classes = get_model_classes(model)
    label = class_value_to_label(pred, classes)
    proba: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
    return pred, label, proba


def resolve_model() -> tuple[Any | None, str | None, bool]:
    for filename, label, is_fallback in MODEL_CANDIDATES:
        path = OUTPUT_DIR / filename
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                model = pickle.load(handle)
            return model, f"{label} (`{path.name}`)", is_fallback
        except Exception as exc:
            st.warning(f"Skipping `{path.name}`: {exc}")
    return None, None, False


def risk_level_from_label(label: str) -> tuple[str, str]:
    bands = {
        "Low": ("LOW", "green"),
        "Medium": ("MEDIUM", "orange"),
        "High": ("HIGH", "red"),
    }
    return bands.get(label, ("UNKNOWN", "gray"))


def procurement_note(label: str) -> str:
    if label == "Low":
        return (
            "Low supplier lead-time risk. Standard procurement monitoring "
            "is appropriate; no immediate escalation required."
        )
    if label == "Medium":
        return (
            "Medium supplier risk. Review buffer stock, alternate suppliers, "
            "and earlier PO placement for critical SKUs."
        )
    return (
        "High supplier risk. Escalate with the supplier, review lead-time "
        "commitments, and prioritize contingency sourcing."
    )


def build_input_row(
    feature_columns: list[str],
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> pd.DataFrame:
    row = {col: defaults.get(col) for col in feature_columns}
    for key, value in overrides.items():
        if key in row:
            row[key] = value
    return pd.DataFrame([row], columns=feature_columns)


def show_image_if_exists(filename: str, caption: str) -> None:
    path = OUTPUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"`{filename}` not found in model outputs.")


def render_sidebar_inputs(
    feature_df: pd.DataFrame,
    available: set[str],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    st.header("Shipment parameters")
    for col in CATEGORICAL_UI_COLS:
        if col not in available:
            continue
        options = sorted(feature_df[col].dropna().astype(str).unique().tolist())
        default = str(defaults.get(col, options[0] if options else ""))
        idx = options.index(default) if default in options else 0
        overrides[col] = st.selectbox(
            "Supplier ID",
            options=options,
            index=idx,
            key=f"input_{col}",
        )
    for col, spec in EDITABLE_SPECS.items():
        if col not in available:
            continue
        default = defaults.get(col, _float_default(feature_df[col]))
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
st.set_page_config(page_title="Supplier Risk Predictor", layout="wide")
st.title("Supplier Lead-Time Risk Predictor")
st.caption(
    "DAP391m — Group 8, FPT University HCMC · 3-class supplier risk classification"
)

missing_core: list[str] = []
if not DATA_CSV.exists():
    missing_core.append(str(DATA_CSV.relative_to(PROJECT_ROOT)))

model, model_label, model_is_fallback = resolve_model()
if model is None:
    missing_core.append("trained model (.pkl in model_outputs/)")

if missing_core:
    st.error("Required artifacts are missing.")
    for item in missing_core:
        st.markdown(f"- `{item}`")
    st.markdown(PIPELINE_HINT)
    st.stop()

if model_is_fallback:
    st.warning(
        "Loaded `best_model.pkl` as fallback. This artifact follows Macro PR-AUC "
        "and may be Logistic Regression — not the official XGBoost primary model. "
        "Run `python src-code/05_modeling.py` to generate `primary_model.pkl`."
    )

raw_df = pd.read_csv(DATA_CSV)
feature_df = prepare_features(raw_df)
feature_columns, column_defaults = load_feature_reference(str(DATA_CSV))
available_cols = set(feature_columns)

with st.sidebar:
    user_overrides = render_sidebar_inputs(feature_df, available_cols, column_defaults)
    run_predict = st.button("Predict supplier risk", type="primary")

# ── prediction ───────────────────────────────────────────────────────────────
st.header("Prediction")
if not run_predict:
    st.info("Adjust parameters in the sidebar, then click **Predict supplier risk**.")
else:
    features = build_input_row(feature_columns, column_defaults, user_overrides)
    _, risk_label, proba = predict_risk(model, features)

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model loaded", model_label or "Unknown")
    with col2:
        st.metric("Predicted risk label", risk_label)
    with col3:
        risk_band, color = risk_level_from_label(risk_label)
        st.metric("Risk level", risk_band)
        st.markdown(f":{color}[{risk_label} risk]")

    if proba is not None:
        prob_map = probability_by_label(model, proba)
        st.markdown("**Risk probabilities**")
        pcols = st.columns(3)
        for idx, lbl in enumerate(CLASS_LABELS):
            with pcols[idx]:
                st.metric(f"P({lbl})", f"{prob_map.get(lbl, 0.0):.1%}")
        max_prob = max(prob_map.values()) if prob_map else 0.0
        st.progress(min(max(max_prob, 0.0), 1.0))
    else:
        st.warning(
            "Model does not expose `predict_proba`; class probabilities unavailable."
        )

    st.markdown("**Procurement note**")
    st.write(procurement_note(risk_label))

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
st.markdown("""
- Predictions use **defaults (median/mode)** for features not edited in the sidebar; \
tune supplier and operational fields for scenario analysis.
- The model is trained on historical supply-chain records; **new suppliers or \
conditions** may behave differently.
- Risk level follows the **predicted class** (Low / Medium / High), not a single \
binary delay threshold.
- Supplier ID and other categoricals stay at training defaults unless changed in \
the sidebar.
- For production use, validate with procurement stakeholders and monitor drift \
against `model_comparison.csv` metrics.
""")
