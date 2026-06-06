"""
DAP391m - Supplier Lead-Time and Late Delivery Risk Prediction (Streamlit).

Loads the primary XGBoost binary classifier (Late_delivery_risk: 1 = late)
trained by src-code/05_modeling.py on the DataCo Smart Supply Chain dataset and
scores a single order line item. The app reports P(late), P(on-time), a risk
tier, and the historical shipping-mode base rate.

Run from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# `streamlit run app/app.py` puts this folder on sys.path, but other launchers
# such as AppTest and `python app/app.py` do not.
_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import ai_assistant as ai  # noqa: E402

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from lifelines import KaplanMeierFitter

    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
DATACO_RAW_CSV = PROJECT_ROOT / "Data" / "dataco_raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

TARGET_COL = "Late_delivery_risk"
GROUP_KEY = "Order Id"
DATE_COL = "order date (DateOrders)"
LEAD_TIME_COL = "Days for shipping (real)"
SHIPPING_MODE_COL = "Shipping Mode"
SCHEDULED_DAYS_COL = "Days for shipment (scheduled)"
POSITIVE_CLASS = 1

LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "shipping date (DateOrders)",
    "Order Status",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
]

PII_COLUMNS = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Customer Id",
    "Order Customer Id",
    "Latitude",
    "Longitude",
    "Product Image",
    "Product Description",
]

NON_FEATURE_COLUMNS = [GROUP_KEY, "Order Item Id"]

HEATMAP_ROW_COL = SHIPPING_MODE_COL
HEATMAP_COL_CANDIDATES = ["Order Region", "Market", "Category Name"]
DATE_COL_CANDIDATES = [DATE_COL, "DateOrders"]

MODEL_CANDIDATES: list[tuple[str, str, bool]] = [
    ("xgb_model.pkl", "XGBoost (primary deployment model)", False),
    ("primary_model.pkl", "XGBoost (legacy primary artifact)", False),
    ("random_forest_model.pkl", "Random Forest (fallback)", True),
    ("decision_tree_model.pkl", "Decision Tree (fallback)", True),
    ("logistic_model.pkl", "Logistic Regression (fallback)", True),
    ("best_model.pkl", "Best model by PR-AUC (fallback)", True),
]
MODEL_LABELS = {filename: label for filename, label, _ in MODEL_CANDIDATES}
NON_FALLBACK_FILES = {"xgb_model.pkl", "primary_model.pkl"}

PIPELINE_HINT = (
    "Train the models first:\n"
    "- `modal run modal_train.py` (GPU), or\n"
    "- `.venv/bin/python3 src-code/05_modeling.py` (local)"
)

SIDEBAR_DEFAULTS = {
    SHIPPING_MODE_COL: "Standard Class",
    "Market": "LATAM",
    "Order Region": "Central America",
    "Customer Segment": "Consumer",
    "Department Name": "Fan Shop",
    "Category Name": "Cleats",
    "Type": "DEBIT",
    "Order Item Quantity": 1,
    "Sales": 199.92,
    "Order Item Discount Rate": 0.10,
}

EXPOSED_CATEGORICAL = [
    (SHIPPING_MODE_COL, "Shipping mode"),
    ("Market", "Market"),
    ("Order Region", "Order region"),
    ("Customer Segment", "Customer segment"),
    ("Department Name", "Department"),
    ("Category Name", "Product category"),
    ("Type", "Payment type"),
]

EXPOSED_NUMERIC = {
    "Order Item Quantity": {
        "label": "Item quantity",
        "min": 1,
        "max": 20,
        "step": 1,
        "kind": "int",
    },
    "Sales": {
        "label": "Sales (USD)",
        "min": 0.0,
        "max": 100000.0,
        "step": 0.01,
        "format": "%.2f",
        "kind": "float",
    },
    "Order Item Discount Rate": {
        "label": "Discount rate",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "kind": "float",
    },
}

SUMMARIZE_PROMPT = (
    "Summarize the current prediction in plain language for a non-technical "
    "reader: the P(late), the risk tier, and how it compares to this shipping "
    "mode's historical base rate. Keep it to a few sentences."
)


def _force_cpu_inference(model: Any) -> Any:
    """Pin an XGBoost booster to CPU for single-row Streamlit inference."""
    final = getattr(model, "named_steps", {}).get("model", model)
    try:
        final.set_params(device="cpu")
    except Exception:  # noqa: BLE001 - non-XGBoost estimators have no device.
        pass
    get_booster = getattr(final, "get_booster", None)
    if callable(get_booster):
        try:
            get_booster().set_param({"device": "cpu"})
        except Exception:  # noqa: BLE001 - booster may be unavailable.
            pass
    return model


@st.cache_resource(show_spinner=False)
def load_model_file(filename: str) -> Any:
    with (OUTPUT_DIR / filename).open("rb") as handle:
        return _force_cpu_inference(pickle.load(handle))


def resolve_model(
    preferred: str | None = None,
) -> tuple[Any | None, str | None, str | None, bool]:
    """Load the preferred model if available, otherwise the first candidate."""
    order: list[str] = []
    if preferred:
        order.append(preferred)
    order += [fn for fn, _, _ in MODEL_CANDIDATES if fn != preferred]

    for filename in order:
        path = OUTPUT_DIR / filename
        if not path.exists():
            continue
        try:
            model = load_model_file(filename)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Skipping `{filename}`: {exc}")
            continue
        label = MODEL_LABELS.get(filename, filename)
        is_fallback = filename not in NON_FALLBACK_FILES
        return model, label, filename, is_fallback
    return None, None, None, False


def available_model_files() -> list[str]:
    return [fn for fn, _, _ in MODEL_CANDIDATES if (OUTPUT_DIR / fn).exists()]


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame | None:
    if not CLEAN_CSV.exists():
        return None
    return pd.read_csv(CLEAN_CSV)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_template_row(clean_df: pd.DataFrame) -> pd.Series:
    """Build robust defaults from medians and modes, never from row position."""
    numeric_medians = clean_df.median(numeric_only=True)
    defaults: dict[str, Any] = {}

    for col in clean_df.columns:
        series = clean_df[col]
        if pd.api.types.is_numeric_dtype(series):
            value = numeric_medians.get(col, np.nan)
            if pd.isna(value):
                value = SIDEBAR_DEFAULTS.get(col, 0)
            defaults[col] = _python_scalar(value)
            continue

        mode = series.dropna().mode()
        if not mode.empty:
            defaults[col] = _python_scalar(mode.iloc[0])
        else:
            defaults[col] = SIDEBAR_DEFAULTS.get(col, "")

    return pd.Series(defaults, index=clean_df.columns)


def get_feature_defaults(clean_df: pd.DataFrame) -> dict[str, Any]:
    return build_template_row(clean_df).to_dict()


@st.cache_data(show_spinner=False)
def load_sidebar_options(clean_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return all sidebar option lists from the already-loaded clean dataset."""
    options: dict[str, list[str]] = {}
    for col, _label in EXPOSED_CATEGORICAL:
        if col not in clean_df.columns:
            options[col] = []
            continue
        values = clean_df[col].dropna().astype(str).unique().tolist()
        options[col] = sorted(values)
    return options


def load_scheduled_days_by_mode(clean_df: pd.DataFrame) -> dict[str, int]:
    if {SHIPPING_MODE_COL, SCHEDULED_DAYS_COL}.issubset(clean_df.columns):
        series = (
            clean_df.groupby(SHIPPING_MODE_COL, dropna=False)[SCHEDULED_DAYS_COL]
            .first()
            .dropna()
        )
        return {str(mode): int(days) for mode, days in series.items()}
    return {"Standard Class": 4}


def load_shipping_mode_base_rates(clean_df: pd.DataFrame) -> dict[str, float]:
    if {SHIPPING_MODE_COL, TARGET_COL}.issubset(clean_df.columns):
        grp = clean_df.groupby(SHIPPING_MODE_COL, dropna=False)[TARGET_COL].mean()
        return {str(mode): float(rate) for mode, rate in grp.items()}
    return {}


def _default_option(options: list[str], preferred: Any, fallback: str) -> str:
    preferred_text = str(preferred)
    if preferred_text in options:
        return preferred_text
    if fallback in options:
        return fallback
    return options[0] if options else fallback


def _bounded_number(value: Any, spec: dict[str, Any]) -> int | float:
    low = spec["min"]
    high = spec["max"]
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(SIDEBAR_DEFAULTS.get(spec["label"], low))
    number = min(max(number, float(low)), float(high))
    if spec.get("kind") == "int":
        return int(round(number))
    return float(number)


def seed_sidebar_state(
    template: pd.Series,
    options_by_col: dict[str, list[str]],
) -> None:
    for col, _label in EXPOSED_CATEGORICAL:
        key = f"in_{col}"
        if key in st.session_state:
            continue
        options = options_by_col.get(col, [])
        fallback = str(SIDEBAR_DEFAULTS.get(col, options[0] if options else ""))
        st.session_state[key] = _default_option(options, template.get(col), fallback)

    for col, spec in EXPOSED_NUMERIC.items():
        key = f"in_{col}"
        if key not in st.session_state:
            st.session_state[key] = _bounded_number(template.get(col), spec)


def apply_pending_inputs(
    options_by_col: dict[str, list[str]],
) -> None:
    pending = st.session_state.get("pending_inputs", {})
    if not pending:
        return

    for col, value in pending.items():
        key = f"in_{col}"
        if col in options_by_col:
            options = options_by_col[col]
            if str(value) in options:
                st.session_state[key] = str(value)
        elif col in EXPOSED_NUMERIC:
            st.session_state[key] = _bounded_number(value, EXPOSED_NUMERIC[col])

    st.session_state["pending_inputs"] = {}


def read_tuned_threshold() -> float:
    path = OUTPUT_DIR / "threshold_analysis.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "tuned on train" in line and ":" in line:
                try:
                    return float(line.split(":")[-1].strip())
                except ValueError:
                    pass
    return 0.5


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror src-code/05_modeling.py feature derivation for prediction rows."""
    out = df.copy()
    if DATE_COL in out.columns:
        dt = pd.to_datetime(out[DATE_COL], errors="coerce")
        out["order_month"] = dt.dt.month
        out["order_day_of_week"] = dt.dt.dayofweek
        out["order_hour"] = dt.dt.hour
        out = out.drop(columns=[DATE_COL])
    return out


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        col
        for col in LEAKAGE_COLUMNS + PII_COLUMNS + [TARGET_COL] + NON_FEATURE_COLUMNS
        if col in df.columns
    ]
    return df.drop(columns=to_drop)


def build_features_from_row(
    row: pd.Series,
    model: Any,
    template: pd.Series | None = None,
) -> pd.DataFrame:
    """Create the one-row model frame and honor model.feature_names_in_ order."""
    features = drop_non_feature_columns(add_engineered_features(pd.DataFrame([row])))
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return features

    expected_cols = list(expected)
    fallback = pd.DataFrame()
    if template is not None:
        fallback = drop_non_feature_columns(
            add_engineered_features(pd.DataFrame([template]))
        )

    missing: list[str] = []
    for col in expected_cols:
        if col in features.columns:
            continue
        if not fallback.empty and col in fallback.columns:
            features[col] = fallback.iloc[0][col]
        else:
            missing.append(col)

    if missing:
        shown = ", ".join(missing[:12])
        extra = "" if len(missing) <= 12 else f" and {len(missing) - 12} more"
        raise ValueError(f"Missing required model feature(s): {shown}{extra}")

    return features[expected_cols]


def build_order_row(
    template: pd.Series,
    input_values: dict[str, Any],
    order_date_value: date,
) -> pd.Series:
    row = template.copy()
    item_quantity = int(input_values.get("Order Item Quantity", 1) or 1)
    sales_usd = float(input_values.get("Sales", 0.0) or 0.0)
    discount_rate = float(input_values.get("Order Item Discount Rate", 0.0) or 0.0)
    product_price = sales_usd / max(item_quantity, 1)
    discount_amount = round(sales_usd * discount_rate, 2)

    row.update(
        {
            SHIPPING_MODE_COL: input_values.get(SHIPPING_MODE_COL),
            SCHEDULED_DAYS_COL: input_values.get(SCHEDULED_DAYS_COL),
            "Market": input_values.get("Market"),
            "Order Region": input_values.get("Order Region"),
            "Customer Segment": input_values.get("Customer Segment"),
            "Department Name": input_values.get("Department Name"),
            "Category Name": input_values.get("Category Name"),
            "Type": input_values.get("Type"),
            "Order Item Quantity": item_quantity,
            "Sales": sales_usd,
            "Order Item Total": sales_usd,
            "Order Item Product Price": product_price,
            "Product Price": product_price,
            "Order Item Discount Rate": discount_rate,
            "Order Item Discount": discount_amount,
            "Sales per customer": sales_usd,
            DATE_COL: datetime.combine(order_date_value, datetime.min.time()).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
    )
    return row


def positive_proba(model: Any, features: pd.DataFrame) -> float:
    proba = model.predict_proba(features)[0]
    final = getattr(model, "named_steps", {}).get("model", model)
    classes = list(getattr(final, "classes_", [0, 1]))
    pos_idx = classes.index(POSITIVE_CLASS) if POSITIVE_CLASS in classes else 1
    return float(proba[pos_idx])


def risk_tier_display(p_late: float) -> tuple[str, str, str]:
    if p_late >= 0.65:
        return "Action", "red", "Expedite review is justified."
    if p_late >= 0.40:
        return "Watch", "orange", "Monitor this shipment closely."
    return "Routine", "green", "Standard monitoring is enough."


def prediction_label(p_late: float, threshold: float) -> str:
    return "Late delivery" if p_late >= threshold else "On-time delivery"


def render_vs_base_badge(p_late: float, base_rate: float) -> None:
    delta_pp = (p_late - base_rate) * 100.0
    color = "#b42318" if delta_pp > 0 else "#177245"
    sign = "+" if delta_pp > 0 else ""
    st.markdown(
        (
            f'<span style="background-color:{color};color:#ffffff;'
            f'padding:3px 10px;border-radius:999px;font-size:0.85rem;">'
            f"{sign}{delta_pp:.1f} pp vs base</span>"
        ),
        unsafe_allow_html=True,
    )


def short_model_label(model_label: str) -> str:
    return model_label if len(model_label) <= 28 else model_label[:25] + "..."


def prediction_snapshot(
    input_values: dict[str, Any],
    with_model: Any,
    model_label: str,
    model_file: str,
    template: pd.Series,
    base_rates: dict[str, float],
    threshold: float,
    order_date_value: date,
) -> dict[str, Any]:
    row = build_order_row(template, input_values, order_date_value)
    features = build_features_from_row(row, with_model, template)
    p_late = positive_proba(with_model, features)
    tier, _color, tier_note = risk_tier_display(p_late)
    mode = str(input_values.get(SHIPPING_MODE_COL, ""))
    base_rate = base_rates.get(mode)

    out: dict[str, Any] = {
        "p_late": round(p_late, 4),
        "p_on_time": round(1.0 - p_late, 4),
        "prediction": prediction_label(p_late, threshold),
        "tier": tier,
        "tier_note": tier_note,
        "shipping_mode": mode,
        "model_label": model_label,
        "model_file": model_file,
        "threshold": threshold,
        "inputs": {
            col: input_values.get(col)
            for col, _label in EXPOSED_CATEGORICAL
            if col in input_values
        },
    }
    if base_rate is not None:
        out["base_rate"] = round(base_rate, 4)
        out["delta_vs_base"] = round(p_late - base_rate, 4)
    return out


def show_image_if_exists(filename: str, caption: str) -> None:
    path = OUTPUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"`{filename}` not found in model outputs.")


def render_sidebar(
    template: pd.Series,
    options_by_col: dict[str, list[str]],
    scheduled_by_mode: dict[str, int],
) -> tuple[pd.Series, dict[str, Any], str, str, bool]:
    input_values: dict[str, Any] = {}

    with st.sidebar:
        st.header("Shipment")
        for col, label in EXPOSED_CATEGORICAL:
            options = options_by_col.get(col, [])
            if not options:
                continue
            input_values[col] = st.selectbox(label, options, key=f"in_{col}")

        shipping_mode = str(input_values.get(SHIPPING_MODE_COL, "Standard Class"))
        scheduled_days = int(scheduled_by_mode.get(shipping_mode, 4))
        input_values[SCHEDULED_DAYS_COL] = scheduled_days
        st.caption(
            f"Scheduled shipping days for **{shipping_mode}**: "
            f"`{scheduled_days}` (fixed by mode)."
        )

        st.divider()
        st.subheader("Order economics")
        if "Order Item Quantity" in EXPOSED_NUMERIC:
            input_values["Order Item Quantity"] = st.number_input(
                EXPOSED_NUMERIC["Order Item Quantity"]["label"],
                min_value=EXPOSED_NUMERIC["Order Item Quantity"]["min"],
                max_value=EXPOSED_NUMERIC["Order Item Quantity"]["max"],
                step=EXPOSED_NUMERIC["Order Item Quantity"]["step"],
                key="in_Order Item Quantity",
            )
        if "Sales" in EXPOSED_NUMERIC:
            input_values["Sales"] = st.number_input(
                EXPOSED_NUMERIC["Sales"]["label"],
                min_value=EXPOSED_NUMERIC["Sales"]["min"],
                max_value=EXPOSED_NUMERIC["Sales"]["max"],
                step=EXPOSED_NUMERIC["Sales"]["step"],
                format=EXPOSED_NUMERIC["Sales"]["format"],
                key="in_Sales",
            )
        if "Order Item Discount Rate" in EXPOSED_NUMERIC:
            input_values["Order Item Discount Rate"] = st.slider(
                EXPOSED_NUMERIC["Order Item Discount Rate"]["label"],
                min_value=EXPOSED_NUMERIC["Order Item Discount Rate"]["min"],
                max_value=EXPOSED_NUMERIC["Order Item Discount Rate"]["max"],
                step=EXPOSED_NUMERIC["Order Item Discount Rate"]["step"],
                key="in_Order Item Discount Rate",
            )

        st.divider()
        order_date_value = st.date_input(
            "Order date",
            value=date.today(),
            key=f"in_{DATE_COL}",
        )

        st.divider()
        st.subheader("AI assistant (OpenRouter)")
        if ai.AI_PROXY_URL:
            st.caption("Routing through the hosted relay; no API key is required.")
        ai_api_key = st.text_input(
            "OpenRouter API key (optional when relay is set)",
            type="password",
            key="openrouter_key",
        )
        ai_model_name = st.text_input(
            "Model",
            value=ai.DEFAULT_MODEL,
            key="openrouter_model",
        )

    order_row = build_order_row(template, input_values, order_date_value)
    assistant_ready = bool(ai.AI_PROXY_URL) or bool(ai_api_key)
    return order_row, input_values, ai_api_key, ai_model_name, assistant_ready


def render_prediction_section(
    model: Any,
    model_label: str,
    model_file: str,
    order_row: pd.Series,
    template: pd.Series,
    base_rates: dict[str, float],
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        features = build_features_from_row(order_row, model, template)
        p_late = positive_proba(model, features)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build a valid prediction row: {exc}")
        st.stop()

    shipping_mode = str(order_row.get(SHIPPING_MODE_COL, ""))
    base_rate = base_rates.get(shipping_mode)
    tier, tier_color, tier_note = risk_tier_display(p_late)
    label = prediction_label(p_late, threshold)

    st.header("Prediction")
    st.subheader("Result")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model", short_model_label(model_label))
        st.caption(f"`{model_file}`")
    with c2:
        st.metric("P(late)", f"{p_late:.1%}")
    with c3:
        st.metric("Risk tier", tier)
        st.markdown(f":{tier_color}[{tier_note}]")

    st.progress(min(max(p_late, 0.0), 1.0))

    pcols = st.columns(3)
    with pcols[0]:
        st.metric("P(on-time)", f"{1 - p_late:.1%}")
    with pcols[1]:
        st.metric("Prediction", label)
    with pcols[2]:
        base_label = f"Base rate - {shipping_mode}"
        if base_rate is None:
            st.metric(base_label, "N/A")
        else:
            st.metric(base_label, f"{base_rate:.1%}")
            render_vs_base_badge(p_late, base_rate)

    st.caption(
        "Base rate = historical late share for this shipping mode. The model's "
        "edge over it comes from ranking shipments within a mode; the dataset "
        "assessment shows ML adds only a small lift over this benchmark."
    )

    with st.expander("Feature vector used for this prediction"):
        feature_view = features.T.reset_index()
        feature_view.columns = ["feature", "value"]
        feature_view["value"] = feature_view["value"].astype(str)
        st.dataframe(feature_view, width="stretch", hide_index=True)

    snapshot = {
        "p_late": round(p_late, 4),
        "p_on_time": round(1.0 - p_late, 4),
        "prediction": label,
        "tier": tier,
        "tier_note": tier_note,
        "shipping_mode": shipping_mode,
        "model_label": model_label,
        "model_file": model_file,
        "threshold": threshold,
    }
    if base_rate is not None:
        snapshot["base_rate"] = round(base_rate, 4)
        snapshot["delta_vs_base"] = round(p_late - base_rate, 4)
    return snapshot, features


@st.cache_data(show_spinner="Loading DataCo dataset for visualizations...")
def load_dataco_for_visuals() -> pd.DataFrame | None:
    if not DATACO_RAW_CSV.exists():
        return None
    return pd.read_csv(DATACO_RAW_CSV, encoding="latin-1")


def clean_dataco_visual_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if LEAD_TIME_COL in out.columns:
        out[LEAD_TIME_COL] = pd.to_numeric(out[LEAD_TIME_COL], errors="coerce")
    if TARGET_COL in out.columns:
        out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    for date_col in DATE_COL_CANDIDATES:
        if date_col in out.columns:
            out["_order_date_parsed"] = pd.to_datetime(out[date_col], errors="coerce")
            break
    return out


def top_categories(series: pd.Series, limit: int = 20) -> list[Any]:
    counts = series.dropna().astype(str).value_counts()
    return counts.head(limit).index.tolist()


def render_lead_time_boxplot(df: pd.DataFrame) -> None:
    group_col = SHIPPING_MODE_COL
    if LEAD_TIME_COL not in df.columns:
        st.warning(f"Column `{LEAD_TIME_COL}` is unavailable; skipping boxplot.")
        return
    if group_col not in df.columns:
        st.warning(f"Column `{group_col}` is unavailable; skipping boxplot.")
        return

    plot_df = df[[LEAD_TIME_COL, group_col]].dropna()
    if plot_df.empty:
        st.warning("No valid rows for the lead-time boxplot.")
        return

    top_groups = top_categories(plot_df[group_col], limit=20)
    plot_df = plot_df[plot_df[group_col].astype(str).isin(top_groups)]

    st.subheader("Boxplot of Actual Shipping Lead Time by Shipping Mode")

    if HAS_PLOTLY:
        fig = px.box(
            plot_df,
            x=group_col,
            y=LEAD_TIME_COL,
            points=False,
            labels={LEAD_TIME_COL: "Actual shipping days", group_col: group_col},
        )
        fig.update_layout(xaxis_tickangle=-35, height=480, showlegend=False)
        st.plotly_chart(fig, width="stretch")
    else:
        grouped = [
            plot_df.loc[plot_df[group_col].astype(str) == g, LEAD_TIME_COL].values
            for g in top_groups
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(grouped, tick_labels=top_groups)
        ax.set_xlabel(group_col)
        ax.set_ylabel("Actual shipping days")
        ax.tick_params(axis="x", rotation=35)
        st.pyplot(fig)
        plt.close(fig)


def _empirical_survival_curve(
    durations: np.ndarray,
    timeline: np.ndarray,
) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        return np.zeros_like(timeline, dtype=float)
    return (durations[:, None] > timeline[None, :]).mean(axis=0)


def render_survival_curves(df: pd.DataFrame) -> None:
    if LEAD_TIME_COL not in df.columns:
        st.warning(
            f"Column `{LEAD_TIME_COL}` is unavailable; skipping survival curves."
        )
        return
    if SHIPPING_MODE_COL not in df.columns:
        st.warning(
            f"Column `{SHIPPING_MODE_COL}` is unavailable; skipping survival curves."
        )
        return

    plot_df = df[[LEAD_TIME_COL, SHIPPING_MODE_COL]].dropna()
    if plot_df.empty:
        st.warning("No valid rows for survival curves.")
        return

    st.subheader("Delivery Time Survival Curves by Shipping Mode")
    st.caption(
        "This is descriptive survival analysis of historical delivery duration, "
        "not a causal model."
    )

    max_day = int(plot_df[LEAD_TIME_COL].max())
    timeline = np.arange(0, max_day + 1)

    if HAS_PLOTLY:
        fig = go.Figure()
        for mode, subset in plot_df.groupby(SHIPPING_MODE_COL, dropna=False):
            durations = subset[LEAD_TIME_COL].to_numpy(dtype=float)
            if HAS_LIFELINES:
                kmf = KaplanMeierFitter()
                kmf.fit(durations, event_observed=np.ones(len(durations), dtype=int))
                surv = kmf.survival_function_.iloc[:, 0].values
                days = kmf.survival_function_.index.to_numpy()
            else:
                days = timeline
                surv = _empirical_survival_curve(durations, timeline)
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=surv,
                    mode="lines",
                    name=str(mode),
                )
            )
        fig.update_layout(
            xaxis_title="Days since order",
            yaxis_title="Survival probability (delivery not yet completed)",
            height=480,
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mode, subset in plot_df.groupby(SHIPPING_MODE_COL, dropna=False):
            durations = subset[LEAD_TIME_COL].to_numpy(dtype=float)
            if HAS_LIFELINES:
                kmf = KaplanMeierFitter()
                kmf.fit(durations, event_observed=np.ones(len(durations), dtype=int))
                ax.step(
                    kmf.survival_function_.index,
                    kmf.survival_function_.iloc[:, 0],
                    where="post",
                    label=str(mode),
                )
            else:
                surv = _empirical_survival_curve(durations, timeline)
                ax.step(timeline, surv, where="post", label=str(mode))
        ax.set_xlabel("Days since order")
        ax.set_ylabel("Survival probability (delivery not yet completed)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


def _resolve_heatmap_column(df: pd.DataFrame) -> str | None:
    for col in HEATMAP_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def render_risk_heatmap(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        st.warning(f"Column `{TARGET_COL}` is unavailable; skipping heatmap.")
        return
    if HEATMAP_ROW_COL not in df.columns:
        st.warning(f"Column `{HEATMAP_ROW_COL}` is unavailable; skipping heatmap.")
        return

    col_dim = _resolve_heatmap_column(df)
    if col_dim is None:
        st.warning("No suitable column dimension found for the risk heatmap.")
        return

    heat_df = df[[HEATMAP_ROW_COL, col_dim, TARGET_COL]].dropna()
    if heat_df.empty:
        st.warning("No valid rows for the risk heatmap.")
        return

    top_cols = top_categories(heat_df[col_dim], limit=15)
    heat_df = heat_df[heat_df[col_dim].astype(str).isin(top_cols)]

    pivot = (
        heat_df.groupby([HEATMAP_ROW_COL, col_dim], dropna=False)[TARGET_COL]
        .mean()
        .unstack(fill_value=np.nan)
    )
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[row_order, col_order]

    st.subheader("Late Delivery Risk Heatmap")
    st.caption(
        "Heatmap shows historical late-delivery rate by operational dimensions. "
        "It is a supplier-risk proxy because DataCo has no supplier master table."
    )

    if HAS_PLOTLY:
        fig = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="Reds",
            labels=dict(color="Late-delivery rate"),
        )
        fig.update_layout(
            xaxis_title=col_dim,
            yaxis_title=HEATMAP_ROW_COL,
            height=max(320, 40 * len(pivot.index)),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(pivot.index))))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(col_dim)
        ax.set_ylabel(HEATMAP_ROW_COL)
        fig.colorbar(im, ax=ax, label="Late-delivery rate")
        st.pyplot(fig)
        plt.close(fig)


def render_late_delivery_trend(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        st.warning(f"Column `{TARGET_COL}` is unavailable; skipping trend chart.")
        return
    if "_order_date_parsed" not in df.columns:
        st.warning("Could not parse order dates; skipping the monthly trend chart.")
        return

    trend_df = df[["_order_date_parsed", TARGET_COL]].dropna()
    if trend_df.empty:
        st.warning("No valid rows for the monthly trend chart.")
        return

    trend_df = trend_df.copy()
    trend_df["order_month"] = trend_df["_order_date_parsed"].dt.to_period("M")
    monthly = (
        trend_df.groupby("order_month", dropna=False)
        .agg(
            late_delivery_rate=(TARGET_COL, "mean"),
            shipment_count=(TARGET_COL, "count"),
        )
        .reset_index()
    )
    monthly["order_month"] = monthly["order_month"].astype(str)

    st.subheader("Monthly Late Delivery Trend")
    st.caption("Descriptive historical trend of late-delivery rate by order month.")

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=monthly["order_month"],
                y=monthly["late_delivery_rate"],
                mode="lines+markers",
                name="Late-delivery rate",
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthly["order_month"],
                y=monthly["shipment_count"],
                name="Row count",
                opacity=0.25,
                yaxis="y2",
            )
        )
        fig.update_layout(
            xaxis_title="Order month",
            yaxis=dict(title="Late-delivery rate", tickformat=".0%"),
            yaxis2=dict(
                title="Row count",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(monthly["order_month"], monthly["late_delivery_rate"], marker="o")
        ax1.set_ylabel("Late-delivery rate")
        ax1.tick_params(axis="x", rotation=35)
        ax2 = ax1.twinx()
        ax2.bar(monthly["order_month"], monthly["shipment_count"], alpha=0.25)
        ax2.set_ylabel("Row count")
        st.pyplot(fig)
        plt.close(fig)

    st.dataframe(
        monthly.rename(
            columns={
                "order_month": "Month",
                "late_delivery_rate": "Late-delivery rate",
                "shipment_count": "Row count",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_exploratory_visualizations() -> None:
    st.header("Exploratory Delivery Risk Visualizations")
    st.caption(
        "DataCo has no explicit supplier master table, so these charts use "
        "operational delivery dimensions as procurement-risk proxies."
    )

    raw_df = load_dataco_for_visuals()
    if raw_df is None:
        st.warning(
            f"Raw DataCo file not found at `{DATACO_RAW_CSV}`. "
            "Skipping exploratory visualizations."
        )
        return

    viz_df = clean_dataco_visual_columns(raw_df)
    render_lead_time_boxplot(viz_df)
    render_survival_curves(viz_df)
    render_risk_heatmap(viz_df)
    render_late_delivery_trend(viz_df)


def render_evaluation_section() -> None:
    st.header("Evaluation plots")
    show_image_if_exists(
        "confusion_matrix_xgboost.png",
        "Confusion matrix - XGBoost (primary)",
    )


def render_explainability_section() -> None:
    st.header("Explainability")
    show_image_if_exists(
        "feature_importance.png",
        "Feature importance - XGBoost (primary)",
    )


def render_ai_assistant_panel(
    clean_df: pd.DataFrame,
    model: Any,
    model_label: str,
    model_file: str,
    input_values: dict[str, Any],
    template: pd.Series,
    scheduled_by_mode: dict[str, int],
    base_rates: dict[str, float],
    threshold: float,
    ai_api_key: str,
    ai_model_name: str,
    assistant_ready: bool,
) -> None:
    st.header("AI assistant")
    st.caption(
        'Ask about this prediction, request scenario changes such as "try Same '
        'Day mode", or look up dataset stats. The assistant uses the hosted '
        "relay when configured, or the optional OpenRouter key from the sidebar."
    )

    def current_order_date() -> date:
        value = st.session_state.get(f"in_{DATE_COL}", date.today())
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return date.today()

    def score_inputs(
        values: dict[str, Any],
        with_model: Any,
        label: str,
        filename: str,
    ) -> dict[str, Any]:
        return prediction_snapshot(
            values,
            with_model,
            label,
            filename,
            template,
            base_rates,
            threshold,
            current_order_date(),
        )

    def _tool_set_shipment_inputs(**changes: Any) -> dict[str, Any]:
        res = ai.set_shipment_inputs_logic(clean_df, changes)
        if res["applied"]:
            merged = {**input_values, **res["applied"]}
            new_mode = str(merged.get(SHIPPING_MODE_COL, ""))
            if new_mode in scheduled_by_mode:
                merged[SCHEDULED_DAYS_COL] = scheduled_by_mode[new_mode]
            try:
                prediction = score_inputs(merged, model, model_label, model_file)
            except Exception as exc:  # noqa: BLE001
                return {**res, "error": str(exc)}

            st.session_state["pending_inputs"].update(res["applied"])
            st.session_state["last_prediction"] = prediction
            st.session_state["rerun_pending"] = True
            res = {**res, "prediction": prediction}
        return res

    def _tool_get_current_prediction() -> dict[str, Any]:
        return ai.describe_prediction_logic(st.session_state.get("last_prediction"))

    def _tool_query_dataset(
        metric: str, group_col: str | None = None
    ) -> dict[str, Any]:
        return ai.query_dataset_logic(clean_df, metric, group_col)

    def _tool_switch_model(model_file: str) -> dict[str, Any]:
        res = ai.switch_model_logic(model_file, available_model_files())
        if "error" in res:
            return res
        new_model, label, filename, _is_fallback = resolve_model(model_file)
        if new_model is None or label is None or filename is None:
            return {"error": f"could not load {model_file}"}
        try:
            prediction = score_inputs(input_values, new_model, label, filename)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

        st.session_state["active_model_file"] = model_file
        st.session_state["last_prediction"] = prediction
        st.session_state["rerun_pending"] = True
        return {
            "applied": model_file,
            "model_label": label,
            "prediction": prediction,
            "summary": f"switched to {label}",
        }

    tools_impl = {
        "set_shipment_inputs": _tool_set_shipment_inputs,
        "get_current_prediction": _tool_get_current_prediction,
        "query_dataset": _tool_query_dataset,
        "switch_model": _tool_switch_model,
    }

    def _stream_assistant_reply(messages: list[dict]) -> tuple[str, list[str]]:
        base_url = ai.AI_PROXY_URL or ai.DEFAULT_BASE_URL
        client = ai.OpenRouterClient(
            ai_api_key,
            ai_model_name or ai.DEFAULT_MODEL,
            base_url=base_url,
        )
        with st.chat_message("assistant"):
            status = st.status("Thinking...", expanded=False)
            placeholder = st.empty()
            pieces: list[str] = []
            final_text, actions = "", []
            try:
                for event in ai.run_agent_stream(client, messages, tools_impl):
                    kind = event["kind"]
                    if kind == "tool_start":
                        status.update(
                            label=f"Running `{event['name']}`...",
                            state="running",
                        )
                    elif kind == "tool_end":
                        status.write(event["summary"])
                    elif kind == "token":
                        pieces.append(event["text"])
                        placeholder.markdown("".join(pieces) + "...")
                    elif kind == "final":
                        final_text, actions = event["text"], event["actions"]
            except ai.OpenRouterError as exc:
                final_text, actions = f"Warning: {exc}", []
            status.update(label="Done", state="complete")
            placeholder.markdown(final_text or "(no response)")
            if actions:
                st.caption("Actions: " + " | ".join(actions))
        return final_text or "(no response)", actions

    def handle_user_message(user_text: str) -> None:
        st.session_state["chat_history"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        if not assistant_ready:
            warn = "Add your OpenRouter API key in the sidebar to enable the assistant."
            with st.chat_message("assistant"):
                st.markdown(warn)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": warn, "actions": []}
            )
            return

        convo = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state["chat_history"]
        ]
        messages = [{"role": "system", "content": ai.SYSTEM_PROMPT}, *convo]
        text, actions = _stream_assistant_reply(messages)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": text, "actions": actions}
        )

    col_summary, col_clear, _ = st.columns([1, 1, 2])
    summarize_clicked = col_summary.button(
        "Summarize this prediction",
        disabled=not assistant_ready,
    )
    if col_clear.button("Clear chat", disabled=not st.session_state["chat_history"]):
        st.session_state["chat_history"] = []
        st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("actions"):
                st.caption("Actions: " + " | ".join(msg["actions"]))

    user_prompt = st.chat_input("Message the assistant...")
    new_message = SUMMARIZE_PROMPT if summarize_clicked else (user_prompt or None)
    if new_message:
        handle_user_message(new_message)


def render_limitations_section() -> None:
    st.header("Limitations")
    st.markdown("""
- DataCo is a simulated teaching retail supply-chain dataset.
- It has no explicit supplier master table, so supplier-level views are
  approximated using operational proxies such as Shipping Mode, Market, Region,
  Category, or Department.
- Shipping Mode and Days for shipment (scheduled) are highly collinear and
  SLA-like, so the app reports a base-rate benchmark for honesty.
- Post-delivery fields such as Days for shipping (real), Delivery Status,
  shipping date, Order Status, and realized profit fields are excluded from
  prediction because they encode outcomes or leakage.
- Model behavior may drift for new markets, carriers, fulfillment processes, or
  future operations.
- The late/on-time threshold should be treated as a business-cost decision,
  balancing missed late deliveries against unnecessary expediting.
- This is an academic decision-support tool, not a production procurement
  system.
        """)


st.set_page_config(
    page_title="Late Delivery Risk Predictor",
    layout="wide",
)
st.title("Supplier Lead-Time and Late Delivery Risk Prediction for Retail Procurement")
st.caption(
    "DAP391m - Group 8, FPT University HCMC | DataCo Smart Supply Chain | "
    "binary Late_delivery_risk classification"
)

st.session_state.setdefault("active_model_file", None)
st.session_state.setdefault("pending_inputs", {})
st.session_state.setdefault("rerun_pending", False)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("last_prediction", None)

model, model_label, model_file, model_is_fallback = resolve_model(
    st.session_state["active_model_file"]
)
if model is None or model_label is None or model_file is None:
    st.error("No trained model found in `Data/filtered/model_outputs/`.")
    st.markdown(PIPELINE_HINT)
    st.stop()

if model_is_fallback:
    st.warning(
        "Loaded a fallback model instead of the XGBoost primary. Run "
        "`modal run modal_train.py` or `python src-code/05_modeling.py` to "
        "regenerate `xgb_model.pkl`."
    )

clean_df = load_clean_data()
if clean_df is None:
    st.error(
        "`clean_data.csv` not found. Run "
        "`python src-code/01_ingestion_cleaning.py` first."
    )
    st.stop()
if TARGET_COL not in clean_df.columns:
    st.error(f"`clean_data.csv` does not contain `{TARGET_COL}`.")
    st.stop()

template_row = build_template_row(clean_df)
sidebar_options = load_sidebar_options(clean_df)
scheduled_days_by_mode = load_scheduled_days_by_mode(clean_df)
base_rate_by_mode = load_shipping_mode_base_rates(clean_df)
decision_threshold = read_tuned_threshold()

seed_sidebar_state(template_row, sidebar_options)
apply_pending_inputs(sidebar_options)

order_row, sidebar_inputs, ai_key, ai_model, is_assistant_ready = render_sidebar(
    template_row,
    sidebar_options,
    scheduled_days_by_mode,
)

current_prediction, _features = render_prediction_section(
    model,
    model_label,
    model_file,
    order_row,
    template_row,
    base_rate_by_mode,
    decision_threshold,
)
current_prediction["inputs"] = {
    col: sidebar_inputs.get(col)
    for col, _label in EXPOSED_CATEGORICAL
    if col in sidebar_inputs
}
st.session_state["last_prediction"] = current_prediction

render_exploratory_visualizations()
render_evaluation_section()
render_explainability_section()
render_ai_assistant_panel(
    clean_df,
    model,
    model_label,
    model_file,
    sidebar_inputs,
    template_row,
    scheduled_days_by_mode,
    base_rate_by_mode,
    decision_threshold,
    ai_key,
    ai_model,
    is_assistant_ready,
)
render_limitations_section()

if st.session_state.get("rerun_pending"):
    st.session_state["rerun_pending"] = False
    st.rerun()
