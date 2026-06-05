"""
DAP391m — Late-Delivery Risk Predictor (Streamlit).

Loads the primary XGBoost binary classifier (Late_delivery_risk: 1 = late)
trained by src-code/05_modeling.py on the DataCo Smart Supply Chain dataset and
scores a single shipment, returning P(late), a risk tier, and an operational
recommendation.

The dominant signal is the shipping mode (faster classes have tight, routinely
missed SLAs); other features only rank shipments *within* a mode. The app makes
this explicit and shows the historical per-mode base rate alongside the model's
prediction (the honest base-rate benchmark from the dataset assessment).

An optional cloud AI assistant (OpenRouter) can read the current prediction,
change shipment inputs / the active model and re-run, query real dataset stats,
and summarize the result — see app/ai_assistant.py.

Run from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# `streamlit run app/app.py` puts this folder on sys.path, but other launchers
# (AppTest, `python app/app.py`) do not — make the sibling import work anyway.
_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import ai_assistant as ai  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

TARGET_COL = "Late_delivery_risk"
GROUP_KEY = "Order Id"
DATE_COL = "order date (DateOrders)"
SHIPPING_MODE_COL = "Shipping Mode"
SCHEDULED_DAYS_COL = "Days for shipment (scheduled)"
CLASS_LABELS = ["On-time", "Late"]  # index == class int (0, 1)
POSITIVE_CLASS = 1

# Columns 05_modeling.py drops before training (target, IDs, PII, raw date).
# Kept here so the clean-data-derived default template matches the model's
# expected feature set even if feature_names_in_ is unavailable.
DROP_FROM_FEATURES = {
    TARGET_COL,
    GROUP_KEY,
    "Order Item Id",
    "Latitude",
    "Longitude",
    DATE_COL,
}
# Engineered date features 05_modeling.py derives from DATE_COL.
ENGINEERED_DATE_FEATURES = ["order_month", "order_day_of_week", "order_hour"]

# Shipping Mode is perfectly collinear with scheduled days (one mode == one SLA).
# Picking a mode fixes the scheduled-days input automatically.
MODE_TO_SCHEDULED_DAYS = {
    "Same Day": 0,
    "First Class": 1,
    "Second Class": 2,
    "Standard Class": 4,
}

MODEL_CANDIDATES: list[tuple[str, str, bool]] = [
    ("primary_model.pkl", "XGBoost (primary deployment model)", False),
    ("xgb_model.pkl", "XGBoost", False),
    ("best_model.pkl", "Best model by PR-AUC (fallback — may not be XGBoost)", True),
    ("random_forest_model.pkl", "Random Forest (fallback)", True),
]
MODEL_LABELS = {fn: label for fn, label, _ in MODEL_CANDIDATES}
NON_FALLBACK_FILES = {"primary_model.pkl", "xgb_model.pkl"}

PIPELINE_HINT = (
    "Train the models first:\n"
    "- `modal run modal_train.py` (GPU), or\n"
    "- `.venv/bin/python3 src-code/05_modeling.py` (local)"
)

ARTIFACT_IMAGES: list[tuple[str, str]] = [
    ("shap_summary.png", "SHAP summary — XGBoost (test sample)"),
    ("feature_importance.png", "Feature importance — XGBoost (primary)"),
    ("model_comparison.png", "Model comparison — binary metrics"),
    ("confusion_matrix_xgboost.png", "Confusion matrix — XGBoost (primary)"),
]

# Features exposed in the sidebar (the rest default silently from clean_data).
# Shipping mode is the primary driver; the others probe residual within-mode
# signal (market/region/segment/department) and order economics.
EXPOSED_CATEGORICAL = [
    (SHIPPING_MODE_COL, "Shipping mode"),
    ("Market", "Market"),
    ("Order Region", "Order region"),
    ("Customer Segment", "Customer segment"),
    ("Department Name", "Department"),
    ("Category Name", "Product category"),
    ("Type", "Payment type"),
]

SUMMARIZE_PROMPT = (
    "Summarize the current prediction in plain language for a non-technical "
    "reader: the P(late), the risk tier, and how it compares to this shipping "
    "mode's historical base rate. Keep it to a few sentences."
)


def _force_cpu_inference(model: Any) -> Any:
    """Pin the XGBoost booster to CPU.

    The models are trained on GPU (Modal T4), so the unpickled booster defaults
    to ``device=cuda``. Scoring a single CPU-resident row then triggers a
    device-mismatch warning and a slow DMatrix fallback. We always infer on CPU
    here, so force it once at load time.
    """
    final = getattr(model, "named_steps", {}).get("model", model)
    try:
        final.set_params(device="cpu")
    except Exception:  # noqa: BLE001 - non-XGBoost estimators have no device
        pass
    get_booster = getattr(final, "get_booster", None)
    if callable(get_booster):
        try:
            get_booster().set_param({"device": "cpu"})
        except Exception:  # noqa: BLE001 - booster may be unavailable
            pass
    return model


@st.cache_resource
def load_model_file(filename: str) -> Any:
    with (OUTPUT_DIR / filename).open("rb") as handle:
        return _force_cpu_inference(pickle.load(handle))


def resolve_model(preferred: str | None = None) -> tuple[Any | None, str | None, bool]:
    """Load the preferred model if given/available, else the first candidate."""
    order: list[str] = []
    if preferred:
        order.append(preferred)
    order += [fn for fn, _, _ in MODEL_CANDIDATES if fn != preferred]
    for filename in order:
        if not (OUTPUT_DIR / filename).exists():
            continue
        try:
            model = load_model_file(filename)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Skipping `{filename}`: {exc}")
            continue
        label = MODEL_LABELS.get(filename, filename)
        is_fallback = filename not in NON_FALLBACK_FILES
        return model, f"{label} (`{filename}`)", is_fallback
    return None, None, False


def available_model_files() -> list[str]:
    return [fn for fn, _, _ in MODEL_CANDIDATES if (OUTPUT_DIR / fn).exists()]


@st.cache_data
def load_clean_data() -> pd.DataFrame | None:
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV)
    return None


def _column_defaults(frame: pd.DataFrame) -> dict[str, Any]:
    """Representative value per column: mode for categoricals, median for numerics."""
    defaults: dict[str, Any] = {}
    for col in frame.columns:
        series = frame[col].dropna()
        if series.empty:
            defaults[col] = 0
        elif series.dtype.kind in "biufc":
            defaults[col] = float(series.median())
        else:
            defaults[col] = str(series.mode().iloc[0])
    return defaults


@st.cache_data
def feature_defaults(_df: pd.DataFrame) -> dict[str, Any]:
    """Global representative row (fallback for modes with sparse coverage)."""
    return _column_defaults(_df)


@st.cache_data
def mode_feature_defaults(_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-shipping-mode defaults so the non-exposed feature row stays *coherent*
    with the chosen mode. Global column-wise defaults are all Standard-Class-typical
    (Standard is ~60% of rows); pairing them with a faster mode yields an
    out-of-distribution record that the model over-predicts as late. Conditioning
    the defaults on the mode keeps predictions calibrated to each mode's base rate."""
    out: dict[str, dict[str, Any]] = {}
    for mode, sub in _df.groupby(SHIPPING_MODE_COL):
        out[str(mode)] = _column_defaults(sub)
    return out


@st.cache_data
def category_options(_df: pd.DataFrame, col: str) -> list[str]:
    return sorted(_df[col].dropna().astype(str).unique().tolist())


@st.cache_data
def mode_base_rates(_df: pd.DataFrame) -> dict[str, float]:
    """Historical late rate per shipping mode — the base-rate benchmark."""
    grp = _df.groupby(SHIPPING_MODE_COL)[TARGET_COL].mean()
    return {str(k): float(v) for k, v in grp.items()}


def expected_feature_columns(model: Any, clean_df: pd.DataFrame | None) -> list[str]:
    """The columns the fitted pipeline expects, in order."""
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    # Fallback: reconstruct from clean data the same way 05_modeling.py does.
    if clean_df is None:
        return []
    base = [c for c in clean_df.columns if c not in DROP_FROM_FEATURES]
    return base + ENGINEERED_DATE_FEATURES


def read_tuned_threshold() -> float:
    """Parse a tuned threshold if 05 wrote one; else the natural 0.5 cutoff."""
    path = OUTPUT_DIR / "threshold_analysis.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "tuned on train" in line and ":" in line:
                try:
                    return float(line.split(":")[-1].strip())
                except ValueError:
                    pass
    return 0.5


def build_feature_row(
    expected_cols: list[str],
    defaults: dict[str, Any],
    overrides: dict[str, Any],
    order_date: pd.Timestamp,
) -> pd.DataFrame:
    """One-row frame with exactly the columns the pipeline was trained on."""
    row: dict[str, Any] = {}
    for col in expected_cols:
        if col == "order_month":
            row[col] = order_date.month
        elif col == "order_day_of_week":
            row[col] = order_date.dayofweek
        elif col == "order_hour":
            row[col] = order_date.hour
        elif col in overrides:
            row[col] = overrides[col]
        elif col in defaults:
            row[col] = defaults[col]
        else:
            row[col] = 0  # unseen column — OHE handle_unknown='ignore' tolerates it
    return pd.DataFrame([row], columns=expected_cols)


def positive_proba(model: Any, features: pd.DataFrame) -> float:
    proba = model.predict_proba(features)[0]
    final = getattr(model, "named_steps", {}).get("model", model)
    classes = list(getattr(final, "classes_", [0, 1]))
    pos_idx = classes.index(POSITIVE_CLASS) if POSITIVE_CLASS in classes else 1
    return float(proba[pos_idx])


def risk_tier(p_late: float) -> tuple[str, str]:
    if p_late < 0.40:
        return "LOW", "green"
    if p_late < 0.65:
        return "MODERATE", "orange"
    return "HIGH", "red"


def show_image_if_exists(filename: str, caption: str) -> None:
    path = OUTPUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"`{filename}` not found in model outputs.")


# ── page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Late-Delivery Risk Predictor", layout="wide")
st.title("Supplier Late-Delivery Risk Predictor")
st.caption(
    "DAP391m — Group 8, FPT University HCMC · DataCo Smart Supply Chain · "
    "binary Late_delivery_risk classification"
)

# Session-state defaults. `pending_inputs` carries AI-requested input changes that
# must be applied *before* the sidebar widgets are instantiated (Streamlit forbids
# writing a widget-backed key after its widget exists in the same run).
st.session_state.setdefault("active_model_file", None)
st.session_state.setdefault("pending_inputs", {})
st.session_state.setdefault("rerun_pending", False)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("last_prediction", None)

model, model_label, model_is_fallback = resolve_model(
    st.session_state["active_model_file"]
)
if model is None:
    st.error("No trained model found in `Data/filtered/model_outputs/`.")
    st.markdown(PIPELINE_HINT)
    st.stop()

if model_is_fallback:
    st.warning(
        "Loaded a fallback model (not the XGBoost primary). Run "
        "`modal run modal_train.py` to regenerate `primary_model.pkl`."
    )

clean_df = load_clean_data()
if clean_df is None:
    st.error(
        "`clean_data.csv` not found — needed to default the non-exposed feature "
        "columns. Run `.venv/bin/python3 src-code/01_ingestion_cleaning.py` first."
    )
    st.stop()

defaults = feature_defaults(clean_df)
mode_defaults = mode_feature_defaults(clean_df)
expected_cols = expected_feature_columns(model, clean_df)
threshold = read_tuned_threshold()
base_rates = mode_base_rates(clean_df)

# Seed widget-backed keys (once), then apply any AI-requested pending changes —
# both must happen before the sidebar widgets below are created.
for col, _label in EXPOSED_CATEGORICAL:
    if col in clean_df.columns:
        key = f"in_{col}"
        if key not in st.session_state:
            options = category_options(clean_df, col)
            seed = str(defaults.get(col, options[0] if options else ""))
            st.session_state[key] = (
                seed if seed in options else (options[0] if options else "")
            )
NUMERIC_SEED = {
    "Order Item Quantity": int(defaults.get("Order Item Quantity", 1) or 1),
    "Sales": float(defaults.get("Sales", 200.0) or 0.0),
    "Order Item Discount Rate": float(
        defaults.get("Order Item Discount Rate", 0.1) or 0.0
    ),
}
for col, seed in NUMERIC_SEED.items():
    if col in clean_df.columns:
        st.session_state.setdefault(f"in_{col}", seed)

if st.session_state["pending_inputs"]:
    for col, value in st.session_state["pending_inputs"].items():
        st.session_state[f"in_{col}"] = value
    st.session_state["pending_inputs"] = {}

# ── sidebar: shipment inputs ─────────────────────────────────────────────────
overrides: dict[str, Any] = {}
with st.sidebar:
    st.header("Shipment")
    for col, label in EXPOSED_CATEGORICAL:
        if col not in clean_df.columns:
            continue
        options = category_options(clean_df, col)
        overrides[col] = st.selectbox(label, options, key=f"in_{col}")

    # Scheduled days is collinear with the mode → derive, don't ask.
    mode = overrides.get(SHIPPING_MODE_COL)
    if mode in MODE_TO_SCHEDULED_DAYS:
        overrides[SCHEDULED_DAYS_COL] = MODE_TO_SCHEDULED_DAYS[mode]
        st.caption(
            f"Scheduled shipping days for **{mode}**: "
            f"`{MODE_TO_SCHEDULED_DAYS[mode]}` (fixed by mode)."
        )

    st.divider()
    st.subheader("Order economics")
    if "Order Item Quantity" in clean_df.columns:
        overrides["Order Item Quantity"] = st.number_input(
            "Item quantity", 1, 50, step=1, key="in_Order Item Quantity"
        )
    if "Sales" in clean_df.columns:
        overrides["Sales"] = st.number_input(
            "Sales (USD)", 0.0, 5000.0, step=10.0, key="in_Sales"
        )
    if "Order Item Discount Rate" in clean_df.columns:
        overrides["Order Item Discount Rate"] = st.slider(
            "Discount rate", 0.0, 0.5, step=0.01, key="in_Order Item Discount Rate"
        )

    st.divider()
    order_date = st.date_input("Order date")

    st.divider()
    st.subheader("AI assistant (OpenRouter)")
    ai_api_key = st.text_input(
        "OpenRouter API key", type="password", key="openrouter_key"
    )
    ai_model_name = st.text_input(
        "Model", value=ai.DEFAULT_MODEL, key="openrouter_model"
    )

order_ts = pd.Timestamp(order_date)


def compute_prediction(override_map: dict[str, Any], with_model: Any) -> dict[str, Any]:
    """Score one shipment from an override map; returns a JSON-friendly summary."""
    mode = override_map.get(SHIPPING_MODE_COL, "")
    row_defaults = {**defaults, **mode_defaults.get(str(mode), {})}
    features = build_feature_row(expected_cols, row_defaults, override_map, order_ts)
    p_late = positive_proba(with_model, features)
    tier, _color = risk_tier(p_late)
    rate = base_rates.get(str(mode))
    out: dict[str, Any] = {
        "p_late": round(p_late, 4),
        "tier": tier,
        "shipping_mode": str(mode),
    }
    if rate is not None:
        out["base_rate"] = round(rate, 4)
        out["delta_vs_base"] = round(p_late - rate, 4)
    out["inputs"] = {k: override_map.get(k) for k, _ in EXPOSED_CATEGORICAL}
    return out


def prediction_snapshot(
    override_map: dict[str, Any], with_model: Any, label: str
) -> dict[str, Any]:
    """Build the full ``last_prediction`` dict the AI's get_current_prediction
    tool reads — ``compute_prediction`` plus the active model label + threshold.

    Both the main render path and the AI's mutating tools write this exact shape
    so a set→get within one agent turn observes the change immediately, instead
    of reading the pre-turn snapshot until the next rerun.
    """
    return {
        **compute_prediction(override_map, with_model),
        "model_label": label,
        "threshold": threshold,
    }


# ── prediction (always computed for current inputs) ──────────────────────────
st.header("Prediction")
mode = overrides.get(SHIPPING_MODE_COL, "")
row_defaults = {**defaults, **mode_defaults.get(str(mode), {})}
features = build_feature_row(expected_cols, row_defaults, overrides, order_ts)
p_late = positive_proba(model, features)
tier, color = risk_tier(p_late)
mode_rate = base_rates.get(str(mode))

# Persist for the AI's get_current_prediction tool and the summarize button.
st.session_state["last_prediction"] = prediction_snapshot(overrides, model, model_label)

st.subheader("Result")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Model loaded", model_label or "Unknown")
with c2:
    st.metric("P(late)", f"{p_late:.1%}")
with c3:
    st.metric("Risk tier", tier)
    st.markdown(f":{color}[{tier} late-delivery risk]")

st.progress(min(max(p_late, 0.0), 1.0))

pcols = st.columns(2)
with pcols[0]:
    st.metric("P(on-time)", f"{1 - p_late:.1%}")
with pcols[1]:
    if mode_rate is not None:
        st.metric(
            f"Base rate · {mode}",
            f"{mode_rate:.1%}",
            delta=f"{(p_late - mode_rate):+.1%} vs base",
        )

if mode_rate is not None:
    st.caption(
        "Base rate = historical late share for this shipping mode. The model's "
        "edge over it comes from ranking shipments *within* a mode — the dataset "
        "assessment shows ML adds only ~+0.01 ROC-AUC over this one-line lookup."
    )

with st.expander("Feature vector used for this prediction"):
    # Transposing yields a single object column of mixed str/int/float values,
    # which Arrow can't serialize — render every value as text.
    feature_view = features.T.reset_index()
    feature_view.columns = ["feature", "value"]
    feature_view["value"] = feature_view["value"].astype(str)
    st.dataframe(feature_view, width="stretch", hide_index=True)


# ── AI assistant ─────────────────────────────────────────────────────────────
st.header("AI assistant")
st.caption(
    'Ask about this prediction, request scenario changes (e.g. "try Same Day '
    'mode" or "compare Same Day vs Standard Class"), or look up dataset stats. '
    "Requires an OpenRouter API key in the sidebar."
)


def _tool_set_shipment_inputs(**changes: Any) -> dict[str, Any]:
    res = ai.set_shipment_inputs_logic(clean_df, changes)
    if res["applied"]:
        merged = {**overrides, **res["applied"]}
        new_mode = merged.get(SHIPPING_MODE_COL)
        if new_mode in MODE_TO_SCHEDULED_DAYS:
            merged[SCHEDULED_DAYS_COL] = MODE_TO_SCHEDULED_DAYS[new_mode]
        prediction = prediction_snapshot(merged, model, model_label)
        st.session_state["pending_inputs"].update(res["applied"])
        # Make the write visible to a same-turn get_current_prediction; the
        # sidebar/UI catch up on the next rerun.
        st.session_state["last_prediction"] = prediction
        st.session_state["rerun_pending"] = True
        res = {**res, "prediction": prediction}
    return res


def _tool_get_current_prediction() -> dict[str, Any]:
    return ai.describe_prediction_logic(st.session_state.get("last_prediction"))


def _tool_query_dataset(metric: str, group_col: str | None = None) -> dict[str, Any]:
    return ai.query_dataset_logic(clean_df, metric, group_col)


def _tool_switch_model(model_file: str) -> dict[str, Any]:
    res = ai.switch_model_logic(model_file, available_model_files())
    if "error" in res:
        return res
    new_model, label, _is_fb = resolve_model(model_file)
    if new_model is None:
        return {"error": f"could not load {model_file}"}
    prediction = prediction_snapshot(overrides, new_model, label)
    st.session_state["active_model_file"] = model_file
    st.session_state["last_prediction"] = prediction
    st.session_state["rerun_pending"] = True
    return {
        "applied": model_file,
        "model_label": label,
        "prediction": prediction,
        "summary": f"switched to {label}",
    }


TOOLS_IMPL = {
    "set_shipment_inputs": _tool_set_shipment_inputs,
    "get_current_prediction": _tool_get_current_prediction,
    "query_dataset": _tool_query_dataset,
    "switch_model": _tool_switch_model,
}


def _stream_assistant_reply(messages: list[dict]) -> tuple[str, list[str]]:
    """Stream one assistant turn into a live chat bubble; return (text, actions).

    Tool steps surface as transient status lines; the final answer streams in
    token-by-token. Returns the assembled text + action list for chat history.
    """
    client = ai.OpenRouterClient(ai_api_key, ai_model_name or ai.DEFAULT_MODEL)
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        placeholder = st.empty()
        pieces: list[str] = []
        final_text, actions = "", []
        try:
            for event in ai.run_agent_stream(client, messages, TOOLS_IMPL):
                kind = event["kind"]
                if kind == "tool_start":
                    status.update(label=f"Running `{event['name']}`…", state="running")
                elif kind == "tool_end":
                    status.write(event["summary"])
                elif kind == "token":
                    pieces.append(event["text"])
                    placeholder.markdown("".join(pieces) + "▌")
                elif kind == "final":
                    final_text, actions = event["text"], event["actions"]
        except ai.OpenRouterError as exc:
            final_text, actions = f"⚠️ {exc}", []
        status.update(label="Done", state="complete")
        placeholder.markdown(final_text or "(no response)")
        if actions:
            st.caption("Actions: " + " · ".join(actions))
    return final_text or "(no response)", actions


def handle_user_message(user_text: str) -> None:
    """Render the user's message, then stream the assistant's reply live."""
    st.session_state["chat_history"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    if not ai_api_key:
        warn = "⚠️ Add your OpenRouter API key in the sidebar to enable the assistant."
        with st.chat_message("assistant"):
            st.markdown(warn)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": warn, "actions": []}
        )
        return

    convo = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["chat_history"]
    ]
    messages = [{"role": "system", "content": ai.SYSTEM_PROMPT}, *convo]
    text, actions = _stream_assistant_reply(messages)
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": text, "actions": actions}
    )


col_btn, _ = st.columns([1, 3])
summarize_clicked = col_btn.button("Summarize this prediction", disabled=not ai_api_key)

# Render the prior conversation, then stream any new turn live below it.
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("actions"):
            st.caption("Actions: " + " · ".join(msg["actions"]))

user_prompt = st.chat_input("Message the assistant…")
new_message = SUMMARIZE_PROMPT if summarize_clicked else (user_prompt or None)
if new_message:
    handle_user_message(new_message)

# ── model performance ────────────────────────────────────────────────────────
st.header("Model performance")
comparison_path = OUTPUT_DIR / "model_comparison.csv"
if comparison_path.exists():
    st.dataframe(pd.read_csv(comparison_path), width="stretch", hide_index=True)
else:
    st.warning(f"`model_comparison.csv` not found. {PIPELINE_HINT}")

st.subheader("Evaluation plots")
plot_cols = st.columns(2)
for idx, (filename, caption) in enumerate(ARTIFACT_IMAGES[2:]):
    with plot_cols[idx % 2]:
        show_image_if_exists(filename, caption)

# ── explainability ───────────────────────────────────────────────────────────
st.header("Explainability")
explain_cols = st.columns(2)
for idx, (filename, caption) in enumerate(ARTIFACT_IMAGES[:2]):
    with explain_cols[idx % 2]:
        show_image_if_exists(filename, caption)

# Apply AI-requested input/model changes by re-running once, after rendering.
if st.session_state.get("rerun_pending"):
    st.session_state["rerun_pending"] = False
    st.rerun()
