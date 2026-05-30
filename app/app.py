"""
DAP391m — Consumer Credit-Default Risk Predictor (Streamlit).

Loads the primary XGBoost binary classifier (loan_status: 1 = default) trained
by src-code/05_modeling.py and scores an individual loan application, returning
the probability of default, a risk tier, and a lending recommendation.

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
CLEAN_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
RAW_CSV = PROJECT_ROOT / "Data" / "credit_risk_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

TARGET_COL = "loan_status"
CLASS_LABELS = ["Non-default", "Default"]
POSITIVE_CLASS = 1

# Engineered features that 04/05 derive from base inputs (kept in sync here so
# the deployed pipeline receives the same columns it was trained on).
HOME_OPTIONS = ["RENT", "OWN", "MORTGAGE", "OTHER"]
INTENT_OPTIONS = [
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
]
GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_FLAG_OPTIONS = ["N", "Y"]

MODEL_CANDIDATES: list[tuple[str, str, bool]] = [
    ("primary_model.pkl", "XGBoost (primary deployment model)", False),
    ("xgb_model.pkl", "XGBoost", False),
    ("best_model.pkl", "Best model by PR-AUC (fallback — may not be XGBoost)", True),
]

PIPELINE_HINT = "Run the modeling step first:\n- `python src-code/05_modeling.py`"

ARTIFACT_IMAGES: list[tuple[str, str]] = [
    ("shap_summary.png", "SHAP summary — XGBoost (test sample)"),
    ("feature_importance.png", "Feature importance — XGBoost (primary)"),
    ("model_comparison.png", "Model comparison — binary metrics"),
    ("confusion_matrix_xgboost.png", "Confusion matrix — XGBoost (primary)"),
]


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


def read_tuned_threshold() -> float:
    """Parse the F1-optimal threshold written by 05_modeling.py, else 0.5."""
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
    """Mirror src-code/04 + 05 feature derivation for a single applicant row."""
    out = df.copy()
    out["high_debt_burden"] = (out["loan_percent_income"] > 0.30).astype(int)
    out["credit_hist_ratio"] = (
        out["cb_person_cred_hist_length"] / out["person_age"]
    ).round(4)
    out["interest_to_income"] = (
        ((out["loan_int_rate"] / 100.0) * out["loan_amnt"]) / out["person_income"].clip(lower=1)
    ).round(4)
    out["income_bracket"] = pd.cut(
        out["person_income"],
        bins=[-1, 30000, 60000, 100000, float("inf")],
        labels=["<30k", "30k-60k", "60k-100k", "100k+"],
    ).astype(str)
    out["age_bracket"] = pd.cut(
        out["person_age"],
        bins=[-1, 25, 35, 50, float("inf")],
        labels=["<=25", "26-35", "36-50", "50+"],
    ).astype(str)
    out["loan_amount_bracket"] = pd.cut(
        out["loan_amnt"],
        bins=[-1, 5000, 10000, 20000, float("inf")],
        labels=["<5k", "5k-10k", "10k-20k", "20k+"],
    ).astype(str)
    return out


def positive_proba(model: Any, features: pd.DataFrame) -> float:
    proba = model.predict_proba(features)[0]
    final = getattr(model, "named_steps", {}).get("model", model)
    classes = list(getattr(final, "classes_", [0, 1]))
    pos_idx = classes.index(POSITIVE_CLASS) if POSITIVE_CLASS in classes else 1
    return float(proba[pos_idx])


def risk_tier(p_default: float) -> tuple[str, str]:
    if p_default < 0.20:
        return "LOW", "green"
    if p_default < 0.50:
        return "MODERATE", "orange"
    return "HIGH", "red"


def lending_note(p_default: float, threshold: float) -> str:
    decision = "DECLINE / manual review" if p_default >= threshold else "APPROVE"
    base = (
        f"At the deployed decision threshold ({threshold:.2f}), this application "
        f"would be **{decision}** (P(default) = {p_default:.1%})."
    )
    if p_default >= 0.50:
        return base + (
            " High default probability — recommend declining, requiring a"
            " co-signer, or repricing with a higher interest rate."
        )
    if p_default >= 0.20:
        return base + (
            " Moderate risk — consider risk-based pricing, a lower loan amount,"
            " or additional income verification."
        )
    return base + " Low default probability — standard approval terms apply."


def show_image_if_exists(filename: str, caption: str) -> None:
    path = OUTPUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"`{filename}` not found in model outputs.")


# ── page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Default Risk Predictor", layout="wide")
st.title("Consumer Credit-Default Risk Predictor")
st.caption(
    "DAP391m — Group 8, FPT University HCMC · binary loan-default classification"
)

model, model_label, model_is_fallback = resolve_model()
if model is None:
    st.error("No trained model found in `Data/filtered/model_outputs/`.")
    st.markdown(PIPELINE_HINT)
    st.stop()

if model_is_fallback:
    st.warning(
        "Loaded `best_model.pkl` as fallback (may not be the XGBoost primary "
        "model). Run `python src-code/05_modeling.py` to generate "
        "`primary_model.pkl`."
    )

threshold = read_tuned_threshold()

# ── sidebar: applicant inputs ────────────────────────────────────────────────
with st.sidebar:
    st.header("Loan application")
    person_age = st.number_input("Age", 18, 100, 30, 1)
    person_income = st.number_input("Annual income (USD)", 0, 1_000_000, 55_000, 1000)
    person_emp_length = st.number_input(
        "Employment length (years)", 0.0, 60.0, 5.0, 0.5
    )
    person_home_ownership = st.selectbox("Home ownership", HOME_OPTIONS)
    loan_intent = st.selectbox("Loan intent", INTENT_OPTIONS)
    loan_grade = st.selectbox("Loan grade", GRADE_OPTIONS)
    loan_amnt = st.number_input("Loan amount (USD)", 500, 40_000, 10_000, 500)
    loan_int_rate = st.number_input("Interest rate (%)", 1.0, 30.0, 11.0, 0.1)
    cb_person_default_on_file = st.selectbox(
        "Prior default on file", DEFAULT_FLAG_OPTIONS
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit history length (years)", 0, 40, 5, 1
    )
    run_predict = st.button("Predict default risk", type="primary")

# ── prediction ───────────────────────────────────────────────────────────────
st.header("Prediction")
if not run_predict:
    st.info(
        "Enter applicant details in the sidebar, then click **Predict default risk**."
    )
else:
    loan_percent_income = round(loan_amnt / max(person_income, 1), 4)
    base_row = pd.DataFrame(
        [
            {
                "person_age": person_age,
                "person_income": person_income,
                "person_home_ownership": person_home_ownership,
                "person_emp_length": person_emp_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
            }
        ]
    )
    features = add_engineered_features(base_row)

    p_default = positive_proba(model, features)
    tier, color = risk_tier(p_default)

    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model loaded", model_label or "Unknown")
    with c2:
        st.metric("P(default)", f"{p_default:.1%}")
    with c3:
        st.metric("Risk tier", tier)
        st.markdown(f":{color}[{tier} default risk]")

    st.progress(min(max(p_default, 0.0), 1.0))

    pcols = st.columns(2)
    with pcols[0]:
        st.metric("P(Non-default)", f"{1 - p_default:.1%}")
    with pcols[1]:
        st.metric("P(Default)", f"{p_default:.1%}")

    st.markdown("**Lending recommendation**")
    st.write(lending_note(p_default, threshold))
    st.caption(f"Loan-to-income ratio: {loan_percent_income:.1%}")

    with st.expander("Feature vector used for this prediction"):
        st.dataframe(features, use_container_width=True, hide_index=True)

# ── model performance ────────────────────────────────────────────────────────
st.header("Model performance")
comparison_path = OUTPUT_DIR / "model_comparison.csv"
if comparison_path.exists():
    st.dataframe(
        pd.read_csv(comparison_path), use_container_width=True, hide_index=True
    )
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

# ── limitations ──────────────────────────────────────────────────────────────
st.header("Limitations")
st.markdown("""
- `loan_grade` and `loan_int_rate` are lender-assigned, decision-time inputs \
that strongly reflect creditworthiness; they are valid predictors but partly \
encode the lender's own risk assessment.
- The model is trained on a historical credit dataset; **new applicant \
populations or economic conditions** may shift default behaviour (drift).
- The approve/decline call uses the **F1-tuned threshold**; production policy \
should set the threshold from the business cost of false approvals vs. lost \
good loans.
- This is an academic decision-support tool, **not** a regulated credit \
adjudication system.
""")
