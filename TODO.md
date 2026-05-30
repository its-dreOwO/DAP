# Project TODO — DAP391m Group 8

> **Status key:** ✅ Done · 🔄 In Progress · ⬜ Not Started
> **2026-05-29:** Project pivoted to **Consumer Credit-Default Risk** (`Data/credit_risk_dataset.csv`). Pipeline `01`–`06`, SQL, and the Streamlit app were rebuilt for the new dataset. Old supply-chain work is in `Data/archived/`.

---

## Week 1 — Data & Cleaning

| Task | File | Status |
|------|------|--------|
| Define data architecture & column roles | `CLAUDE.md` | ✅ |
| Ingest & clean credit dataset (impossible-value + dup removal) | `src-code/01_ingestion_cleaning.py` | ✅ |
| Produce `clean_data.csv` (32,409 rows) | `Data/filtered/clean_data.csv` | ✅ |

---

## Week 2 — SQL & Modeling

| Task | File | Status |
|------|------|--------|
| 6 credit SQL queries (grade/home/intent/income/amount/penetration) | `sql/analysis.sql` | ✅ |
| Execute queries & save results | `src-code/02_sql_analysis.py` | ✅ |
| EDA: distributions, correlation heatmap, default-rate-by-category | `src-code/03_eda.py` | ✅ |
| Feature engineering (debt burden, ratios, brackets, OHE) | `src-code/04_feature_engineering.py` | ✅ |
| Train 4 binary models + SHAP on XGBoost | `src-code/05_modeling.py` | ✅ |
| Evaluate: PR-AUC, Recall, ROC-AUC, F1 + threshold tuning | `src-code/05_modeling.py` | ✅ |
| Baseline + imbalance-weighted experiments | `Data/filtered/model_outputs/` | ✅ |

---

## Week 3 — Visualisation

| Task | File | Status |
|------|------|--------|
| Plotly: default rate by grade/intent, income×grade matrix, model comparison | `src-code/06_visualization_advanced.py` | ✅ |
| Odds-ratio analysis (Logistic Regression interpretation) | `src-code/05_modeling.py` | ⬜ optional report add-on |

---

## Week 4 — Power BI

| Task | Status |
|------|--------|
| Credit risk scorecard with default-rate alerts | ⬜ |
| Drill-down by grade / intent / income band | ⬜ |
| Connect to exported model output CSV | ⬜ |

---

## Week 4–5 — Streamlit App

| Task | File | Status |
|------|------|--------|
| Applicant input form → P(default) + risk tier + recommendation | `app/app.py` | ✅ |
| Load saved XGBoost model (`primary_model.pkl`) | `app/app.py` | ✅ |
| Run locally end-to-end | — | ✅ prediction logic smoke-tested |

---

## Report (`report/main.tex`)

| Section | Status |
|---------|--------|
| Structure / skeleton | 🔄 needs re-framing for credit domain |
| Introduction & research questions | ⬜ |
| Literature review & frameworks | ⬜ (credit-risk literature) |
| Data description & leakage discussion (grade/int_rate stance) | ⬜ |
| Methodology (pipeline, models, threshold) | ⬜ |
| Results & evaluation tables | ⬜ baseline + weighted |
| Discussion & limitations (imbalance trade-off, drift) | ⬜ |
| Conclusion | ⬜ |

---

## AI Audit Log

| Task | Status |
|------|--------|
| Template in place | ✅ |
| Dataset-pivot VERIFICATION entry logged | ✅ |
| 15–20 core prompts logged | 🔄 |
| ≥3 hallucination checks documented | ⬜ |
| Human Delta filled in by students | ⬜ ongoing |
| Entries exported to `.xlsx` before submission | ⬜ |

---

## Submission Checklist

- [x] All 6 pipeline scripts run clean end-to-end (credit dataset)
- [x] `sql/analysis.sql` matches output from `02_sql_analysis.py`
- [x] 4-model comparison with PR-AUC primary (`model_comparison.csv`)
- [x] Baseline vs imbalance-weighted table (`model_comparison_all_experiments.csv`)
- [x] SHAP summary saved (`shap_summary.png`)
- [ ] Power BI `.pbix` committed
- [x] Streamlit app runs: `streamlit run app/app.py`
- [ ] Report PDF compiled from `main.tex` (10–12 pages)
- [ ] AI Audit Log `.xlsx` complete (15–20 entries, ≥3 hallucination logs)
- [ ] CI passes (black + flake8)
