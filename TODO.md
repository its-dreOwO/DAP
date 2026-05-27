# Project TODO — DAP391m Group 8

> **Status key:** ✅ Done · 🔄 In Progress · ⬜ Not Started

---

## Week 1 — Data & Cleaning

| Task | File | Status |
|------|------|--------|
| Define data architecture & column roles | `CLAUDE.md` | ✅ |
| Ingest & clean raw dataset | `src-code/01_ingestion_cleaning.py` | ✅ |
| Produce `clean_data.csv` | `Data/filtered/clean_data.csv` | ✅ |

---

## Week 2 — SQL & Modeling

| Task | File | Status |
|------|------|--------|
| Write all 6 required SQL queries | `sql/analysis.sql` | ✅ |
| Execute queries & save results in Python | `src-code/02_sql_analysis.py` | ✅ |
| EDA: distributions, heatmap, class balance | `src-code/03_eda.py` | ✅ |
| Feature engineering (temporal, expanding window, OHE) | `src-code/04_feature_engineering.py` | ✅ |
| Train 4 models + SHAP on XGBoost | `src-code/05_modeling.py` | ✅ |
| Evaluate: PR-AUC, Recall, ROC-AUC, F1 | `src-code/05_modeling.py` | ✅ |

---

## Week 3 — Visualisation

| Task | File | Status |
|------|------|--------|
| Plotly supplier ranking, boxplots, trend lines | `src-code/06_visualization_advanced.py` | ✅ |
| Odds-ratio analysis (Logistic Regression interpretation) | `src-code/05_modeling.py` | ⬜ optional report add-on |

---

## Week 4 — Power BI

| Task | Status |
|------|--------|
| Supplier scorecard with risk alerts | ⬜ |
| Drill-down by supplier / region / carrier | ⬜ |
| Connect to exported model output CSV | ⬜ |

---

## Week 4–5 — Streamlit App

| Task | File | Status |
|------|------|--------|
| 3-class input form → predict Low/Medium/High + probabilities | `app/app.py` | ✅ |
| Load saved XGBoost model (`primary_model.pkl`) | `app/app.py` | ✅ |
| Deploy / run locally end-to-end | — | ✅ smoke-tested locally |

---

## Report (`report/main.tex`)

| Section | Status |
|---------|--------|
| Structure / skeleton | ✅ (166 lines) |
| Introduction & research questions | ⬜ |
| Literature review & theoretical frameworks | ⬜ |
| Data description & leakage analysis | ⬜ |
| Methodology (pipeline, models) | ⬜ |
| Results & evaluation tables | ⬜ |
| Discussion & limitations | ⬜ |
| Conclusion | ⬜ |

---

## AI Audit Log

| Task | Status |
|------|--------|
| Template in place | ✅ |
| 15–20 core prompts logged | 🔄 in progress |
| ≥3 hallucination checks documented | ⬜ |
| Human Delta filled in by students | ⬜ ongoing |
| Entries exported to `.xlsx` before submission | ⬜ |

---

## Submission Checklist

- [x] All 6 pipeline scripts run clean end-to-end
- [x] `sql/analysis.sql` matches output from `02_sql_analysis.py`
- [x] 4 model comparison table with PR-AUC as primary metric (`Data/filtered/model_outputs/model_comparison.csv`)
- [x] SHAP summary plot saved (`Data/filtered/model_outputs/shap_summary.png`)
- [ ] Power BI `.pbix` file committed
- [x] Streamlit app runs: `streamlit run app/app.py`
- [ ] Report PDF compiled from `main.tex` (10–12 pages)
- [ ] AI Audit Log `.xlsx` complete (15–20 entries, ≥3 hallucination logs)
- [ ] CI passes (black + flake8)
