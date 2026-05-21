# Project TODO — DAP391m Group 8

> **Status key:** ✅ Done · 🔄 In Progress · ⬜ Not Started

---

## Week 1 — Data & Cleaning

| Task | File | Status |
|------|------|--------|
| Define data architecture & column roles | `CLAUDE.md` | ✅ |
| Ingest & clean raw dataset | `src-code/01_ingestion_cleaning.py` | 🔄 minimal (28 lines) |
| Produce `clean_data.csv` | `Data/filtered/clean_data.csv` | ⬜ |

---

## Week 2 — SQL & Modeling

| Task | File | Status |
|------|------|--------|
| Write all 6 required SQL queries | `sql/analysis.sql` | ✅ |
| Execute queries & save results in Python | `src-code/02_sql_analysis.py` | ⬜ empty |
| EDA: distributions, heatmap, class balance | `src-code/03_eda.py` | ⬜ empty |
| Feature engineering (temporal, expanding window, OHE) | `src-code/04_feature_engineering.py` | ✅ |
| Train 4 models + SHAP on XGBoost | `src-code/05_modeling.py` | ⬜ empty |
| Evaluate: PR-AUC, Recall, ROC-AUC, F1, CV | `src-code/05_modeling.py` | ⬜ empty |

---

## Week 3 — Visualisation

| Task | File | Status |
|------|------|--------|
| Plotly/Folium risk heatmap, boxplots, trend lines | `src-code/06_visualization_advanced.py` | ⬜ empty |
| Odds-ratio analysis (Logistic Regression interpretation) | `src-code/05_modeling.py` | ⬜ |

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
| Input form → predict risk + SHAP waterfall | `app/app.py` | 🔄 stub (67 lines) |
| Load saved XGBoost model (`.pkl` / `.json`) | `app/app.py` | ⬜ model not trained yet |
| Deploy / run locally end-to-end | — | ⬜ |

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

- [ ] All 6 pipeline scripts run clean end-to-end
- [ ] `sql/analysis.sql` matches output from `02_sql_analysis.py`
- [ ] 4 model comparison table with PR-AUC as primary metric
- [ ] SHAP waterfall plot saved
- [ ] Power BI `.pbix` file committed
- [ ] Streamlit app runs: `streamlit run app/app.py`
- [ ] Report PDF compiled from `main.tex` (10–12 pages)
- [ ] AI Audit Log `.xlsx` complete (15–20 entries, ≥3 hallucination logs)
- [ ] CI passes (black + flake8)
