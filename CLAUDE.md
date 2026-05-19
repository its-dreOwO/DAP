# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**DAP391m — Project 8: Supplier Lead-Time Risk Prediction for Retail Procurement**
FPT University HCMC, Summer 2026. Group 8: Nguyễn Hoài Khánh, Hồ Lâm Bảo Đăng, Dương Gia Bảo.
Supervisor: Mr. Nguyen Hoai Linh.

## Python Environment

All Python must run inside the project venv. Always use:
```bash
.venv/bin/python3 script.py
```

To activate in fish shell:
```bash
source .venv/bin/activate.fish
```

Run a pipeline script:
```bash
.venv/bin/python3 src-code/01_ingestion_cleaning.py
```

Install dependencies:
```bash
.venv/bin/pip install pandas numpy scikit-learn xgboost sqlalchemy matplotlib seaborn plotly folium streamlit shap openpyxl
```

## Data Architecture

Three source CSVs in `Data/` — `shipment.csv` is the fact table:

| File | Rows | Key columns | Role |
|---|---|---|---|
| `customer.csv` | 750 | `supplier_id`, `lead_time_days`, `market_segment`, `satisfaction_score` | Supplier attributes |
| `shipment.csv` | 728 | `delivery_status` (target), `customs_clearance_time_days`, `freight_cost`, `supplier_id` | Fact table |
| `logistics_performance.csv` | 100 | `carrier`, `region`, `delay_hours_avg`, `warehouse_utilization_percent` | Regional carrier performance |

**Join strategy:**
- `customer` joined to `shipment` on `supplier_id` (direct lookup)
- `shipment.D_Country` → mapped to region via `country_to_region` dict
- `logistics_performance` joined on `region`
- Inner joins only — unmatched rows dropped. Final dataset: **704 rows, 23 features**

**Target:** `delivery_status == "Delayed"` → binary 1/0.
Class imbalance: ~16% Delayed, 84% On-Time — handled via `class_weight='balanced'` and threshold tuning at evaluation.

## Filtered Data

`Data/filtered/extract_model_data.py` — merges all 3 CSVs and outputs `Data/filtered/model_features.csv`.
```bash
.venv/bin/python3 Data/filtered/extract_model_data.py
```

## Pipeline Order (`src-code/`)

Scripts must be run in sequence:
1. `01_ingestion_cleaning.py` — load raw CSVs, type-fix, null handling, output clean CSVs
2. `02_sql_analysis.py` — SQLite in-memory queries: avg lead time, delay frequency, volume-delay correlation, YoY growth, penetration index
3. `03_eda.py` — distributions, correlation heatmap, class balance, time-series plots
4. `04_feature_engineering.py` — label encoding, standard scaling, train/test split (stratified)
5. `05_modeling.py` — 4 models: LogReg, DecisionTree, RandomForest, XGBoost; evaluate PR-AUC + recall (primary), ROC-AUC + F1 + confusion matrix (secondary); stratified 5-fold CV; SHAP on XGBoost
6. `06_visualization_advanced.py` — Plotly/Folium risk heatmap, boxplots by supplier, lead-time histograms, trend lines

## Models

**4 classifiers benchmarked** (no MLP):
- Logistic Regression (baseline + odds-ratio interpretation)
- Decision Tree
- Random Forest
- XGBoost ← primary model, used in Streamlit app

**Primary evaluation metrics:** PR-AUC and recall on the Delayed class (imbalance-aware).
**Secondary:** ROC-AUC, F1-score, confusion matrix.
**Validation:** Stratified 5-fold cross-validation.

## SQL (`sql/analysis.sql`)

Required queries per project spec:
- Average lead time per supplier
- Delay frequency per supplier
- Volume–delay correlation
- Monthly delay trends over time
- Year-over-year growth rate
- Penetration index

## Deliverables (course requirements)

- **4 models** compared on PR-AUC, Recall, ROC-AUC, F1 — primary metric is PR-AUC
- **SQL** queries in `sql/analysis.sql`
- **AI Audit Log** (`docs/AI_AuditLog_Template_DAP391m.xlsx`) — 15–20 prompts + ≥3 hallucination checks
- **Final report** in `report/main.tex` — LaTeX, 10–12 pages
- **Power BI dashboard** — supplier scorecard with risk alerts, drill-down by supplier/region/carrier
- **Streamlit app** in `app/app.py` — input shipment → predict delay probability + SHAP waterfall explanation

## AI Audit Log workflow

Live log is maintained at `report/ai_audit_log.md`. Claude updates this file automatically when a conversation qualifies as a **core prompt** (DECISION / PROBLEM-SOLVING / VERIFICATION) per the framework in `docs/AI_AuditLog_Template_DAP391m.xlsx`.

**Claude's responsibility:**
- Add new entries with Entry #, Prompt Type, Stage/Component, Problem/Context, Prompt to AI, AI Response (Summary)
- Suggest what to write in Human Delta and Evidence (in `[brackets]`) but never fill them in
- Update the coverage tracker table at the top of the file
- Update the "Last updated" line

**Student's responsibility:**
- Fill in Human Delta & Reflection (all 4 questions: Critical Thinking, Contextualization, Creative Synthesis, Decision Ownership)
- Fill in Evidence (screenshots, metrics, comparisons)
- Copy finalized entries into `docs/AI_AuditLog_Template_DAP391m.xlsx` before submission
- Log hallucinations in the hallucination table when found (project requires ≥3)

## Weekly Timeline

| Week | Focus |
|---|---|
| 1 | Business understanding, research questions, data collection & cleaning |
| 2 | SQL analysis, Python modeling (4 models + SHAP) |
| 3 | Visualisation (Plotly/Folium), regression analysis (odds ratios) |
| 4 | Power BI supplier scorecard |
| 4–5 | Streamlit web application |

##source code managment##
run lint / black max-length=88  test before pushing to github 
