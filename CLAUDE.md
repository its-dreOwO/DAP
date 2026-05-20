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

**Active dataset:** `Data/supply_chain_risk_dataset.csv` — 2,478 rows, 17 raw columns, 51 unique suppliers.

| Column | Type | Role |
|---|---|---|
| `timestamp` | datetime | Order time → extract temporal features |
| `machine_id` | categorical | **DROPPED** — 201 values, no procurement signal |
| `temperature_C`, `vibration_level`, `machine_runtime_hours` | float | Machine operational signals |
| `inventory_level_units`, `pending_orders` | int | Demand pressure |
| `supplier_id` | categorical | One-Hot Encoded (50 binary cols) |
| `supplier_lead_time_days` | float | Core supplier metric |
| `supplier_quality_score` | float | Proxy for defect rate |
| `supplier_reliability_index` | float | Delivery consistency |
| `fuel_price_index` | float | External cost pressure |
| `port_delay_days` | float | **DROPPED** — target leakage (r=0.898 with risk_probability) |
| `market_demand_index`, `weather_disruption_score` | float | Environmental disruption |
| `risk_probability` | float | **DROPPED** — direct target leakage |
| `risk_label` | categorical | **TARGET** — High/Medium/Low (3-class) |

**Target:** `risk_label` (High=2, Medium=1, Low=0) — 3-class classification.
Class distribution: 63% High, 20% Low, 18% Medium — handled via `class_weight='balanced'`.
Binary secondary: High vs. not-High for comparability with original proposal.

**Critical leakage warning:** `port_delay_days` correlates with `risk_probability` at r=0.898 — the target is synthetically derived from port delay. Never use `port_delay_days` or `risk_probability` as model features.

## Engineered Features (added in `04_feature_engineering.py`)

| Feature | How | Source |
|---|---|---|
| `order_month`, `order_weekday`, `order_hour` | Extract from `timestamp` | Paper (Orajaka & Okolie 2025) |
| `supplier_avg_lead` | Expanding mean of `supplier_lead_time_days` per supplier, `.shift(1)` | Paper — top SHAP feature |
| `supplier_std_lead` | Expanding std of `supplier_lead_time_days` per supplier, `.shift(1)` | Paper — top SHAP feature |
| `supplier_risk_score` | Composite: avg_lead×0.4 + std_lead×0.3 + (100−quality)/100×0.2 + (1−reliability)×0.1 | New |
| `external_risk_score` | Composite: weather×0.5 + fuel×0.3 + (1−demand)×0.2 (after dropping port_delay) | New |

## Pipeline Order (`src-code/`)

Scripts must be run in sequence:
1. `01_ingestion_cleaning.py` — load `supply_chain_risk_dataset.csv`, parse timestamp, type-fix, null guard, output `Data/filtered/clean_data.csv`
2. `02_sql_analysis.py` — SQLite in-memory queries on new schema: avg lead time by supplier, High-risk frequency, volume-delay correlation, monthly trends, YoY growth, penetration index
3. `03_eda.py` — distributions, leakage correlation heatmap, 3-class balance chart, per-supplier boxplots
4. `04_feature_engineering.py` — drop leakage cols, extract temporal features, expanding window supplier features, composite scores, OHE for supplier_id, IQR capping, 80/20 stratified split
5. `05_modeling.py` — 4 models (3-class + binary secondary): LogReg (`multi_class='ovr'`), DecisionTree, RandomForest, XGBoost (`objective='multi:softprob'`, `num_class=3`); macro PR-AUC + per-class recall (primary); ROC-AUC, F1, confusion matrix (secondary); stratified 5-fold CV; SHAP on XGBoost
6. `06_visualization_advanced.py` — Plotly/Folium risk heatmap by supplier, boxplots, lead-time histograms, trend lines

**Reference spec:** `report/project_pipeline.pdf` — full pipeline decision record.

## Models

**4 classifiers benchmarked** (no MLP):
- Logistic Regression (baseline + odds-ratio interpretation, `multi_class='ovr'`)
- Decision Tree
- Random Forest
- XGBoost (`objective='multi:softprob'`, `num_class=3`) ← primary model, used in Streamlit app

**Primary evaluation metrics:** Macro PR-AUC and per-class recall (imbalance-aware).
**Secondary:** Macro ROC-AUC, weighted F1-score, confusion matrix; binary analysis (High vs. not-High).
**Validation:** Stratified 5-fold cross-validation.

## Reference Paper

Orajaka & Okolie (2025) — WJARR-2025-3753 (`docs/WJARR-2025-3753.pdf`).
Key adoptions: expanding-window supplier features, OHE encoding, temporal features, 5 theoretical frameworks (TCE, RBV, Lean SCM, CAS, Decision Theory).
Key rejection: regression framing (R²=−0.09 in paper) — project keeps classification.

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
