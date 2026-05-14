# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**DAP391m — Project 8: Supplier Lead-Time Risk Prediction for Retail Procurement**
FPT University HCMC, Summer 2026. Group 8: Hồ Lâm Bảo Đăng, Nguyễn Hoài Khánh, Dương Gia Bảo.

## Python Environment

All Python must run inside the project venv. Always use:
```bash
.venv/bin/python3 script.py
```

To activate in fish shell:
```bash
source .venv/bin/activate.fish
```

To run a notebook:
```bash
.venv/bin/jupyter notebook notebooks/<name>.ipynb
```

Install dependencies:
```bash
.venv/bin/pip install pandas numpy scikit-learn xgboost sqlalchemy matplotlib seaborn plotly folium streamlit shap prophet boto3 ipykernel jupyter openpyxl
```

## Data Architecture

Three source CSVs in `Data/` with no shared primary key — joined by region/date proxies:

| File | Rows | Key columns | Role |
|---|---|---|---|
| `customer.csv` | 750 | `supplier_id`, `lead_time_days`, `market_segment`, `satisfaction_score` | Supplier & customer features |
| `shipment.csv` | 728 | `delivery_status` (target), `customs_clearance_time_days`, `freight_cost` | Base table for modeling |
| `logistics_performance.csv` | 100 | `carrier`, `region`, `delay_hours_avg`, `warehouse_utilization_percent` | Carrier/region conditions (Jun 2024 only) |

**Join strategy:**
- `shipment.D_Country` → mapped to region via `country_to_region` dict
- `logistics_performance` joined on `year_month` + `region` (sparse — falls back to global means)
- `customer` aggregated by `market_segment` (treated as region proxy) then joined on `region`

**Target:** `delivery_status == "Delayed"` → binary 1/0. Class imbalance: ~84% On-Time, 16% Delayed — use `class_weight='balanced'` or SMOTE.

## Filtered Data

`Data/filtered/extract_model_data.py` — merges all 3 CSVs and outputs `Data/filtered/model_features.csv` (704 rows, 23 columns, zero nulls). Run from `Data/filtered/`:
```bash
cd Data/filtered && ../../.venv/bin/python3 extract_model_data.py
```

## Notebook Pipeline Order

Notebooks must be run in sequence — each builds on the previous:
1. `01_ingestion_cleaning.ipynb` — load, type-fix, null handling
2. `02_sql_analysis.ipynb` — SQLite queries, supplier/carrier aggregations
3. `03_eda.ipynb` — distributions, correlations, time series
4. `04_feature_engineering.ipynb` — encoding, scaling, engineered features
5. `05_modeling.ipynb` — 5 models (LogReg, DTree, RF, XGBoost, MLP), SHAP
6. `06_visualization_advanced.ipynb` — Plotly/Folium dashboards

## Deliverables (course requirements)

- **≥5 models** compared on Accuracy, F1, AUC-ROC vs a baseline paper
- **SQL** queries in `sql/analysis.sql` covering time series and aggregations
- **AI Audit Log** (`docs/AI_AuditLog_Template_DAP391m.xlsx`) — 15–20 prompts + ≥3 hallucination checks, mandatory for every graded submission
- **Final report** in `report/main.tex` — LaTeX Springer 1-column, 10–12 pages
- **Streamlit app** in `app/app.py` — input shipment → predict delay probability using trained XGBoost + SHAP explanation
