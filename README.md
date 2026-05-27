#  Supplier Lead-Time Risk Prediction


For a full description of the project scope and research questions, see [`report/main-1.pdf`](file:///home/dre/Desktop/study/DAP/report/main-1.pdf). For the complete pipeline specification (dataset decisions, feature engineering, model choices, evaluation plan), see [`report/project\_pipeline.pdf`](file:///home/dre/Desktop/study/DAP/report/project_pipeline.pdf).

## Setup

```
python3 -m venv .venv  
source .venv/bin/activate        \# bash/zsh  
\# source .venv/bin/activate.fish \# fish shell  
  
pip install pandas numpy scikit-learn xgboost sqlalchemy \\  
            matplotlib seaborn plotly folium streamlit shap openpyxl
```

Run the pipeline in order:

```
.venv/bin/python3 src-code/01\_ingestion\_cleaning.py  
.venv/bin/python3 src-code/02\_sql\_analysis.py  
.venv/bin/python3 src-code/03\_eda.py  
.venv/bin/python3 src-code/04\_feature\_engineering.py  
.venv/bin/python3 src-code/05\_modeling.py  
.venv/bin/python3 src-code/06\_visualization\_advanced.py
```

## Pipeline Outputs

- `Data/filtered/clean_data.csv` — cleaned active modeling dataset.
- `Data/filtered/sql_outputs/*.csv` — six outputs from `sql/analysis.sql`.
- `Data/filtered/eda_outputs/*` — class balance, numeric summaries, leakage correlation, supplier summaries, and EDA figures.
- `Data/filtered/engineered_features.csv` and `Data/filtered/processed/*` — leakage-safe engineered features and train/test matrices.
- `Data/filtered/model_outputs/*` — model comparison, reports, SHAP summary, predictions, and pickle artifacts.
- `Data/filtered/visualization_outputs/*` — Plotly HTML charts and supplier scorecard CSVs for dashboard/report use.

`02_sql_analysis.py` uses the older business-analysis CSVs (`customer.csv`,
`shipment.csv`, `logistics_performance.csv`). Because `shipment.csv` does not
ship with `supplier_id` or `carrier`, the script derives deterministic analysis
fields before executing the SQL queries.

## Streamlit App

```
.venv/bin/streamlit run app/app.py
```

The app loads `primary_model.pkl` / `xgb_model.pkl` from
`Data/filtered/model_outputs/` and predicts Low / Medium / High supplier risk
for a scenario entered in the sidebar.

## Project Status

See [`TODO.md`](TODO.md) for the full task checklist (what's done, in progress, and not started).

See [`PROBLEMS.md`](PROBLEMS.md) for open decisions that need group discussion before work can proceed.

**Current state:**
- Scripts `01`-`06` run end-to-end with `.venv/bin/python3`.
- Generated outputs are written under `Data/filtered/`.
- Streamlit loads the saved XGBoost pickle artifacts from `Data/filtered/model_outputs/`.
- Remaining open work is mainly report ownership, Power BI, optional logistic odds-ratio interpretation, and audit-log completion.
