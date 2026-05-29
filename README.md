# Consumer Credit-Default Risk Prediction

DAP391m — Project 8, Group 8, FPT University HCMC.

> **Note:** This project pivoted from *Supplier Lead-Time Risk* to *Consumer Credit-Default Risk* on 2026-05-29. The original supply-chain dataset had a synthetic, leakage-derived target that left no learnable signal after cleaning (ROC-AUC ≈ 0.52). With supervisor approval we adopted the Kaggle `laotse/credit-risk-dataset` (verified ROC-AUC ≈ 0.93). The previous dataset and all its outputs are archived under `Data/archived/`.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate          # bash/zsh
# source .venv/bin/activate.fish   # fish shell

pip install pandas numpy scikit-learn xgboost sqlalchemy \
            matplotlib seaborn plotly folium streamlit shap openpyxl
```

Run the pipeline in order:

```
.venv/bin/python3 src-code/01_ingestion_cleaning.py
.venv/bin/python3 src-code/02_sql_analysis.py
.venv/bin/python3 src-code/03_eda.py
.venv/bin/python3 src-code/04_feature_engineering.py
.venv/bin/python3 src-code/05_modeling.py
.venv/bin/python3 src-code/06_visualization_advanced.py
```

## Pipeline Outputs

- `Data/filtered/clean_data.csv` — cleaned credit dataset (32,409 rows).
- `Data/filtered/sql_outputs/*.csv` — six credit-analysis outputs from `sql/analysis.sql`.
- `Data/filtered/eda_outputs/*` — class balance, numeric summaries, correlation heatmap, default-rate-by-category tables and figures.
- `Data/filtered/engineered_features.csv` and `Data/filtered/processed/*` — engineered features and train/test matrices.
- `Data/filtered/model_outputs/*` — baseline 4-model comparison, reports, SHAP, predictions, threshold analysis, and pickle artifacts.
- `Data/filtered/model_outputs/imbalance_weighted/*` — separate imbalance-weighted experiment (same layout).
- `Data/filtered/visualization_outputs/*` — Plotly HTML charts and the grade risk scorecard.

`02_sql_analysis.py` loads the single cleaned credit table; there is no date/carrier column, so the supply-chain time-series queries were replaced by credit-default analyses (default rate by grade/home/intent/income band/loan-amount band + grade penetration index).

## Streamlit App

```
.venv/bin/streamlit run app/app.py
```

The app loads `primary_model.pkl` from `Data/filtered/model_outputs/`, takes an applicant's loan details, and predicts the probability of default with a risk tier and an approve/decline recommendation at the tuned decision threshold. The imbalance-weighted artifacts are experimental and are not used by the app.

## Results Summary (baseline, test set)

| Model | PR-AUC | ROC-AUC | F1 | Recall |
|-------|--------|---------|----|--------|
| Logistic Regression | 0.764 | 0.885 | 0.696 | 0.625 |
| Decision Tree | 0.839 | 0.895 | 0.820 | 0.707 |
| Random Forest | 0.890 | 0.934 | 0.831 | 0.731 |
| **XGBoost** (primary) | **0.911** | **0.952** | **0.843** | 0.740 |

## Project Status

See [`TODO.md`](TODO.md) for the task checklist and [`PROBLEMS.md`](PROBLEMS.md) for open group decisions.

**Current state:**
- Scripts `01`–`06` run end-to-end with `.venv/bin/python3` on the credit dataset.
- Streamlit loads the XGBoost primary pickle and scores individual applications.
- Modeling keeps both the baseline PR-AUC result and the imbalance-weighted comparison.
- Remaining open work: Power BI credit scorecard, report writing for the new domain, and audit-log completion.
