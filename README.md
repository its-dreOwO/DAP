# Supplier Lead-Time and Late Delivery Risk Prediction

DAP391m — Project 8, Group 8, FPT University HCMC.  
Supervisor: Mr. Nguyen Hoai Linh.  
Team: Nguyễn Hoài Khánh, Hồ Lâm Bảo Đăng, Dương Gia Bảo.

Predicts **late delivery risk** for retail procurement orders using the
[DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
dataset (CC0). Target: `Late_delivery_risk` (binary, 1 = late).

> **Dataset note:** DataCo is a simulated teaching dataset — limitations are
> disclosed in the report.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # bash/zsh
# source .venv/bin/activate.fish  # fish

pip install pandas numpy scikit-learn xgboost sqlalchemy \
            matplotlib seaborn plotly folium streamlit shap openpyxl
```

Place the raw dataset at:

```
Data/dataco_raw/DataCoSupplyChainDataset.csv
```

Download via Kaggle CLI:

```bash
kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis \
  -p Data/dataco_raw --unzip
```

## Pipeline

Run scripts in order:

```bash
.venv/bin/python3 src-code/01_ingestion_cleaning.py
.venv/bin/python3 src-code/02_sql_analysis.py
.venv/bin/python3 src-code/03_eda.py
.venv/bin/python3 src-code/04_feature_engineering.py
.venv/bin/python3 src-code/05_modeling.py
.venv/bin/python3 src-code/06_visualization_advanced.py
```

`05_modeling.py` reads `Data/filtered/clean_data.csv` if it exists (output of
script 01), otherwise falls back to the raw CSV directly.

## Key Design Decisions

- **Group-aware split:** `GroupShuffleSplit` on `Order Id` — the dataset is
  order-item grain (~2.75 items/order), so all items from one order stay on
  the same side of the train/test boundary.
- **Leakage removed:** post-outcome columns (`Days for shipping (real)`,
  `Delivery Status`, realized profit fields, `shipping date`) are dropped
  before modeling.
- **Collinearity note:** `Shipping Mode` and `Days for shipment (scheduled)`
  are perfectly collinear in DataCo. A per-mode base-rate lookup scores
  ROC-AUC 0.725 — the four ML models add ~+0.08 PR-AUC on top.

## Results (test set, GroupShuffleSplit)

| Model | PR-AUC | ROC-AUC | F1 | Recall |
|-------|--------|---------|----|--------|
| Shipping-mode baseline | 0.749 | 0.725 | — | — |
| Logistic Regression | 0.796 | 0.726 | 0.677 | 0.630 |
| Decision Tree | 0.795 | 0.769 | 0.692 | 0.584 |
| Random Forest | 0.821 | 0.755 | 0.692 | 0.615 |
| **XGBoost** (primary) | **0.833** | **0.767** | **0.693** | 0.590 |

XGBoost trained with `device="cuda"` (GPU). Primary metric is PR-AUC.

## Pipeline Outputs

| Path | Contents |
|------|----------|
| `Data/filtered/clean_data.csv` | Cleaned DataCo dataset (180,519 rows, 29 cols) |
| `Data/filtered/sql_outputs/*.csv` | Six supply-chain SQL analyses |
| `Data/filtered/eda_outputs/*` | EDA plots and summary tables |
| `Data/filtered/model_outputs/` | Model artifacts, comparison, SHAP, predictions |
| `Data/filtered/model_outputs/primary_model.pkl` | XGBoost deployment pickle |
| `Data/filtered/model_outputs/model_comparison.csv` | All 4 models × all metrics |
| `Data/filtered/model_outputs/shipping_mode_model_performance.csv` | Per-mode XGBoost breakdown |

## Streamlit App

```bash
.venv/bin/streamlit run app/app.py
```

## Remote Training (Modal)

To run `05_modeling.py` on a cloud GPU:

```bash
pip install modal
modal run modal_train.py
```

Artifacts are saved to the `dap-outputs` Modal Volume and downloaded to
`Data/filtered/model_outputs/` automatically.

## Project Status

See [`TODO.md`](TODO.md) for the task checklist and [`PROBLEMS.md`](PROBLEMS.md)
for open group decisions.
