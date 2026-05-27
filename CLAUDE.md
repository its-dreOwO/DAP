# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**DAP391m — Project 8: Supplier Lead-Time Risk Prediction for Retail Procurement** FPT University HCMC, Summer 2026. Group 8: Nguyễn Hoài Khánh, Hồ Lâm Bảo Đăng, Dương Gia Bảo. Supervisor: Mr. Nguyen Hoai Linh.

## Python Environment

All Python must run inside the project venv. Always use:

```
.venv/bin/python3 script.py
```

To activate in fish shell:

```
source .venv/bin/activate.fish
```

Run a pipeline script:

```
.venv/bin/python3 src-code/01\_ingestion\_cleaning.py
```

Install dependencies:

```
.venv/bin/pip install pandas numpy scikit-learn xgboost sqlalchemy matplotlib seaborn plotly folium streamlit shap openpyxl
```

## Data Architecture

**Active dataset:** `Data/supply\_chain\_risk\_dataset.csv` — 2,478 rows, 17 raw columns, 51 unique suppliers.

| Column | Type | Role |
| - | - | - |
| `timestamp` | datetime | Order time → extract temporal features |
| `machine\_id` | categorical | **DROPPED** — 201 values, no procurement signal |
| `temperature\_C`, `vibration\_level`, `machine\_runtime\_hours` | float | Machine operational signals |
| `inventory\_level\_units`, `pending\_orders` | int | Demand pressure |
| `supplier\_id` | categorical | One-Hot Encoded (50 binary cols) |
| `supplier\_lead\_time\_days` | float | Core supplier metric |
| `supplier\_quality\_score` | float | Proxy for defect rate |
| `supplier\_reliability\_index` | float | Delivery consistency |
| `fuel\_price\_index` | float | External cost pressure |
| `port\_delay\_days` | float | **DROPPED** — target leakage (r=0.898 with risk\_probability) |
| `market\_demand\_index`, `weather\_disruption\_score` | float | Environmental disruption |
| `risk\_probability` | float | **DROPPED** — direct target leakage |
| `risk\_label` | categorical | **TARGET** — High/Medium/Low (3-class) |


**Target:** `risk\_label` (High=2, Medium=1, Low=0) — 3-class classification. Class distribution: 63% High, 20% Low, 18% Medium — handled via `class\_weight='balanced'`. Binary secondary: High vs. not-High for comparability with original proposal.

**Critical leakage warning:** `port\_delay\_days` correlates with `risk\_probability` at r=0.898 — the target is synthetically derived from port delay. Never use `port\_delay\_days` or `risk\_probability` as model features.

## Engineered Features (added in `04\_feature\_engineering.py`)

| Feature | How | Source |
| - | - | - |
| `order\_month`, `order\_weekday`, `order\_hour` | Extract from `timestamp` | Paper (Orajaka & Okolie 2025) |
| `supplier\_avg\_lead` | Expanding mean of `supplier\_lead\_time\_days` per supplier, `.shift(1)` | Paper — top SHAP feature |
| `supplier\_std\_lead` | Expanding std of `supplier\_lead\_time\_days` per supplier, `.shift(1)` | Paper — top SHAP feature |
| `supplier\_risk\_score` | Composite: avg\_lead×0.4 + std\_lead×0.3 + (100−quality)/100×0.2 + (1−reliability)×0.1 | New |
| `external\_risk\_score` | Composite: weather×0.5 + fuel×0.3 + (1−demand)×0.2 (after dropping port\_delay) | New |


## Pipeline Order (`src-code/`)

Scripts must be run in sequence:

1. `01\_ingestion\_cleaning.py` — load `supply\_chain\_risk\_dataset.csv`, parse timestamp, type-fix, null guard, output `Data/filtered/clean\_data.csv`

2. `02\_sql\_analysis.py` — SQLite in-memory queries on `Data/customer.csv`, `Data/shipment.csv`, and `Data/logistics_performance.csv`: average lead time by supplier, delay frequency, volume-delay summary, monthly trends, YoY growth, penetration index. The source shipment file does not contain `supplier_id` or `carrier`, so the script derives deterministic analysis fields from the customer and logistics files before running `sql/analysis.sql`.

3. `03\_eda.py` — distributions, leakage correlation heatmap, 3-class balance chart, per-supplier boxplots

4. `04\_feature\_engineering.py` — drop leakage cols, extract temporal features, expanding-window supplier features, composite scores, OHE for supplier\_id, IQR capping, 80/20 stratified split, and write dual processed matrices under `Data/filtered/processed/`

5. `05\_modeling.py` — **standalone script**, reads `Data/supply\_chain\_risk\_dataset.csv` directly (drops leakage cols + machine\_id, extracts temporal features). Trains 4 classifiers (3-class `risk\_label`): LogReg (`multi\_class='ovr'`), DecisionTree, RandomForest, XGBoost (`objective='multi:softprob'`, `num\_class=3`). Primary metric: Macro PR-AUC. Outputs all artifacts to `Data/filtered/model\_outputs/`: `primary\_model.pkl` (XGBoost deployment), `best\_model.pkl` (highest PR-AUC, may differ from XGBoost), `xgb\_model.pkl`, per-model classification reports, confusion matrices, SHAP summary, feature importance, model comparison CSV/PNG. Does **not** require `04\_feature\_engineering.py` to run first.

6. `06\_visualization\_advanced.py` — Plotly supplier risk ranking, monthly risk trend, lead-time distribution, model comparison export, and supplier scorecard under `Data/filtered/visualization_outputs/`

Generated outputs from the current runnable pipeline:

- `Data/filtered/clean_data.csv`
- `Data/filtered/sql_outputs/*.csv`
- `Data/filtered/eda_outputs/*`
- `Data/filtered/engineered_features.csv`
- `Data/filtered/processed/*`
- `Data/filtered/model_outputs/*`
- `Data/filtered/visualization_outputs/*`

**Reference spec:** `report/project\_pipeline.pdf` — full pipeline decision record.

## Models

**4 classifiers benchmarked** (no MLP):

- Logistic Regression (baseline + odds-ratio interpretation, `multi\_class='ovr'`)

- Decision Tree

- Random Forest

- XGBoost (`objective='multi:softprob'`, `num\_class=3`) ← primary model, used in Streamlit app

**Primary evaluation metrics:** Macro PR-AUC (imbalance-aware). **Secondary:** Macro ROC-AUC OvR, weighted F1, per-class precision/recall, confusion matrix. **Validation:** Stratified 80/20 train-test split (`random_state=42`). No cross-validation in the current pipeline.

**Model artifacts** (`Data/filtered/model_outputs/`):

| File | Contents |
| - | - |
| `primary\_model.pkl` | XGBoost pipeline — official deployment + Streamlit model |
| `xgb\_model.pkl` | XGBoost pipeline (explainability / SHAP) |
| `best\_model.pkl` | Highest Macro PR-AUC pipeline (may be Logistic Regression) |
| `predictions\_test.csv` | XGBoost test-set predictions + per-class probabilities |
| `model\_comparison.csv` / `.png` | All 4 models side-by-side on every metric |
| `shap\_summary.png` | SHAP beeswarm on XGBoost (top 20 features) |
| `feature\_importance.png` | XGBoost feature importances |
| `confusion\_matrix\_xgboost.png` | XGBoost confusion matrix heatmap |

## Reference Paper

Orajaka & Okolie (2025) — WJARR-2025-3753 (`docs/WJARR-2025-3753.pdf`). Key adoptions: expanding-window supplier features, OHE encoding, temporal features, 5 theoretical frameworks (TCE, RBV, Lean SCM, CAS, Decision Theory). Key rejection: regression framing (R²=−0.09 in paper) — project keeps classification.

## SQL (`sql/analysis.sql`)

Required queries per project spec:

- Average lead time per supplier

- Delay frequency per supplier

- Volume–delay correlation

- Monthly delay trends over time

- Year-over-year growth rate

- Penetration index

`02_sql_analysis.py` executes these queries directly from `sql/analysis.sql`
and saves one CSV per query under `Data/filtered/sql_outputs/`.

## Deliverables (course requirements)

- **4 models** compared on PR-AUC, Recall, ROC-AUC, F1 — primary metric is PR-AUC

- **SQL** queries in `sql/analysis.sql`

- **AI Audit Log** (`docs/AI\_AuditLog\_Template\_DAP391m.xlsx`) — 15–20 prompts + ≥3 hallucination checks

- **Final report** in `report/main.tex` — LaTeX, 10–12 pages

- **Power BI dashboard** — supplier scorecard with risk alerts, drill-down by supplier/region/carrier

- **Streamlit app** in `app/app.py` — 3-class risk prediction (Low/Medium/High) with P(Low)/P(Medium)/P(High) breakdown, procurement notes, SHAP summary, and model comparison display

## AI Audit Log workflow

Live log is maintained at `report/ai\_audit\_log.md`. Claude updates this file automatically when a conversation qualifies as a **core prompt** (DECISION / PROBLEM-SOLVING / VERIFICATION) per the framework in `docs/AI\_AuditLog\_Template\_DAP391m.xlsx`.

**Claude's responsibility:**

- Add new entries with Entry \#, Prompt Type, Stage/Component, Problem/Context, Prompt to AI, AI Response (Summary)

- Suggest what to write in Human Delta and Evidence (in `\[brackets\]`) but never fill them in

- Update the coverage tracker table at the top of the file

- Update the "Last updated" line

**Student's responsibility:**

- Fill in Human Delta & Reflection (all 4 questions: Critical Thinking, Contextualization, Creative Synthesis, Decision Ownership)

- Fill in Evidence (screenshots, metrics, comparisons)

- Copy finalized entries into `docs/AI\_AuditLog\_Template\_DAP391m.xlsx` before submission

- Log hallucinations in the hallucination table when found (project requires ≥3)

## Weekly Timeline

| Week | Focus |
| - | - |
| 1 | Business understanding, research questions, data collection & cleaning |
| 2 | SQL analysis, Python modeling (4 models + SHAP) |
| 3 | Visualisation (Plotly exports), optional regression/odds-ratio interpretation |
| 4 | Power BI supplier scorecard |
| 4–5 | Streamlit web application |


## Project Tracking

- **`TODO.md`** — task checklist by week and deliverable (✅ Done / 🔄 In Progress / ⬜ Not Started)
- **`PROBLEMS.md`** — open decisions requiring group consensus; resolve before proceeding

## Source Code Management

Important: run lint / black max-length=88 test before pushing to GitHub.
