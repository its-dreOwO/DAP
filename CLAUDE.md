# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**DAP391m — Project 8: Consumer Credit-Default Risk Prediction** FPT University HCMC, Summer 2026. Group 8: Nguyễn Hoài Khánh, Hồ Lâm Bảo Đăng, Dương Gia Bảo. Supervisor: Mr. Nguyen Hoai Linh.

> **Domain pivot (2026-05-29):** The project moved from *Supplier Lead-Time Risk* to *Consumer Credit-Default Risk*. The original `supply_chain_risk_dataset.csv` had a synthetically-derived target (`risk_label` was generated from `port_delay_days`); after removing leakage columns the four models scored ROC-AUC ≈ 0.52 (near-random). The supervisor approved switching to a real dataset. All supply-chain data and outputs are preserved under `Data/archived/`. The supervisor's first suggested dataset (`uciml/german-credit`) was rejected because that Kaggle upload has **no target column**; the `laotse/credit-risk-dataset` was verified to have genuine signal (quick RF ROC-AUC ≈ 0.93) and was adopted.

## Python Environment

All Python must run inside the project venv. Always use:

```
.venv/bin/python3 script.py
```

To activate in fish shell:

```
source .venv/bin/activate.fish
```

Install dependencies:

```
.venv/bin/pip install pandas numpy scikit-learn xgboost sqlalchemy matplotlib seaborn plotly folium streamlit shap openpyxl
```

## Data Architecture

**Active dataset:** `Data/credit_risk_dataset.csv` (from Kaggle `laotse/credit-risk-dataset`, CC0). 32,581 raw rows, 12 columns. After cleaning: **32,409 rows** (5 impossible ages >100, 2 impossible employment lengths >60y, and 165 duplicates removed).

| Column | Type | Role |
| - | - | - |
| `person_age` | int | Applicant age |
| `person_income` | int | Annual income (USD) |
| `person_home_ownership` | categorical | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | float | Employment length (years); has nulls |
| `loan_intent` | categorical | PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION |
| `loan_grade` | categorical | A–G lender-assigned grade |
| `loan_amnt` | int | Loan amount (USD) |
| `loan_int_rate` | float | Interest rate (%); has nulls |
| `loan_status` | int | **TARGET** — 1 = default, 0 = non-default |
| `loan_percent_income` | float | Loan amount / income |
| `cb_person_default_on_file` | categorical | Y / N prior default |
| `cb_person_cred_hist_length` | int | Credit history length (years) |

**Target:** `loan_status` — binary. Class distribution: **78.1% non-default / 21.9% default** (natural imbalance).

**Leakage stance (important):** Unlike the old dataset, `loan_status` is a real observed outcome, so there are **no columns to drop for leakage**. `loan_grade` and `loan_int_rate` are lender-assigned and partly encode the lender's own risk assessment, but they are **decision-time inputs** (known at origination, before default is observed) — they are valid predictors and are kept. This is the key distinction from the dropped `port_delay_days`, which was derived *from* the outcome. State this clearly in the report; do not quietly drop these features (the quoted 0.93 baseline includes them).

**Missing values:** `person_emp_length` (~887 nulls) and `loan_int_rate` (~3094 nulls) are preserved through cleaning and imputed (median) inside the modeling/feature pipelines.

## Engineered Features (added in `04_feature_engineering.py`)

| Feature | How |
| - | - |
| `high_debt_burden` | `loan_percent_income > 0.30` → 1/0 |
| `credit_hist_ratio` | `cb_person_cred_hist_length / person_age` |
| `interest_to_income` | `(loan_int_rate × loan_amnt) / person_income` |
| `income_bracket` | bins: <30k / 30k-60k / 60k-100k / 100k+ |
| `age_bracket` | bins: ≤25 / 26-35 / 36-50 / 50+ |
| `loan_amount_bracket` | bins: <5k / 5k-10k / 10k-20k / 20k+ |

All categoricals (base + engineered) are One-Hot Encoded; numeric features are median-imputed and (for scaled models) standardized. IQR capping is applied on the train split with bounds reused on test. **These engineered features feed the trained models:** `05_modeling.py` derives them inline (a copy of `04`'s `add_engineered_features`, kept in sync), and `app/app.py` recomputes the identical columns so the deployed pipeline receives exactly what it was trained on.

## Pipeline Order (`src-code/`)

Scripts must be run in sequence:

1. `01_ingestion_cleaning.py` — load `credit_risk_dataset.csv`, validate schema, remove impossible-value rows + duplicates, type-fix, write `Data/filtered/clean_data.csv` and a cleaning report. Nulls preserved for downstream imputation.
2. `02_sql_analysis.py` — load `clean_data.csv` into one in-memory SQLite table `credit`, run the six queries in `sql/analysis.sql`, save one CSV per query under `Data/filtered/sql_outputs/`.
3. `03_eda.py` — class balance, numeric distributions, feature-correlation heatmap (signal confirmation), default rate by category, numeric means by class.
4. `04_feature_engineering.py` — engineered features, OHE, IQR capping, 80/20 stratified split, dual processed matrices (tree + scaled) under `Data/filtered/processed/`.
5. `05_modeling.py` — **reads `Data/filtered/clean_data.csv`**, trains 4 binary classifiers, evaluates (PR-AUC primary), tunes the decision threshold, runs two experiments (baseline + imbalance-weighted), and writes artifacts. Self-contained pipeline objects for Streamlit. Does **not** require `04` to run first.
6. `06_visualization_advanced.py` — Plotly: default rate by grade, by intent, income×grade default matrix, loan-to-income distribution, model comparison export, grade scorecard.

Generated outputs:

- `Data/filtered/clean_data.csv`, `cleaning_report.txt`
- `Data/filtered/sql_outputs/*.csv`
- `Data/filtered/eda_outputs/*`
- `Data/filtered/engineered_features.csv`, `Data/filtered/processed/*`
- `Data/filtered/model_outputs/*` and `Data/filtered/model_outputs/imbalance_weighted/*`
- `Data/filtered/visualization_outputs/*`

## Models

**4 classifiers benchmarked** (binary `loan_status`, positive class = default):

- Logistic Regression (baseline + odds-ratio interpretation)
- Decision Tree
- Random Forest
- XGBoost (`objective='binary:logistic'`, `eval_metric='aucpr'`) ← primary model, used in Streamlit

**Primary metric:** PR-AUC (average precision, imbalance-aware). **Secondary:** ROC-AUC, F1, precision, recall, accuracy, confusion matrix. **Validation:** Stratified 80/20 split (`random_state=42`). **Threshold tuning** now applies (binary): the F1-optimal threshold is tuned on the train split and reported in `threshold_analysis.txt`.

**Current results (baseline experiment, test set):**

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| - | - | - | - | - | - | - |
| Logistic Regression | 0.881 | 0.785 | 0.625 | 0.696 | 0.764 | 0.885 |
| Decision Tree | 0.932 | 0.978 | 0.707 | 0.820 | 0.839 | 0.895 |
| Random Forest | 0.935 | 0.965 | 0.731 | 0.831 | 0.890 | 0.934 |
| **XGBoost** | **0.940** | **0.979** | 0.740 | **0.843** | **0.911** | **0.952** |

XGBoost is best on every metric → official report/deployment model. The **imbalance-weighted** experiment (`class_weight='balanced'` / `scale_pos_weight`) raises XGBoost recall (0.740 → 0.818) at the cost of precision (0.979 → 0.819); F1 and PR-AUC are marginally lower. Report it as a recall/precision trade-off, not the deployed model.

**Model artifacts** (`Data/filtered/model_outputs/`):

| File | Contents |
| - | - |
| `primary_model.pkl` | XGBoost pipeline — official deployment + Streamlit model |
| `xgb_model.pkl` | XGBoost pipeline (explainability / SHAP) |
| `best_model.pkl` | Highest PR-AUC pipeline |
| `predictions_test.csv` | XGBoost test predictions + P(default) |
| `threshold_analysis.txt` | F1-optimal threshold + precision/recall trade-off table |
| `model_comparison.csv` / `.png` | All 4 models on every metric |
| `model_comparison_all_experiments.csv` | Baseline vs imbalance-weighted |
| `shap_summary.png`, `feature_importance.png`, `confusion_matrix_xgboost.png` | Explainability / evaluation plots |

The `imbalance_weighted/` subfolder mirrors this layout but is experimental only — Streamlit always loads from `Data/filtered/model_outputs/`.

## SQL (`sql/analysis.sql`)

The original supply-chain queries (monthly delay trends, year-over-year growth, carrier penetration) **do not apply** — the credit dataset has no date column and no carrier. They are replaced by six credit-meaningful queries (still exactly six; `02_sql_analysis.py` asserts this):

1. Default rate by loan grade
2. Default rate by home ownership
3. Default rate by loan intent
4. Default rate by income band
5. Default rate by loan-amount band
6. Grade penetration index (grade's share of defaults vs share of volume)

## Deliverables (course requirements)

- **4 models** compared on PR-AUC (primary), Recall, ROC-AUC, F1
- **SQL** queries in `sql/analysis.sql`
- **AI Audit Log** (`report/ai_audit_log.md` → `docs/AI_AuditLog_Template_DAP391m.xlsx`) — 15–20 prompts + ≥3 hallucination checks
- **Final report** in `report/main.tex` — LaTeX, 10–12 pages
- **Power BI dashboard** — credit risk scorecard with drill-down by grade/intent/income band
- **Streamlit app** in `app/app.py` — single-application default-risk predictor with P(default), risk tier, and a lending recommendation at the deployed threshold

## AI Audit Log workflow

Live log at `report/ai_audit_log.md`. Claude adds DECISION / PROBLEM-SOLVING / VERIFICATION entries (Entry #, Prompt Type, Stage, Problem/Context, Prompt, AI Response summary), updates the coverage tracker and "Last updated" line, and suggests Human Delta / Evidence in `[brackets]` without filling them. Students fill Human Delta & Reflection, Evidence, and the hallucination table.

## Streamlit App

```
.venv/bin/streamlit run app/app.py
```

Loads `primary_model.pkl` from `Data/filtered/model_outputs/`, takes an applicant's loan details in the sidebar, derives the engineered features inline (kept in sync with `04`/`05`), and returns P(default), a risk tier, and an approve/decline recommendation at the tuned threshold. `app/data_quality_dashboard.py` compares raw `credit_risk_dataset.csv` vs cleaned `clean_data.csv`.

## Project Tracking

- **`TODO.md`** — task checklist by week and deliverable.
- **`PROBLEMS.md`** — open decisions requiring group consensus.

## Source Code Management

Important: run `black` and `flake8` (max-line-length=88) before pushing to GitHub.
