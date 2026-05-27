# AI Audit Log — DAP391m Project 8

Tracks **core prompts only** (DECISION / PROBLEM-SOLVING / VERIFICATION) per the
`docs/AI_AuditLog_Template_DAP391m.xlsx` framework.

**Human Delta & Evidence columns are left for the student to complete.**
Suggestions in `[brackets]` indicate what to write — do not copy verbatim.

---

## Coverage tracker

| Component | Logged | Required |
|---|---|---|
| Business & Problem Understanding | 0 | ≥ 2 |
| Data Understanding & Preparation | 1 | ≥ 3 |
| EDA | 1 | ≥ 2 |
| Modeling & Regression Analysis | 1 | ≥ 4 |
| Evaluation, Visualization & Reporting | 1 | ≥ 3 |

---

## Entry 001

| Field | Content |
|---|---|
| **Entry #** | 001 |
| **Prompt Type** | DECISION |
| **Stage/Component** | Modeling & Regression Analysis |
| **Problem/Context** | Cần xác định chiến lược xử lý dữ liệu cho mô hình XGBoost để tránh overfitting và tăng khả năng dự đoán trên tập test |
| **Prompt to AI** | "considering the XGBOOST model is the main focus how do we standardize/process the data to help the model in training not to be overfitting and good at predicting?" |
| **AI Response (Summary)** | AI đề xuất: (1) Không cần StandardScaler cho XGBoost vì tree-based; dùng scaler riêng cho LogReg. (2) Target encoding cho O_Country/D_Country thay vì one-hot do high cardinality. (3) `scale_pos_weight ≈ 5.25` cho class imbalance 16/84. (4) Cảnh báo leakage risk trên các cột avg_* nếu tính từ toàn bộ dataset trước khi split. (5) Key hyperparameters: `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=30`, `eval_metric='aucpr'`. |
| **Human Delta & Reflection** | *(To complete — answer all 4 questions):* [Critical Thinking: e.g. did you agree/disagree with target encoding? Why?] [Contextualization: how does this apply specifically to your 704-row dataset?] [Creative Synthesis: what did you add/change beyond the AI suggestion?] [Decision Ownership: state the final decisions YOU made] |
| **Evidence** | *(To complete):* [Screenshot of final `src-code/04_feature_engineering.py` using leakage-safe temporal/supplier features, `ColumnTransformer`, OHE, `StandardScaler` for scaled outputs, IQR capping fit on train data, and dual processed matrices in `Data/filtered/processed/`. Add `05_modeling.py` output showing the 3-class XGBoost parameters and `Data/filtered/model_outputs/model_comparison.csv`. Current split: 1,982 train / 496 test from 2,478 rows.] |

---

## Entry 002

| Field | Content |
|---|---|
| **Entry #** | 002 |
| **Prompt Type** | VERIFICATION |
| **Stage/Component** | Data Understanding & Preparation; EDA; Evaluation, Visualization & Reporting |
| **Problem/Context** | The tracker said key scripts were empty or blocked, but the repository already contained modeling and Streamlit work. The pipeline needed verification and reconciliation before adding more deliverables. |
| **Prompt to AI** | "Implement the previous work plan: make the six pipeline scripts runnable in order, reconcile modeling/app state, update TODO/PROBLEMS, and add an AI audit log entry." |
| **AI Response (Summary)** | AI verified the active dataset as `Data/supply_chain_risk_dataset.csv`, implemented ingestion, SQL execution, EDA, active-dataset feature engineering, and Plotly visualization exports, preserved leakage columns only for EDA, reran `05_modeling.py`, kept XGBoost as the deployment model through `primary_model.pkl`/`xgb_model.pkl`, and smoke-tested the Streamlit app startup. |
| **Human Delta & Reflection** | *(To complete — answer all 4 questions):* [Critical Thinking: check whether the SQL shipment-to-supplier normalization is acceptable for the business-analysis CSVs.] [Contextualization: explain why the active modeling dataset differs from the older customer/shipment/logistics SQL files.] [Creative Synthesis: note any report or Power BI changes the team adds after reviewing the outputs.] [Decision Ownership: state which generated artifacts the team will commit versus regenerate locally.] |
| **Evidence** | *(To complete):* [Terminal screenshots of scripts `01`-`06` running successfully, `Data/filtered/sql_outputs/`, `Data/filtered/eda_outputs/`, `Data/filtered/model_outputs/model_comparison.csv`, `Data/filtered/visualization_outputs/`, and Streamlit startup at `http://localhost:8501`.] |

---

<!-- Add new entries below as work progresses. Use the template:

## Entry 00X

| Field | Content |
|---|---|
| **Entry #** | 00X |
| **Prompt Type** | DECISION / PROBLEM-SOLVING / VERIFICATION |
| **Stage/Component** | |
| **Problem/Context** | |
| **Prompt to AI** | |
| **AI Response (Summary)** | |
| **Human Delta & Reflection** | *(To complete)* |
| **Evidence** | *(To complete)* |

-->

---

## Hallucination log

| Entry # | Type | AI Claim | Reality Check | How Detected | Corrective Action |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

*Project requires ≥ 3 detected hallucinations. Log here when found.*

---

*Last updated: 2026-05-27 — Entry 002 added (pipeline reconciliation and verification); scripts 01-06 run end-to-end*
