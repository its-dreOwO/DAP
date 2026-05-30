# AI Audit Log — DAP391m Project 8

Tracks **core prompts only** (DECISION / PROBLEM-SOLVING / VERIFICATION) per the
`docs/AI_AuditLog_Template_DAP391m.xlsx` framework.

**Human Delta & Evidence columns are left for the student to complete.**
Suggestions in `[brackets]` indicate what to write — do not copy verbatim.

---

## Coverage tracker

| Component | Logged | Required |
|---|---|---|
| Business & Problem Understanding | 1 | ≥ 2 |
| Data Understanding & Preparation | 2 | ≥ 3 |
| EDA | 1 | ≥ 2 |
| Modeling & Regression Analysis | 2 | ≥ 4 |
| Evaluation, Visualization & Reporting | 2 | ≥ 3 |

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

## Entry 003

| Field | Content |
|---|---|
| **Entry #** | 003 |
| **Prompt Type** | DECISION / PROBLEM-SOLVING |
| **Stage/Component** | Modeling & Regression Analysis; Evaluation, Visualization & Reporting |
| **Problem/Context** | The first XGBoost result had the best baseline Macro PR-AUC but weak Low/Medium class recognition. The team needed to know whether recalibration or imbalance handling should replace the official report model. |
| **Prompt to AI** | "can you recalibrate / try to improve it"; then "what do you think the best solution for this would be"; then "keep both" |
| **AI Response (Summary)** | AI tested an imbalance-weighted modeling variant, compared it against the baseline PR-AUC run, and updated `05_modeling.py` to preserve both outputs. Baseline XGBoost remains the official deployment/report model because it is best by Macro PR-AUC (0.3571). The imbalance-weighted experiment is saved separately because it improves XGBoost macro F1 (0.3587 vs. 0.3102) but lowers XGBoost Macro PR-AUC (0.3473) and accuracy. |
| **Human Delta & Reflection** | *(To complete — answer all 4 questions):* [Critical Thinking: decide whether the group agrees that Macro PR-AUC should remain the primary metric.] [Contextualization: explain whether supplier-risk decisions value overall PR-AUC or minority-class recall/F1 more.] [Creative Synthesis: describe how the team will present both variants in the report.] [Decision Ownership: state the final model-framing decision made by the group.] |
| **Evidence** | *(To complete):* [Add screenshots or snippets from `Data/filtered/model_outputs/model_comparison.csv`, `Data/filtered/model_outputs/imbalance_weighted/model_comparison.csv`, and `Data/filtered/model_outputs/model_comparison_all_experiments.csv`. Include `PROBLEMS.md` P7 discussion outcome once resolved.] |

---

## Entry 004

| Field | Content |
|---|---|
| **Entry #** | 004 |
| **Prompt Type** | VERIFICATION / DECISION |
| **Stage/Component** | Business & Problem Understanding; Data Understanding & Preparation |
| **Problem/Context** | XGBoost appeared to classify almost everything as the majority class. The student asked whether the dataset (63% majority) was the problem and whether to find a new one. Needed to verify the true cause before committing to a fix. |
| **Prompt to AI** | "looking that dataset the high is taking about 63% explaining why XGBOOST is classifying everyone is high risk should we find another dataset?"; then "can you use the kaggle skills and see what feature each dataset has and is it really suitable" |
| **AI Response (Summary)** | AI checked the actual predictions/metrics and found the real cause was not imbalance but **near-zero signal**: all 4 models scored ROC-AUC ≈ 0.52 because the target `risk_label` was synthetically derived from the dropped leakage column `port_delay_days`. After supervisor approval to switch datasets, AI downloaded both supervisor-suggested Kaggle datasets and verified them: `uciml/german-credit` has **no target column** (10 feature columns only) → rejected; `laotse/credit-risk-dataset` (32,581 rows, target `loan_status`, 78/22 imbalance) showed genuine signal (quick RandomForest ROC-AUC ≈ 0.93) → adopted. The full pipeline (`01`–`06`), SQL, and Streamlit app were rebuilt for binary credit-default classification; final XGBoost PR-AUC = 0.911, ROC-AUC = 0.952. |
| **Human Delta & Reflection** | *(To complete — answer all 4 questions):* [Critical Thinking: explain why ROC-AUC ≈ 0.52 points to a signal problem rather than an imbalance problem.] [Contextualization: distinguish leakage `port_delay_days` (outcome-derived) from `loan_grade`/`loan_int_rate` (decision-time inputs).] [Creative Synthesis: describe how the team re-framed the project from supply-chain to credit risk.] [Decision Ownership: record the supervisor's confirmation and the group's final dataset choice.] |
| **Evidence** | *(To complete):* [Old `predictions_test.csv` showing 456/496 predicted High and `model_comparison.csv` with ROC-AUC ≈ 0.52; Kaggle inspection output showing German Credit has no target column and Credit Risk RF ROC-AUC ≈ 0.93; new `Data/filtered/model_outputs/model_comparison.csv` (XGBoost PR-AUC 0.911). This is a candidate hallucination-check: the supervisor's suggested dataset was empirically unusable.] |

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

*Last updated: 2026-05-29 — Entry 004 added (dataset pivot to credit-default risk: signal verification + Kaggle dataset suitability check)*
