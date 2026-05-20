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
| Data Understanding & Preparation | 0 | ≥ 3 |
| EDA | 0 | ≥ 2 |
| Modeling & Regression Analysis | 1 | ≥ 4 |
| Evaluation, Visualization & Reporting | 0 | ≥ 3 |

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
| **Evidence** | *(To complete):* [Screenshot of final `src-code/04_feature_engineering.py` (already implemented — uses sklearn `ColumnTransformer` + `TargetEncoder` + `StandardScaler`, fits only on train fold, produces dual outputs in `Data/filtered/processed/`) + `05_modeling.py` showing XGBoost params + train/val PR-AUC comparison before/after tuning. Reference script output: 563 train / 141 test, pos rate 15.63% / 15.60%, scale_pos_weight=5.400] |

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

*Last updated: 2026-05-19 — Entry 001 added (XGBoost preprocessing strategy); 04_feature_engineering.py implemented*
