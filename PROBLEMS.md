# Open Problems — Group 8 Discussion

This file tracks decisions and issues that need everyone's input before we proceed.
Add your name and opinion next to each item. Once consensus is reached, mark it
**RESOLVED** and note the decision.

---

## P1 — Primary Evaluation Metric: PR-AUC vs. ROC-AUC

**Context:** The credit dataset is imbalanced (78% non-default, 22% default).
PR-AUC is more informative than ROC-AUC for imbalanced data. `CLAUDE.md` sets
PR-AUC as primary; ROC-AUC is reported as a secondary metric.

**Question:** Do we lead with PR-AUC in the report tables, or report both
side-by-side and let the reader judge?

| Name | Opinion |
|------|---------|
| Khánh | |
| Đăng | |
| Bảo | |

**Status:** ⬜ Open

---

## P7 — Report Framing: Baseline XGBoost vs. Imbalance-Weighted Experiment

**Context:** `src-code/05_modeling.py` now preserves two result sets. The
baseline PR-AUC experiment is saved in `Data/filtered/model_outputs/` and keeps
XGBoost as the best model by the primary metric:

| Experiment | XGBoost PR-AUC | XGBoost Recall | XGBoost Precision | XGBoost F1 |
|------------|----------------|----------------|-------------------|------------|
| Baseline | 0.911 | 0.740 | 0.979 | 0.843 |
| Imbalance-weighted | 0.910 | 0.818 | 0.819 | 0.817 |

The imbalance-weighted experiment improves XGBoost recall (catches more
defaults) at the cost of precision; PR-AUC and F1 are marginally lower. Since
`CLAUDE.md` defines PR-AUC as the primary metric, the baseline XGBoost remains
the official model. For credit, the recall/precision balance is a business
policy choice (cost of a missed default vs. a rejected good loan).

**Question:** Should the report use baseline XGBoost as the official deployed
model and present the imbalance-weighted run as a limitation/future-work
experiment, or should the team prioritize minority-class F1/recall over
Macro PR-AUC?

**Recommended discussion position:** Keep baseline XGBoost as the official
model for report, Streamlit, SHAP, and deployment artifacts. Use the
imbalance-weighted results to show that class-balance improvements are possible
but involve a metric tradeoff.

| Name | Opinion |
|------|---------|
| Khánh | |
| Đăng | |
| Bảo | |

**Status:** ⬜ Open

---

## P6 — Report Length & Section Ownership

**Context:** `report/main.tex` has a skeleton (166 lines) but the course
requires 10-12 pages.

**Question:** Who writes which sections? Suggested split:

| Section | Suggested Owner | Agreed? |
|---------|----------------|---------|
| Introduction & research questions | | |
| Literature review (5 frameworks) | | |
| Data description & leakage analysis | | |
| Methodology | | |
| Results & evaluation | | |
| Discussion & limitations | | |
| Conclusion | | |

**Status:** ⬜ Open

---

## RESOLVED

| # | Decision | Date |
|---|----------|------|
| P2 | `01_ingestion_cleaning.py` now validates `Data/supply_chain_risk_dataset.csv`, parses timestamps, trims `risk_label`, guards missing targets/numeric fields, preserves leakage columns for EDA, and writes `Data/filtered/clean_data.csv`. | 2026-05-27 |
| P3 | `clean_data.csv` is generated locally by script `01`; generated CSV/HTML/PNG outputs now exist under `Data/filtered/` for reproducible local runs. The group can still decide later whether to commit generated artifacts. | 2026-05-27 |
| P4 | Python pipeline ownership blocker is resolved for scripts `02`, `03`, `05`, `06`, and the Streamlit app. Power BI and report authorship remain tracked in their own checklist sections. | 2026-05-27 |
| P5 | Streamlit app model format is resolved as pickle artifacts under `Data/filtered/model_outputs/`: `primary_model.pkl` and `xgb_model.pkl` for XGBoost, with `best_model.pkl` as fallback. | 2026-05-27 |
| P8 | **Dataset pivot.** Old `supply_chain_risk_dataset.csv` had a synthetic leakage-derived target (ROC-AUC ≈ 0.52 after removing leakage). Supervisor approved switching to `laotse/credit-risk-dataset` (ROC-AUC ≈ 0.93). His other suggestion, `uciml/german-credit`, was rejected — that Kaggle upload has no target column. Pipeline rebuilt for binary `loan_status`; old work archived under `Data/archived/`. **Pending:** supervisor's written confirmation of the full title/topic change. | 2026-05-29 |
