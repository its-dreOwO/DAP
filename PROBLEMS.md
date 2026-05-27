# Open Problems — Group 8 Discussion

This file tracks decisions and issues that need everyone's input before we proceed.
Add your name and opinion next to each item. Once consensus is reached, mark it
**RESOLVED** and note the decision.

---

## P1 — Primary Evaluation Metric: PR-AUC vs. ROC-AUC

**Context:** The dataset is imbalanced (63% High, 20% Low, 18% Medium). PR-AUC
is more informative than ROC-AUC for imbalanced data, but the reference paper
reports ROC-AUC. `CLAUDE.md` sets PR-AUC as primary.

**Question:** Do we lead with PR-AUC in the report tables, or report both
side-by-side and let the reader judge?

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
