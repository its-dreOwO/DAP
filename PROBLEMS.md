# Open Problems — Group 8 Discussion

This file tracks decisions and issues that need everyone's input before we proceed.
Add your name and opinion next to each item. Once consensus is reached, mark it **RESOLVED** and note the decision.

---

## P1 — Primary Evaluation Metric: PR-AUC vs. ROC-AUC

**Context:** The dataset is heavily imbalanced (63% High, 20% Low, 18% Medium). PR-AUC is more informative than ROC-AUC for imbalanced data, but the reference paper reports ROC-AUC. CLAUDE.md sets PR-AUC as primary.

**Question:** Do we lead with PR-AUC in the report tables, or report both side-by-side and let the reader judge?

| Name | Opinion |
|------|---------|
| Khánh | |
| Đăng | |
| Bảo | |

**Status:** ⬜ Open

---

## P2 — `01_ingestion_cleaning.py` Is Only 28 Lines

**Context:** The ingestion script is minimal and does not yet output `clean_data.csv`. The rest of the pipeline (`02`–`06`) cannot run without it.

**Question:** Who takes ownership of completing this script? It needs: timestamp parsing, type-fixing, null guard, drop `machine_id`, output to `Data/filtered/clean_data.csv`.

| Name | Opinion / Assignment |
|------|---------------------|
| Khánh | |
| Đăng | |
| Bảo | |

**Status:** ⬜ Open — **blocking all downstream scripts**

---

## P3 — `clean_data.csv` Not Present in `Data/filtered/`

**Context:** Even though `04_feature_engineering.py` is written, it reads `clean_data.csv` which doesn't exist yet. Scripts `02`, `03`, `05`, `06` are all empty.

**Question:** Should we run `01` first and commit the output CSV to the repo, or `.gitignore` generated data and require everyone to run the pipeline locally?

| Name | Opinion |
|------|---------|
| Khánh | |
| Đăng | |
| Bảo | |

**Status:** ⬜ Open

---

## P4 — Work Division for Empty Scripts

**Context:** Scripts `02_sql_analysis.py`, `03_eda.py`, `05_modeling.py`, `06_visualization_advanced.py` are all empty (0 lines). These are the core of the project.

**Question:** How do we split this work? Suggested split below — revise as needed.

| Script | Suggested Owner | Agreed? |
|--------|----------------|---------|
| `02_sql_analysis.py` | | |
| `03_eda.py` | | |
| `05_modeling.py` | | |
| `06_visualization_advanced.py` | | |
| `app/app.py` (complete) | | |
| Power BI dashboard | | |
| `report/main.tex` sections | | |

**Status:** ⬜ Open — **critical to assign before Week 2 ends**

---

## P5 — Streamlit App: Model Not Trained Yet

**Context:** `app/app.py` is a stub. It can't run until `05_modeling.py` saves a trained XGBoost model (`.pkl` or `.json`). The app loads the model and runs SHAP waterfall explanations.

**Question:** What format should the saved model use (`.pkl` via joblib vs. XGBoost native `.json`)? Where should it be saved (`models/xgb_model.json`)?

| Name | Opinion |
|------|---------|
| Khánh | |
| Đăng | |
| Bảo | |

**Status:** ⬜ Open

---

## P6 — Report Length & Section Ownership

**Context:** `report/main.tex` has a skeleton (166 lines) but all content sections are empty. The course requires 10–12 pages.

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

*(Move items here once the group agrees on a decision)*

| # | Decision | Date |
|---|----------|------|
| — | — | — |
