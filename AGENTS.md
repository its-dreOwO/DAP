# Repository Guidelines

## Project Structure & Module Organization

This repository supports a DAP391m supplier lead-time risk prediction project.
Raw CSV inputs live in `Data/`, while cleaned and derived datasets live in
`Data/filtered/`. Keep generated modeling tables there, especially
`Data/filtered/model_features.csv`. The notebook pipeline is in `notebooks/`
and is numbered in execution order from ingestion through visualization. SQL
deliverables belong in `sql/schema.sql` and `sql/analysis.sql`. The planned
Streamlit application entry points are `app/app.py` and `app/api_client.py`.
Course documents, audit logs, and supporting references are in `docs/`; the
final LaTeX report belongs in `report/main.tex`.

## Build, Test, and Development Commands

Use the project virtual environment for all Python work:

```bash
.venv/bin/python3 script.py
.venv/bin/jupyter notebook notebooks/01_ingestion_cleaning.ipynb
.venv/bin/pip install pandas numpy scikit-learn xgboost sqlalchemy matplotlib seaborn plotly folium streamlit shap prophet boto3 ipykernel jupyter openpyxl
```

Run notebooks in numeric order. If the model feature extraction script is
restored, run it from `Data/filtered/`:

```bash
cd Data/filtered && ../../.venv/bin/python3 extract_model_data.py
```

When the Streamlit app is implemented, run it with:

```bash
.venv/bin/streamlit run app/app.py
```

## Coding Style & Naming Conventions

Prefer Python 3 with 4-space indentation, clear function names, and explicit
dataframe column transformations. Use `snake_case` for Python files, functions,
variables, and generated CSV names; avoid spaces in new filenames. Keep notebook
names numbered and descriptive, for example `04_feature_engineering.ipynb`.
Favor reproducible transformations over manual spreadsheet edits.

## Testing Guidelines

No automated test suite is currently configured. Validate data changes by
rerunning the affected notebook or extraction script and checking row counts,
null counts, and target distribution. For future Python modules, add tests under
`tests/` using `pytest`, with names like `test_feature_engineering.py`.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase summaries such as `cleaned_data` and
`risk_final_v2`. Keep commit subjects concise and focused on one change. Pull
requests should describe the affected pipeline stage, list regenerated files,
include important metrics or row-count changes, and attach screenshots for app
or visualization updates. Link related course issues or deliverables when
available.

## Security & Configuration Tips

Do not commit secrets, API keys, or local credentials. Treat files in `Data/` as
source data; document any derived dataset changes in `docs/notes/` or the
relevant notebook markdown. Update the AI audit log in `docs/` for graded
submissions that use AI assistance.
