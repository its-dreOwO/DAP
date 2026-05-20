#  Supplier Lead-Time Risk Prediction


For a full description of the project scope and research questions, see [`report/main-1.pdf`](file:///home/dre/Desktop/study/DAP/report/main-1.pdf). For the complete pipeline specification (dataset decisions, feature engineering, model choices, evaluation plan), see [`report/project\_pipeline.pdf`](file:///home/dre/Desktop/study/DAP/report/project_pipeline.pdf).

## Setup

```
python3 -m venv .venv  
source .venv/bin/activate        \# bash/zsh  
\# source .venv/bin/activate.fish \# fish shell  
  
pip install pandas numpy scikit-learn xgboost sqlalchemy \\  
            matplotlib seaborn plotly folium streamlit shap openpyxl
```

Run the pipeline in order:

```
.venv/bin/python3 src-code/01\_ingestion\_cleaning.py  
.venv/bin/python3 src-code/02\_sql\_analysis.py  
.venv/bin/python3 src-code/03\_eda.py  
.venv/bin/python3 src-code/04\_feature\_engineering.py  
.venv/bin/python3 src-code/05\_modeling.py  
.venv/bin/python3 src-code/06\_visualization\_advanced.py
```

## Streamlit App

```
.venv/bin/streamlit run app/app.py
```

