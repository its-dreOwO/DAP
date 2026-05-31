"""
DAP391m Project 8 - Feature Engineering & Preprocessing
=======================================================

Builds engineered credit-risk features from Data/filtered/clean_data.csv and
exports train/test matrices for downstream experiments.

The official deployment model is trained by src-code/05_modeling.py, which
contains its own pipeline object for Streamlit compatibility. This script keeps
the reproducible feature-engineering deliverables required by the project plan.

There is no target leakage to drop here (loan_status is a real outcome); all
engineered features are derived from decision-time applicant/loan attributes.

Run:
    .venv/bin/python3 src-code/04_feature_engineering.py
    (run src-code/01_ingestion_cleaning.py first)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
ARTIFACT_DIR = OUT_DIR / "artifacts"
ENGINEERED_CSV = PROJECT_ROOT / "Data" / "filtered" / "engineered_features.csv"

TARGET_COL = "loan_status"
CLASS_LABELS = ["Non-default", "Default"]

BASE_CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
ENGINEERED_CATEGORICAL = ["income_bracket", "age_bracket", "loan_amount_bracket"]


def load_clean_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run src-code/01_ingestion_cleaning.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in {INPUT_CSV.name}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Debt-burden and credit-history ratios (decision-time, leakage-safe).
    out["high_debt_burden"] = (out["loan_percent_income"] > 0.30).astype(int)
    out["credit_hist_ratio"] = (
        out["cb_person_cred_hist_length"] / out["person_age"]
    ).round(4)
    out["interest_to_income"] = (
        ((out["loan_int_rate"] / 100.0) * out["loan_amnt"])
        / out["person_income"].clip(lower=1)
    ).round(4)

    out["income_bracket"] = pd.cut(
        out["person_income"],
        bins=[-1, 30000, 60000, 100000, float("inf")],
        labels=["<30k", "30k-60k", "60k-100k", "100k+"],
    ).astype(str)
    out["age_bracket"] = pd.cut(
        out["person_age"],
        bins=[-1, 25, 35, 50, float("inf")],
        labels=["<=25", "26-35", "36-50", "50+"],
    ).astype(str)
    out["loan_amount_bracket"] = pd.cut(
        out["loan_amnt"],
        bins=[-1, 5000, 10000, 20000, float("inf")],
        labels=["<5k", "5k-10k", "10k-20k", "20k+"],
    ).astype(str)

    return out


def cap_iqr(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    train_out = train.copy()
    test_out = test.copy()
    bounds: dict[str, dict[str, float]] = {}

    for col in columns:
        q1 = train_out[col].quantile(0.25)
        q3 = train_out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        train_out[col] = train_out[col].clip(lower, upper)
        test_out[col] = test_out[col].clip(lower, upper)
        bounds[col] = {"lower": float(lower), "upper": float(upper)}

    return train_out, test_out, bounds


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def transform_to_frame(
    preprocessor: ColumnTransformer, X: pd.DataFrame
) -> pd.DataFrame:
    arr = preprocessor.transform(X)
    cols = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=cols, index=X.index)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_clean_data()
    engineered = add_engineered_features(raw)
    engineered.to_csv(ENGINEERED_CSV, index=False)

    y = engineered[TARGET_COL].astype(int)
    X = engineered.drop(columns=[TARGET_COL])

    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test, iqr_bounds = cap_iqr(X_train, X_test, numeric_cols)

    tree_preprocessor = build_preprocessor(numeric_cols, categorical_cols, False)
    scaled_preprocessor = build_preprocessor(numeric_cols, categorical_cols, True)

    X_train_tree = pd.DataFrame(
        tree_preprocessor.fit_transform(X_train),
        columns=tree_preprocessor.get_feature_names_out(),
        index=X_train.index,
    )
    X_test_tree = transform_to_frame(tree_preprocessor, X_test)

    X_train_scaled = pd.DataFrame(
        scaled_preprocessor.fit_transform(X_train),
        columns=scaled_preprocessor.get_feature_names_out(),
        index=X_train.index,
    )
    X_test_scaled = transform_to_frame(scaled_preprocessor, X_test)

    X_train_tree.to_csv(OUT_DIR / "X_train_tree.csv", index=False)
    X_test_tree.to_csv(OUT_DIR / "X_test_tree.csv", index=False)
    X_train_scaled.to_csv(OUT_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(OUT_DIR / "X_test_scaled.csv", index=False)
    y_train.to_frame(TARGET_COL).to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_frame(TARGET_COL).to_csv(OUT_DIR / "y_test.csv", index=False)

    with (ARTIFACT_DIR / "preprocessor_tree.pkl").open("wb") as handle:
        pickle.dump(tree_preprocessor, handle)
    with (ARTIFACT_DIR / "preprocessor_scaled.pkl").open("wb") as handle:
        pickle.dump(scaled_preprocessor, handle)

    metadata = {
        "source": str(INPUT_CSV.relative_to(PROJECT_ROOT)),
        "engineered_features": str(ENGINEERED_CSV.relative_to(PROJECT_ROOT)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target": TARGET_COL,
        "class_labels": CLASS_LABELS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_columns_tree": list(X_train_tree.columns),
        "feature_columns_scaled": list(X_train_scaled.columns),
        "iqr_bounds": iqr_bounds,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved {ENGINEERED_CSV.relative_to(PROJECT_ROOT)} ({engineered.shape})")
    print(f"Saved processed matrices to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(
        f"  train={len(X_train)} test={len(X_test)} "
        f"features_tree={X_train_tree.shape[1]}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
