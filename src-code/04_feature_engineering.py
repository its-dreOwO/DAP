"""
DAP391m Project 8 - Feature Engineering & Preprocessing
=======================================================

Builds leakage-safe engineered features from Data/filtered/clean_data.csv and
exports train/test matrices for downstream experiments.

The official deployment model is trained by src-code/05_modeling.py, which
contains its own pipeline object for Streamlit compatibility. This script keeps
the reproducible feature-engineering deliverables required by the project plan.

Run:
    .venv/bin/python3 src-code/04_feature_engineering.py
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

TARGET_COL = "risk_label"
TARGET_INT_COL = "target"
CLASS_LABELS = ["Low", "Medium", "High"]
LABEL_TO_INT = {label: idx for idx, label in enumerate(CLASS_LABELS)}

LEAKAGE_COLS = ["risk_probability", "port_delay_days"]
DROP_FEATURE_COLS = ["machine_id", "timestamp"]


def load_clean_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run src-code/01_ingestion_cleaning.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("clean_data.csv contains invalid timestamps")
    return df


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    low = values.min()
    high = values.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(0.0, index=series.index)
    return (values - low) / (high - low)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("timestamp").reset_index(drop=True).copy()

    out["order_month"] = out["timestamp"].dt.month
    out["order_weekday"] = out["timestamp"].dt.dayofweek
    out["order_hour"] = out["timestamp"].dt.hour

    grouped_lead = out.groupby("supplier_id")["supplier_lead_time_days"]
    out["supplier_avg_lead"] = grouped_lead.transform(
        lambda s: s.expanding().mean().shift(1)
    )
    out["supplier_std_lead"] = grouped_lead.transform(
        lambda s: s.expanding().std().shift(1)
    )

    out["supplier_avg_lead"] = out["supplier_avg_lead"].fillna(
        out["supplier_lead_time_days"].median()
    )
    out["supplier_std_lead"] = out["supplier_std_lead"].fillna(0.0)

    lead_avg_norm = _minmax(out["supplier_avg_lead"])
    lead_std_norm = _minmax(out["supplier_std_lead"])
    quality_risk = ((100 - out["supplier_quality_score"]) / 100).clip(0, 1)
    reliability_risk = (1 - out["supplier_reliability_index"]).clip(0, 1)
    out["supplier_risk_score"] = (
        lead_avg_norm * 0.4
        + lead_std_norm * 0.3
        + quality_risk * 0.2
        + reliability_risk * 0.1
    )

    weather_norm = _minmax(out["weather_disruption_score"])
    fuel_norm = _minmax(out["fuel_price_index"])
    demand_norm = _minmax(out["market_demand_index"])
    out["external_risk_score"] = (
        weather_norm * 0.5 + fuel_norm * 0.3 + (1 - demand_norm) * 0.2
    )

    out[TARGET_COL] = out[TARGET_COL].astype(str).str.strip()
    unknown = sorted(set(out[TARGET_COL]) - set(CLASS_LABELS))
    if unknown:
        raise ValueError(f"Unexpected risk_label values: {unknown}")
    out[TARGET_INT_COL] = out[TARGET_COL].map(LABEL_TO_INT)

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

    feature_df = engineered.drop(columns=[*LEAKAGE_COLS, *DROP_FEATURE_COLS])
    feature_df.to_csv(ENGINEERED_CSV, index=False)

    y = feature_df[TARGET_INT_COL].astype(int)
    X = feature_df.drop(columns=[TARGET_COL, TARGET_INT_COL])
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
    y_train.to_frame(TARGET_INT_COL).to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_frame(TARGET_INT_COL).to_csv(OUT_DIR / "y_test.csv", index=False)

    with (ARTIFACT_DIR / "preprocessor_tree.pkl").open("wb") as handle:
        pickle.dump(tree_preprocessor, handle)
    with (ARTIFACT_DIR / "preprocessor_scaled.pkl").open("wb") as handle:
        pickle.dump(scaled_preprocessor, handle)

    metadata = {
        "source": str(INPUT_CSV.relative_to(PROJECT_ROOT)),
        "engineered_features": str(ENGINEERED_CSV.relative_to(PROJECT_ROOT)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "class_labels": CLASS_LABELS,
        "label_to_int": LABEL_TO_INT,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_columns_tree": list(X_train_tree.columns),
        "feature_columns_scaled": list(X_train_scaled.columns),
        "leakage_columns_dropped": LEAKAGE_COLS,
        "iqr_bounds": iqr_bounds,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved {ENGINEERED_CSV.relative_to(PROJECT_ROOT)} ({feature_df.shape})")
    print(f"Saved processed matrices to {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
