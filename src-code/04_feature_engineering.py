"""
DAP391m Project 8 — Feature Engineering & Preprocessing
=======================================================

Strict standardization pipeline for supplier-lead-time delay prediction.

Design rules (read before changing):
    1. Reproducibility — every random op uses RANDOM_STATE = 42.
    2. No data leakage — every encoder/scaler is fit ONLY on the training
       fold, then transform() is applied to the test fold.
    3. Stratified train/test split — preserves the ~16/84 class balance.
    4. Two parallel outputs:
         * `X_*_scaled`  — for distance-based / linear models (LogReg).
         * `X_*_tree`    — unscaled, target-encoded for tree models (XGB, RF, DT).
       Trees split on thresholds, not distances; scaling is a no-op for them.
    5. Persisted artifacts — all fitted transformers are pickled so the
       Streamlit app and any downstream script use the *exact* same preprocessing.

Run:
    .venv/bin/python3 src-code/04_feature_engineering.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

# ── config ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "model_features.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
ARTIFACT_DIR = OUT_DIR / "artifacts"

DROP_COLS = ["shipment_id", "date"]
TARGET_COL = "target"

# tree-model friendly: keep month/quarter as integers, no scaling needed
NUMERIC_COLS = [
    "value",
    "freight_cost",
    "freight_ratio",
    "customs_clearance_time_days",
    "month",
    "quarter",
    "avg_delay_hours",
    "avg_fuel_price",
    "avg_warehouse_util",
    "avg_damage_claims",
    "avg_lead_time_days",
    "avg_satisfaction",
    "avg_support_tickets",
    "avg_order_value",
    "avg_days_to_pay",
]

BINARY_COLS = ["is_import"]                            # already 0/1
LOW_CARD_COLS = ["product_category", "region"]         # ordinal encode
HIGH_CARD_COLS = ["O_Country", "D_Country"]            # target encode (fit on train only)


# ── helpers ─────────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run Data/filtered/extract_model_data.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    expected = (
        DROP_COLS
        + BINARY_COLS
        + LOW_CARD_COLS
        + HIGH_CARD_COLS
        + NUMERIC_COLS
        + [TARGET_COL]
    )
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"Schema check failed — missing columns: {missing}")
    return df


def build_tree_preprocessor() -> ColumnTransformer:
    """Tree models: ordinal + target encode, NO scaling."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
            (
                "lowcard",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                LOW_CARD_COLS,
            ),
            (
                "highcard",
                TargetEncoder(
                    target_type="binary",
                    smooth="auto",
                    random_state=RANDOM_STATE,
                ),
                HIGH_CARD_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_scaled_preprocessor() -> ColumnTransformer:
    """Linear models: same encoding + StandardScaler on numeric block."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
            (
                "lowcard",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                LOW_CARD_COLS,
            ),
            (
                "highcard",
                TargetEncoder(
                    target_type="binary",
                    smooth="auto",
                    random_state=RANDOM_STATE,
                ),
                HIGH_CARD_COLS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_to_frame(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    fitted: bool,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    """Run transform/fit_transform and rebuild a DataFrame with named columns."""
    if fitted:
        arr = preprocessor.transform(X)
    else:
        arr = preprocessor.fit_transform(X, y)
    cols = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=cols, index=X.index)


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print(f"Loaded {INPUT_CSV.name} — shape={df.shape}")

    # ── separate target & drop ID/date ──────────────────────────────────────
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=DROP_COLS + [TARGET_COL])

    pos_rate = float(y.mean())
    scale_pos_weight = (1.0 - pos_rate) / pos_rate
    print(
        f"Class balance: pos={int(y.sum())} ({pos_rate:.2%}) | "
        f"neg={int((1 - y).sum())} ({1 - pos_rate:.2%})"
    )
    print(f"scale_pos_weight = {scale_pos_weight:.3f}  (use in XGBoost)")

    # ── stratified split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(
        f"Train pos rate: {y_train.mean():.2%}  "
        f"Test pos rate: {y_test.mean():.2%}  (should match)"
    )

    # ── tree preprocessor (unscaled) ────────────────────────────────────────
    tree_pre = build_tree_preprocessor()
    X_train_tree = transform_to_frame(tree_pre, X_train, fitted=False, y=y_train)
    X_test_tree = transform_to_frame(tree_pre, X_test, fitted=True)

    # ── linear preprocessor (scaled) ────────────────────────────────────────
    scaled_pre = build_scaled_preprocessor()
    X_train_scaled = transform_to_frame(scaled_pre, X_train, fitted=False, y=y_train)
    X_test_scaled = transform_to_frame(scaled_pre, X_test, fitted=True)

    # ── persist data ────────────────────────────────────────────────────────
    X_train_tree.to_csv(OUT_DIR / "X_train_tree.csv", index=False)
    X_test_tree.to_csv(OUT_DIR / "X_test_tree.csv", index=False)
    X_train_scaled.to_csv(OUT_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(OUT_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    # ── persist fitted transformers ─────────────────────────────────────────
    with open(ARTIFACT_DIR / "preprocessor_tree.pkl", "wb") as f:
        pickle.dump(tree_pre, f)
    with open(ARTIFACT_DIR / "preprocessor_scaled.pkl", "wb") as f:
        pickle.dump(scaled_pre, f)

    # ── metadata for downstream scripts ─────────────────────────────────────
    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
        "scale_pos_weight": scale_pos_weight,
        "feature_columns_tree": list(X_train_tree.columns),
        "feature_columns_scaled": list(X_train_scaled.columns),
        "numeric_cols": NUMERIC_COLS,
        "binary_cols": BINARY_COLS,
        "low_card_cols": LOW_CARD_COLS,
        "high_card_cols": HIGH_CARD_COLS,
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("Saved to", OUT_DIR.relative_to(PROJECT_ROOT))
    print("  X_train_tree.csv / X_test_tree.csv      (for XGB, RF, DT)")
    print("  X_train_scaled.csv / X_test_scaled.csv  (for LogReg)")
    print("  y_train.csv / y_test.csv")
    print("  artifacts/preprocessor_tree.pkl")
    print("  artifacts/preprocessor_scaled.pkl")
    print("  artifacts/metadata.json")


if __name__ == "__main__":
    main()
