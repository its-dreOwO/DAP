"""
DAP391m Project 8 — Late-Delivery Risk Modeling (DataCo)
========================================================

Trains four binary classifiers on the DataCo Smart Supply Chain dataset
(target: Late_delivery_risk, 1 = late), evaluates metrics (PR-AUC primary),
persists artifacts for reporting, explainability, and Streamlit deployment
(XGBoost primary).

Uses GroupShuffleSplit on Order Id so line-items from the same order never
leak across train and test.

Run:
    python src-code/05_modeling.py
    (reads Data/filtered/clean_data.csv if present, else raw DataCo CSV)
"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── config ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "Data" / "dataco_raw" / "DataCoSupplyChainDataset.csv"
CLEAN_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

TARGET_COL = "Late_delivery_risk"
GROUP_KEY = "Order Id"
SHIPPING_MODE_COL = "Shipping Mode"
DATE_COL = "order date (DateOrders)"
SOURCE_DATASET_NAME = "DataCoSupplyChainDataset.csv"

CLASS_LABELS = ["On-time", "Late"]  # index == class int (0, 1)
POSITIVE_CLASS = 1

LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "shipping date (DateOrders)",
    "Order Status",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
]

PII_COLUMNS = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Customer Id",
    "Order Customer Id",
    "Latitude",
    "Longitude",
    "Product Image",
    "Product Description",
]

NON_FEATURE_COLUMNS = [GROUP_KEY, "Order Item Id"]

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "key": "logistic_regression",
        "display": "Logistic Regression",
        "pkl": "logistic_model.pkl",
        "scaled": True,
        "report": "classification_report_logistic_regression.txt",
        "cm": "confusion_matrix_logistic_regression.csv",
    },
    {
        "key": "decision_tree",
        "display": "Decision Tree",
        "pkl": "decision_tree_model.pkl",
        "scaled": False,
        "report": "classification_report_decision_tree.txt",
        "cm": "confusion_matrix_decision_tree.csv",
    },
    {
        "key": "random_forest",
        "display": "Random Forest",
        "pkl": "random_forest_model.pkl",
        "scaled": False,
        "report": "classification_report_random_forest.txt",
        "cm": "confusion_matrix_random_forest.csv",
    },
    {
        "key": "xgboost",
        "display": "XGBoost",
        "pkl": "xgb_model.pkl",
        "scaled": False,
        "report": "classification_report_xgboost.txt",
        "cm": "confusion_matrix_xgboost.csv",
        "optional": True,
    },
]


# ── data loading & feature prep ─────────────────────────────────────────────
def load_dataco_dataframe() -> pd.DataFrame:
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV)
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected {CLEAN_CSV} (run 01 first) or {RAW_CSV}."
        )
    return pd.read_csv(RAW_CSV, encoding="latin-1")


def drop_leakage_and_pii(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        c for c in LEAKAGE_COLUMNS + PII_COLUMNS if c in df.columns and c != TARGET_COL
    ]
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DATE_COL in out.columns:
        dt = pd.to_datetime(out[DATE_COL], errors="coerce")
        out["order_month"] = dt.dt.month
        out["order_day_of_week"] = dt.dt.dayofweek
        out["order_hour"] = dt.dt.hour
        out = out.drop(columns=[DATE_COL])
    return out


def load_and_prepare_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    df = load_dataco_dataframe()
    df = drop_leakage_and_pii(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    if GROUP_KEY not in df.columns:
        raise ValueError(f"Group column '{GROUP_KEY}' not found.")

    print(f"Dataset shape: {df.shape}")
    counts = df[TARGET_COL].value_counts().sort_index()
    print("Target distribution (Late_delivery_risk):")
    print(counts.to_string())
    late_pct = 100.0 * counts.get(1, 0) / len(df)
    print(f"Late-delivery rate: {late_pct:.1f}%")
    print(f"Unique orders: {df[GROUP_KEY].nunique():,}")
    print()

    shipping_df = (
        df[[SHIPPING_MODE_COL, TARGET_COL]].copy()
        if SHIPPING_MODE_COL in df.columns
        else None
    )

    df = add_engineered_features(df)
    y = df[TARGET_COL].astype(int)
    groups = df[GROUP_KEY]
    order_ids = df[GROUP_KEY]
    shipping_modes = (
        df[SHIPPING_MODE_COL]
        if SHIPPING_MODE_COL in df.columns
        else pd.Series(dtype=object)
    )

    feature_drop = [TARGET_COL] + [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    X = df.drop(columns=feature_drop)
    return X, y, groups, order_ids, shipping_modes, shipping_df


def split_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    order_ids: pd.Series,
    shipping_modes: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
]:
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()
    order_ids_test = order_ids.iloc[test_idx]
    shipping_test = shipping_modes.iloc[test_idx]
    return X_train, X_test, y_train, y_test, order_ids_test, shipping_test


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(key: str) -> Any:
    if key == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    if key == "decision_tree":
        return DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if key == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model key: {key}")


def make_model_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    scale_numeric: bool,
    key: str,
) -> Pipeline:
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric)
    estimator = build_estimator(key)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


# ── evaluation ───────────────────────────────────────────────────────────────
def positive_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["model"].classes_)
    pos_idx = classes.index(POSITIVE_CLASS)
    return proba[:, pos_idx]


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray | None,
    display_name: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model": display_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "y_pred": y_pred,
        "proba_pos": proba_pos,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_LABELS, zero_division=0
        ),
    }
    if proba_pos is not None:
        metrics["pr_auc"] = average_precision_score(y_true, proba_pos)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, proba_pos)
        except ValueError:
            metrics["roc_auc"] = np.nan
    return metrics


def select_best_model(comparison: pd.DataFrame) -> str:
    ranked = comparison.sort_values(
        by=["pr_auc", "recall", "f1"],
        ascending=[False, False, False],
        na_position="last",
    )
    return str(ranked.iloc[0]["model"])


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    try:
        return (
            float(roc_auc_score(y_true, scores)),
            float(average_precision_score(y_true, scores)),
        )
    except ValueError:
        return np.nan, np.nan


# ── shipping mode analysis ───────────────────────────────────────────────────
def compute_late_rate_by_shipping_mode(
    shipping_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if shipping_df is None or SHIPPING_MODE_COL not in shipping_df.columns:
        return pd.DataFrame(columns=[SHIPPING_MODE_COL, "count", "late_rate"])

    summary = (
        shipping_df.groupby(SHIPPING_MODE_COL, dropna=False)[TARGET_COL]
        .agg(count="count", late_rate="mean")
        .reset_index()
        .sort_values("late_rate", ascending=False)
    )
    return summary


def shipping_mode_base_rate_benchmark(
    train_modes: pd.Series,
    y_train: np.ndarray,
    test_modes: pd.Series,
    y_test: np.ndarray,
) -> dict[str, float]:
    train_df = pd.DataFrame(
        {SHIPPING_MODE_COL: train_modes.values, TARGET_COL: y_train}
    )
    global_rate = float(y_train.mean())
    mode_rates = train_df.groupby(SHIPPING_MODE_COL)[TARGET_COL].mean()
    proba = test_modes.map(mode_rates).fillna(global_rate).to_numpy()
    roc, pr = safe_auc(y_test, proba)
    return {"roc_auc": roc, "pr_auc": pr}


def xgboost_performance_by_shipping_mode(
    shipping_test: pd.Series,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_df = pd.DataFrame(
        {
            SHIPPING_MODE_COL: shipping_test.values,
            "y_true": y_test,
            "y_pred": y_pred,
            "proba": proba_pos,
        }
    )
    for mode, subset in test_df.groupby(SHIPPING_MODE_COL, dropna=False):
        yt = subset["y_true"].to_numpy()
        yp = subset["y_pred"].to_numpy()
        proba = subset["proba"].to_numpy()
        roc, pr = safe_auc(yt, proba)
        rows.append(
            {
                SHIPPING_MODE_COL: mode,
                "n_rows": len(subset),
                "accuracy": accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )
    return pd.DataFrame(rows).sort_values(SHIPPING_MODE_COL)


# ── persistence ──────────────────────────────────────────────────────────────
def clear_output_dir(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in CLASS_LABELS],
        columns=[f"Pred_{c}" for c in CLASS_LABELS],
    )
    df.to_csv(path)


def save_model_pickle(model: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def write_model_selection_summary(
    path: Path,
    comparison: pd.DataFrame,
    best_name: str,
    benchmark: dict[str, float],
    n_train_rows: int,
    n_test_rows: int,
    n_train_orders: int,
    n_test_orders: int,
) -> None:
    best_row = comparison.loc[comparison["model"] == best_name].iloc[0]
    lines = [
        "DAP391m — Model Selection Summary (DataCo Late Delivery)",
        "=" * 72,
        "",
        f"Final dataset: {SOURCE_DATASET_NAME} (DataCo Smart Supply Chain)",
        f"Target: {TARGET_COL} (0 = on-time, 1 = late delivery)",
        "Problem: binary late-delivery risk prediction for retail procurement",
        "",
        "Split: GroupShuffleSplit on Order Id (order-item grain; ~2.75 items/order)",
        f"  Train: {n_train_rows:,} rows, {n_train_orders:,} unique orders",
        f"  Test:  {n_test_rows:,} rows, {n_test_orders:,} unique orders",
        "",
        "Leakage / unsafe columns removed before modeling:",
        "  Post-outcome: real shipping days, delivery status, shipping date,",
        "  order status, realized profit fields.",
        "  PII / geo / media: customer identifiers, passwords, lat/long, etc.",
        f"  {GROUP_KEY} used only for grouping, not as a model feature.",
        "",
        "Feature engineering:",
        "  Parsed order date → order_month, order_day_of_week, order_hour.",
        "  Kept Shipping Mode and Days for shipment (scheduled).",
        "  Note: Shipping Mode and scheduled days are perfectly collinear in DataCo;",
        "  a per-mode base-rate benchmark is reported below.",
        "",
        "Models trained:",
        "  • Logistic Regression — baseline model",
        "  • Decision Tree — simple baseline",
        "  • Random Forest — ensemble comparison model",
        "  • XGBoost — primary model (deployment, SHAP, report)",
        "",
        "Primary metric: PR-AUC (average precision)",
        "",
        "Shipping-mode base-rate benchmark (test set, train-set mode late rates):",
        (
            f"  ROC-AUC: {benchmark['roc_auc']:.4f}"
            if not np.isnan(benchmark["roc_auc"])
            else "  ROC-AUC: n/a"
        ),
        (
            f"  PR-AUC:  {benchmark['pr_auc']:.4f}"
            if not np.isnan(benchmark["pr_auc"])
            else "  PR-AUC:  n/a"
        ),
        "",
        "Model comparison (test set):",
        comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        f"Best model by PR-AUC: {best_name}",
        f"  PR-AUC: {best_row['pr_auc']:.4f}",
        "",
        "Saved artifacts:",
        "  • best_model.pkl — highest PR-AUC pipeline",
        "  • primary_model.pkl — XGBoost pipeline (official primary)",
        "  • xgb_model.pkl — XGBoost pipeline (explainability / deployment)",
        "  • predictions_test.csv — XGBoost test predictions",
        "  • late_rate_by_shipping_mode.csv / shipping_mode_model_performance.csv",
        "",
        "Shipping Mode analysis:",
        "  Operationally important and strongly related to delivery timeliness in",
        "  DataCo; per-mode metrics and base-rate benchmark are included.",
        "",
        "Limitation:",
        "  DataCo is a simulated teaching dataset; disclose limitations in the report.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        n_in = pipeline.named_steps["model"].n_features_in_
        return [f"feature_{i}" for i in range(n_in)]


# ── plots ────────────────────────────────────────────────────────────────────
def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    plot_df = comparison[["model"] + metrics].set_index("model")
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df.plot(kind="bar", ax=ax, rot=20)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Late_delivery_risk (Test Set)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    pipeline: Pipeline, title: str, path: Path, top_n: int = 20
) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    names = get_feature_names(pipeline)
    values = model.feature_importances_
    order = np.argsort(values)[::-1][:top_n]
    top_names = [names[i] for i in order]
    top_values = values[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names[::-1], top_values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_late_rate_by_shipping_mode(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary[SHIPPING_MODE_COL], summary["late_rate"] * 100.0)
    ax.set_ylabel("Late rate (%)")
    ax.set_xlabel(SHIPPING_MODE_COL)
    ax.set_title("Late Delivery Rate by Shipping Mode")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_shap_summary(
    pipeline: Pipeline, X_test: pd.DataFrame, path: Path, max_samples: int = 500
) -> None:
    if not HAS_SHAP:
        warnings.warn("SHAP not installed — skipping SHAP summary plot.", stacklevel=2)
        return
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocess"]
        feature_names = get_feature_names(pipeline)

        sample = X_test
        if len(sample) > max_samples:
            sample = sample.sample(max_samples, random_state=RANDOM_STATE)
        X_sample = preprocessor.transform(sample)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary saved to {path.relative_to(PROJECT_ROOT)}")
    except Exception as exc:
        warnings.warn(f"SHAP failed ({exc}) — continuing without SHAP plot.")


def save_xgboost_predictions(
    order_ids_test: pd.Series,
    shipping_test: pd.Series,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray,
    path: Path,
) -> None:
    pred_df = pd.DataFrame(
        {
            GROUP_KEY: order_ids_test.values,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba_late": proba_pos,
            "model_name": "XGBoost",
        }
    )
    if len(shipping_test):
        pred_df[SHIPPING_MODE_COL] = shipping_test.values
    pred_df.to_csv(path, index=False)


def train_and_evaluate(
    output_dir: Path,
    num_cols: list[str],
    cat_cols: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    order_ids_test: pd.Series,
    shipping_test: pd.Series,
    train_modes: pd.Series,
    late_rate_summary: pd.DataFrame,
    n_train_rows: int,
    n_test_rows: int,
    n_train_orders: int,
    n_test_orders: int,
) -> pd.DataFrame:
    trained: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    print("Training binary models...")
    for spec in MODEL_SPECS:
        display = spec["display"]
        print(f"  • {display}")

        if spec.get("optional") and spec["key"] == "xgboost" and not HAS_XGBOOST:
            warnings.warn("XGBoost not installed — skipping XGBoost.", stacklevel=2)
            continue

        try:
            pipeline = make_model_pipeline(
                num_cols, cat_cols, spec["scaled"], spec["key"]
            )
            pipeline.fit(X_train, y_train)
        except Exception as exc:
            warnings.warn(
                f"{display} training failed ({exc}) — skipping.", stacklevel=2
            )
            continue

        proba_pos = positive_proba(pipeline, X_test)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_binary(y_test, y_pred, proba_pos, display)

        save_model_pickle(pipeline, output_dir / spec["pkl"])
        save_confusion_matrix(metrics["confusion_matrix"], output_dir / spec["cm"])
        (output_dir / spec["report"]).write_text(
            metrics["classification_report"], encoding="utf-8"
        )

        comparison_rows.append(
            {
                "model": display,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        trained[display] = {"spec": spec, "pipeline": pipeline, "metrics": metrics}

    if not trained:
        raise RuntimeError("No models were trained successfully.")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    best_name = select_best_model(comparison)
    save_model_pickle(trained[best_name]["pipeline"], output_dir / "best_model.pkl")

    benchmark = shipping_mode_base_rate_benchmark(
        train_modes, y_train, shipping_test, y_test
    )
    print(
        "\nShipping-mode base-rate benchmark (test): "
        f"ROC-AUC={benchmark['roc_auc']:.4f}, PR-AUC={benchmark['pr_auc']:.4f}"
    )

    late_rate_summary.to_csv(output_dir / "late_rate_by_shipping_mode.csv", index=False)

    if "XGBoost" in trained:
        xgb_pipeline = trained["XGBoost"]["pipeline"]
        xgb_metrics = trained["XGBoost"]["metrics"]
        save_model_pickle(xgb_pipeline, output_dir / "primary_model.pkl")
        save_model_pickle(xgb_pipeline, output_dir / "xgb_model.pkl")

        if xgb_metrics["proba_pos"] is not None:
            mode_perf = xgboost_performance_by_shipping_mode(
                shipping_test,
                y_test,
                xgb_metrics["y_pred"],
                xgb_metrics["proba_pos"],
            )
            mode_perf.to_csv(
                output_dir / "shipping_mode_model_performance.csv", index=False
            )
            save_xgboost_predictions(
                order_ids_test,
                shipping_test,
                y_test,
                xgb_metrics["y_pred"],
                xgb_metrics["proba_pos"],
                output_dir / "predictions_test.csv",
            )
    else:
        warnings.warn(
            "XGBoost not available — primary_model.pkl / predictions not created.",
            stacklevel=2,
        )
        pd.DataFrame().to_csv(
            output_dir / "shipping_mode_model_performance.csv", index=False
        )

    write_model_selection_summary(
        output_dir / "model_selection_summary.txt",
        comparison,
        best_name,
        benchmark,
        n_train_rows,
        n_test_rows,
        n_train_orders,
        n_test_orders,
    )

    print()
    print("Model comparison (test set):")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model (by PR-AUC): {best_name}")
    print("Primary model: XGBoost")
    print("Saved outputs to", output_dir.relative_to(PROJECT_ROOT))

    try:
        plot_model_comparison(comparison, output_dir / "model_comparison.png")
    except Exception as exc:
        warnings.warn(f"model_comparison plot failed ({exc}).", stacklevel=2)

    if "XGBoost" in trained:
        try:
            plot_confusion_matrix(
                trained["XGBoost"]["metrics"]["confusion_matrix"],
                "Confusion Matrix — XGBoost (Primary)",
                output_dir / "confusion_matrix_xgboost.png",
            )
        except Exception as exc:
            warnings.warn(f"confusion matrix plot failed ({exc}).", stacklevel=2)
        try:
            plot_feature_importance(
                trained["XGBoost"]["pipeline"],
                "Feature Importance — XGBoost (Primary)",
                output_dir / "feature_importance.png",
            )
        except Exception as exc:
            warnings.warn(f"feature importance plot failed ({exc}).", stacklevel=2)
        try:
            run_shap_summary(
                trained["XGBoost"]["pipeline"],
                X_test,
                output_dir / "shap_summary.png",
            )
        except Exception as exc:
            warnings.warn(f"SHAP step failed ({exc}).", stacklevel=2)

    try:
        plot_late_rate_by_shipping_mode(
            late_rate_summary, output_dir / "late_rate_by_shipping_mode.png"
        )
    except Exception as exc:
        warnings.warn(f"late_rate_by_shipping_mode plot failed ({exc}).", stacklevel=2)

    return comparison


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    X, y, groups, order_ids, shipping_modes, shipping_df = load_and_prepare_data()

    clear_output_dir(OUTPUT_DIR)

    late_rate_summary = compute_late_rate_by_shipping_mode(shipping_df)
    if not late_rate_summary.empty:
        print("Late rate by Shipping Mode:")
        print(
            late_rate_summary.to_string(
                index=False, float_format=lambda x: f"{x:.4f}" if x < 1 else f"{x:.0f}"
            )
        )
        print()

    num_cols, cat_cols = split_features(X)
    X_train, X_test, y_train, y_test, order_ids_test, shipping_test = (
        group_train_test_split(X, y, groups, order_ids, shipping_modes)
    )

    train_groups = groups.iloc[X_train.index]
    test_groups = groups.iloc[X_test.index]
    train_modes = shipping_modes.iloc[X_train.index]

    print(
        f"Train size: {len(X_train):,} rows ({train_groups.nunique():,} unique orders)"
    )
    print(f"Test size:  {len(X_test):,} rows ({test_groups.nunique():,} unique orders)")
    print()

    train_and_evaluate(
        OUTPUT_DIR,
        num_cols,
        cat_cols,
        X_train,
        X_test,
        y_train,
        y_test,
        order_ids_test,
        shipping_test,
        train_modes,
        late_rate_summary,
        len(X_train),
        len(X_test),
        int(train_groups.nunique()),
        int(test_groups.nunique()),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
