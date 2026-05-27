"""
DAP391m Project 8 — 3-Class Supplier Lead-Time Risk Modeling
=============================================================

Trains four classifiers on supply_chain_risk_dataset.csv (target: risk_label),
evaluates multiclass metrics (Macro PR-AUC primary), and persists artifacts
for reporting, explainability, and Streamlit deployment (XGBoost primary).

Run:
    .venv/bin/python3 src-code/05_modeling.py
"""

from __future__ import annotations

import inspect
import os
import pickle
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    OneHotEncoder,
    StandardScaler,
    label_binarize,
)
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
DATA_CSV = PROJECT_ROOT / "Data" / "supply_chain_risk_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

TARGET_COL = "risk_label"
CLASS_LABELS = ["Low", "Medium", "High"]
LABEL_TO_INT = {label: i for i, label in enumerate(CLASS_LABELS)}
INT_TO_LABEL = {i: label for label, i in LABEL_TO_INT.items()}

LEAKAGE_COLS = ["risk_probability", "port_delay_days"]
DROP_FEATURE_COLS = ["machine_id"]

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "key": "logistic_regression",
        "display": "Logistic Regression",
        "role": "baseline",
        "pkl": "logistic_model.pkl",
        "scaled": True,
        "report": "classification_report_logistic_regression.txt",
        "cm": "confusion_matrix_logistic_regression.csv",
    },
    {
        "key": "decision_tree",
        "display": "Decision Tree",
        "role": "simple baseline",
        "pkl": "decision_tree_model.pkl",
        "scaled": False,
        "report": "classification_report_decision_tree.txt",
        "cm": "confusion_matrix_decision_tree.csv",
    },
    {
        "key": "random_forest",
        "display": "Random Forest",
        "role": "ensemble comparison",
        "pkl": "random_forest_model.pkl",
        "scaled": False,
        "report": "classification_report_random_forest.txt",
        "cm": "confusion_matrix_random_forest.csv",
    },
    {
        "key": "xgboost",
        "display": "XGBoost",
        "role": "primary model",
        "pkl": "xgb_model.pkl",
        "scaled": False,
        "report": "classification_report_xgboost.txt",
        "cm": "confusion_matrix_xgboost.csv",
        "optional": True,
    },
]


# ── data loading & feature prep ─────────────────────────────────────────────
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_CSV.name}")

    print(f"Dataset shape: {df.shape}")
    print("Class distribution (risk_label):")
    print(df[TARGET_COL].value_counts().to_string())
    print()

    for col in LEAKAGE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in DROP_FEATURE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["order_month"] = df["timestamp"].dt.month
        df["order_day_of_week"] = df["timestamp"].dt.dayofweek
        df["order_hour"] = df["timestamp"].dt.hour
        df = df.drop(columns=["timestamp"])

    y_raw = df[TARGET_COL].astype(str).str.strip()
    unknown = set(y_raw.unique()) - set(CLASS_LABELS)
    if unknown:
        raise ValueError(f"Unexpected risk_label values: {unknown}")

    y = y_raw.map(LABEL_TO_INT)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def split_features(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    cat_cols = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


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
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
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
        params: dict[str, Any] = {
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        }
        if "multi_class" in inspect.signature(LogisticRegression).parameters:
            params["multi_class"] = "ovr"
        return LogisticRegression(**params)
    if key == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if key == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_LABELS),
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
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
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    display_name: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model": display_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2]
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2]
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2]
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=[0, 1, 2]
        ),
        "macro_pr_auc": np.nan,
        "roc_auc_ovr_macro": np.nan,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASS_LABELS,
            zero_division=0,
        ),
    }

    if y_proba is not None and y_proba.ndim == 2:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        metrics["macro_pr_auc"] = average_precision_score(
            y_bin, y_proba, average="macro"
        )
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=[0, 1, 2],
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = np.nan

    return metrics


def select_best_model(comparison: pd.DataFrame) -> str:
    ranked = comparison.sort_values(
        by=["macro_pr_auc", "macro_recall", "macro_f1"],
        ascending=[False, False, False],
        na_position="last",
    )
    return str(ranked.iloc[0]["model"])


# ── persistence ──────────────────────────────────────────────────────────────
def clear_output_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


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
) -> None:
    best_row = comparison.loc[comparison["model"] == best_name].iloc[0]
    lines = [
        "DAP391m — Model Selection Summary (3-Class Risk Classification)",
        "=" * 68,
        "",
        "Dataset: Data/supply_chain_risk_dataset.csv",
        "Target: risk_label (Low, Medium, High)",
        "Problem: 3-class multiclass classification",
        "",
        "Models trained:",
        "  • Logistic Regression — baseline model",
        "  • Decision Tree — simple baseline",
        "  • Random Forest — ensemble comparison model",
        "  • XGBoost — PRIMARY model (deployment, SHAP, final report)",
        "",
        "Primary metric: Macro PR-AUC (imbalance-aware multiclass evaluation)",
        "",
        "Model comparison (test set):",
        comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        f"Best model by Macro PR-AUC: {best_name}",
        f"  Macro PR-AUC: {best_row['macro_pr_auc']:.4f}",
        "",
        "Saved artifacts:",
        "  • best_model.pkl — highest Macro PR-AUC pipeline",
        "  • primary_model.pkl — XGBoost pipeline (official deployment model)",
        "  • xgb_model.pkl — XGBoost pipeline (explainability / Streamlit)",
        "  • predictions_test.csv — XGBoost test-set predictions + probabilities",
        "",
        "Threshold tuning is NOT used as core model selection for this",
        "3-class pipeline (no single binary threshold applies to Low/Medium/High).",
        "",
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
    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "macro_pr_auc",
        "roc_auc_ovr_macro",
    ]
    plot_df = comparison[["model"] + metrics].set_index("model")
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_df.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — 3-Class risk_label (Test Set)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=7)
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
    pipeline: Pipeline,
    title: str,
    path: Path,
    top_n: int = 20,
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


def run_shap_summary(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    path: Path,
    max_samples: int = 300,
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
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    path: Path,
) -> None:
    pred_df = pd.DataFrame(
        {
            "y_true": [INT_TO_LABEL[i] for i in y_test],
            "y_pred": [INT_TO_LABEL[i] for i in y_pred],
            "model_name": "XGBoost",
        }
    )
    for idx, label in enumerate(CLASS_LABELS):
        pred_df[f"proba_{label}"] = y_proba[:, idx]
    pred_df.to_csv(path, index=False)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    X, y = load_and_prepare_data()
    clear_output_dir(OUTPUT_DIR)
    num_cols, cat_cols = split_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    trained: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    print("Training 3-class models...")
    for spec in MODEL_SPECS:
        display = spec["display"]
        print(f"  • {display}")

        if spec.get("optional") and spec["key"] == "xgboost":
            if not HAS_XGBOOST:
                warnings.warn("XGBoost not installed — skipping XGBoost.", stacklevel=2)
                continue
            try:
                pipeline = make_model_pipeline(
                    num_cols, cat_cols, spec["scaled"], spec["key"]
                )
                pipeline.fit(X_train, y_train_arr)
            except Exception as exc:
                warnings.warn(
                    f"XGBoost training failed ({exc}) — continuing.", stacklevel=2
                )
                continue
        else:
            pipeline = make_model_pipeline(
                num_cols, cat_cols, spec["scaled"], spec["key"]
            )
            pipeline.fit(X_train, y_train_arr)

        y_pred = pipeline.predict(X_test)
        y_proba = (
            pipeline.predict_proba(X_test)
            if hasattr(pipeline, "predict_proba")
            else None
        )

        metrics = evaluate_multiclass(y_test_arr, y_pred, y_proba, display)

        save_model_pickle(pipeline, OUTPUT_DIR / spec["pkl"])
        save_confusion_matrix(metrics["confusion_matrix"], OUTPUT_DIR / spec["cm"])
        (OUTPUT_DIR / spec["report"]).write_text(
            metrics["classification_report"], encoding="utf-8"
        )

        comparison_rows.append(
            {
                "model": display,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_pr_auc": metrics["macro_pr_auc"],
                "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
            }
        )

        trained[display] = {
            "spec": spec,
            "pipeline": pipeline,
            "metrics": metrics,
        }

    if not trained:
        print("ERROR: No models were trained successfully.")
        sys.exit(1)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_name = select_best_model(comparison)
    best_pipeline = trained[best_name]["pipeline"]
    save_model_pickle(best_pipeline, OUTPUT_DIR / "best_model.pkl")

    if "XGBoost" in trained:
        xgb_pipeline = trained["XGBoost"]["pipeline"]
        xgb_metrics = trained["XGBoost"]["metrics"]
        save_model_pickle(xgb_pipeline, OUTPUT_DIR / "primary_model.pkl")
        if xgb_metrics["y_proba"] is not None:
            save_xgboost_predictions(
                y_test_arr,
                xgb_metrics["y_pred"],
                xgb_metrics["y_proba"],
                OUTPUT_DIR / "predictions_test.csv",
            )
    else:
        warnings.warn(
            "XGBoost not available — primary_model.pkl and predictions_test.csv "
            "not created.",
            stacklevel=2,
        )

    write_model_selection_summary(
        OUTPUT_DIR / "model_selection_summary.txt",
        comparison,
        best_name,
    )

    print()
    print("Model comparison (test set):")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Best model (by Macro PR-AUC): {best_name}")
    print("Primary model (deployment / SHAP / report): XGBoost")
    print()
    print("Saved outputs to", OUTPUT_DIR.relative_to(PROJECT_ROOT))

    try:
        plot_model_comparison(comparison, OUTPUT_DIR / "model_comparison.png")
    except Exception as exc:
        warnings.warn(f"model_comparison plot failed ({exc}).", stacklevel=2)

    if "XGBoost" in trained:
        xgb_cm = trained["XGBoost"]["metrics"]["confusion_matrix"]
        try:
            plot_confusion_matrix(
                xgb_cm,
                "Confusion Matrix — XGBoost (Primary)",
                OUTPUT_DIR / "confusion_matrix_xgboost.png",
            )
        except Exception as exc:
            warnings.warn(f"confusion matrix plot failed ({exc}).", stacklevel=2)

        try:
            plot_feature_importance(
                trained["XGBoost"]["pipeline"],
                "Feature Importance — XGBoost (Primary)",
                OUTPUT_DIR / "feature_importance.png",
            )
        except Exception as exc:
            warnings.warn(f"feature importance plot failed ({exc}).", stacklevel=2)

        try:
            run_shap_summary(
                trained["XGBoost"]["pipeline"],
                X_test,
                OUTPUT_DIR / "shap_summary.png",
            )
        except Exception as exc:
            warnings.warn(f"SHAP step failed ({exc}).", stacklevel=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
