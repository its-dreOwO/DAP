"""
DAP391m Project 8 — Binary Delay Classification Modeling
========================================================

Trains four classifiers on processed features, evaluates on the held-out
test set, selects the best model by PR-AUC, and persists artifacts for
downstream reporting and the Streamlit app.

Run (after feature engineering):
    .venv/bin/python3 src-code/05_modeling.py
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

REQUIRED_FILES = [
    "X_train_scaled.csv",
    "X_test_scaled.csv",
    "X_train_tree.csv",
    "X_test_tree.csv",
    "y_train.csv",
    "y_test.csv",
]

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "key": "logistic_regression",
        "display": "Logistic Regression",
        "pkl": "logistic_model.pkl",
        "feature_set": "scaled",
        "report": "classification_report_logistic_regression.txt",
        "cm": "confusion_matrix_logistic_regression.csv",
    },
    {
        "key": "decision_tree",
        "display": "Decision Tree",
        "pkl": "decision_tree_model.pkl",
        "feature_set": "tree",
        "report": "classification_report_decision_tree.txt",
        "cm": "confusion_matrix_decision_tree.csv",
    },
    {
        "key": "random_forest",
        "display": "Random Forest",
        "pkl": "random_forest_model.pkl",
        "feature_set": "tree",
        "report": "classification_report_random_forest.txt",
        "cm": "confusion_matrix_random_forest.csv",
    },
    {
        "key": "xgboost",
        "display": "XGBoost",
        "pkl": "xgb_model.pkl",
        "feature_set": "tree",
        "report": "classification_report_xgboost.txt",
        "cm": "confusion_matrix_xgboost.csv",
        "optional": True,
    },
]


# ── data loading ─────────────────────────────────────────────────────────────
def _missing_files(directory: Path, names: list[str]) -> list[str]:
    return [n for n in names if not (directory / n).exists()]


def load_metadata() -> dict[str, Any]:
    """Load metadata.json from processed/ or processed/artifacts/."""
    candidates = [
        PROCESSED_DIR / "metadata.json",
        PROCESSED_DIR / "artifacts" / "metadata.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _read_target(path: Path) -> np.ndarray:
    series = pd.read_csv(path).squeeze("columns")
    return series.astype(int).to_numpy()


def load_datasets() -> dict[str, Any]:
    missing = _missing_files(PROCESSED_DIR, REQUIRED_FILES)
    if missing:
        print("ERROR: Missing required processed files:")
        for name in missing:
            print(f"  - {PROCESSED_DIR / name}")
        print()
        print("Run feature engineering first:")
        print("  python src-code/04_feature_engineering.py")
        sys.exit(1)

    data: dict[str, Any] = {
        "X_train_scaled": pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv"),
        "X_test_scaled": pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv"),
        "X_train_tree": pd.read_csv(PROCESSED_DIR / "X_train_tree.csv"),
        "X_test_tree": pd.read_csv(PROCESSED_DIR / "X_test_tree.csv"),
        "y_train": _read_target(PROCESSED_DIR / "y_train.csv"),
        "y_test": _read_target(PROCESSED_DIR / "y_test.csv"),
        "metadata": load_metadata(),
    }
    return data


def feature_matrices(
    data: dict[str, Any], feature_set: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if feature_set == "scaled":
        return data["X_train_scaled"], data["X_test_scaled"]
    if feature_set == "tree":
        return data["X_train_tree"], data["X_test_tree"]
    raise ValueError(f"Unknown feature_set: {feature_set}")


# ── training ─────────────────────────────────────────────────────────────────
def build_logistic() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


def build_decision_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
    )


def train_model(
    spec: dict[str, Any],
    data: dict[str, Any],
    scale_pos_weight: float,
) -> Any | None:
    key = spec["key"]
    X_train, X_test = feature_matrices(data, spec["feature_set"])
    y_train = data["y_train"]

    if key == "logistic_regression":
        model = build_logistic()
    elif key == "decision_tree":
        model = build_decision_tree()
    elif key == "random_forest":
        model = build_random_forest()
    elif key == "xgboost":
        if not HAS_XGBOOST:
            warnings.warn("XGBoost not installed — skipping XGBoost.", stacklevel=2)
            return None
        model = build_xgboost(scale_pos_weight)
    else:
        raise ValueError(f"Unknown model key: {key}")

    model.fit(X_train, y_train)
    return model


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    display_name: str,
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics: dict[str, Any] = {
        "model": display_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "y_pred": y_pred,
        "y_proba": None,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["y_proba"] = y_proba
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["pr_auc"] = average_precision_score(y_test, y_proba)

    return metrics


def select_best_model(comparison: pd.DataFrame) -> str:
    ranked = comparison.sort_values(
        by=["pr_auc", "recall", "f1"],
        ascending=[False, False, False],
        na_position="last",
    )
    return str(ranked.iloc[0]["model"])


# ── persistence ────────────────────────────────────────────────────────────────
def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Pred_0", "Pred_1"],
    )
    df.to_csv(path)


def save_classification_report_text(report: str, path: Path) -> None:
    path.write_text(report, encoding="utf-8")


def save_model_pickle(model: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    plot_df = comparison[["model"] + metrics].set_index("model")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Test Set)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["On-Time", "Delayed"],
        yticklabels=["On-Time", "Delayed"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_precision_recall(
    y_test: np.ndarray,
    probas: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (Test Set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(
    y_test: np.ndarray,
    probas: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    title: str,
    path: Path,
) -> None:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        return

    order = np.argsort(values)[::-1][:20]
    top_names = [feature_names[i] for i in order]
    top_values = values[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_names[::-1], top_values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_shap_summary(
    model: Any,
    X_test: pd.DataFrame,
    path: Path,
    max_samples: int = 300,
) -> None:
    if not HAS_SHAP:
        warnings.warn("SHAP not installed — skipping SHAP summary plot.", stacklevel=2)
        return

    try:
        sample = X_test
        if len(sample) > max_samples:
            sample = sample.sample(max_samples, random_state=RANDOM_STATE)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary saved to {path.relative_to(PROJECT_ROOT)}")
    except Exception as exc:
        warnings.warn(f"SHAP failed ({exc}) — continuing without SHAP plot.")


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    global RANDOM_STATE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_datasets()
    metadata = data["metadata"]

    if metadata.get("random_state") is not None:
        RANDOM_STATE = int(metadata["random_state"])

    scale_pos_weight = float(metadata.get("scale_pos_weight", 1.0))
    y_test = data["y_test"]

    trained: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    probas_for_curves: dict[str, np.ndarray] = {}

    print("Training models on processed data...")
    for spec in MODEL_SPECS:
        display = spec["display"]
        print(f"  • {display}")

        if spec.get("optional") and spec["key"] == "xgboost" and not HAS_XGBOOST:
            continue

        model = train_model(spec, data, scale_pos_weight)
        if model is None:
            continue

        _, X_test = feature_matrices(data, spec["feature_set"])
        metrics = evaluate_model(model, X_test, y_test, display)

        save_model_pickle(model, OUTPUT_DIR / spec["pkl"])
        save_confusion_matrix(metrics["confusion_matrix"], OUTPUT_DIR / spec["cm"])
        save_classification_report_text(
            metrics["classification_report"], OUTPUT_DIR / spec["report"]
        )

        comparison_rows.append(
            {
                "model": display,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

        if metrics["y_proba"] is not None:
            probas_for_curves[display] = metrics["y_proba"]

        trained[display] = {
            "spec": spec,
            "model": model,
            "metrics": metrics,
            "X_test": X_test,
        }

    if not trained:
        print("ERROR: No models were trained successfully.")
        sys.exit(1)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_name = select_best_model(comparison)
    best_entry = trained[best_name]
    best_model = best_entry["model"]
    best_metrics = best_entry["metrics"]

    save_model_pickle(best_model, OUTPUT_DIR / "best_model.pkl")
    print()
    print("Model comparison (test set):")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Best model (by PR-AUC): {best_name}")

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": best_metrics["y_pred"],
            "y_proba": best_metrics["y_proba"],
        }
    )
    for name, proba in probas_for_curves.items():
        col = name.lower().replace(" ", "_") + "_proba"
        pred_df[col] = proba
    pred_df.to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)

    plot_model_comparison(comparison, OUTPUT_DIR / "model_comparison.png")
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        f"Confusion Matrix — {best_name}",
        OUTPUT_DIR / "confusion_matrix_best_model.png",
    )
    if probas_for_curves:
        plot_precision_recall(y_test, probas_for_curves, OUTPUT_DIR / "precision_recall_curve.png")
        plot_roc_curves(y_test, probas_for_curves, OUTPUT_DIR / "roc_curve.png")

    best_spec = best_entry["spec"]
    _, X_test_best = feature_matrices(data, best_spec["feature_set"])
    plot_feature_importance(
        best_model,
        list(X_test_best.columns),
        f"Feature Importance — {best_name}",
        OUTPUT_DIR / "feature_importance.png",
    )

    if "XGBoost" in trained and HAS_SHAP:
        xgb_model = trained["XGBoost"]["model"]
        xgb_X_test = trained["XGBoost"]["X_test"]
        run_shap_summary(xgb_model, xgb_X_test, OUTPUT_DIR / "shap_summary.png")
    elif "XGBoost" not in trained:
        warnings.warn("XGBoost not trained — skipping SHAP.", stacklevel=2)
    elif not HAS_SHAP:
        warnings.warn("SHAP not installed — skipping SHAP.", stacklevel=2)

    print()
    print("Saved outputs to", OUTPUT_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
