"""
DAP391m Project 8 — Binary Credit-Default Risk Modeling
========================================================

Trains four classifiers on credit_risk_dataset.csv (target: loan_status,
1 = default), evaluates binary metrics (PR-AUC primary), tunes the decision
threshold, and persists artifacts for reporting, explainability, and Streamlit
deployment (XGBoost primary).

The main output directory keeps the default-threshold baseline. An
imbalance-weighted experiment (class weighting / scale_pos_weight) is saved
separately for report discussion.

Run:
    .venv/bin/python3 src-code/05_modeling.py
    (reads Data/filtered/clean_data.csv; run 01_ingestion_cleaning.py first)
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
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
DATA_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

TARGET_COL = "loan_status"
CLASS_LABELS = ["Non-default", "Default"]  # index == class int (0, 1)
POSITIVE_CLASS = 1

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

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "key": "baseline",
        "display": "Baseline PR-AUC",
        "output_dir": OUTPUT_DIR,
        "weighted": False,
    },
    {
        "key": "imbalance_weighted",
        "display": "Imbalance-weighted",
        "output_dir": OUTPUT_DIR / "imbalance_weighted",
        "weighted": True,
    },
]


# ── data loading & feature prep ─────────────────────────────────────────────
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_CSV}. Run 01_ingestion_cleaning.py first."
        )

    df = pd.read_csv(DATA_CSV)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {DATA_CSV.name}")

    print(f"Dataset shape: {df.shape}")
    counts = df[TARGET_COL].value_counts().sort_index()
    print("Class distribution (loan_status):")
    print(counts.to_string())
    print(f"Default rate: {100.0 * counts.get(1, 0) / len(df):.1f}%")
    print()

    df = add_engineered_features(df)
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decision-time engineered features. Kept in sync with
    src-code/04_feature_engineering.py and app/app.py."""
    out = df.copy()
    out["high_debt_burden"] = (out["loan_percent_income"] > 0.30).astype(int)
    out["credit_hist_ratio"] = (
        out["cb_person_cred_hist_length"] / out["person_age"]
    ).round(4)
    out["interest_to_income"] = (
        ((out["loan_int_rate"] / 100.0) * out["loan_amnt"]) / out["person_income"]
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


def split_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
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
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(key: str, weighted: bool, scale_pos_weight: float) -> Any:
    if key == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced" if weighted else None,
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    if key == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced" if weighted else None,
            max_depth=8,
            random_state=RANDOM_STATE,
        )
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced" if weighted else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if key == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight if weighted else 1.0,
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
    weighted: bool,
    scale_pos_weight: float,
) -> Pipeline:
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric)
    estimator = build_estimator(key, weighted, scale_pos_weight)
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


def tune_threshold(y_true: np.ndarray, proba_pos: np.ndarray) -> tuple[float, float]:
    """Pick the probability threshold that maximizes F1 for the positive class."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba_pos)
    # precision/recall have len(thresholds)+1; align by dropping the last point.
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def select_best_model(comparison: pd.DataFrame) -> str:
    ranked = comparison.sort_values(
        by=["pr_auc", "recall", "f1"],
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
    experiment_name: str,
    is_primary_output: bool,
    tuned_threshold: float | None,
) -> None:
    best_row = comparison.loc[comparison["model"] == best_name].iloc[0]
    xgb_role = (
        "official deployment + Streamlit model"
        if is_primary_output
        else "experimental comparison model"
    )
    lines = [
        "DAP391m — Model Selection Summary (Binary Credit-Default Classification)",
        "=" * 72,
        "",
        f"Experiment: {experiment_name}",
        "Dataset: Data/credit_risk_dataset.csv (cleaned)",
        "Target: loan_status (0 = Non-default, 1 = Default)",
        "Problem: binary classification (positive class = Default)",
        "",
        "Models trained:",
        "  • Logistic Regression — baseline model",
        "  • Decision Tree — simple baseline",
        "  • Random Forest — ensemble comparison model",
        f"  • XGBoost — {xgb_role}",
        "",
        "Primary metric: PR-AUC (average precision, imbalance-aware)",
        "",
        "Model comparison (test set):",
        comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        f"Best model by PR-AUC: {best_name}",
        f"  PR-AUC: {best_row['pr_auc']:.4f}",
        "",
    ]
    if tuned_threshold is not None:
        lines += [
            f"XGBoost F1-optimal threshold (tuned on train): {tuned_threshold:.3f}",
            "(default 0.5 used for the comparison table; tuned threshold reported",
            "in threshold_analysis.txt for deployment decisions.)",
            "",
        ]
    lines += [
        "Saved artifacts:",
        "  • best_model.pkl — highest PR-AUC pipeline",
        f"  • primary_model.pkl — XGBoost pipeline ({xgb_role})",
        "  • xgb_model.pkl — XGBoost pipeline (explainability / Streamlit)",
        "  • predictions_test.csv — XGBoost test predictions + default probability",
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
    metrics = ["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]
    plot_df = comparison[["model"] + metrics].set_index("model")
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df.plot(kind="bar", ax=ax, rot=20)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — loan_status (Test Set)")
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
    y_test: np.ndarray, y_pred: np.ndarray, proba_pos: np.ndarray, path: Path
) -> None:
    pred_df = pd.DataFrame(
        {
            "y_true": [CLASS_LABELS[i] for i in y_test],
            "y_pred": [CLASS_LABELS[i] for i in y_pred],
            "model_name": "XGBoost",
            "proba_default": proba_pos,
        }
    )
    pred_df.to_csv(path, index=False)


def write_threshold_analysis(
    path: Path,
    y_test: np.ndarray,
    proba_pos: np.ndarray,
    tuned_threshold: float,
) -> None:
    rows = []
    for thr in [0.3, 0.4, 0.5, tuned_threshold, 0.6, 0.7]:
        pred = (proba_pos >= thr).astype(int)
        rows.append(
            {
                "threshold": round(thr, 3),
                "precision": round(precision_score(y_test, pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, pred, zero_division=0), 4),
            }
        )
    table = (
        pd.DataFrame(rows).drop_duplicates(subset="threshold").sort_values("threshold")
    )
    lines = [
        "XGBoost Threshold Analysis (test set)",
        "=" * 40,
        f"F1-optimal threshold (tuned on train): {tuned_threshold:.3f}",
        "",
        table.to_string(index=False),
        "",
        "Lower thresholds catch more defaults (higher recall) at the cost of",
        "precision — a business trade-off for credit approval policy.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    experiment: dict[str, Any],
    num_cols: list[str],
    cat_cols: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_arr: np.ndarray,
    y_test_arr: np.ndarray,
    scale_pos_weight: float,
) -> pd.DataFrame:
    output_dir = Path(experiment["output_dir"])
    clear_output_dir(output_dir)
    trained: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    weighted = bool(experiment["weighted"])

    print(f"Training binary models ({experiment['display']})...")
    for spec in MODEL_SPECS:
        display = spec["display"]
        print(f"  • {display}")

        if spec.get("optional") and spec["key"] == "xgboost" and not HAS_XGBOOST:
            warnings.warn("XGBoost not installed — skipping XGBoost.", stacklevel=2)
            continue

        try:
            pipeline = make_model_pipeline(
                num_cols,
                cat_cols,
                spec["scaled"],
                spec["key"],
                weighted,
                scale_pos_weight,
            )
            pipeline.fit(X_train, y_train_arr)
        except Exception as exc:
            warnings.warn(
                f"{display} training failed ({exc}) — skipping.", stacklevel=2
            )
            continue

        proba_pos = positive_proba(pipeline, X_test)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_binary(y_test_arr, y_pred, proba_pos, display)

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
        print("ERROR: No models were trained successfully.")
        sys.exit(1)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    best_name = select_best_model(comparison)
    save_model_pickle(trained[best_name]["pipeline"], output_dir / "best_model.pkl")

    tuned_threshold: float | None = None
    if "XGBoost" in trained:
        xgb_pipeline = trained["XGBoost"]["pipeline"]
        xgb_metrics = trained["XGBoost"]["metrics"]
        save_model_pickle(xgb_pipeline, output_dir / "primary_model.pkl")
        if xgb_metrics["proba_pos"] is not None:
            save_xgboost_predictions(
                y_test_arr,
                xgb_metrics["y_pred"],
                xgb_metrics["proba_pos"],
                output_dir / "predictions_test.csv",
            )
            train_proba = positive_proba(xgb_pipeline, X_train)
            tuned_threshold, _ = tune_threshold(y_train_arr, train_proba)
            write_threshold_analysis(
                output_dir / "threshold_analysis.txt",
                y_test_arr,
                xgb_metrics["proba_pos"],
                tuned_threshold,
            )
    else:
        warnings.warn(
            "XGBoost not available — primary_model.pkl / predictions not created.",
            stacklevel=2,
        )

    write_model_selection_summary(
        output_dir / "model_selection_summary.txt",
        comparison,
        best_name,
        str(experiment["display"]),
        output_dir == OUTPUT_DIR,
        tuned_threshold,
    )

    print()
    print("Model comparison (test set):")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model (by PR-AUC): {best_name}")
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

    comparison.insert(0, "experiment", experiment["display"])
    return comparison


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    X, y = load_and_prepare_data()
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

    n_pos = int((y_train_arr == POSITIVE_CLASS).sum())
    n_neg = int((y_train_arr != POSITIVE_CLASS).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    combined: list[pd.DataFrame] = []
    for experiment in EXPERIMENTS:
        combined.append(
            run_experiment(
                experiment,
                num_cols,
                cat_cols,
                X_train,
                X_test,
                y_train_arr,
                y_test_arr,
                scale_pos_weight,
            )
        )

    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "model_comparison_all_experiments.csv", index=False)
    print("\nCombined experiment comparison saved to")
    print(
        (OUTPUT_DIR / "model_comparison_all_experiments.csv").relative_to(PROJECT_ROOT)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
