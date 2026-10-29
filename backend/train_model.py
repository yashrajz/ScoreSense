"""Train and evaluate the two models using the existing CSV only."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .passing_rules import ETE_THRESHOLD, is_passing

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "lpu_student_performance.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RANDOM_SEED = 42
TEST_SIZE = 0.2
FEATURE_COLUMNS = [
    "study_hours_per_week",
    "attendance_percentage",
    "ca_marks_30",
    "midterm_marks_25",
]
REQUIRED_COLUMNS = [
    "student_id",
    *FEATURE_COLUMNS,
    "endterm_marks_45",
    "total_marks_100",
]
RANGES = {
    "study_hours_per_week": (0, 40),
    "attendance_percentage": (0, 100),
    "ca_marks_30": (0, 30),
    "midterm_marks_25": (0, 25),
    "endterm_marks_45": (0, 45),
    "total_marks_100": (0, 100),
}


def load_and_validate_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Required dataset was not found: {CSV_PATH}")
    data = pd.read_csv(CSV_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if data[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    for column in REQUIRED_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        low, high = RANGES[column]
        if not data[column].between(low, high).all():
            raise ValueError(f"Column '{column}' contains values outside {low}-{high}.")
    if data["student_id"].duplicated().any():
        raise ValueError("Duplicate student_id values were found.")
    computed = data["ca_marks_30"] + data["midterm_marks_25"] + data["endterm_marks_45"]
    inconsistent = ~np.isclose(computed, data["total_marks_100"], atol=1e-9)
    if inconsistent.any():
        rows = (data.index[inconsistent] + 2).tolist()
        raise ValueError(f"total_marks_100 disagrees with component sums on CSV rows: {rows}")
    return data


def _save_scatter(actual: pd.Series, predicted: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(actual, predicted, color="#e56832", alpha=0.7, edgecolors="white")
    axis.plot([0, 45], [0, 45], "--", color="#17324d", label="Perfect estimate")
    axis.set(xlabel="Actual ETE marks / 45", ylabel="Predicted ETE marks / 45")
    axis.set_title("ETE regression: held-out test data")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(figure)


def _save_confusion(matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks([0, 1], ["Fail", "Pass"])
    axis.set_yticks([0, 1], ["Fail", "Pass"])
    axis.set(xlabel="Predicted", ylabel="Actual", title="Pass/fail classifier")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def _save_feature_analysis(data: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    correlations = data[FEATURE_COLUMNS + ["endterm_marks_45"]].corr()["endterm_marks_45"]
    correlations.drop("endterm_marks_45").sort_values().plot.barh(ax=axis, color="#e56832")
    axis.set(xlabel="Pearson correlation", title="Feature correlation with ETE marks")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "feature_analysis.png", dpi=160)
    plt.close(figure)


def train_models(data: pd.DataFrame) -> dict:
    X = data[FEATURE_COLUMNS]
    regression_target = data["endterm_marks_45"]
    labels = np.where(
        [
            is_passing(total, ete)
            for total, ete in zip(data["total_marks_100"], data["endterm_marks_45"])
        ],
        "Pass",
        "Fail",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, regression_target, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    regression = LinearRegression().fit(X_train, y_train)
    predicted_ete = np.clip(regression.predict(X_test), 0, 45)
    _save_scatter(y_test, predicted_ete)

    classifier = None
    classification_metrics = {"skipped": False}
    class_counts = pd.Series(labels).value_counts()
    if len(class_counts) < 2:
        classification_metrics = {
            "skipped": True,
            "reason": "The dataset contains only one passing-status class.",
        }
    else:
        stratified = class_counts.min() >= 2
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels if stratified else None,
        )
        classifier = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)
        classifier.fit(Xc_train, yc_train)
        predicted_status = classifier.predict(Xc_test)
        matrix = confusion_matrix(yc_test, predicted_status, labels=["Fail", "Pass"])
        _save_confusion(matrix)
        classification_metrics = {
            "skipped": False,
            "stratified_split": bool(stratified),
            "positive_class": "Pass",
            "accuracy": float(accuracy_score(yc_test, predicted_status)),
            "precision": float(precision_score(yc_test, predicted_status, pos_label="Pass", zero_division=0)),
            "recall": float(recall_score(yc_test, predicted_status, pos_label="Pass", zero_division=0)),
            "f1_score": float(f1_score(yc_test, predicted_status, pos_label="Pass", zero_division=0)),
            "confusion_matrix": {"labels": ["Fail", "Pass"], "matrix": matrix.tolist()},
            "feature_importance": dict(zip(FEATURE_COLUMNS, classifier.feature_importances_.tolist())),
            "train_size": len(Xc_train),
            "test_size": len(Xc_test),
        }

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": regression, "features": FEATURE_COLUMNS, "model_name": "LinearRegression-ETE-v1"},
        MODELS_DIR / "ete_regression.joblib",
    )
    if classifier is not None:
        joblib.dump(
            {"model": classifier, "features": FEATURE_COLUMNS, "model_name": "DecisionTreeClassifier-v1"},
            MODELS_DIR / "pass_classifier.joblib",
        )
    _save_feature_analysis(data)
    results = {
        "dataset": {"source": CSV_PATH.name, "rows": len(data), "features": FEATURE_COLUMNS},
        "passing_rules": {"overall_threshold": 40, "ete_threshold": ETE_THRESHOLD, "maximum_ete": 45},
        "regression": {
            "model_name": "LinearRegression-ETE-v1",
            "target": "endterm_marks_45",
            "r2_score": float(r2_score(y_test, predicted_ete)),
            "mean_absolute_error": float(mean_absolute_error(y_test, predicted_ete)),
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "classification": classification_metrics,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    try:
        data = load_and_validate_dataset()
        results = train_models(data)
        print(json.dumps(results, indent=2))
        print("Training complete. Artifacts saved in models/ and results/.")
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
