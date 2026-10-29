"""Flask REST API for prediction serving."""

from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask import send_from_directory
from flask_cors import CORS

if __package__:
    from .database import recent_predictions, save_prediction
    from .passing_rules import minimum_ete_required, projected_outcome
    from .train_model import BASE_DIR, FEATURE_COLUMNS
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.database import recent_predictions, save_prediction
    from backend.passing_rules import minimum_ete_required, projected_outcome
    from backend.train_model import BASE_DIR, FEATURE_COLUMNS

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        }
    },
)
MODELS_DIR = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "results" / "evaluation_results.json"
INPUT_RANGES = {
    "study_hours_per_week": (0, 40),
    "attendance_percentage": (0, 100),
    "ca_marks_30": (0, 30),
    "midterm_marks_25": (0, 25),
}


def _load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python main.py` first.")
    artifact = joblib.load(path)
    if artifact.get("features") != FEATURE_COLUMNS:
        raise RuntimeError("Saved model features do not match the current API contract.")
    return artifact


def _validate_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    student_id = str(payload.get("student_id", "")).strip()
    if not student_id:
        raise ValueError("student_id is required.")
    values = {}
    for field, (low, high) in INPUT_RANGES.items():
        if field not in payload:
            raise ValueError(f"{field} is required.")
        try:
            value = float(payload[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be numeric.") from error
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{field} must be between {low} and {high}.")
        values[field] = value
    return {"student_id": student_id, **values}


@app.post("/api/predict")
def predict():
    try:
        raw_payload = request.get_json(silent=True)
        if not isinstance(raw_payload, dict):
            raise ValueError("Request body must be a JSON object.")
        payload = _validate_payload(raw_payload)
        regression = _load_artifact(MODELS_DIR / "ete_regression.joblib")
        classifier = _load_artifact(MODELS_DIR / "pass_classifier.joblib")
        row = pd.DataFrame([[payload[field] for field in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        predicted_ete = float(np.clip(regression["model"].predict(row)[0], 0, 45))
        outcome = projected_outcome(payload["ca_marks_30"], payload["midterm_marks_25"], predicted_ete)
        class_prediction = str(classifier["model"].predict(row)[0])
        probabilities = classifier["model"].predict_proba(row)[0]
        classes = classifier["model"].classes_
        probability = float(probabilities[list(classes).index(class_prediction)])
        calculator = minimum_ete_required(payload["ca_marks_30"], payload["midterm_marks_25"])
        response = {
            **payload,
            "predicted_ete_marks": predicted_ete,
            "regression_projected_status": outcome["projected_status"],
            "classifier_projected_status": outcome["projected_status"],
            "classifier_model_status": class_prediction,
            "classifier_probability": probability,
            **outcome,
            "calculator": calculator,
            "is_estimate": True,
        }
        try:
            save_prediction(response)
            response["database"] = {"saved": True}
        except Exception as error:
            response["database"] = {"saved": False, "message": str(error)}
        return jsonify(response)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Unexpected prediction failure")
        return jsonify({"error": "The prediction service encountered an unexpected error."}), 500


@app.get("/api/model-performance")
def model_performance():
    if not RESULTS_PATH.exists():
        return jsonify({"error": "Metrics are unavailable. Run `python main.py` first."}), 404
    try:
        return jsonify(json.loads(RESULTS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return jsonify({"error": "Saved evaluation metrics are invalid."}), 500


@app.get("/api/prediction-history")
def prediction_history():
    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400
    if not 1 <= limit <= 100:
        return jsonify({"error": "limit must be between 1 and 100."}), 400
    try:
        return jsonify({"predictions": recent_predictions(limit)})
    except Exception as error:
        return jsonify({"error": f"Prediction history is unavailable: {error}"}), 503


@app.get("/api/charts/<path:filename>")
def chart(filename):
    if filename not in {"actual_vs_predicted.png", "confusion_matrix.png", "feature_analysis.png"}:
        return jsonify({"error": "Chart not found."}), 404
    return send_from_directory(BASE_DIR / "results", filename)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "models_ready": (MODELS_DIR / "ete_regression.joblib").exists()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
