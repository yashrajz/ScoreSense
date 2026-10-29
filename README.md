# LPU Student Performance Prediction System

An academic analytics dashboard built with Python, Flask, scikit-learn, MongoDB/PyMongo, and React.

The application estimates ETE marks before the final examination. It is an educational project and does not issue official grades.

## Passing rules

A projected result is **Pass** only when both conditions are true:

1. Total marks are at least `40/100`.
2. ETE marks are at least `13.5/45` (a whole-number score therefore needs at least `14/45`).

The backend calculates:

```text
total_marks = CA + MTE + ETE
required_ete = max(13.5, 40 - CA - MTE)
```

If `CA + MTE + 45 < 40`, passing is mathematically impossible. A low model prediction alone is never treated as mathematical impossibility.

## How the project works

- [lpu_student_performance.csv](./lpu_student_performance.csv) is the existing input dataset. It is never generated, replaced, or modified by training.
- [backend/train_model.py](./backend/train_model.py) validates the CSV, creates labels with the two passing rules, trains both models, evaluates held-out data, saves model artifacts, and generates charts.
- [backend/passing_rules.py](./backend/passing_rules.py) contains the authoritative passing and minimum-ETE calculations.
- [backend/app.py](./backend/app.py) serves predictions and saved evaluation metrics through Flask.
- [backend/database.py](./backend/database.py) handles optional MongoDB writes and history reads.
- [frontend/src/main.jsx](./frontend/src/main.jsx) provides the React dashboard.
- [frontend/src/styles.css](./frontend/src/styles.css) contains the responsive visual system and accessible states.

## Models

### ETE regression

Linear Regression predicts `endterm_marks_45` from:

- study hours per week
- attendance percentage
- CA marks
- MTE marks

ETE and total marks are never used as input features. Predictions are bounded to `0–45`.

### Pass/fail classification

Decision Tree Classifier predicts a likely `Pass` or `Fail` using the same four pre-ETE features. The label is created from the deterministic passing rules. `Pass` is the positive class.

The API retains the classifier's raw historical signal as `classifier_model_status` for evaluation, but the user-facing `classifier_projected_status` is aligned with the regression-based projected outcome. This keeps every displayed result consistent with the authoritative passing rules: `total_marks >= 40` and `ETE_marks >= 13.5`.

## Installation

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

The original CSV must remain in the project root.

## Train the models

Training is explicit and does not run when Flask starts:

```powershell
python main.py
```

Successful training creates:

- `models/ete_regression.joblib`
- `models/pass_classifier.joblib`
- `results/evaluation_results.json`
- `results/actual_vs_predicted.png`
- `results/confusion_matrix.png`
- `results/feature_analysis.png`

If the CSV is missing, malformed, contains missing values, invalid ranges, duplicate IDs, or inconsistent totals, training stops with a clear error. It does not create replacement records.

## Start the backend

After training:

```powershell
python -m backend.app
```

The Flask API runs at `http://localhost:5000`.

Endpoints:

- `POST /api/predict`
- `GET /api/model-performance`
- `GET /api/prediction-history?limit=20`
- `GET /api/health`

## Configure MongoDB

Copy [.env.example](./.env.example) to `.env` and set your own URI:

```powershell
Copy-Item .env.example .env
```

For local MongoDB:

```text
MONGODB_URI=mongodb://localhost:27017
```

For MongoDB Atlas, use the URI from Atlas and URL-encode special characters in the password. The database is `lpu_student_performance_db`, with `students` and `predictions` collections. MongoDB is optional for model predictions; if unavailable, the API clearly reports that a prediction was not saved.

Never commit `.env` or expose the URI in React.

## Start the React frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local frontend URL printed by Vite, normally `http://localhost:5173`.

To use a different backend URL, create `frontend\.env`:

```text
VITE_API_URL=http://localhost:5000/api
```

## Understanding the dashboard

- **Predict performance** shows the ETE estimate, predicted total, minimum ETE requirement, mathematical possibility, and both model readouts.
- **Model performance** shows actual held-out R², MAE, accuracy, precision, recall, F1, and generated charts.
- **Prediction history** reads recent records from MongoDB. An empty state is shown when MongoDB is not configured.

R² describes explained variation. MAE is the average mark error. Accuracy is the share of correct classifications. Precision measures how often predicted Pass is correct. Recall measures how many actual Pass records were found. F1 balances precision and recall.

## Validation and tests

Run the focused Python tests:

```powershell
python -m unittest discover -s tests -v
```

The tests cover the central passing calculator, whole-number rounding, mathematical impossibility, exact thresholds, invalid API inputs, and feature selection.

Frontend production build:

```powershell
cd frontend
npm run build
```

## Limitations

- Predictions are estimates, not official or guaranteed grades.
- Model quality depends on the existing CSV and may change when that dataset changes.
- MongoDB integration requires a running local server or valid Atlas configuration.
- Browser-level visual testing requires the backend and frontend to be running.
