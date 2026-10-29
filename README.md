# LPU Student Performance Prediction System

An academic analytics application that estimates a student's End Term Examination
(ETE) marks from their current academic profile. The project combines:

- A React/Vite landing page and prediction workspace.
- A Flask REST API.
- A dataset-trained scikit-learn regression model.
- A dataset-trained scikit-learn pass/fail classifier.
- Authoritative passing-rule calculations shared by training and serving.
- Optional MongoDB persistence for submitted predictions.
- Evaluation charts and metrics generated from held-out test data.

This is an educational planning tool. It does not issue official university
grades, does not guarantee outcomes, and should not be used as an official
academic record.

---

## Table of contents

1. [What the application does](#what-the-application-does)
2. [Important project facts](#important-project-facts)
3. [Passing rules](#passing-rules)
4. [Application architecture](#application-architecture)
5. [Repository structure](#repository-structure)
6. [Dataset contract](#dataset-contract)
7. [Feature and target contract](#feature-and-target-contract)
8. [Machine-learning pipeline](#machine-learning-pipeline)
9. [Regression model](#regression-model)
10. [Classification model](#classification-model)
11. [Why model signals can differ](#why-model-signals-can-differ)
12. [Evaluation artifacts](#evaluation-artifacts)
13. [Frontend experience](#frontend-experience)
14. [Backend API](#backend-api)
15. [Prediction request and response](#prediction-request-and-response)
16. [Minimum ETE calculator](#minimum-ete-calculator)
17. [MongoDB integration](#mongodb-integration)
18. [Installation](#installation)
19. [Training the models](#training-the-models)
20. [Starting the Flask API](#starting-the-flask-api)
21. [Starting the React frontend](#starting-the-react-frontend)
22. [Environment configuration](#environment-configuration)
23. [Running the complete application](#running-the-complete-application)
24. [Testing and validation](#testing-and-validation)
25. [Example API calls](#example-api-calls)
26. [Frontend behavior](#frontend-behavior)
27. [Error handling](#error-handling)
28. [Troubleshooting](#troubleshooting)
29. [Security and privacy](#security-and-privacy)
30. [Reproducibility](#reproducibility)
31. [Known limitations](#known-limitations)
32. [Changing the project](#changing-the-project)
33. [Frequently asked questions](#frequently-asked-questions)

---

## What the application does

The application accepts five values:

1. Student ID.
2. Study hours per week.
3. Attendance percentage.
4. CA marks out of 30.
5. MTE marks out of 25.

It sends the four numeric academic features to the Flask API. The API:

1. Loads the saved ETE regression model.
2. Predicts ETE marks out of 45.
3. Calculates the predicted total out of 100.
4. Calculates the minimum ETE marks needed to satisfy both passing conditions.
5. Determines whether the predicted outcome passes.
6. Loads the saved classifier and records its raw historical signal.
7. Returns the rule-consistent user-facing status.
8. Attempts to save the prediction to MongoDB if configured.

The frontend then presents:

- Predicted ETE marks.
- Predicted total marks.
- Projected Pass/Fail status.
- Minimum ETE marks required.
- Difference between predicted and required ETE.
- Whether passing is mathematically possible.
- The two passing conditions.
- Model evaluation metrics and charts.

---

## Important project facts

### The CSV is an existing input dataset

The application uses the existing root-level file
[`lpu_student_performance.csv`](./lpu_student_performance.csv). Training validates
and reads this file; it does not generate a replacement CSV.

The current evaluation artifact records 800 rows. The project does not claim that
the rows represent official university records. Treat the dataset as project
data unless its provenance has been independently verified.

### Training is explicit

Starting Flask does not retrain the models. Training happens only when you run:

```powershell
python main.py
```

### Models are artifacts

Training writes model files into `models/`. The API loads those files when a
prediction is requested. If they are missing, the API returns an explicit error
asking you to run training.

### MongoDB is optional

Predictions do not require MongoDB. If `MONGODB_URI` is missing or MongoDB is
unavailable, the prediction response still returns the model result and reports
that persistence was not saved.

### Results are estimates

The frontend labels model output as an estimate. A prediction is not a guarantee,
an official grade, or a substitute for academic advice.

---

## Passing rules

The authoritative rules live in
[`backend/passing_rules.py`](./backend/passing_rules.py).

A projected result is **Pass** only if both conditions are true:

```text
total_marks >= 40
ETE_marks >= 13.5
```

The total is calculated as:

```text
total_marks = CA_marks + MTE_marks + predicted_ETE_marks
```

The minimum ETE requirement is:

```text
required_ETE = max(13.5, 40 - CA_marks - MTE_marks)
```

The rules use:

| Rule | Value |
|---|---:|
| Overall passing threshold | 40/100 |
| ETE passing threshold | 13.5/45 |
| Whole-number ETE minimum | 14/45 |
| Maximum possible ETE | 45/45 |

### Mathematical possibility

Passing is mathematically possible when:

```text
CA_marks + MTE_marks + 45 >= 40
```

If that expression is below 40, passing is impossible even with a perfect ETE.
A low model prediction by itself is **not** mathematical impossibility.

For example:

```text
CA = 3
MTE = 7
maximum possible total = 3 + 7 + 45 = 55
```

Passing remains mathematically possible. If the model predicts ETE `15.1`, the
projected total is approximately `25.1`, so the predicted outcome is Fail while
the mathematical possibility remains Yes.

---

## Application architecture

```text
lpu_student_performance.csv
          |
          v
backend/train_model.py
          |
          +--> models/ete_regression.joblib
          +--> models/pass_classifier.joblib
          +--> results/evaluation_results.json
          +--> results/actual_vs_predicted.png
          +--> results/confusion_matrix.png
          +--> results/feature_analysis.png
                         |
                         v
                    backend/app.py
                         |
                 Flask REST API /api/*
                         |
                         v
                    frontend/src
                         |
                 React/Vite browser UI
```

MongoDB is an optional side path used by the API:

```text
POST /api/predict
        |
        +--> calculate and return prediction
        |
        +--> attempt save_prediction()
                 |
                 +--> MongoDB predictions collection
```

---

## Repository structure

```text
.
├── .env.example
├── .gitignore
├── README.md
├── lpu_student_performance.csv
├── main.py
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── database.py
│   ├── passing_rules.py
│   ├── requirements.txt
│   └── train_model.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── src/
│       ├── assets/
│       │   └── hero-landscape.jpg
│       ├── main.jsx
│       └── styles.css
├── models/
│   ├── ete_regression.joblib
│   └── pass_classifier.joblib
├── results/
│   ├── actual_vs_predicted.png
│   ├── confusion_matrix.png
│   ├── evaluation_results.json
│   └── feature_analysis.png
└── tests/
    ├── test_passing_rules.py
    └── test_training_contract.py
```

Generated files such as `frontend/dist/`, Python bytecode caches, virtual
environments, and `frontend/node_modules/` are ignored by `.gitignore`.

---

## Dataset contract

The CSV must contain these columns:

```text
student_id
study_hours_per_week
attendance_percentage
ca_marks_30
midterm_marks_25
endterm_marks_45
total_marks_100
```

### Allowed ranges

| Column | Minimum | Maximum |
|---|---:|---:|
| `study_hours_per_week` | 0 | 40 |
| `attendance_percentage` | 0 | 100 |
| `ca_marks_30` | 0 | 30 |
| `midterm_marks_25` | 0 | 25 |
| `endterm_marks_45` | 0 | 45 |
| `total_marks_100` | 0 | 100 |

### Validation performed before training

[`load_and_validate_dataset()`](./backend/train_model.py) checks:

- The CSV exists.
- Every required column exists.
- Required cells are not missing.
- Numeric columns can be converted to numbers.
- Numeric values are within their configured ranges.
- `student_id` values are unique.
- `total_marks_100` equals CA + MTE + ETE.

Training stops with a clear error if any check fails. It does not silently repair
or replace the source data.

---

## Feature and target contract

Both saved models use exactly these four input features, in this order:

```python
FEATURE_COLUMNS = [
    "study_hours_per_week",
    "attendance_percentage",
    "ca_marks_30",
    "midterm_marks_25",
]
```

The regression target is:

```text
endterm_marks_45
```

The classifier target is a derived label:

```text
Pass if total_marks_100 >= 40 and endterm_marks_45 >= 13.5
Fail otherwise
```

The following are deliberately excluded from model input:

- `endterm_marks_45`
- `total_marks_100`
- Any future Pass/Fail label

This prevents target leakage. The API checks that the feature list saved inside
each joblib artifact matches the current feature contract before using it.

---

## Machine-learning pipeline

Training is implemented in
[`backend/train_model.py`](./backend/train_model.py).

The pipeline is:

1. Read the CSV.
2. Validate schema and value ranges.
3. Select the four pre-ETE features.
4. Split data into training and test sets.
5. Train a linear regression model for ETE.
6. Evaluate regression R² and MAE.
7. Create deterministic Pass/Fail labels from the authoritative rules.
8. Train a decision-tree classifier.
9. Evaluate classification metrics and confusion matrix.
10. Generate evaluation charts.
11. Save model artifacts.
12. Save `results/evaluation_results.json`.

The configured constants are:

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
```

For the current 800-row dataset, the expected split is:

- 640 training rows.
- 160 test rows.

The classifier uses a stratified split when both classes have enough examples.

---

## Regression model

The ETE model is:

```text
LinearRegression-ETE-v1
```

It predicts `endterm_marks_45` from:

- Study hours per week.
- Attendance percentage.
- CA marks.
- MTE marks.

The API clips the raw regression result to the valid ETE range:

```text
0 <= predicted_ete_marks <= 45
```

The current saved evaluation artifact records:

| Metric | Value |
|---|---:|
| R² | 0.6709464964944141 |
| MAE | 4.077324333205772 |
| Training rows | 640 |
| Test rows | 160 |

### Meaning of regression metrics

**R²** describes the proportion of target variation explained by the model.
Higher is generally better, but it does not mean that every individual
prediction is accurate.

**MAE** is the average absolute distance between the actual and predicted ETE
marks. An MAE of approximately 4.08 means the average absolute error is about
4 marks on the 45-mark ETE scale.

---

## Classification model

The classifier is:

```text
DecisionTreeClassifier-v1
```

It uses the same four pre-ETE features but is trained against historical labels
created from the actual dataset's ETE and total marks.

The saved evaluation artifact currently records:

| Metric | Value |
|---|---:|
| Accuracy | 0.95625 |
| Precision | 1.0 |
| Recall | 0.9461538461538461 |
| F1 | 0.9723320158102767 |
| Training rows | 640 |
| Test rows | 160 |

### Classification metric meanings

- **Accuracy**: share of all classifications that were correct.
- **Precision**: among records classified as Pass, share that were actually Pass.
- **Recall**: among actual Pass records, share identified by the classifier.
- **F1**: harmonic balance of precision and recall.

The positive class is `Pass`.

---

## Why model signals can differ

The regression and classifier answer different statistical questions:

### Regression path

```text
Inputs
  -> predicted ETE
  -> predicted total
  -> exact passing rules
  -> projected Pass/Fail
```

### Classifier training path

```text
Inputs
  -> learned historical Pass/Fail pattern
  -> raw classifier signal
```

The raw classifier does not receive the regression model's predicted ETE. It can
therefore identify an input profile as similar to historical Pass records even
when the regression estimate does not reach the exact passing threshold.

To prevent contradictory user-facing results, the API returns:

- `classifier_model_status`: the raw classifier output for evaluation.
- `classifier_projected_status`: aligned with the authoritative regression-based
  passing calculation.

The frontend displays the rule-consistent status. It does not allow the raw
classifier signal to override the deterministic rules.

---

## Evaluation artifacts

Training creates three PNG charts.

### `actual_vs_predicted.png`

Shows held-out actual ETE values against regression predictions. The diagonal
reference line represents a perfect prediction.

### `confusion_matrix.png`

Shows the classifier's held-out counts for:

- Actual Fail predicted Fail.
- Actual Fail predicted Pass.
- Actual Pass predicted Fail.
- Actual Pass predicted Pass.

### `feature_analysis.png`

Shows Pearson correlation between the four input features and ETE marks. This is
an analysis chart; correlation is not proof of causation.

The frontend loads these files through:

```text
GET /api/charts/actual_vs_predicted.png
GET /api/charts/confusion_matrix.png
GET /api/charts/feature_analysis.png
```

The JSON metrics file is served through:

```text
GET /api/model-performance
```

If metrics are unavailable, the frontend shows an honest empty state rather than
inventing values.

---

## Frontend experience

The React application is implemented in
[`frontend/src/main.jsx`](./frontend/src/main.jsx), with styling in
[`frontend/src/styles.css`](./frontend/src/styles.css).

### Hero

The hero includes:

- LPU Student Analytics product identity.
- Main prediction value proposition.
- Primary scroll-to-predictor CTA.
- Model-insights CTA.
- Estimate disclaimer.
- Responsive navigation.
- Academic performance visualization.
- Interactive illustrative scenario controls.

The scenario controls in the hero are illustrative UI examples. They are not
backend predictions and must not be interpreted as actual submitted results.

### Predictor

The predictor immediately follows the hero. It contains:

- Student ID input.
- Study hours input.
- Attendance input.
- CA input.
- MTE input.
- Browser range validation.
- Loading state.
- API error state.
- Empty results state.
- Live result state.

### Results

After a successful API response, the interface shows:

- Regression-based projected outcome.
- Predicted ETE.
- Predicted total.
- ETE difference from the minimum requirement.
- Minimum ETE calculator.
- Required and predicted ETE comparison bar.
- Overall threshold.
- ETE threshold.
- Mathematical possibility.
- Rule-consistent regression and classifier projections.

### Model insights

The model-insights section fetches live evaluation JSON and displays:

- Regression R².
- Regression MAE.
- Classifier accuracy.
- Classifier F1.
- Actual-versus-predicted chart.
- Confusion matrix.
- Feature-analysis chart.
- Train/test size explanation.

### How it works

The workflow section explains:

1. Enter the current academic profile.
2. Estimate ETE with regression.
3. Calculate total and minimum ETE requirements.
4. Show the rule-consistent model readouts.

### Responsive behavior

The page supports:

- Desktop two-column predictor layout.
- Tablet single-column fallback.
- Mobile stacked form/results layout.
- Mobile navigation menu.
- Keyboard-visible focus states.
- Reduced-motion support.

---

## Backend API

The Flask service is implemented in
[`backend/app.py`](./backend/app.py).

Default address:

```text
http://localhost:5000
```

Allowed frontend development origins:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

### `GET /api/health`

Checks whether the service is running and whether the ETE regression artifact
exists.

Example response:

```json
{
  "status": "ok",
  "models_ready": true
}
```

### `POST /api/predict`

Validates an input profile, loads both models, calculates the result, and
optionally saves the response to MongoDB.

Required JSON fields:

```json
{
  "student_id": "12324661",
  "study_hours_per_week": 35,
  "attendance_percentage": 99,
  "ca_marks_30": 3,
  "midterm_marks_25": 7
}
```

### `GET /api/model-performance`

Returns the saved `results/evaluation_results.json` content.

Returns HTTP 404 when the evaluation file has not been generated.

### `GET /api/prediction-history?limit=20`

Reads recent prediction documents from MongoDB. The `limit` query parameter must
be an integer from 1 through 100.

MongoDB failures return HTTP 503 with an explanatory error message.

### `GET /api/charts/<filename>`

Only these filenames are accepted:

- `actual_vs_predicted.png`
- `confusion_matrix.png`
- `feature_analysis.png`

Other filenames return HTTP 404.

---

## Prediction request and response

### Input validation

The API requires:

- A non-empty `student_id`.
- Numeric `study_hours_per_week` from 0 to 40.
- Numeric `attendance_percentage` from 0 to 100.
- Numeric `ca_marks_30` from 0 to 30.
- Numeric `midterm_marks_25` from 0 to 25.

The API rejects:

- Missing fields.
- Non-numeric values.
- NaN and infinite numeric values.
- Values outside configured ranges.
- Non-object JSON request bodies.

### Response fields

A successful response includes:

| Field | Meaning |
|---|---|
| `student_id` | Submitted identifier |
| `study_hours_per_week` | Validated study-hours input |
| `attendance_percentage` | Validated attendance input |
| `ca_marks_30` | Validated CA input |
| `midterm_marks_25` | Validated MTE input |
| `predicted_ete_marks` | Regression estimate from 0 to 45 |
| `predicted_total_marks` | CA + MTE + predicted ETE |
| `projected_status` | Rule-based projected Pass/Fail |
| `regression_projected_status` | Same rule-based projected status |
| `classifier_projected_status` | User-facing rule-consistent status |
| `classifier_model_status` | Raw classifier signal |
| `classifier_probability` | Classifier probability for its raw signal |
| `minimum_ete_marks` | Minimum ETE required under both rules |
| `ete_gap` | Predicted ETE minus required ETE |
| `passing_possible` | Whether 45/45 could still reach total 40 |
| `maximum_possible_total` | CA + MTE + 45 |
| `calculator_message` | Human-readable possibility message |
| `calculator` | Detailed minimum-ETE calculation object |
| `is_estimate` | Always true for model predictions |
| `database` | Whether MongoDB persistence succeeded |

### Database response

When MongoDB is configured:

```json
{
  "database": {
    "saved": true
  }
}
```

When MongoDB is unavailable:

```json
{
  "database": {
    "saved": false,
    "message": "MONGODB_URI is not configured."
  }
}
```

The model response is still returned when persistence fails.

---

## Minimum ETE calculator

The calculator is implemented in
[`backend/passing_rules.py`](./backend/passing_rules.py).

For CA `24` and MTE `19`:

```text
current_marks = 24 + 19 = 43
required_ETE = max(13.5, 40 - 43) = 13.5
maximum_possible_total = 43 + 45 = 88
```

For CA `3` and MTE `7`:

```text
current_marks = 3 + 7 = 10
required_ETE = max(13.5, 40 - 10) = 30
maximum_possible_total = 10 + 45 = 55
```

If the UI displays a different result, verify that the request contains both
`ca_marks_30` and `midterm_marks_25` and that the browser is using the current
frontend build.

---

## MongoDB integration

MongoDB support is implemented in
[`backend/database.py`](./backend/database.py).

The integration:

- Reads `MONGODB_URI` from `.env`.
- Uses a three-second server selection timeout.
- Pings MongoDB before use.
- Writes predictions to the `predictions` collection.
- Reads recent predictions ordered by `created_at`.
- Excludes MongoDB's `_id` from history responses.
- Closes the client in a `finally` block.

Database name:

```text
lpu_student_performance_db
```

Collection currently used by the code:

```text
predictions
```

The application does not expose MongoDB credentials to the browser.

---

## Installation

### Requirements

Install:

- Python 3.10 or newer recommended.
- Node.js and npm.
- Optional MongoDB local server or MongoDB Atlas account.

### Create a Python environment

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, either adjust the local execution policy or
run the Python executable directly from `.venv\Scripts\python.exe`.

### Install Python dependencies

```powershell
python -m pip install -r backend\requirements.txt
```

The root `requirements.txt` delegates to the backend requirements file.

### Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

The frontend uses:

- React.
- React DOM.
- Lucide React icons.
- Vite.

---

## Training the models

From the project root:

```powershell
python main.py
```

This command:

1. Validates the existing CSV.
2. Trains the regression model.
3. Trains the classifier.
4. Generates charts.
5. Writes evaluation JSON.
6. Writes joblib artifacts.

Expected output includes:

```text
Training complete. Artifacts saved in models/ and results/.
```

Do not run training while the API is actively serving requests if you need
strict artifact consistency. Stop the API, train, then restart it.

---

## Starting the Flask API

Recommended command from the project root:

```powershell
python -m backend.app
```

The API starts on:

```text
http://127.0.0.1:5000
```

Direct startup from the backend folder is also supported:

```powershell
cd backend
python app.py
```

The direct-entry import fallback exists specifically to preserve this command.

Do not start the API with both commands at the same time because both processes
try to bind port 5000.

---

## Starting the React frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Vite normally serves the page at:

```text
http://localhost:5173
```

The frontend reads the backend base URL from `VITE_API_URL`. If it is not set,
the default is:

```text
http://localhost:5000/api
```

---

## Environment configuration

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Then replace the placeholder:

```text
MONGODB_URI=mongodb://localhost:27017
```

For Atlas, use your private connection string and URL-encode special characters
in passwords.

Never:

- Commit `.env`.
- Put the MongoDB URI in React code.
- Paste credentials into screenshots or issue reports.
- Store credentials in the CSV.

### Frontend API override

Create `frontend\.env` only when the backend is not at the default URL:

```text
VITE_API_URL=http://127.0.0.1:5000/api
```

Restart Vite after changing environment variables.

---

## Running the complete application

### Terminal 1: train

```powershell
python main.py
```

### Terminal 2: start Flask

```powershell
python -m backend.app
```

### Terminal 3: start React

```powershell
cd frontend
npm run dev
```

Open the Vite URL in a browser. The browser should be able to:

1. Load model evaluation data.
2. Scroll from the hero to the predictor.
3. Submit the five-field form.
4. Receive a live Flask response.
5. Display rule-consistent results.
6. Display saved evaluation charts.

---

## Testing and validation

### Python tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover:

- Exact passing thresholds.
- Failing totals.
- Minimum ETE calculations.
- Whole-number rounding to 14.
- Mathematical possibility.
- ETE gap calculations.
- Dataset schema.
- Feature leakage prevention.

### Frontend production build

```powershell
cd frontend
npm run build
```

The build validates that Vite can transform and bundle the current React
application.

### Health check

```powershell
curl.exe http://127.0.0.1:5000/api/health
```

Expected response:

```json
{
  "models_ready": true,
  "status": "ok"
}
```

### CORS check

If the frontend uses `127.0.0.1`, the API must allow:

```text
http://127.0.0.1:5173
```

If it uses `localhost`, the API must allow:

```text
http://localhost:5173
```

The current Flask configuration allows both.

---

## Example API calls

### PowerShell request

```powershell
$body = @{
  student_id = "12324661"
  study_hours_per_week = 35
  attendance_percentage = 99
  ca_marks_30 = 3
  midterm_marks_25 = 7
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/api/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Equivalent curl request

```powershell
curl.exe -X POST http://127.0.0.1:5000/api/predict `
  -H "Content-Type: application/json" `
  -d "{\"student_id\":\"12324661\",\"study_hours_per_week\":35,\"attendance_percentage\":99,\"ca_marks_30\":3,\"midterm_marks_25\":7}"
```

### Expected calculation for the example

The exact model ETE value depends on the saved artifact. The deterministic
portion is:

```text
current marks = 3 + 7 = 10
minimum ETE = 30
maximum possible total = 55
```

Therefore:

- Passing is mathematically possible.
- A predicted ETE below 30 produces a projected Fail.
- A predicted ETE of at least 30 produces a projected Pass.

### Retrieve metrics

```powershell
curl.exe http://127.0.0.1:5000/api/model-performance
```

### Retrieve history

```powershell
curl.exe "http://127.0.0.1:5000/api/prediction-history?limit=20"
```

---

## Frontend behavior

### Hero scenarios

The hero contains selectable illustrative student scenarios. Selecting a
scenario updates the visual preview and can populate the predictor form. These
values are presentation examples and are not a substitute for a submitted
backend prediction.

### Form behavior

Inputs are required. Numeric fields include HTML minimum and maximum constraints.
The backend repeats validation and must be treated as the source of truth.

### Loading behavior

During prediction:

- The submit button is disabled.
- A loading icon and `Calculating...` label are shown.

During evaluation loading:

- The model-insights section shows a loading state.

### Error behavior

Fetch failures, API errors, unavailable metrics, and invalid requests are shown
as visible messages. The UI does not fabricate successful values when the
backend is unavailable.

### Accessibility behavior

The interface includes:

- Labels associated with form inputs.
- Descriptive input hints.
- `role="alert"` for errors.
- Keyboard focus-visible styles.
- Navigation labels.
- Reduced-motion media-query handling.
- Alt text for evaluation charts.

---

## Error handling

### Invalid prediction request

Returns HTTP 400 with a JSON error:

```json
{
  "error": "midterm_marks_25 must be between 0 and 25."
}
```

### Missing model artifacts

Returns HTTP 400 with:

```text
Model artifacts are missing. Run `python main.py` first.
```

### Invalid artifact contract

If the saved feature list does not match `FEATURE_COLUMNS`, the API refuses to
use the artifact rather than silently making a prediction with the wrong inputs.

### Missing evaluation file

`GET /api/model-performance` returns HTTP 404 and asks you to run training.

### MongoDB unavailable

Prediction still returns, with:

```json
{
  "database": {
    "saved": false,
    "message": "..."
  }
}
```

### Unexpected server failure

The Flask application logs the exception and returns a generic HTTP 500 message
instead of exposing an internal traceback to the browser.

---

## Troubleshooting

### `ImportError: attempted relative import with no known parent package`

Use the recommended package command from the project root:

```powershell
python -m backend.app
```

The current `app.py` also supports:

```powershell
cd backend
python app.py
```

If using an older copy of the project, update `backend/app.py`.

### `Failed to fetch`

Check all of these:

1. Flask is running.
2. The browser URL uses the correct API URL.
3. The frontend and backend ports are correct.
4. The frontend origin is `localhost:5173` or `127.0.0.1:5173`.
5. Flask CORS includes the actual frontend origin.
6. Port 5000 is not occupied by a stale server.

Use:

```powershell
curl.exe http://127.0.0.1:5000/api/health
```

### `Model artifacts are missing`

Run:

```powershell
python main.py
```

Then restart Flask.

### Metrics section is empty

Run training and verify:

```text
results/evaluation_results.json
```

exists and contains valid JSON.

### Chart images do not load

Verify the three files exist in `results/` and that Flask is running. Charts are
served by Flask; they are not bundled into the React application.

### MongoDB save reports false

This is expected when MongoDB is not configured. To enable saving:

1. Create `.env`.
2. Set `MONGODB_URI`.
3. Make sure MongoDB is reachable.
4. Restart Flask.

### Port already in use

Do not start multiple Flask processes. Stop the previous process or configure a
different port and update the frontend API URL.

### Frontend changes do not appear

Try:

```powershell
cd frontend
npm run build
npm run dev
```

Then hard-refresh the browser. Do not edit `frontend/dist` manually; it is
generated output.

---

## Security and privacy

- `.env` is ignored by Git.
- MongoDB credentials stay on the backend.
- The browser receives prediction results, not the database URI.
- The API validates numeric ranges before model execution.
- File serving is restricted to a fixed allowlist of chart filenames.
- The API does not expose internal tracebacks in normal responses.
- Student IDs should be treated as sensitive project data.
- Do not use real student personal information without proper authorization.

This project is not a production-grade identity, authorization, audit, or
student-record management system.

---

## Reproducibility

Training uses:

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
```

The model split and evaluation process are therefore reproducible when the CSV,
Python package versions, and source code remain unchanged.

The `generated_at_utc` field in
[`results/evaluation_results.json`](./results/evaluation_results.json) changes
each time training runs.

Model numbers can change if:

- The CSV changes.
- Package versions change.
- Feature order changes.
- Training logic changes.
- Passing rules change.

Always retrain after changing the training contract or passing rules.

---

## Known limitations

1. The model predicts from a relatively small feature set.
2. A prediction is an estimate, not a guarantee.
3. Regression MAE means individual predictions can be several marks away.
4. The classifier's raw historical signal is not a mathematical guarantee.
5. The frontend hero scenarios are illustrative.
6. MongoDB persistence is optional and environment-dependent.
7. The development Flask server is not intended for production deployment.
8. CORS is configured for local development origins.
9. Evaluation charts are generated PNGs rather than interactive chart data.
10. Feature correlation does not establish causation.
11. The application does not authenticate users.
12. The application does not provide official university integration.
13. Dataset provenance and representativeness must be assessed separately.

---

## Changing the project

### Change the passing threshold

Update the constants in
[`backend/passing_rules.py`](./backend/passing_rules.py), then:

1. Update tests.
2. Update training label generation.
3. Retrain with `python main.py`.
4. Review the frontend copy.
5. Update this README.

Do not change only the frontend display. The backend is authoritative.

### Add an input feature

Update all of these together:

1. `FEATURE_COLUMNS` in `backend/train_model.py`.
2. `REQUIRED_COLUMNS` and ranges if the CSV needs a new column.
3. Training DataFrame selection.
4. API validation in `backend/app.py`.
5. Frontend fields in `frontend/src/main.jsx`.
6. Tests.
7. README feature contract.
8. Retrained model artifacts.

### Change the API port

Change the Flask `app.run()` port and set the matching frontend:

```text
VITE_API_URL=http://localhost:<port>/api
```

Also update CORS origins when the frontend origin changes.

### Change the frontend design

Primary files:

- [`frontend/src/main.jsx`](./frontend/src/main.jsx)
- [`frontend/src/styles.css`](./frontend/src/styles.css)
- [`frontend/src/assets/hero-landscape.jpg`](./frontend/src/assets/hero-landscape.jpg)

Keep the API request field names and response interpretation aligned with
`backend/app.py`.

---

## Frequently asked questions

### Does the application use the ETE score as an input?

No. ETE is the regression target. The user enters only pre-ETE information.

### Can a low prediction be called mathematically impossible?

No. Mathematical impossibility depends on whether CA + MTE + 45 can reach 40.

### Why is the minimum ETE sometimes 13.5 and sometimes higher?

The minimum is the larger of:

```text
13.5
40 - CA - MTE
```

High current marks leave the 13.5 ETE floor as the limiting requirement. Lower
current marks make the overall 40-mark condition stricter.

### Why does the raw classifier sometimes say Pass when the projected result is Fail?

The raw classifier learns historical Pass/Fail patterns from the four inputs. It
does not use the regression prediction. The API preserves that raw signal as
`classifier_model_status` for evaluation but aligns the displayed
`classifier_projected_status` with the exact rule-based projected outcome.

### Does MongoDB have to be running?

No. Predictions and model evaluation work without MongoDB. Only persistence and
history require a reachable MongoDB instance.

### Does training modify the CSV?

No. Training validates and reads the CSV, then writes models and results.

### Are the hero numbers real predictions?

No. Hero scenarios are illustrative UI examples. Real predictions are generated
only after submitting the predictor form to the Flask API.

### Is this production-ready?

No. It is an educational application and development setup. Production use
would require authentication, authorization, HTTPS, a production WSGI server,
secret management, database indexes and retention policies, monitoring, audit
controls, privacy review, and validated data provenance.

---

## Project status

The current project includes:

- Existing CSV validation.
- Explicit model training.
- Saved regression and classifier artifacts.
- Real evaluation JSON and charts.
- Flask prediction and evaluation endpoints.
- Optional MongoDB prediction persistence.
- React/Vite landing page.
- Responsive predictor interface.
- Rule-consistent results.
- Passing-rule unit tests.
- Training-contract tests.

The project intentionally does not claim that predictions are official grades or
that the dataset represents real university records.
