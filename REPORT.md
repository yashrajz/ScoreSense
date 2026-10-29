# LPU Student Performance Prediction System

## Project Report

## 1. Introduction

Student performance is influenced by several measurable factors, including
attendance, study habits, internal assessment marks, and midterm examination
marks. Before the final examination, a student usually knows some of these
values but does not yet know the End Term Examination (ETE) result. This
project uses historical student-performance data and machine-learning models to
estimate the expected ETE marks from the information that is already available.

The application is an academic analytics and planning tool. A user enters:

1. A student ID.
2. Study hours per week.
3. Attendance percentage.
4. Continuous Assessment (CA) marks out of 30.
5. Midterm Examination (MTE) marks out of 25.

The system then:

1. Estimates ETE marks out of 45 using a linear regression model.
2. Calculates the expected total marks out of 100.
3. Calculates the minimum ETE marks required to satisfy the configured passing
   rules.
4. Determines whether the predicted result is Pass or Fail.
5. Uses a separate decision-tree classifier to produce a historical
   Pass/Fail signal.
6. Stores the prediction in MongoDB when MongoDB is configured and available.
7. Displays model-performance metrics and charts generated from held-out test
   data.

The project has three main layers:

- **Frontend:** A React/Vite browser interface where users enter data and read
  results.
- **Backend:** A Flask REST API that validates input, loads models, performs
  calculations, and saves predictions.
- **Data and machine learning layer:** Pandas, NumPy, and scikit-learn code
  that validates the CSV, trains the models, and creates evaluation artifacts.

This project is intended for educational planning. It does not issue official
university grades, guarantee a student's result, or replace official academic
records.

## 2. Project Objective

### 2.1 Main objective

The main objective is to build a complete web-based student-performance
prediction system that converts a student's current academic profile into a
clear, understandable estimate of likely ETE performance.

### 2.2 Specific objectives

The project aims to:

1. Read and validate a structured student-performance dataset.
2. Prevent invalid or inconsistent data from entering the training process.
3. Select only information known before the ETE as model input.
4. Predict the ETE mark on a 45-mark scale.
5. Classify the student's likely historical Pass/Fail pattern.
6. Apply explicit mathematical passing rules consistently.
7. Expose prediction functionality through a REST API.
8. Provide a usable React interface for non-technical users.
9. Save submitted prediction records in MongoDB.
10. Make stored records searchable by `student_id`.
11. Generate objective evaluation metrics and visual charts.
12. Keep the system reproducible through explicit scripts and tests.

### 2.3 Problem statement

Without a prediction tool, a student may know current CA and MTE marks but
still not know:

- What ETE mark might be expected.
- What total mark might be achieved.
- How many ETE marks are mathematically required to pass.
- Whether passing is still possible even with a perfect ETE.

The application addresses this problem by combining statistical prediction with
transparent rule-based calculations.

## 3. Scope of the Project

### Included

- CSV-based historical data loading.
- Dataset validation.
- Linear regression for ETE estimation.
- Decision-tree classification for historical Pass/Fail prediction.
- Deterministic passing-rule calculations.
- Flask API endpoints.
- React/Vite frontend.
- MongoDB persistence through PyMongo.
- Prediction history endpoint.
- Training and evaluation charts.
- Unit tests for passing rules and training contracts.

### Not included

- Official university grade processing.
- Authentication or role-based access control.
- Editing or deleting MongoDB records through the frontend.
- Automatic model retraining after every prediction.
- Real-time learning from newly submitted predictions.
- A causal study proving that any input directly causes a grade outcome.
- A guarantee that a model prediction will match a student's actual result.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend and machine-learning implementation |
| Flask | REST API server |
| Flask-CORS | Allows the configured frontend origins to call the API |
| Pandas | Reading, validating, and manipulating tabular data |
| NumPy | Numeric validation, finite-number checks, clipping, and comparisons |
| scikit-learn | Regression, classification, train/test splitting, and metrics |
| joblib | Saving and loading trained model artifacts |
| Matplotlib | Creating evaluation charts |
| PyMongo | Connecting to MongoDB and writing/querying documents |
| python-dotenv | Loading `MONGODB_URI` from `.env` |
| React | Frontend user interface |
| Vite | Frontend development and build tool |
| Lucide React | Interface icons |
| unittest | Automated Python tests |
| MongoDB Compass | Optional graphical tool for inspecting stored documents |

The Python dependencies are listed in
[`backend/requirements.txt`](./backend/requirements.txt). The root
[`requirements.txt`](./requirements.txt) delegates to that file.

## 5. System Architecture

The main data flow is:

```text
lpu_student_performance.csv
              |
              v
    backend/train_model.py
              |
              +--> models/ete_regression.joblib
              +--> models/pass_classifier.joblib
              +--> results/evaluation_results.json
              +--> results/*.png
                              |
                              v
                       backend/app.py
                       Flask REST API
                              |
                              v
                    React/Vite frontend
```

MongoDB is an additional persistence path:

```text
Browser form
    |
    | POST /api/predict
    v
Flask validates the request
    |
    v
Models calculate prediction and rules calculate status
    |
    +--> JSON response to browser
    |
    +--> PyMongo insert_one(...)
              |
              v
MongoDB: lpu_student_performance_db.predictions
```

There are two different kinds of records in the same MongoDB collection:

1. **Seeded dataset records:** Imported from the CSV by
   [`backend/seed_database.py`](./backend/seed_database.py). They have
   `record_type: "dataset_record"` and `is_estimate: false`.
2. **New prediction records:** Created after a user submits the frontend form.
   They have `is_estimate: true` and contain the model output.

## 6. Repository Structure

Important project files are:

| File or directory | Purpose |
|---|---|
| [`lpu_student_performance.csv`](./lpu_student_performance.csv) | Historical input dataset |
| [`main.py`](./main.py) | Entry point for explicit model training |
| [`backend/train_model.py`](./backend/train_model.py) | Dataset validation, training, metrics, and charts |
| [`backend/passing_rules.py`](./backend/passing_rules.py) | Authoritative passing calculations |
| [`backend/app.py`](./backend/app.py) | Flask REST API |
| [`backend/database.py`](./backend/database.py) | MongoDB/PyMongo integration |
| [`backend/seed_database.py`](./backend/seed_database.py) | CSV-to-MongoDB import utility |
| [`models/`](./models/) | Saved joblib model artifacts |
| [`results/`](./results/) | JSON metrics and PNG evaluation charts |
| [`frontend/src/main.jsx`](./frontend/src/main.jsx) | React application and form behavior |
| [`frontend/src/styles.css`](./frontend/src/styles.css) | Frontend styling |
| [`tests/test_training_contract.py`](./tests/test_training_contract.py) | Dataset and target-leakage tests |
| [`tests/test_passing_rules.py`](./tests/test_passing_rules.py) | Passing-rule tests |

## 7. Dataset Description

### 7.1 Source and size

The dataset is the root-level file
[`lpu_student_performance.csv`](./lpu_student_performance.csv).

Verified dataset facts:

- Number of rows: **800**
- Number of columns: **7**
- Missing values in required columns: **0**
- Duplicate student IDs: **0**
- Student IDs are numeric in the CSV.
- The dataset contains historical ETE and total marks.

The dataset is used as project data. The application does not claim that it
represents official university records.

### 7.2 Dataset columns

| Column | Meaning | Role | Valid range |
|---|---|---|---:|
| `student_id` | Student identifier | Identifier only | Non-empty/unique |
| `study_hours_per_week` | Weekly study time | Model feature | 0–40 |
| `attendance_percentage` | Attendance percentage | Model feature | 0–100 |
| `ca_marks_30` | Continuous Assessment marks | Model feature | 0–30 |
| `midterm_marks_25` | Midterm Examination marks | Model feature | 0–25 |
| `endterm_marks_45` | End Term Examination marks | Regression target | 0–45 |
| `total_marks_100` | Final total | Validation and label construction | 0–100 |

### 7.3 Dataset summary statistics

The following values were calculated from the current CSV:

| Variable | Mean | Standard deviation | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|---:|
| `study_hours_per_week` | 16.7672 | 7.2097 | 0.0 | 17.5 | 37.7 |
| `attendance_percentage` | 72.8222 | 13.7351 | 20.4 | 74.8 | 100.0 |
| `ca_marks_30` | 16.3288 | 5.5508 | 0 | 17 | 30 |
| `midterm_marks_25` | 13.7762 | 4.8250 | 0 | 14 | 25 |
| `endterm_marks_45` | 24.5362 | 8.8113 | 0 | 25 | 45 |
| `total_marks_100` | 54.6412 | 17.7826 | 0 | 57 | 100 |

The configured passing rules classify:

- Passing records: **652**
- Failing records: **148**
- Passing percentage: **81.5%**

These are labels calculated using the project's rules, not a separate
`pass_status` column in the CSV.

### 7.4 Observed correlations

The current Pearson correlations with `endterm_marks_45` are:

| Input | Correlation with ETE |
|---|---:|
| `study_hours_per_week` | 0.615494 |
| `attendance_percentage` | 0.677926 |
| `ca_marks_30` | 0.795876 |
| `midterm_marks_25` | 0.740535 |

A positive correlation means that higher values tend to be associated with
higher ETE values in this dataset. Correlation does not prove that one variable
causes another. The project displays this relationship as an analysis chart,
not as a causal conclusion.

## 8. Data Preprocessing and Validation

Data preprocessing is implemented in
`load_and_validate_dataset()` inside
[`backend/train_model.py`](./backend/train_model.py).

### 8.1 Reading the data

Pandas reads the CSV into a `DataFrame`. A DataFrame is a table-like Python
object with named columns and rows, similar to a spreadsheet.

### 8.2 Schema validation

The required columns are:

```python
[
    "student_id",
    "study_hours_per_week",
    "attendance_percentage",
    "ca_marks_30",
    "midterm_marks_25",
    "endterm_marks_45",
    "total_marks_100",
]
```

If any required column is missing, training stops with an explicit error. This
prevents the model from training on an incomplete or incorrectly named dataset.

### 8.3 Missing-value validation

The required columns are checked for null or missing values. If a required
value is missing, training stops instead of silently filling it with a guessed
value.

This is important because silently replacing missing academic marks could
change the meaning of the dataset and introduce incorrect training examples.

### 8.4 Numeric conversion

Every required numeric column except `student_id` is converted using
`pandas.to_numeric(..., errors="raise")`. The `raise` behavior means invalid
text such as `"unknown"` causes a visible error instead of being silently
converted to a missing value.

### 8.5 Range validation

Each numeric column has an allowed range:

```text
Study hours:       0 to 40
Attendance:        0 to 100
CA marks:          0 to 30
MTE marks:         0 to 25
ETE marks:         0 to 45
Total marks:       0 to 100
```

Values outside these limits are rejected.

### 8.6 Duplicate-ID validation

The CSV must not contain duplicate `student_id` values. Duplicate IDs would
make it difficult to identify a unique source record and would undermine the
dataset's identity assumptions.

### 8.7 Total-mark consistency validation

The project verifies:

```text
CA + MTE + ETE = total_marks_100
```

The comparison uses NumPy's `isclose` with a small tolerance to avoid false
failures caused by floating-point representation. If a row's recorded total
does not match its component marks, training stops and reports the affected CSV
rows.

### 8.8 No imputation or automatic repair

The preprocessing code intentionally does not:

- Invent missing marks.
- Clip invalid CSV values into range.
- Remove bad rows silently.
- Rewrite the source CSV.
- Generate replacement data.

Invalid source data is reported so that it can be corrected deliberately.

## 9. Feature and Target Selection

### 9.1 Model features

Both models use exactly these four features, in this order:

```python
[
    "study_hours_per_week",
    "attendance_percentage",
    "ca_marks_30",
    "midterm_marks_25",
]
```

These values are available before the ETE and are also the values accepted by
the frontend form and `/api/predict` endpoint.

### 9.2 Regression target

The regression model predicts:

```text
endterm_marks_45
```

This is a numeric value from 0 to 45.

### 9.3 Classification target

There is no stored `pass_status` column in the CSV. The classifier label is
created from the authoritative passing rules:

```text
Pass if:
    total_marks_100 >= 40
    AND endterm_marks_45 >= 13.5

Fail otherwise
```

### 9.4 Avoiding target leakage

Target leakage occurs when information that would not be available at
prediction time is accidentally provided to a model. The project avoids this
by excluding:

- `endterm_marks_45`
- `total_marks_100`
- Any future Pass/Fail label

from the model input features. The test
[`tests/test_training_contract.py`](./tests/test_training_contract.py) verifies
that these target fields are not present in `FEATURE_COLUMNS`.

## 10. Passing Rules

The authoritative rules are implemented in
[`backend/passing_rules.py`](./backend/passing_rules.py).

### 10.1 Passing conditions

A projected result is **Pass** only if both conditions are true:

```text
Total marks >= 40
ETE marks >= 13.5
```

The total is:

```text
total = CA marks + MTE marks + ETE marks
```

### 10.2 Minimum ETE calculation

The minimum ETE required is:

```text
minimum ETE = max(13.5, 40 - CA marks - MTE marks)
```

The first value, 13.5, enforces the minimum ETE condition. The second value,
`40 - CA - MTE`, enforces the overall total condition. Taking the maximum means
both conditions are satisfied.

### 10.3 Mathematical possibility

Passing remains mathematically possible when:

```text
CA + MTE + 45 >= 40
```

The value 45 is the maximum possible ETE mark. A model prediction can be below
the passing requirement while passing is still mathematically possible.

For example:

```text
CA = 10
MTE = 10
Maximum possible total = 10 + 10 + 45 = 65
```

Passing is possible, but if the predicted ETE is 15, the predicted total is 35
and the projected outcome is Fail.

Under the current valid API ranges, CA and MTE are both non-negative and the
maximum ETE is 45. Consequently, the maximum possible total is always at least
45, so the mathematically-impossible branch cannot be reached by valid current
API inputs. The branch is retained as defensive logic for future callers or
changed passing rules.

### 10.4 Whole-number option

The helper supports a `whole_number=True` option. When enabled, a fractional
minimum is rounded upward. For example, 13.5 becomes 14. The API's standard
response reports the exact configured threshold, while the helper can provide a
whole-mark requirement where needed.

### 10.5 Floating-point tolerance

The rules use a small tolerance of `1e-9` so that values such as 40.0 are not
incorrectly rejected because of tiny floating-point representation differences.

## 11. Model Implementation

Training is performed by
[`backend/train_model.py`](./backend/train_model.py), normally started through
[`main.py`](./main.py).

The explicit training command is:

```powershell
python main.py
```

Starting Flask does not retrain the models. Training is a separate, deliberate
operation.

### 11.1 Train/test split

The project uses:

```python
TEST_SIZE = 0.2
RANDOM_SEED = 42
```

For 800 rows:

- Training set: 640 rows.
- Test set: 160 rows.

The training set is used to fit the models. The test set is held out and used
to measure performance on records the models did not see during fitting.

The fixed random seed makes the split reproducible when the same data and
software behavior are used.

### 11.2 Linear regression model

The regression model is:

```text
LinearRegression-ETE-v1
```

It learns a linear relationship between the four input features and ETE marks.
Conceptually, it estimates:

```text
predicted ETE =
    intercept
    + coefficient1 * study hours
    + coefficient2 * attendance
    + coefficient3 * CA marks
    + coefficient4 * MTE marks
```

The exact coefficients are learned from the training records by
scikit-learn's `LinearRegression`.

The API clips the model result to the valid ETE range:

```text
0 <= predicted ETE <= 45
```

Clipping is applied at serving time because a mathematical regression line can
occasionally produce a value outside the real examination scale.

### 11.3 Decision-tree classifier

The classification model is:

```text
DecisionTreeClassifier-v1
```

It learns a series of feature-based decision rules from the four input
features. Its target is the derived `Pass` or `Fail` label.

The classifier uses:

```python
DecisionTreeClassifier(max_depth=4, random_state=42)
```

The maximum depth limits the number of decision levels and helps keep the tree
more controlled and interpretable than an unrestricted tree.

### 11.4 Stratification

The classifier split uses stratification when each class has at least two
examples. Stratification attempts to preserve the Pass/Fail class balance in
both the training and test partitions.

The current evaluation artifact records:

```text
stratified_split: true
```

### 11.5 Model artifact validation

Each saved joblib artifact stores its feature list. Before the API uses a model,
it checks that the saved list matches the current `FEATURE_COLUMNS` list.

This protects against loading a model trained with a different input contract.
If the artifact is missing or incompatible, the API reports an explicit error.

### 11.6 Regression and classifier signals

The two models answer different questions:

```text
Regression:
inputs -> predicted ETE -> predicted total -> passing rules
```

```text
Classifier:
inputs -> learned historical Pass/Fail signal
```

The classifier does not receive the regression model's predicted ETE. Because
of this, the raw classifier signal can differ from the deterministic
rule-based outcome.

The API therefore returns both:

- `classifier_model_status`: the classifier's raw signal.
- `classifier_projected_status`: aligned with the authoritative projected
  result.

The user-facing status is based on the transparent regression-plus-rules path,
not on silently allowing the classifier to override the configured passing
conditions.

## 12. Evaluation Results

The current saved evaluation artifact is
[`results/evaluation_results.json`](./results/evaluation_results.json).

It was generated at:

```text
2026-10-29T18:11:37.107381+00:00
```

### 12.1 Regression results

| Metric | Value |
|---|---:|
| Model | LinearRegression-ETE-v1 |
| Target | `endterm_marks_45` |
| Training rows | 640 |
| Test rows | 160 |
| R² | 0.6709464964944141 |
| Mean Absolute Error | 4.077324333205772 |

#### Meaning of R²

R² measures how much of the variation in the held-out ETE values is explained
by the model relative to a simple baseline. A value of approximately 0.671
indicates that the model captures a substantial portion of the observed
variation, but it does not explain everything.

R² is not the probability that a prediction is correct, and it does not mean
that every prediction is within a fixed number of marks.

#### Meaning of MAE

Mean Absolute Error is the average absolute difference between actual and
predicted ETE marks. An MAE of approximately 4.08 means the average absolute
error is about 4 marks on a 45-mark ETE scale.

### 12.2 Classification results

| Metric | Value |
|---|---:|
| Model | DecisionTreeClassifier-v1 |
| Positive class | `Pass` |
| Training rows | 640 |
| Test rows | 160 |
| Stratified split | Yes |
| Accuracy | 0.95625 |
| Precision | 1.0 |
| Recall | 0.9461538461538461 |
| F1 score | 0.9723320158102767 |

#### Meaning of classification metrics

- **Accuracy:** The proportion of all test records classified correctly.
- **Precision:** Among records predicted as Pass, the proportion that actually
  belongs to the Pass class.
- **Recall:** Among actual Pass records, the proportion identified by the
  classifier.
- **F1 score:** A combined measure balancing precision and recall.

### 12.3 Confusion matrix

The saved matrix uses labels in this order:

```text
["Fail", "Pass"]
```

The matrix is:

```text
                 Predicted Fail  Predicted Pass
Actual Fail              30              0
Actual Pass               7            123
```

Interpretation:

- 30 failing records were correctly identified as Fail.
- 0 failing records were incorrectly identified as Pass.
- 7 passing records were incorrectly identified as Fail.
- 123 passing records were correctly identified as Pass.

### 12.4 Feature correlation analysis

The project also creates a feature-analysis chart, but it is important to
distinguish this chart from classifier feature importance. The file
`results/feature_analysis.png` plots Pearson correlations between each input
feature and the actual `endterm_marks_45` column. It does **not** plot the
decision tree's `feature_importances_` values.

The correlations are:

| Feature | Pearson correlation with ETE |
|---|---:|
| `study_hours_per_week` | 0.615494 |
| `attendance_percentage` | 0.677926 |
| `ca_marks_30` | 0.795876 |
| `midterm_marks_25` | 0.740535 |

These are correlation coefficients, not classifier importance values. They
describe linear association in this dataset and should not be interpreted as
proof that a feature causes academic performance.

### 12.5 Evaluation charts

Training creates:

1. [`results/actual_vs_predicted.png`](./results/actual_vs_predicted.png):
   Compares actual held-out ETE marks with regression predictions. The
   diagonal reference line represents perfect prediction.
2. [`results/confusion_matrix.png`](./results/confusion_matrix.png):
   Visualizes the four confusion-matrix counts.
3. [`results/feature_analysis.png`](./results/feature_analysis.png):
   Shows correlations between input features and ETE marks.

## 13. Flask API Implementation

The API is implemented in
[`backend/app.py`](./backend/app.py).

### 13.1 CORS configuration

Cross-Origin Resource Sharing (CORS) is enabled only for:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

This allows the local Vite frontend to call the local Flask backend.

### 13.2 `POST /api/predict`

This endpoint accepts a JSON object with:

```json
{
  "student_id": "12324751",
  "study_hours_per_week": 18,
  "attendance_percentage": 82,
  "ca_marks_30": 22,
  "midterm_marks_25": 18
}
```

Processing sequence:

1. Read the JSON body.
2. Confirm that the body is an object.
3. Validate `student_id`.
4. Convert numeric fields to floating-point values.
5. Reject missing, non-numeric, NaN, infinite, or out-of-range values.
6. Load the regression artifact.
7. Load the classifier artifact.
8. Construct a one-row Pandas DataFrame with the exact feature order.
9. Predict ETE marks.
10. Clip predicted ETE to 0–45.
11. Calculate the projected total and passing information.
12. Calculate classifier status and probability.
13. Mark the response as an estimate.
14. Attempt to save the complete response in MongoDB.
15. Return JSON to the frontend.

### 13.3 Input validation

The accepted ranges are:

| Input | Required range |
|---|---:|
| `study_hours_per_week` | 0–40 |
| `attendance_percentage` | 0–100 |
| `ca_marks_30` | 0–30 |
| `midterm_marks_25` | 0–25 |

The API rejects:

- Missing fields.
- An empty student ID.
- Non-object JSON bodies.
- Non-numeric academic values.
- NaN values.
- Infinite values.
- Values outside the configured ranges.

### 13.4 Response fields

A successful response includes the submitted inputs and:

| Field | Explanation |
|---|---|
| `predicted_ete_marks` | Regression estimate from 0 to 45 |
| `predicted_total_marks` | CA + MTE + predicted ETE |
| `projected_status` | Rule-based projected Pass/Fail |
| `regression_projected_status` | Same rule-based status |
| `classifier_projected_status` | User-facing rule-consistent classifier field |
| `classifier_model_status` | Raw decision-tree output |
| `classifier_probability` | Probability for the raw classifier class |
| `minimum_ete_marks` | Minimum ETE required by both passing rules |
| `ete_gap` | Predicted ETE minus minimum required ETE |
| `passing_possible` | Whether a total of 40 is still possible with ETE 45 |
| `maximum_possible_total` | CA + MTE + 45 |
| `calculator_message` | Human-readable mathematical possibility message |
| `calculator` | Detailed calculator object |
| `is_estimate` | Always `true` for model predictions |
| `database` | Whether MongoDB persistence succeeded |

### 13.5 Other endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/model-performance` | GET | Returns saved evaluation JSON |
| `/api/prediction-history` | GET | Returns recent MongoDB predictions |
| `/api/charts/<filename>` | GET | Serves one of three allowed PNG charts |
| `/api/health` | GET | Reports API health and model readiness |

The history endpoint accepts a `limit` query parameter from 1 to 100:

```text
GET /api/prediction-history?limit=20
```

The chart endpoint uses an allowlist and only serves:

- `actual_vs_predicted.png`
- `confusion_matrix.png`
- `feature_analysis.png`

## 14. MongoDB and PyMongo Implementation

### 14.1 Why MongoDB is used

MongoDB is a document database. Instead of storing data in rows and columns
inside fixed relational tables, it stores JSON-like documents in collections.
This suits prediction responses because a prediction contains inputs, output
values, calculator details, status values, and timestamps.

PyMongo is the official Python driver used by the backend to communicate with
MongoDB.

### 14.2 Connection configuration

The backend reads `MONGODB_URI` from the environment. The
`python-dotenv` package loads values from the backend `.env` file.

The application uses:

```text
Database:   lpu_student_performance_db
Collection: predictions
```

The connection uses a 3-second server-selection timeout and sends a `ping`
command before returning the client. This makes unavailable configuration or
an unavailable MongoDB server fail explicitly.

MongoDB is not exposed directly to the browser. The browser only communicates
with Flask; Flask communicates with MongoDB.

### 14.3 Saving a new prediction

The function `save_prediction()` in
[`backend/database.py`](./backend/database.py) performs:

```python
collection.insert_one(
    {**record, "created_at": datetime.now(timezone.utc)}
)
```

This means the record returned by the prediction calculation is copied and
stored with a UTC `created_at` timestamp.

A successful API response contains:

```json
{
  "database": {
    "saved": true
  }
}
```

If MongoDB is not configured or unavailable, the model result is still returned
but the response contains:

```json
{
  "database": {
    "saved": false,
    "message": "..."
  }
}
```

This separates prediction availability from optional persistence availability.

### 14.4 Seeded CSV records

[`backend/seed_database.py`](./backend/seed_database.py) imports the validated
CSV into the same MongoDB collection. For each CSV row it adds:

```json
{
  "record_type": "dataset_record",
  "source_dataset": "lpu_student_performance.csv",
  "is_estimate": false,
  "created_at": "...",
  "seeded_at": "..."
}
```

The seed operation uses `UpdateOne(..., upsert=True)`. In simple terms:

- If the matching dataset record exists, it is updated.
- If it does not exist, it is inserted.

The matching identity is:

```text
record_type + source_dataset + student_id
```

The script creates a partial unique compound index using these fields. The
index applies only when `record_type` is `dataset_record` and
`source_dataset` is `lpu_student_performance.csv`. It prevents duplicate
seeded records for the same source dataset and student while allowing multiple
live predictions with the same student ID.

If a database was seeded with the older non-partial index, the seeding helper
replaces that project-owned index with the partial version. This is necessary
because the old compound index also treated live records with missing seed
fields as duplicate index keys.

The seed script is not required for normal prediction requests. It is a
repeatable utility for importing or refreshing the original CSV data. Deleting
the script would not delete records already in MongoDB, but it would remove the
convenient CSV import/rebuild operation.

### 14.5 Searching in MongoDB Compass

To find a newly predicted record:

1. Open MongoDB Compass.
2. Connect using the same MongoDB URI.
3. Open database `lpu_student_performance_db`.
4. Open collection `predictions`.
5. Enter this filter:

```javascript
{ "student_id": "12324751", "is_estimate": true }
```

Use the exact stored type. New API predictions normalize the ID to a string,
so the Compass filter should normally use quotes:

```javascript
{ "student_id": "12324751" }
```

To find seeded CSV records:

```javascript
{
  "student_id": "12324751",
  "record_type": "dataset_record"
}
```

To find the newest predictions:

```javascript
{ "is_estimate": true }
```

and sort by:

```text
created_at: descending
```

## 15. Example Stored Prediction Document

A newly submitted prediction is conceptually similar to:

```json
{
  "_id": "MongoDB-generated identifier",
  "student_id": "12324751",
  "study_hours_per_week": 18.0,
  "attendance_percentage": 82.0,
  "ca_marks_30": 22.0,
  "midterm_marks_25": 18.0,
  "predicted_ete_marks": 29.4,
  "predicted_total_marks": 69.4,
  "regression_projected_status": "Pass",
  "classifier_projected_status": "Pass",
  "classifier_model_status": "Pass",
  "classifier_probability": 0.98,
  "minimum_ete_marks": 13.5,
  "ete_gap": 15.9,
  "passing_possible": true,
  "maximum_possible_total": 85.0,
  "calculator_message": "Passing remains mathematically possible under the configured rules.",
  "calculator": {
    "current_marks": 40.0,
    "minimum_ete_marks": 13.5,
    "raw_minimum_ete_marks": 13.5,
    "maximum_possible_total": 85.0,
    "passing_possible": true,
    "message": "Passing remains mathematically possible under the configured rules."
  },
  "is_estimate": true,
  "created_at": "UTC timestamp"
}
```

The numeric prediction values in this example are illustrative. The real
values are calculated by the saved model at request time.

## 16. Frontend Implementation

The frontend is implemented in
[`frontend/src/main.jsx`](./frontend/src/main.jsx).

### 16.1 Form fields

The form contains:

- Student ID text input.
- Study hours number input with 0–40 limits.
- Attendance number input with 0–100 limits.
- CA marks number input with 0–30 limits.
- MTE marks number input with 0–25 limits.

The browser form uses HTML required fields and numeric constraints, but the
backend repeats validation because browser validation alone cannot be trusted.

### 16.2 API request

When the form is submitted, React sends a `POST` request to:

```text
${VITE_API_URL}/predict
```

If `VITE_API_URL` is not configured, the default is:

```text
http://localhost:5000/api
```

The form displays a loading state while the request is in progress. Errors are
shown in an alert area. A successful response is rendered in the results
panel.

### 16.3 Evaluation section

The frontend requests:

```text
GET /api/model-performance
```

It displays evaluation results and chart images when artifacts are available.
If artifacts are missing, it reports that the models need to be trained rather
than displaying invented values.

### 16.4 User-facing disclaimer

The interface clearly describes outputs as estimates and not official grades.
This is important because a statistical prediction has uncertainty and should
not be confused with an official academic decision.

## 17. Error Handling

The backend uses explicit error responses:

- Invalid user input returns HTTP 400 with a readable message.
- Missing model artifacts return an explicit training-related error.
- Invalid model feature contracts return an explicit compatibility error.
- Unexpected prediction failures are logged and return a generic HTTP 500
  response.
- Prediction-history database failures return HTTP 503.
- Missing evaluation metrics return HTTP 404.
- Invalid evaluation JSON returns HTTP 500.
- MongoDB persistence failure is reported in the `database` field while the
  model result remains available.

The implementation does not silently report a successful database save when the
insert fails.

## 18. Testing and Validation

The project contains:

### 18.1 Passing-rule tests

[`tests/test_passing_rules.py`](./tests/test_passing_rules.py) checks:

- Minimum ETE calculations.
- Whole-number rounding.
- Mathematical possibility with zero current marks.
- Exact pass thresholds.
- Predicted ETE gap calculation.

### 18.2 Training-contract tests

[`tests/test_training_contract.py`](./tests/test_training_contract.py) checks:

- ETE and total marks are not included as model features.
- The CSV contains the expected target columns.

### 18.3 Recommended test command

From the project root:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## 19. Running the Project

### 19.1 Install backend dependencies

From the project root:

```powershell
python -m pip install -r requirements.txt
```

### 19.2 Configure MongoDB

Create a backend `.env` file based on
[`.env.example`](./.env.example) and set:

```text
MONGODB_URI=your_mongodb_connection_string
```

Do not commit credentials or secrets to source control.

### 19.3 Train models

```powershell
python main.py
```

This validates the CSV, trains the models, saves joblib artifacts, writes
evaluation JSON, and generates PNG charts.

### 19.4 Seed the CSV into MongoDB

```powershell
python -m backend.seed_database
```

This step is optional for prediction serving. It is needed only when the
historical CSV should also be available in MongoDB.

### 19.5 Start the backend

```powershell
python -m backend.app
```

The Flask API runs on port 5000.

### 19.6 Start the frontend

From the frontend directory:

```powershell
cd frontend
npm run dev
```

The Vite development server normally runs on port 5173.

### 19.7 Complete workflow

The complete development workflow is:

```text
Install dependencies
        |
        v
Configure MONGODB_URI
        |
        v
python main.py
        |
        v
python -m backend.seed_database  (optional)
        |
        v
python -m backend.app
        |
        v
cd frontend
npm run dev
        |
        v
Open the frontend and submit academic inputs
```

## 20. Limitations

### 20.1 Dataset limitations

The system depends on the quality and representativeness of the 800-row CSV.
If the dataset does not represent future students, departments, grading
patterns, or academic years, prediction quality may decrease.

### 20.2 Prediction uncertainty

The regression MAE is approximately 4.08 marks. Individual predictions can be
wrong by more or less than that average. A prediction should therefore be
treated as an estimate and not a promise.

### 20.3 Historical classifier labels

The classifier is trained using labels derived from historical ETE and total
marks. It learns patterns in those records; it does not understand personal
circumstances or future changes in teaching and assessment.

### 20.4 No automatic retraining

New predictions are saved to MongoDB, but they are not automatically added to
the training CSV or used to retrain the model. The deployed models remain the
saved artifacts until `python main.py` is run again.

### 20.5 Shared collection

Seeded historical records and new prediction estimates are stored in the same
`predictions` collection. The `record_type` and `is_estimate` fields are
important for distinguishing them.

### 20.6 Security and privacy

The current project is suitable for a local or controlled educational
environment. A production deployment would additionally need:

- Authentication and authorization.
- Stronger secret management.
- HTTPS.
- Rate limiting.
- Audit logging.
- Data-retention rules.
- Access controls in MongoDB.
- Privacy review for student identifiers and academic information.

### 20.7 Fairness and responsible use

Model performance can differ across groups or circumstances not represented in
the dataset. The application should not be used to make high-stakes academic
decisions automatically.

## 21. Conclusion

The LPU Student Performance Prediction System is a complete machine-learning
application that connects data validation, model training, REST serving,
rule-based academic calculations, frontend presentation, and MongoDB
persistence.

The system uses 800 validated student records. It predicts ETE marks with a
linear regression model and provides a separate decision-tree Pass/Fail signal.
The current held-out evaluation reports an R² of approximately 0.671 and a
mean absolute error of approximately 4.08 marks. The classifier reports
95.625% accuracy and an F1 score of approximately 0.972 on its test split.

The project is especially useful because it does not show only a model output.
It also explains:

- The predicted ETE mark.
- The predicted total.
- The minimum ETE needed.
- The gap between predicted and required ETE.
- Whether passing remains mathematically possible.
- Whether the database save succeeded.

MongoDB stores both the imported CSV records and newly submitted predictions.
New records can be searched in MongoDB Compass using:

```javascript
{ "student_id": "YOUR_STUDENT_ID", "is_estimate": true }
```

The most important interpretation is that the application is a planning aid.
Its predictions are estimates based on historical data, while the configured
passing rules provide transparent mathematical guidance. Official academic
results must always come from the institution's authoritative assessment
system.
