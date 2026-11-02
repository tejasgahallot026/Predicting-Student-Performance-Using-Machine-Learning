# Student Performance Prediction

An end-to-end machine learning project that predicts a student's **maths score** from their background and their reading and writing scores, served through a Flask web form.

## Results

Evaluated on a held-out test set of 200 students (80/20 split of 1,000):

| Metric | Model (Linear Regression) | Baseline (always predict the average) |
|---|---|---|
| R² | **0.874** | about 0 |
| Mean absolute error | **4.37 marks** | 12.34 marks |

The training pipeline compares Linear Regression, XGBoost, CatBoost, AdaBoost and other regressors with grid-searched hyperparameters, and keeps the best by R² on the test split. Linear Regression won, which makes sense: reading and writing scores are strongly and almost linearly related to maths scores. These figures are checked automatically by the test suite (`tests/test_app.py`).

## How it works

1. **Ingestion** (`src/components/data_ingestion.py`) reads `Notebook/data/stud.csv` and writes the train/test split to `artifacts/`.
2. **Transformation** (`src/components/data_transformation.py`) scales the two numeric scores and one-hot encodes the five categorical fields (gender, ethnicity group, parental education, lunch type, test preparation), then saves the fitted preprocessor.
3. **Training** (`src/components/model_trainer.py`) runs the model comparison and saves the best model to `artifacts/model.pkl`.
4. **Serving** (`app.py`) loads the preprocessor and model once at startup. The form at `/predictdata` validates input (all fields required, scores 0 to 100) and returns the predicted maths score.

Exploratory analysis and model experiments are in `Notebook/`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python app.py              # http://localhost:5000/predictdata
pytest -v                  # tests, including the accuracy check
```

To retrain from scratch: `pip install -r requirements-train.txt`, then `python src/components/data_ingestion.py`.

## Tests

`tests/test_app.py` checks that the committed model still scores R² above 0.85 on the held-out set, that form fields reach the model in the right columns (a regression test for a bug where reading and writing scores were swapped), input validation, and the `/health` endpoint.

## Limitations

- The dataset is the public "Students Performance in Exams" dataset of 1,000 rows, which is widely described as generated teaching data rather than real student records, so the model shows the workflow rather than real-world predictive power.
- Predicting maths from reading and writing scores is close to predicting one exam from two others. The background fields add relatively little on their own.
