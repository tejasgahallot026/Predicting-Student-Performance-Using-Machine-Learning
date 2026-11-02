import re

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, r2_score

import app as webapp
from src.pipeline.predict_pipeline import PredictPipeline

FORM = {
    "gender": "female",
    "ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "none",
    "reading_score": "90",
    "writing_score": "40",
}


@pytest.fixture
def client():
    return webapp.app.test_client()


def test_committed_model_accuracy_on_held_out_test_set():
    test = pd.read_csv("artifacts/test.csv")
    pipeline = PredictPipeline()
    preds = pipeline.predict(test.drop(columns=["math_score"]))
    assert r2_score(test["math_score"], preds) > 0.85
    assert mean_absolute_error(test["math_score"], preds) < 5


def test_form_maps_reading_and_writing_scores_correctly(client, monkeypatch):
    """Regression test: the old app passed reading as writing and writing as reading."""
    seen = {}

    def fake_predict(df):
        seen["row"] = df.iloc[0].to_dict()
        return [70.0]

    monkeypatch.setattr(webapp.pipeline, "predict", fake_predict)
    assert client.post("/predictdata", data=FORM).status_code == 200
    assert seen["row"]["reading_score"] == 90.0
    assert seen["row"]["writing_score"] == 40.0
    assert seen["row"]["race_ethnicity"] == "group B"


def test_end_to_end_prediction(client):
    resp = client.post("/predictdata", data=FORM)
    assert resp.status_code == 200
    assert b"The predicted maths score is" in resp.data


def test_rejects_out_of_range_and_missing_input(client):
    assert client.post("/predictdata", data={**FORM, "reading_score": "150"}).status_code == 400
    assert client.post("/predictdata", data={**FORM, "writing_score": "abc"}).status_code == 400
    assert client.post("/predictdata", data={**FORM, "lunch": ""}).status_code == 400


def test_form_page_shows_no_result_before_submitting(client):
    resp = client.get("/predictdata")
    assert resp.status_code == 200 and b"The predicted maths score is" not in resp.data


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok", "model": "LinearRegression"}


def test_form_labels_match_input_names(client):
    """Regression: the template labelled the reading_score box 'Writing Score' and vice versa."""
    html = client.get("/predictdata").data.decode()
    assert re.search(r'for="reading-score"[^>]*>\s*Reading Score', html)
    assert re.search(r'for="writing-score"[^>]*>\s*Writing Score', html)
