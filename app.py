import logging
import os

from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
log = logging.getLogger(__name__)

# Load the trained preprocessor and model once at startup, not on every request.
pipeline = PredictPipeline()

CATEGORICAL_FIELDS = ["gender", "ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
SCORE_ERROR = "Reading and writing scores must be numbers from 0 to 100."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok", "model": type(pipeline.model).__name__}


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    if any(not request.form.get(field) for field in CATEGORICAL_FIELDS):
        return render_template("home.html", error="Please fill in every field."), 400
    try:
        reading = float(request.form["reading_score"])
        writing = float(request.form["writing_score"])
    except (KeyError, ValueError):
        return render_template("home.html", error=SCORE_ERROR), 400
    if not (0 <= reading <= 100 and 0 <= writing <= 100):
        return render_template("home.html", error=SCORE_ERROR), 400

    data = CustomData(
        gender=request.form["gender"],
        race_ethnicity=request.form["ethnicity"],
        parental_level_of_education=request.form["parental_level_of_education"],
        lunch=request.form["lunch"],
        test_preparation_course=request.form["test_preparation_course"],
        reading_score=reading,
        writing_score=writing,
    )
    prediction = float(pipeline.predict(data.get_data_as_data_frame())[0])
    log.info("Predicted maths score %.1f", prediction)
    return render_template("home.html", results=round(prediction, 1))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
