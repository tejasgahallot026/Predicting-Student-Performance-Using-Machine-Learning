import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))


class PredictPipeline:
    """Loads the trained preprocessor and model once, then reuses them for every prediction."""

    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        try:
            self.model = load_object(file_path=os.path.join(artifacts_dir, "model.pkl"))
            self.preprocessor = load_object(file_path=os.path.join(artifacts_dir, "preprocessor.pkl"))
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            return self.model.predict(self.preprocessor.transform(features))
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """One student's details from the web form, in the column layout the model was trained on."""

    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str,
                 test_preparation_course: str, reading_score: float, writing_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            })
        except Exception as e:
            raise CustomException(e, sys)
