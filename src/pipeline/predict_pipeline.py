import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# THis class is responsible for loading trained artifacts and making predictions
# NO initialization needed
# exists for future scalability
class PredictPipeline:
    def __init__(self):
        pass
# features is panda dataframe and output is model prediction
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
# preprocessing pipeline
# ENsures same preprocessing as training
            # Load model & preprocessor
            # load serialized object
            # prevents data leakage and mismatch
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform input features
            #encodes categorucal variables
            # scales numerical values
            # converts dataframe into numpy array
            # we have to use only transform here as training is already done
            data_scaled = preprocessor.transform(features)

            # 🔥 MAKE PREDICTION (THIS WAS MISSING)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)# prevents app crash

# These values comes from flask form,streamlit ui and api request
class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ): # stores input as variables
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
# maintains feature order
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
