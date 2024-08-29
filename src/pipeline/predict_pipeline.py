import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessing.pkl'

            # Load the model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform the features using the preprocessor
            features = self.encode_features(features)
            data_scaled = preprocessor.transform(features)

            # Predict using the model
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def encode_features(self, df: pd.DataFrame):
        try:
            df['model'] = df['model'].astype('category').cat.codes
            return df
        except Exception as e:
            logging.error(f"Error during feature encoding: {e}")
            raise CustomException(e, sys)

class CustomData:
    """ Responsible for mapping all the input data to the expected format for prediction """

    def __init__(self,
                 model: str,
                 vehicle_age: int,
                 km_driven: float,
                 mileage: float,
                 engine: float,
                 max_power: float,
                 seats: int,
                 seller_type: str,
                 fuel_type: str,
                 transmission_type: str):
        self.model = model
        self.vehicle_age = vehicle_age
        self.km_driven = km_driven
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.seats = seats
        self.seller_type = seller_type
        self.fuel_type = fuel_type
        self.transmission_type = transmission_type

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "model": [self.model],
                "vehicle_age": [self.vehicle_age],
                "km_driven": [self.km_driven],
                "mileage": [self.mileage],
                "engine": [self.engine],
                "max_power": [self.max_power],
                "seats": [self.seats],
                "seller_type": [self.seller_type],
                "fuel_type": [self.fuel_type],
                "transmission_type": [self.transmission_type]
            }
            df = pd.DataFrame(custom_data_input_dict)
            return df
        except Exception as e:
            raise CustomException(e, sys)
