import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ## Handels missing values in df
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessing.pkl')

class DataTransformation:
    """ This class is responsible for data transformation based on different types of data """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            # Updated column names
            numerical_columns = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
            categorical_columns = ['seller_type', 'fuel_type', 'transmission_type']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Encoding and scaling completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error(f"Error while getting transformer object: {e}")
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading train and test data')

            logging.info(f"Train DataFrame columns: {train_df.columns.tolist()}")
            logging.info(f"Test DataFrame columns: {test_df.columns.tolist()}")

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_transformer_object()

            target_column_name = "selling_price"

            if target_column_name not in train_df.columns:
                logging.error(f"Target column '{target_column_name}' is missing from training DataFrame.")
            if target_column_name not in test_df.columns:
                logging.error(f"Target column '{target_column_name}' is missing from testing DataFrame.")

            # Train Data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_data = train_df[target_column_name]
            # Test Data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_data = test_df[target_column_name]

            logging.info("Applying preprocessing on training dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("Preprocessing on training dataframe complete.")

            logging.info("Applying preprocessing on testing dataframe")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Preprocessing on testing dataframe complete.")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_data)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_data)
            ]

            logging.info('Saving Processing Object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f'Preprocessing object saved successfully to {self.data_transformation_config.preprocessor_obj_file_path}')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys) from e
#