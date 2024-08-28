import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # use to create class variable

##Transformation
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig



# Creating a path eg; 'artifacts/train.csv'.....
@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifact','train.csv')
    test_data_path : str = os.path.join('artifact','test.csv')
    raw_data_path : str = os.path.join('artifact','data.csv')


# Injecting data to tha path created above
class DataIngestion:
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion method')
        try:
            df = pd.read_csv('Notebook/cars_data.csv',index_col=0)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path,index = False, header=True)

            logging.info("Train test split initioated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False , header = True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
               raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))