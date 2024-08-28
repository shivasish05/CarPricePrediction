import dill
import os
import sys
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Saving object to {file_path}")
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save object to {file_path}")
        raise CustomException(e, sys)
