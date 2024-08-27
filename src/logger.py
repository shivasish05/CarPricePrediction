import logging
import os
from datetime import datetime
import sys

# Define CustomException
class CustomException(Exception):
    def __init__(self, message, module):
        super().__init__(message)
        self.module = module

# Create a log directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Generate the log file path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s]%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    try:
        a = 1 / 0
    except ZeroDivisionError as e:
        raise CustomException("Divided By Zero", sys)
    
   