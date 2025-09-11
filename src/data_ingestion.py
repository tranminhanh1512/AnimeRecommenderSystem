import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_names = self.config["bucket_file_name"]
    
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"DataIngestion initialized with bucket: {self.bucket_name} and files: {self.bucket_file_names}")
    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.bucket_file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    data = pd.read_csv(file_path, nrows = 10000000)
                    data.to_csv(file_path, index=False)
                    logger.info("Large file detected. Downloaded first 5 million rows of animelist.csv")
                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    logger.info(f"Downloaded {file_name} to {file_path}")
        except Exception as e:
            logger.error(f"Error downloading files from GCP: {e}")
            raise CustomException("Failed to download files from GCP", e)
    
    def run(self):
        try:
            logger.info("Starting data ingestion process")
            self.download_csv_from_gcp()
            logger.info("Data ingestion process completed successfully")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        finally:
            logger.info("Data ingestion process finished")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()