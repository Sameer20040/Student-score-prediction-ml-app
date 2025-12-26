import os 
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):  ## make config accessible inside the class
        self.ingestion_config = DataIngestionConfig()

    # This method executes the entire data ingestion pipeline
    def initiate_data_ingestion(self):  # this is the main function to be called for data ingestion
        logger.info("Entered the data ingestion method or component")  # Helps track execution flow in log files
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logger.info('Read the dataset as dataframe')

            ## Create the artifacts folder if not exists
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )  # avoids error if folder already exists

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logger.info("Train test split initiated")
            # random_state ensures same split every time
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            ## now datasets are reusable by other pipeline stages and components
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info("Ingestion of the data is completed")

            ## These paths can be passed to other components like data transformation and model training
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # Logs errors and raises custom exception
        # Wraps original exception and Adds system info (file name, line number)
        # This is production-grade error handling
        except Exception as e:
            logger.info("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)
