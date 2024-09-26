import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion')
        try:
            df = pd.read_csv('Notebooks/Data/default of credit card clients.csv')
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info('Train-test split')
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=10)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
  
        except Exception as e:
            logging.info('Exception occurred during data ingestion')
            raise CustomException(e, sys)


# from src.components.data_transformation import DataTransformation

# if __name__=='__main__':
#     obj=DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr, test_arr,_= data_transformation.initiate_data_transformation(train_data_path,test_data_path)


