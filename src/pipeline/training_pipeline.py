import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Main script to execute data ingestion, transformation, and model training
if __name__ == '__main__':
    try:
        # Step 1: Data Ingestion
        logging.info('Starting the data ingestion process.')
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        logging.info('Starting the data transformation process.')
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Step 3: Model Training
        logging.info('Starting the model training process.')
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)

        logging.info('Model training completed successfully.')
        
    except Exception as e:
        logging.error('An error occurred during model training.')
        raise CustomException(e, sys)
