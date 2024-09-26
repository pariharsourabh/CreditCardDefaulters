import os
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save a Python object (like a trained model) to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object (like a trained model) from a file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple models, then return a report with their accuracy scores.
    """
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            # Train the model
            model.fit(X_train, y_train)

            # Predict on test data
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)
            confusion_mat = confusion_matrix(y_test, y_pred)

            logging.info(f"Model {model_name} - Accuracy: {accuracy}")
            logging.info(f"Model {model_name} - Classification Report: \n{classification_rep}")
            logging.info(f"Model {model_name} - Confusion Matrix: \n{confusion_mat}")

            # Save results in the report
            report[model_name] = accuracy

        return report

    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise CustomException(e, sys)
