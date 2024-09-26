import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting data into features and target')
            # X_train, y_train = train_array
            # X_test, y_test = test_array
            # Assuming last column is the target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Logistic Regression Model
            model = LogisticRegression()

            logging.info('Training Logistic Regression Model')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            logging.info('Evaluating model performance')
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f'Accuracy: {accuracy}')
            print(classification_report(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info('Model training and saving completed')

        except Exception as e:
            logging.info('Exception during model training')
            raise CustomException(e, sys)
