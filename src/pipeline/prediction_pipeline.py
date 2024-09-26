import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

# Class for running the prediction pipeline
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths for the preprocessor and model objects
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load the preprocessor and model
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions with the model
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

# Class for organizing custom input data into a DataFrame
class CustomData:
    def __init__(self,
                 limit_bal: float,
                 sex: str,
                 education: str,
                 marriage: str,
                 age: int,
                 pay_0: int,
                 pay_2: int,
                 pay_3: int,
                 pay_4: int,
                 pay_5: int,
                 pay_6: int,
                 bill_amt1: float,
                 bill_amt2: float,
                 bill_amt3: float,
                 bill_amt4: float,
                 bill_amt5: float,
                 bill_amt6: float,
                 pay_amt1: float,
                 pay_amt2: float,
                 pay_amt3: float,
                 pay_amt4: float,
                 pay_amt5: float,
                 pay_amt6: float):

        self.limit_bal = limit_bal
        self.sex = sex
        self.education = education
        self.marriage = marriage
        self.age = age
        self.pay_0 = pay_0
        self.pay_2 = pay_2
        self.pay_3 = pay_3
        self.pay_4 = pay_4
        self.pay_5 = pay_5
        self.pay_6 = pay_6
        self.bill_amt1 = bill_amt1
        self.bill_amt2 = bill_amt2
        self.bill_amt3 = bill_amt3
        self.bill_amt4 = bill_amt4
        self.bill_amt5 = bill_amt5
        self.bill_amt6 = bill_amt6
        self.pay_amt1 = pay_amt1
        self.pay_amt2 = pay_amt2
        self.pay_amt3 = pay_amt3
        self.pay_amt4 = pay_amt4
        self.pay_amt5 = pay_amt5
        self.pay_amt6 = pay_amt6

    # Convert the collected data into a DataFrame
    def get_data_as_dataframe(self):
        try:
            # Create a dictionary of all the data
            custom_data_input_dict = {
                'LIMIT_BAL': [self.limit_bal],
                'SEX': [self.sex],
                'EDUCATION': [self.education],
                'MARRIAGE': [self.marriage],
                'AGE': [self.age],
                'PAY_0': [self.pay_0],
                'PAY_2': [self.pay_2],
                'PAY_3': [self.pay_3],
                'PAY_4': [self.pay_4],
                'PAY_5': [self.pay_5],
                'PAY_6': [self.pay_6],
                'BILL_AMT1': [self.bill_amt1],
                'BILL_AMT2': [self.bill_amt2],
                'BILL_AMT3': [self.bill_amt3],
                'BILL_AMT4': [self.bill_amt4],
                'BILL_AMT5': [self.bill_amt5],
                'BILL_AMT6': [self.bill_amt6],
                'PAY_AMT1': [self.pay_amt1],
                'PAY_AMT2': [self.pay_amt2],
                'PAY_AMT3': [self.pay_amt3],
                'PAY_AMT4': [self.pay_amt4],
                'PAY_AMT5': [self.pay_amt5],
                'PAY_AMT6': [self.pay_amt6]
            }
            
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)
