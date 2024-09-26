import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')
            
            # Define columns for transformation
            categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
            numerical_cols = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                              'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
                              'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            # Define ordinal categories
            sex_cat = ['1','2']
            edu_cat = ['0','1','2','3','4','5','6']
            marriage_cat =['0','1','2','3']

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[sex_cat, edu_cat, marriage_cat])),
                ('scaler', StandardScaler())
            ])

            # Column transformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Error during data transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')

            # # Get preprocessor
            # preprocessing_obj = self.get_data_transformation_object()

            # target_column = 'default payment next month'

            # # Input and target separation
            # input_feature_train_df = train_df.drop(columns=[target_column])
            # target_feature_train_df = train_df[target_column]

            # input_feature_test_df = test_df.drop(columns=[target_column])
            # target_feature_test_df = test_df[target_column]

            # # Apply transformations
            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # logging.info("Transformation applied on training and testing datasets")

            # # Save preprocessing object
            # save_object(
            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj
            # )

            # return (
            #     input_feature_train_arr,
            #     input_feature_test_arr,
            #     target_feature_train_df,
            #     target_feature_test_df,
            #     self.data_transformation_config.preprocessor_obj_file_path
            # )
        

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'default payment next month'

            input_feature_train_df = train_df
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df
            target_feature_test_df=test_df[target_column_name]
            
            ## Transformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception during data transformation initiation")
            raise CustomException(e, sys)
