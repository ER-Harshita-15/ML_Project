import sys
from dataclasses import dataclass     #Always remember why these dataclass(like DataTransformationConfig etc) are used. They are used to create classes that are mainly used to store data and have little functionality.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scr.logger import logging
from scr.exception import CustomerException
from scr.utils import save_object
import os

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
       

    def get_data_transformer_object(self):

        """"
        This function is responsible for data transformation

        """""
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns =['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            # Numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Fill missing values with median
                ('scaler', StandardScaler())  # Scale numerical features
            ]
            )
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing values with most frequent
                ('one_hot_encoder', OneHotEncoder()), # One-hot encode categorical features
                ('scaler', StandardScaler(with_mean=False))  # Scale categorical features
            ]
            
            )

            logging.info(f"Numerical columns: {numerical_columns} standard scaling completed.")
            logging.info(f"Categorical columns: {categorical_columns} standard scaling completed.")

            preprocessor= ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomerException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(
                f"Applying Preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr= np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr= np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )




        except Exception as e:
            raise CustomerException(e,sys)