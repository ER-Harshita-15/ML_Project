#common functionalities that our project will be using:
import os
import sys

import numpy as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the performance of different regression models and return the best model based on R2 score.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target variable
    - X_test: Testing features
    - y_test: Testing target variable
    - models: Dictionary of models to evaluate
    
    Returns:
    - model_report: Dictionary containing R2 scores for each model
    """
    
    try:
        report = {}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)