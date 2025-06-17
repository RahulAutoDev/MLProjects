import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys.exc_info())

def evaluate_models(X_train, y_train, X_test,y_test, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score
            logging.info(f"{name} - Train Score: {train_model_score}, Test Score: {test_model_score}")
            logging.info("Model evaluation completed successfully.")

        return report
    
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys.exc_info())
   


