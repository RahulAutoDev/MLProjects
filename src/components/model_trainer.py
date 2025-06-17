import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np

#from xgboost import XGBClassifier, XGBRegressor
#from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'AdaBoost Regression': AdaBoostRegressor(),
                #'XGB Regressor': XGBRegressor(eval_metric='rmse'),
                #'CatBoost Regressor': CatBoostRegressor(verbose=0),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
            }

            logging.info("Model dictionary created with various regression models")
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,y_test= y_test, 
                                                models = models)
            
            best_model_Score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_Score)
                ]

            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_Score}")

            if best_model_Score < 0.6:
                raise CustomException("No best model found with sufficient score", sys.exc_info())
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")
            
            predicted = best_model.predict(X_test)

            r2_score_value = r2_score(y_test, predicted)
            mean_absolute_error_value = mean_absolute_error(y_test, predicted)
            mean_squared_error_value = mean_squared_error(y_test, predicted)
            #accuracy = accuracy_score(y_test, predicted) if hasattr(best_model, 'predict') else None
            #f1 = f1_score(y_test, predicted, average='weighted') if hasattr(best_model, 'predict') else None
            logging.info(f"Model evaluation metrics - R2: {r2_score_value}, MAE: {mean_absolute_error_value}, MSE: {mean_squared_error_value}")
            
            return r2_score_value,mean_absolute_error_value,mean_squared_error_value



        except Exception as e:
            raise CustomException(e, sys.exc_info())
