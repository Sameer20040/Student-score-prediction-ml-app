import os 
import sys 
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logger

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self, train_array, test_array):
        
        try:
            logger.info("spliting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models={
                'Linear Regression':LinearRegression(),
                'Ridge Regression':Ridge(),
                'Lasso Regression':Lasso(),
                'KNN Regressor':KNeighborsRegressor(),
                'Decision Tree Regressor':DecisionTreeRegressor(),
                'Random Forest Regressor':RandomForestRegressor(),
                'SVR':SVR(),
                'XGB Regressor':XGBRegressor(),
                'CatBoost Regressor':CatBoostRegressor(verbose=False),
                'AdaBoost Regressor':AdaBoostRegressor()
            }
            
            
            params={
                "Decision Tree Regressor":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                "Linear Regression": {},

                "Ridge Regression": {
                    "alpha": [0.01, 0.1, 1, 10, 100]
                },

                "Lasso Regression": {
                    "alpha": [0.01, 0.1, 1, 10]
                },

                "KNN Regressor":{
                    'n_neighbors':[5,7,9,11]
                },
                "SVR": {
                    "kernel": ["rbf", "linear"],
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto"]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                },

                "XGB Regressor": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0]
                },

                "CatBoost Regressor": {
                    "iterations": [200, 500],
                    "depth": [4, 6],
                    "learning_rate": [0.05, 0.1]
                }
            }

            model_report,trained_models=evaluate_models(
                X_train,y_train,X_test,y_test,models,params
            )

            # To get the best model score from the dict
            # To get the best model name from the dict
            best_model_name = max(model_report,key=model_report.get)
            best_model=trained_models[best_model_name]
            best_model_score=model_report[best_model_name]
        
            logger.info(f'Best model found: {best_model_name} with r2 score: {best_model_score}')

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)

            y_pred=best_model.predict(X_test)
            final_r2=r2_score(y_test,y_pred)

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            return final_r2

        except Exception as e:
            logger.info("Exception occurred at model training")
            raise CustomException(e,sys)