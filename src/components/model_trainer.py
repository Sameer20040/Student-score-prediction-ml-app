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
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # To get the best model score from the dict
            best_model_score = max(sorted(model_report.values()))
            # To get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logger.info(f'Best model found: {best_model_name} with r2 score: {best_model_score}')

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logger.info("Evaluating best model on test data")
            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)


            logger.info(f'R2 score of best model on test data: {r2_square}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2_square

        except Exception as e:
            logger.info("Exception occurred at model training")
            raise CustomException(e,sys)