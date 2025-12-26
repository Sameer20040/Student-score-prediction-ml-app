import os
import sys
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")

            params = param.get(model_name, {})

            # Skip GridSearch for CatBoost
            if model_name == "CatBoost Regressor":
                model.fit(X_train, y_train)
                best_model = model

            elif params:
                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score
            trained_models[model_name] = best_model

            logger.info(f"{model_name} R2 score: {score}")

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
