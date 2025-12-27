import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils import load_object


# ---------------- PATHS ----------------
ARTIFACTS_DIR = "artifacts"
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
DATA_PATH = os.path.join("notebook","data", "stud.csv")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "model_metrics.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["math_score"])
y = df["math_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------- PREPROCESS ----------------
preprocessor = load_object(PREPROCESSOR_PATH)

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# ---------------- MODELS ----------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}


# ---------------- EVALUATION ----------------
results = []

for model_name, model in models.items():
    model.fit(X_train_transformed, y_train)
    preds = model.predict(X_test_transformed)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append({
        "Model": model_name,
        "R2 Score": round(r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })


# ---------------- SAVE METRICS ----------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(METRICS_PATH, index=False)

print("✅ Model evaluation completed")
print(metrics_df)
