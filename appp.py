import streamlit as st
import pandas as pd
import numpy as np
import os
import dill
import logging

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
# save runtime logs to app.log
# helps debug predictions issues in production
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------
#improve app performance
@st.cache_resource
def load_object(file_path):
    with open(file_path, "rb") as f:
        return dill.load(f)

@st.cache_resource
def load_artifacts():
    model = load_object("artifacts/model.pkl")
    preprocessor = load_object("artifacts/preprocessor.pkl")
    return model, preprocessor

# ----------------------------------------------------
# Streamlit Config
# ----------------------------------------------------

# set browser tab title 
# wide layout for betetr ui
# app heading
st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("📘 Student Math Score Prediction")

# ----------------------------------------------------
# Load Model & Preprocessor
# ----------------------------------------------------

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load model or preprocessor")
    logging.error(e)
    st.stop()

# ----------------------------------------------------
# Single Prediction
# ----------------------------------------------------

st.subheader("🔢 Single Student Prediction")

with st.form("single_input"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race = st.selectbox(
            "Race/Ethnicity",
            ["group A", "group B", "group C", "group D", "group E"]
        )
        parent_edu = st.selectbox(
            "Parental Education",
            [
                "some high school", "high school", "some college",
                "associate's degree", "bachelor's degree", "master's degree"
            ]
        )
        lunch = st.selectbox("Lunch", ["standard", "free/reduced"])

    with col2:
        test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
        reading_score = st.slider("Reading Score", 0, 100, 70)
        writing_score = st.slider("Writing Score", 0, 100, 70)

    predict_btn = st.form_submit_button("Predict Math Score")

if predict_btn:
    try:
        input_df = pd.DataFrame({
            "gender": [gender],
            "race_ethnicity": [race],
            "parental_level_of_education": [parent_edu],
            "lunch": [lunch],
            "test_preparation_course": [test_prep],
            "reading_score": [reading_score],
            "writing_score": [writing_score]
        })

        transformed_data = preprocessor.transform(input_df)
        prediction = model.predict(transformed_data)

        st.success(f"🎯 Predicted Math Score: **{prediction[0]:.2f}**")
        logging.info(f"Single prediction: {prediction[0]}")

    except Exception as e:
        st.error("❌ Prediction failed")
        logging.error(e)

# ----------------------------------------------------
# Batch CSV Prediction
# ----------------------------------------------------
st.divider()
st.subheader("📂 Batch CSV Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)

        transformed_batch = preprocessor.transform(batch_df)
        batch_preds = model.predict(transformed_batch)

        batch_df["predicted_math_score"] = batch_preds

        st.dataframe(batch_df.head())

        st.download_button(
            label="⬇️ Download Predictions",
            data=batch_df.to_csv(index=False),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )

        logging.info("Batch prediction completed")

    except Exception as e:
        st.error("❌ Batch prediction failed")
        logging.error(e)
st.divider()
st.caption("Built by Sameer Gandhi | ML • Streamlit • Docker")
