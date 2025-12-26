import streamlit as st
import pandas as pd
import numpy as np
import os
import dill # Requires 'pip install dill'

# Function to load objects using dill
def load_object(file_path):
    with open(file_path, "rb") as f:
        return dill.load(f)

# --- App Configuration ---
st.set_page_config(page_title="Student Score Predictor")
st.title("Math Score Prediction")

# Define paths to your artifacts
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

# --- User Input Form ---
with st.form("input_form"):
    st.subheader("Enter Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
        parent_edu = st.selectbox("Parental Education", [
            "some high school", "high school", "some college", 
            "associate's degree", "bachelor's degree", "master's degree"
        ])
        lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
        
    with col2:
        test_prep = st.selectbox("Test Prep Course", ["none", "completed"])
        reading_score = st.slider("Reading Score", 0, 100, 70)
        writing_score = st.slider("Writing Score", 0, 100, 70)

    submit = st.form_submit_button("Predict Score")

if submit:
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        try:
            # 1. Map inputs to a DataFrame
            input_data = pd.DataFrame({
                "gender": [gender],
                "race_ethnicity": [race],
                "parental_level_of_education": [parent_edu],
                "lunch": [lunch],
                "test_preparation_course": [test_prep],
                "reading_score": [reading_score],
                "writing_score": [writing_score]
            })

            # 2. Load the artifacts
            model = load_object(MODEL_PATH)
            preprocessor = load_object(PREPROCESSOR_PATH)

            # 3. Transform and Predict
            transformed_data = preprocessor.transform(input_data)
            prediction = model.predict(transformed_data)

            # 4. Show Result
            st.success(f"### Predicted Math Score: {np.round(prediction[0], 2)}")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Artifacts not found! Please check the 'artifacts/' folder.")