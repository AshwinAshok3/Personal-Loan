# Personal Loan Predictor Web App
# Required Features: 'Experience', 'Income', 'CCAvg', 'Education', 'Mortgage'

import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import sys
from src.exception import CustomException
from src.logger import logger
# import tensorflow as tf  # Uncomment if using NN

# Function to load and predict using models
def predict_inputs(transformed_input):
    base_dir = os.path.dirname(__file__)
    ml_model_path = os.path.join(base_dir, "models", "gradientboostC_model.pkl")

    if not os.path.exists(ml_model_path):
        st.error(f"️ML model not found at: {ml_model_path}")
        return

    try:
        ml_model = joblib.load(ml_model_path)
        input_reshaped = np.array(transformed_input).reshape(1, -1)
        ml_output = ml_model.predict(input_reshaped)
        ml_result = ml_output[0] if isinstance(ml_output, (list, np.ndarray)) else ml_output
        st.markdown("### Prediction Result")
        st.markdown(f"**Gradient Boosting Classifier:** `{ml_result}`")
        st.markdown('''
        If `0` means you are not going to end up with a `PERSONAL LOAN`,
        but `1` means you do....
        ''')
    except Exception as e:
        st.error("Prediction failed.")
        logger.error("Prediction error", exc_info=True)
        raise CustomException(sys, e)

# Function to transform inputs
def transformation_input(Xperience, Income, CCAvg, Education, Mortgage):
    try:
        transformer = joblib.load("models/column_transformer.pkl")
        inputs_df = pd.DataFrame([{
            'Experience': Xperience,
            'Income': Income,
            'CCAvg': CCAvg,
            'Education': Education,
            'Mortgage': Mortgage
        }])
        return transformer.transform(inputs_df)
    except Exception as e:
        st.error("Input transformation failed.")
        logger.error("Transformation error", exc_info=True)
        raise CustomException(sys, e)

# Initialize Streamlit app
st.title("PERSONAL LOAN PREDICTOR")
st.write("Predict your chances of taking a personal loan based on financial and demographic details.")

# Initialize session state
if 'transformed_input' not in st.session_state:
    st.session_state.transformed_input = None

# Input form
with st.form(key='personal_loan_form'):
    Xperience = st.number_input("Experience (Years):", min_value=-5.0, max_value=90.0, step=0.1)

    raw_income = st.number_input("Annual Income (e.g., enter 2400 for $2400):", min_value=0, step=100)
    income = raw_income / 1000

    ccavg = st.slider("CCAvg (Avg Credit Card Spend in $1000s):", 0.0, 10.0, step=0.1)

    education_map = {
        "Undergrad (1)": 1,
        "Graduate (2)": 2,
        "Advanced/Professional (3)": 3,
        "Other": 0
    }
    education_choice = st.selectbox("Education Level:", list(education_map.keys()))
    education = education_map[education_choice]

    mortgage = st.number_input("Mortgage Amount (0–1000):", min_value=0, max_value=1000, step=1, format="%d")

    submit = st.form_submit_button("Capture Input")

    if submit:
        try:
            transformed = transformation_input(Xperience, income, ccavg, education, mortgage)
            st.session_state.transformed_input = transformed
            st.success("Inputs captured and transformed!")
        except Exception:
            st.session_state.transformed_input = None

# Prediction
if st.button("Predict Input"):
    if st.session_state.transformed_input is None:
        st.warning("Please submit the form first.")
    else:
        predict_inputs(st.session_state.transformed_input)
