'''
a web app for personal Loan predictor
Features Required : [Income,Family,CCAvg,Education,Mortgage]
'''
#------------------------------------#
#       LIBRARIES
#------------------------------------#
import joblib
import pandas as pd
import streamlit as st
import sys
from src.exception import CustomException
from src.logger import logger



# prediction input
def predict_inputs(transformed_input):

    pass




# input transformation
def transformation_input(income_f1, family_f2, education_f4, mortgage_f5, ccavg_f3):
    # load the column transformer
    transformer = joblib.load("models/column_transformer.pkl")

    # Inputs dictionary
    inputs_raw = {
        'Income': income_f1,
        'Family': family_f2,
        'CCAvg': ccavg_f3,
        'Education': education_f4,
        'Mortgage': mortgage_f5
    }
    inputs_df = pd.DataFrame(inputs_raw, index=[0])

    # Perform transformation
    new_inputs = transformer.transform(inputs_df)

    # Fix column names
    try:
        # Get scaled feature names from the StandardScaler part
        scaled_names = transformer.named_transformers_['Numbers'].get_feature_names_out(['Income', 'CCAvg', 'Mortgage'])
    except:
        scaled_names = ['Income', 'CCAvg', 'Mortgage']  # fallback in case method is unavailable

    passthrough_cols = ['Family', 'Education']  # columns that weren't transformed
    all_feature_names = list(scaled_names) + passthrough_cols

    # Rebuild into DataFrame
    transformed_df = pd.DataFrame(new_inputs, columns=all_feature_names)

    return transformed_df






# input train


st.title("PERSONAL LOAN PREDICTOR",)
st.write("Chances that you will end up taking a Personal Loan !!.")

custom_inputs = []
# taking the inputs
with st.form(key='personal_loan_predictor'):

    # Income: Convert input to 'in thousands'
    raw_income = st.number_input('Income p.a (actual value, e.g., 2400 means $2400):\n', min_value=0.0, step=100.0)
    income = raw_income / 1000  # Convert to 'in thousands' format

    # Family: Categorical, integers only, 0 to 100
    family = st.number_input('Family Size (0–100):', min_value=0, max_value=100, step=1, format="%d")

    # Education: Categorical dropdown with 1, 2, 3, or 'Other'
    education_map = {
        "Undergrad (1)": 1,
        "Graduate (2)": 2,
        "Advanced/Professional (3)": 3,
        "Other": 0
    }
    education_choice = st.selectbox('Education:', list(education_map.keys()))
    education = education_map[education_choice]

    # Mortgage: Integer 0 to 1000
    mortgage = st.number_input('Mortgage Amount (0–1000):', min_value=0, max_value=1000, step=1, format="%d")

    # CCAvg: Float between 0.0 and 10.0
    ccavg = st.slider('CCAvg (Average Credit Card Spend, in $1000s):', 0.0, 10.0, step=0.1)

    # Submit button
    submit = st.form_submit_button(label='Capture Input')

    # Optional: Display processed values after submit
    if submit:
        st.success("Inputs captured successfully!")
        st.write(f"Income (in $1000s): {income}")
        st.write(f"Family Size: {family}")
        st.write(f"Education Category: {education}")
        st.write(f"Mortgage: {mortgage}")
        st.write(f"CCAvg: {ccavg}")
        # transform the input
        custom_inputs = transformation_input(income, family, education, mortgage, ccavg)


# print results for the custom input
st.write(custom_inputs)
