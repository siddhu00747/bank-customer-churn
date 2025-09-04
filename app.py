import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# -----------------------------
# Load model and scaler
model = load_model('churn_model.h5')  # Save this model after training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -----------------------------
# Page Title
st.title("🔍 Customer Churn Prediction App")
st.write("Enter customer details below to check whether they will leave the bank or not.")

# -----------------------------
# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
tenure = st.number_input("Tenure (Years with bank)", min_value=0, max_value=10, step=1)
balance = st.number_input("Account Balance", min_value=0.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)

gender_male = st.selectbox("Gender", ["Male", "Female"]) == "Male"
geo_germany = st.selectbox("Geography", ["France", "Germany", "Spain"]) == "Germany"
geo_spain = st.selectbox("Again select Geography (for demo)", ["France", "Germany", "Spain"]) == "Spain"  # to generate both flags

# -----------------------------
# Prediction
if st.button("Predict"):
    # Encode binary fields
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
    gender_male = 1 if gender_male else 0
    geography_germany = 1 if geo_germany else 0
    geography_spain = 1 if geo_spain else 0

    # Arrange features in correct order (same as training)
    input_data = np.array([[credit_score, age, tenure, balance, num_of_products,
                            has_cr_card, is_active_member, estimated_salary,
                            geography_germany, geography_spain, gender_male]])

    input_data_scaled = scaler.transform(input_data)
    pred = model.predict(input_data_scaled)[0][0]

    if pred > 0.5:
        st.error(f"❌ Customer is likely to leave the bank. (Probability: {pred:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay. (Probability: {pred:.2f})")
