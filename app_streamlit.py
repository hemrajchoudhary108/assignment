import streamlit as st
import requests

# API URL (Ensure FastAPI is running)
API_URL = "http://127.0.0.1:8001/predict/"

# Streamlit UI Title
st.title("Enrollment Prediction System")

# Input Fields  
st.header("Enter the details below:")

# Collecting raw inputs from the user (original values)
age = st.number_input("Age", value=43.0)
salary = st.number_input("Salary", value=65000.0)
tenure_years = st.number_input("Tenure (Years)", value=4.0)

gender = st.selectbox("Gender", ["Female", "Male", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Widowed"])
employment_type = st.selectbox("Employment Type", ["Part-time", "Full-time", "Contract"])
region = st.selectbox("Region", ["West", "Midwest", "Northeast", "South"])
has_dependents = st.selectbox("Has Dependents", ["No", "Yes"])

# Submit button
if st.button("Predict Enrollment"):
    # Prepare original data to send
    original_data = {
        "age": age,
        "salary": salary,
        "tenure_years": tenure_years,
        "gender": gender,
        "marital_status": marital_status,
        "employment_type": employment_type,
        "region": region,
        "has_dependents": has_dependents
    }

    # Send request to FastAPI server
    response = requests.post(API_URL, json=original_data)

    if response.status_code == 200:
        prediction = response.json()["enrollment_prediction"]
        result_text = "✅ Enrolled" if prediction == 1 else "❌ Not Enrolled"
        st.success(f"Predicted Enrollment Status: {result_text}")
    else:
        st.error("Error in prediction. Please try again.")
