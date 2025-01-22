import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('svm_model (1).pkl')
scaler = joblib.load('scaler.pkl')

# Function to make predictions
def predict_purchase(gender, age, salary):
    # Encode gender (0 for Female, 1 for Male)
    gender_encoded = [1, 0] if gender == "Male" else [0, 1]
    
    # Prepare input data
    input_data = gender_encoded + [age, salary]
    scaled_data = scaler.transform([input_data])  # Scale the input
    prediction = model.predict(scaled_data)  # Make prediction
    return "Purchased" if prediction[0] == 1 else "Not Purchased"


# Streamlit App
st.title("Purchase Prediction App")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=60, step=1) 
salary = st.number_input("Estimated Salary", min_value=15000, step=1000)

# Predict button
if st.button("Predict"):
    result = predict_purchase(gender, age, salary)
    st.success(f"Prediction: {result}")



