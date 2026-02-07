import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="VeerendraManikonda/predictive_maintenance", filename="best_predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Predictive Maintenance for Engine Health")
st.write("""
This application predicts the potential Machine Failures based on the pitch parameters.
Please enter the Machine Sensor details.
""")

# Input form
with st.form("prediction_form"):
    engine_rpm = st.number_input("Engine RPM", min_value=0.0, max_value=3000, value=700)
    lub_oil_pressure = st.number_input("Lub Oil Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=2.5)
    fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, max_value=30.0, value=12.0)
    coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=3.0)
    lub_oil_temperature = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, max_value=150.0, value=85.0)
    coolant_temperature = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=150.0, value=80.0)

    submit = st.form_submit_button("Predict")

if submit:
    # Convert inputs to DataFrame
    input_data = pd.DataFrame([{
        "engine_rpm": engine_rpm,
        "lub_oil_pressure": lub_oil_pressure,
        "fuel_pressure": fuel_pressure,
        "coolant_pressure": coolant_pressure,
        "lub_oil_temperature": lub_oil_temperature,
        "coolant_temperature": coolant_temperature
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Engine is likely to be failing")
    else:
        st.error("Engine is in good condition.")
