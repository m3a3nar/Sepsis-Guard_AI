import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page Configuration & Title
st.title("🩺 SpaceTrack AI - Early Sepsis Detection")
st.write("A smart tool for early sepsis prediction and false alarm reduction in the ICU.")

st.sidebar.header("Current Patient Data")

# Input fields for patient features
hr = st.sidebar.slider("Heart Rate (HR)", 40, 200, 80)
o2sat = st.sidebar.slider("Oxygen Saturation (O2Sat)", 70, 100, 98)
temp = st.sidebar.slider("Temperature (Temp)", 35.0, 42.0, 37.0)
sbp = st.sidebar.slider("Systolic BP (SBP)", 70, 190, 120)
map_val = st.sidebar.slider("Mean Arterial BP (MAP)", 40, 140, 80)
dbp = st.sidebar.slider("Diastolic BP (DBP)", 40, 120, 70)
resp = st.sidebar.slider("Respiration Rate (Resp)", 10, 40, 18)
glucose = st.sidebar.number_input("Glucose", 50, 300, 100)
wbc = st.sidebar.number_input("White Blood Cells (WBC)", 1.0, 30.0, 7.0)
creatinine = st.sidebar.number_input("Creatinine", 0.1, 10.0, 1.0)
platelets = st.sidebar.number_input("Platelets", 10, 500, 200)
age = st.sidebar.number_input("Age", 18, 100, 50)

# Prediction Button
if st.button("Check Patient Status"):
    input_data = pd.DataFrame([[hr, o2sat, temp, sbp, map_val, dbp, resp, glucose, wbc, creatinine, platelets, age]],
                              columns=['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Glucose', 'WBC', 'Creatinine', 'Platelets', 'Age'])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error("Alert: High risk of Sepsis! Close monitoring required.")
    else:
        st.success(" Patient status is stable and vitals are within normal range.")

