import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Page Architecture
st.set_page_config(page_title="Metabolic Risk Analytics App", layout="centered")

st.title("🧬 Predictive Machine Learning Dashboard")
st.markdown("### Clinical Diabetes & Metabolic Risk Profiling System")
st.write("This application implements an optimized Random Forest Pipeline to calculate a patient's personalized probability of diabetic classification status.")

# Load saved pipeline components safely
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error(f"Error loading files: {e}")

# UI Inputs for Patient Diagnostic Attributes
st.sidebar.header("Patient Diagnostic Entry")
gender = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
age = st.sidebar.slider("Age (Years)", 1, 100, 45)
hypertension = st.sidebar.selectbox("History of Hypertension?", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("History of Heart Disease?", ["No", "Yes"])
smoking_history = st.sidebar.selectbox("Smoking Profile", ["Never", "Current", "Former", "No Info", "not current", "ever"])
bmi = st.sidebar.slider("Body Mass Index (BMI Value)", 10.0, 60.0, 26.5, step=0.1)
HbA1c_level = st.sidebar.slider("Glycated Hemoglobin level (HbA1c %)", 3.5, 9.0, 5.5, step=0.1)
blood_glucose_level = st.sidebar.slider("Fasting/Random Glucose (mg/dL)", 60, 300, 130)

# Build a comprehensive dictionary map covering both lowercase and uppercase variations
raw_inputs = {
    'gender': gender, 'Gender': gender,
    'age': age, 'Age': age,
    'hypertension': 1 if hypertension == "Yes" else 0, 'Hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0, 'Heart_disease': 1 if heart_disease == "Yes" else 0, 'Heart_Disease': 1 if heart_disease == "Yes" else 0,
    'smoking_history': 'Unknown' if smoking_history == 'No Info' else smoking_history, 'Smoking_history': 'Unknown' if smoking_history == 'No Info' else smoking_history, 'Smoking_History': 'Unknown' if smoking_history == 'No Info' else smoking_history,
    'bmi': bmi, 'BMI': bmi,
    'hbA1c_level': HbA1c_level, 'HbA1c_level': HbA1c_level, 'Hba1c_level': HbA1c_level, 'HbA1c': HbA1c_level,
    'blood_glucose_level': blood_glucose_level, 'Blood_glucose_level': blood_glucose_level, 'Blood_Glucose_Level': blood_glucose_level
}

# DYNAMIC MATCHING: Extract the exact feature names the preprocessor expects
try:
    if hasattr(preprocessor, 'feature_names_in_'):
        expected_features = preprocessor.feature_names_in_
    elif hasattr(preprocessor, 'get_feature_names_out'):
        expected_features = preprocessor.get_feature_names_out()
    else:
        # Fallback to standard lowercase format if features aren't exposed
        expected_features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'hbA1c_level', 'blood_glucose_level']
except Exception:
    expected_features = ['gender', 'age', 'hyheading', 'heart_disease', 'smoking_history', 'bmi', 'hbA1c_level', 'blood_glucose_level']

# Construct the input row strictly using the requested column names and ordering
ordered_inputs = {}
for col in expected_features:
    if col in raw_inputs:
        ordered_inputs[col] = raw_inputs[col]
    else:
        # Dynamic fallback if there's a highly specific feature engineered column in the training set
        ordered_inputs[col] = 0

input_data = pd.DataFrame([ordered_inputs])

# Calculate internal risk metric score safely
high_bmi_flag = 1 if bmi >= 25 else 0
senior_flag = 1 if age >= 50 else 0
metabolic_risk_score = (ordered_inputs.get('hypertension', 0) or ordered_inputs.get('Hypertension', 0)) + \
                       (ordered_inputs.get('heart_disease', 0) or ordered_inputs.get('Heart_disease', 0) or ordered_inputs.get('Heart_Disease', 0)) + \
                       high_bmi_flag + senior_flag

# Run Operational Pipeline Elements on Predict Request
if st.button("Calculate Metabolic Risk Profile"):
    # Apply standard saved pipeline data processing transformation matrix
    processed_input = preprocessor.transform(input_data)
    
    # Run classification inferences
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]
    
    st.markdown("---")
    st.markdown("### Clinical Inferences")
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("Classification Target: **HIGH DIABETIC RISK**")
        else:
            st.success("Classification Target: **LOW/NORMAL RISK**")
            
    with col2:
        st.metric(label="Evaluated Probability Matrix Score", value=f"{probability * 100:.2f}%")
        
    st.info(f"**Engineered Metabolic Risk Index Score:** {metabolic_risk_score} / 4 points (Based on systemic tracking variables).")
