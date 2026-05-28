import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
hypertension_choice = st.sidebar.selectbox("History of Hypertension?", ["No", "Yes"])
heart_disease_choice = st.sidebar.selectbox("History of Heart Disease?", ["No", "Yes"])
smoking_history = st.sidebar.selectbox("Smoking Profile", ["Never", "Current", "Former", "No Info", "not current", "ever"])
bmi = st.sidebar.slider("Body Mass Index (BMI Value)", 10.0, 60.0, 26.5, step=0.1)
HbA1c_level = st.sidebar.slider("Glycated Hemoglobin level (HbA1c %)", 3.5, 9.0, 5.5, step=0.1)
blood_glucose_level = st.sidebar.slider("Fasting/Random Glucose (mg/dL)", 60, 300, 130)

# Convert UI selections to match numerical metrics
hypertension = 1 if hypertension_choice == "Yes" else 0
heart_disease = 1 if heart_disease_choice == "Yes" else 0

# Calculate internal risk metric score safely matching training calculations
high_bmi_flag = 1 if bmi >= 25 else 0
senior_flag = 1 if age >= 50 else 0
metabolic_risk_score = hypertension + heart_disease + high_bmi_flag + senior_flag
clean_smoking = 'Unknown' if smoking_history == 'No Info' else smoking_history

# -------------------------------------------------------------
# SCHEMA STRATEGY: Build exact structural versions for fallback
# -------------------------------------------------------------

# Version A: Completely Lowercase Layout
df_lowercase = pd.DataFrame([{
    'gender': gender,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'smoking_history': clean_smoking,
    'bmi': bmi,
    'hbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level,
    'metabolic_risk_score': metabolic_risk_score
}])

# Version B: Standard Capitalized Layout
df_capitalized = pd.DataFrame([{
    'gender': gender,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'smoking_history': clean_smoking,
    'bmi': bmi,
    'HbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level,
    'metabolic_risk_score': metabolic_risk_score
}])

# Run Operational Pipeline Elements on Predict Request
if st.button("Calculate Metabolic Risk Profile"):
    processed_input = None
    
    # Execution Attempt 1: Try processing with the Lowercase Layout
    try:
        processed_input = preprocessor.transform(df_lowercase)
        input_data_used = df_lowercase
    except Exception:
        # Execution Attempt 2: Fallback cleanly to Capitalized Layout if version A rejects capitalization
        try:
            processed_input = preprocessor.transform(df_capitalized)
            input_data_used = df_capitalized
        except Exception as final_trans_error:
            st.error(f"Execution Error during pipeline array transformation: {final_trans_error}")
            st.info("Tip: Ensure your training notebook column names explicitly match: gender, age, hypertension, heart_disease, smoking_history, bmi, hbA1c_level, blood_glucose_level, metabolic_risk_score")

    # If transformation succeeds, run classification
    if processed_input is not None:
        try:
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
        except Exception as pred_error:
            st.error(f"Prediction execution failed: {pred_error}")
