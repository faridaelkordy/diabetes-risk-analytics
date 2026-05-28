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

# Demographic Backfills required by the current preprocessor artifact
st.sidebar.markdown("---")
st.sidebar.subheader("Demographic Alignments")
race_selection = st.sidebar.selectbox("Demographic Cohort Group", ["Caucasian", "Asian", "AfricanAmerican", "Hispanic", "Other"])
location_val = st.sidebar.text_input("Regional Evaluation Hub Location", "Downtown")
year_val = st.sidebar.number_input("Analysis Year Record", value=2026)

# Convert UI selections to match numerical metrics
hypertension = 1 if hypertension_choice == "Yes" else 0
heart_disease = 1 if heart_disease_choice == "Yes" else 0

# Calculate internal risk metric score safely matching training calculations
high_bmi_flag = 1 if bmi >= 25 else 0
senior_flag = 1 if age >= 50 else 0
metabolic_risk_score = hypertension + heart_disease + high_bmi_flag + senior_flag
clean_smoking = 'Unknown' if smoking_history == 'No Info' else smoking_history

# Construct the row explicitly satisfying both schemas
input_row = {
    'gender': gender,
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'smoking_history': clean_smoking,
    'bmi': bmi,
    'hbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level,
    'metabolic_risk_score': metabolic_risk_score,
    
    # Injected fields to eliminate the missing column array errors
    'location': location_val,
    'year': year_val,
    'race:Asian': 1 if race_selection == "Asian" else 0,
    'race:Caucasian': 1 if race_selection == "Caucasian" else 0,
    'race:AfricanAmerican': 1 if race_selection == "AfricanAmerican" else 0,
    'race:Hispanic': 1 if race_selection == "Hispanic" else 0,
    'race:Other': 1 if race_selection == "Other" else 0
}

# -------------------------------------------------------------
# AUTOMATED STRUCTURAL ALIGNMENT
# -------------------------------------------------------------
base_df = pd.DataFrame([input_row])

if 'preprocessor' in locals() or 'preprocessor' in globals():
    try:
        if hasattr(preprocessor, 'feature_names_in_'):
            expected_cols = list(preprocessor.feature_names_in_)
            aligned_row = {}
            for col in expected_cols:
                # Direct casing check
                match = [k for k in input_row.keys() if k.lower() == col.lower()]
                if match:
                    aligned_row[col] = input_row[match[0]]
                else:
                    # Avoid numeric zeros for string categories to prevent isnan crashes
                    aligned_row[col] = "Unknown"
            input_data = pd.DataFrame([aligned_row])[expected_cols]
        else:
            input_data = base_df
    except Exception:
        input_data = base_df
else:
    input_data = base_df

# Run Operational Pipeline Elements on Predict Request
if st.button("Calculate Metabolic Risk Profile"):
    try:
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
        
    except Exception as eval_error:
        st.error(f"Execution Error during pipeline array transformation: {eval_error}")
