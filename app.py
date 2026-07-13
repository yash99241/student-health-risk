import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoder
import pickle

pipe = pickle.load(open("student_health_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
st.set_page_config(
    page_title="Student Health Risk Predictor",
    page_icon="🏥",
    layout="centered"
)
st.title("🏥 Student Health Risk Predictor")
st.write("Enter your health and lifestyle information to predict your health condition.")

# -------------------------
# User Inputs
# -------------------------

sleep_duration = st.number_input(
    "Sleep Duration (hours)",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=40,
    max_value=220,
    value=72
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=22.0
)

calorie_expenditure = st.number_input(
    "Calories Burned",
    min_value=0,
    value=500
)

step_count = st.number_input(
    "Step Count",
    min_value=0,
    value=8000
)

exercise_duration = st.number_input(
    "Exercise Duration (minutes)",
    min_value=0,
    value=45
)

water_intake = st.number_input(
    "Water Intake (Liters)",
    min_value=0.0,
    value=2.5
)

diet_type = st.selectbox(
    "Diet Type",
    ["Balanced", "High Protein", "Vegetarian", "Vegan", "Junk Food"]
)

stress_level = st.selectbox(
    "Stress Level",
    ["Low", "Medium", "High"]
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    ["Poor", "Average", "Good"]
)

physical_activity_level = st.selectbox(
    "Physical Activity Level",
    ["Sedentary", "Moderate", "Active"]
)

smoking_alcohol = st.selectbox(
    "Smoking / Alcohol",
    ["No", "Yes"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# -------------------------
# Feature Engineering
# -------------------------

calorie_per_step = calorie_expenditure / max(step_count, 1)

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Health Condition"):

    input_df = pd.DataFrame({
    "sleep_duration": [sleep_duration],
    "heart_rate": [heart_rate],
    "bmi": [bmi],
    "calorie_expenditure": [calorie_expenditure],
    "step_count": [step_count],
    "exercise_duration": [exercise_duration],
    "water_intake": [water_intake],
    "diet_type": [diet_type],
    "stress_level": [stress_level],
    "sleep_quality": [sleep_quality],
    "physical_activity_level": [physical_activity_level],
    "smoking_alcohol": [smoking_alcohol],
    "gender": [gender],
    "calorie_per_step": [calorie_per_step],
    "bmi_category": [bmi_category]
})
    prediction = pipe.predict(input_df)

    prediction = label_encoder.inverse_transform(prediction)

    st.success(f"Predicted Health Condition: {prediction[0]}")
