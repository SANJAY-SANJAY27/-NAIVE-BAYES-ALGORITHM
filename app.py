import streamlit as st
import numpy as np
import pickle

st.title("Naive Bayes - Diabetes Prediction")

# Load model
with open("naive_bayes_diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

st.write("Enter patient details")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 33)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("❌ Diabetic")
    else:
        st.success("✅ Not Diabetic")
