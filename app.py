import streamlit as st
import numpy as np
import pickle

# Load saved model, scaler, pca
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))

st.title("❤️ Heart Disease Risk Assessment")
st.write("Enter the patient details below to predict heart disease risk.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST (0-2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thal (1-3)", options=[1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: This patient is likely to have Heart Disease.")
    else:
        st.success("✅ Low Risk: This patient is unlikely to have Heart Disease.")