import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="centered"
)

# Train model directly
@st.cache_resource
def train_model():
    df = pd.read_csv('data/heart.csv')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    model = NearestCentroid()
    model.fit(X_pca, y)

    return model, scaler, pca

model, scaler, pca = train_model()

# Header
st.markdown("""
    <h1 style='text-align: center; color: #e63946;'>❤️ Heart Disease Risk Assessment</h1>
    <p style='text-align: center; color: gray;'>Enter patient clinical details to predict heart disease risk using Machine Learning</p>
    <hr>
""", unsafe_allow_html=True)

st.subheader("🧾 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST (0-2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thal (1-3)", options=[1, 2, 3])

st.markdown("<br>", unsafe_allow_html=True)

col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

if predict_btn:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Result")

    if prediction[0] == 1:
        st.markdown("""
            <div style='background-color:#ffe0e0; padding:20px; border-radius:10px; border-left: 6px solid #e63946;'>
                <h3 style='color:#e63946;'>⚠️ High Risk Detected</h3>
                <p style='color:#333;'>This patient is <b>likely to have Heart Disease</b>. Immediate medical consultation is recommended.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background-color:#e0ffe0; padding:20px; border-radius:10px; border-left: 6px solid #2dc653;'>
                <h3 style='color:#2dc653;'>✅ Low Risk</h3>
                <p style='color:#333;'>This patient is <b>unlikely to have Heart Disease</b>. Continue maintaining a healthy lifestyle.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⚠️ This tool is for assistive purposes only. Always consult a qualified medical professional.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Built by HEMANTH S | VTU - CSE (ISE) | ML Project</p>", unsafe_allow_html=True)