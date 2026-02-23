import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# -----------------------
# CUSTOM VLABS STYLE CSS
# -----------------------
st.markdown("""
<style>

/* Background */
html, body, [class*="css"] {
    background-color: #f7f7fb;
    font-family: 'Segoe UI', sans-serif;
}

/* Container spacing */
.main-container {
    margin-top: 80px;
}

/* Title */
.main-title {
    font-size: 52px;
    font-weight: 700;
    color: #1f1f1f;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-text {
    font-size: 20px;
    color: #6b6b6b;
    text-align: center;
    margin-bottom: 50px;
}

/* Upload Box */
.stFileUploader {
    background-color: #808080;   /* full grey */
    padding: 50px;
    border-radius: 18px;
    border: none;                /* remove dashed border */
    box-shadow: 0px 15px 40px rgba(0,0,0,0.06);
    width: 650px;
    margin: auto;
}

/* Hide default label */
.stFileUploader label {
    display: none;
}

/* Button */
.stButton > button {
    background: linear-gradient(#e5e7eb, #8e6cf4, #6c4cf1);
    color: white;
    border-radius: 10px;
    padding: 14px 28px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton > button:hover {
    opacity: 0.85;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(90deg, #8e6cf4, #6c4cf1);
    color: white;
    border-radius: 10px;
    padding: 14px 28px;
    font-weight: 600;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
model = joblib.load(model_path)

# -----------------------
# HEADER SECTION
# -----------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="main-title">Next Level Fraud Detection</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-text">Upload your transaction dataset to detect fraudulent activity instantly.</div>', unsafe_allow_html=True)

# -----------------------
# UPLOAD
# -----------------------
uploaded_file = st.file_uploader("", type=["csv"])

# -----------------------
# PROCESS DATA
# -----------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.markdown("### Preview of Uploaded Data")
    st.dataframe(data.head())

    if "Class" in data.columns:
        data_features = data.drop("Class", axis=1)
    else:
        data_features = data

    predictions = model.predict(data_features)
    probabilities = model.predict_proba(data_features)[:, 1]

    data["Fraud Prediction"] = predictions
    data["Risk Score"] = probabilities

    fraud_count = sum(predictions)
    total = len(predictions)

    st.markdown("### Detection Summary")
    st.write(f"Total Transactions: {total}")
    st.write(f"Fraud Detected: {fraud_count}")
    st.write(f"Fraud Percentage: {(fraud_count/total)*100:.2f}%")

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="fraud_results.csv",
        mime="text/csv"
    )

st.markdown('</div>', unsafe_allow_html=True)