import streamlit as st
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load Trained Model
model = joblib.load("fraud_model.pkl")

# Sidebar
with st.sidebar:
    st.header("📊 Project Information")
    st.write("**Model:** Random Forest")
    st.write("**Accuracy:** 99.81%")
    st.write("**Fraud Recall:** 88%")
    st.write("**Dataset:** Credit Card Transactions")

# Main Title
st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
This application uses a Random Forest Machine Learning model
to predict whether a credit card transaction is fraudulent or genuine.
""")

st.divider()

# Input Section
st.subheader("Enter Transaction Details")

features = []

col1, col2 = st.columns(2)

for i in range(30):
    if i % 2 == 0:
        with col1:
            value = st.number_input(
                f"Feature {i+1}",
                value=0.0,
                format="%.6f"
            )
    else:
        with col2:
            value = st.number_input(
                f"Feature {i+1}",
                value=0.0,
                format="%.6f"
            )

    features.append(value)

st.divider()

# Prediction
if st.button("🔍 Predict Transaction"):

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]

    fraud_prob = probability[1] * 100
    genuine_prob = probability[0] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{fraud_prob:.2f}%"
        )

    with col2:
        st.metric(
            "Genuine Probability",
            f"{genuine_prob:.2f}%"
        )

    st.progress(int(max(fraud_prob, genuine_prob)))

st.divider()

st.caption(
    "Credit Card Fraud Detection Project using Random Forest Classifier"
)