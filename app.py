import streamlit as st
import joblib
import numpy as np
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Risk Intelligence - Pro Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

# ---------------- BASE PATTERNS ----------------
BASE_NORMAL = np.array([
    0.0, 0.0, 0.1, -0.1, 0.0,
    0.05, -0.05, 0.1, -0.1, 0.0,
    0.02, -0.02, 0.03, -0.03, 0.0,
    0.01, -0.01, 0.02, -0.02, 0.0,
    0.0, 0.01, -0.01, 0.02, -0.02,
    0.01, -0.01, 0.0, 0.01, -0.01
])

BASE_FRAUD = np.array([
    406.0,
    -2.31222654,
    1.95199201,
    -1.60985073,
    3.99790559,
    -0.522187865,
    -1.42654532,
    -2.53738731,
    1.39165725,
    -2.77008928,
    -2.77227214,
    3.20203321,
    -2.89990739,
    -0.595221881,
    -4.28925378,
    0.38972412,
    -1.14074718,
    -2.83005567,
    -0.0168224682,
    0.416955705,
    0.126910559,
    0.517232371,
    -0.0350493686,
    -0.465211076,
    0.320198199,
    0.0445191675,
    0.177839798,
    0.261145003,
    -0.143275875,
    0.0
])

# ---------------- HEADER ----------------
st.title("🏦 Fraud Risk Intelligence System (Pro Version)")
st.caption("Interview-ready AI dashboard for credit card fraud detection")

st.write("APP STARTED SUCCESSFULLY 🚀")

st.divider()

# ---------------- SESSION STATE ----------------
if "features" not in st.session_state:
    st.session_state["features"] = BASE_NORMAL

# ---------------- SAMPLE BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🧪 Load Normal Transaction"):
        st.session_state["features"] = BASE_NORMAL

with col2:
    if st.button("🚨 Load Fraud Transaction"):
        st.session_state["features"] = BASE_FRAUD

# ---------------- INPUT SLIDERS ----------------
st.subheader("🎛️ Transaction Features (V1–V30)")

features = []
default_vals = st.session_state["features"]

with st.expander("Adjust Features", expanded=True):
    for i in range(30):
        features.append(
            st.slider(
                f"V{i+1}",
                -5.0, 5.0,
                float(default_vals[i]),
                0.1
            )
        )

features = np.array(features).reshape(1, -1)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🚀 Run Fraud Analysis"):

    st.subheader("📊 Prediction Result")

    prediction = model.predict(features)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0]
        fraud_prob = float(prob[1])
        legit_prob = float(prob[0])
    else:
        st.error("Model does not support probability prediction.")
        st.stop()

        # ---------------- RISK LOGIC ----------------
    if fraud_prob >= 0.65:
        risk = "🔴 HIGH RISK"
    elif fraud_prob >= 0.35:
        risk = "🟡 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    st.write("FINAL RISK:", risk)

    # ---------------- METRICS ----------------
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(
            "Fraud Probability",
            f"{fraud_prob:.2%}"
        )

    with colB:
        st.metric(
            "Legit Probability",
            f"{legit_prob:.2%}"
        )

    with colC:
        st.metric(
            "Risk Level",
            risk
        )

    # ---------------- STATUS CARD ----------------
    if fraud_prob >= 0.65:
        st.error("🚨 HIGH RISK TRANSACTION")
    elif fraud_prob >= 0.35:
        st.warning("⚠️ MEDIUM RISK TRANSACTION")
    else:
        st.success("✅ LOW RISK TRANSACTION")

    # ---------------- VISUALIZATION ----------------
    st.subheader("📈 Probability Breakdown")

    st.bar_chart({
        "Fraud": [fraud_prob],
        "Legit": [legit_prob]
    })

    st.progress(fraud_prob)

    # ---------------- MODEL INSIGHT ----------------
    st.subheader("🧠 Model Insight")

    if fraud_prob >= 0.7:
        st.write("""
        The transaction exhibits strong anomalous patterns.
        Multiple features significantly differ from normal
        transaction behavior, resulting in a high fraud score.
        """)

    elif fraud_prob >= 0.4:
        st.write("""
        The transaction contains moderately suspicious
        characteristics. Manual review is recommended.
        """)

    else:
        st.write("""
        The transaction appears consistent with normal
        transaction behavior and shows low fraud risk.
        """)

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built for Interviews | Fraud Detection ML System | Streamlit + Sklearn")