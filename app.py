import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

# Load model & scaler
model = pickle.load(open("credit_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.write("Enter transaction details or load a sample transaction:")

n_features = 30  # Time + V1..V28 + Amount

# Default empty inputs
inputs = [0.0] * n_features

# --- New: Load Sample Transaction ---
if st.button("🔄 Load Sample Fraud Transaction"):
    # Example suspicious values (you can adjust based on dataset stats)
    sample = np.array([
        100000,   # Time
        -3.5, 2.0, -4.0, 1.5, 0.2, -2.5,  # V1..V6
        1.7, -1.2, 0.9, -3.0, 2.8, -2.9,  # V7..V12
        1.2, -2.1, 0.5, -0.7, 2.4, -1.5,  # V13..V18
        0.8, -3.2, 1.1, 0.4, -2.6, 1.9,   # V19..V24
        -1.4, 2.7, -0.5, 0.6,             # V25..V28
        5000,    # Amount
    ])
    inputs = sample.tolist()
    st.success("Loaded sample fraud transaction ✅")

elif st.button("🔄 Load Sample Legit Transaction"):
    sample = np.array([
        20000,   # Time
        0.2, -0.1, 0.3, -0.2, 0.1, 0.0,   # V1..V6
        0.4, -0.2, 0.1, 0.3, -0.1, 0.2,   # V7..V12
        0.0, -0.3, 0.1, 0.2, -0.1, 0.1,   # V13..V18
        0.2, -0.1, 0.1, 0.0, 0.1, -0.2,   # V19..V24
        0.3, -0.1, 0.0, 0.1,              # V25..V28
        50,     # Amount
    ])
    inputs = sample.tolist()
    st.success("Loaded sample legit transaction ✅")

# Input fields
for i in range(n_features):
    inputs[i] = st.number_input(f"Feature {i+1}", value=float(inputs[i]))

# Prediction
if st.button("Predict"):
    X_input = scaler.transform([inputs])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected (Prob: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Prob Fraud: {prob:.2f})")
