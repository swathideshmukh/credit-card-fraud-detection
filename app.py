import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

# Load model, scaler, and sample transactions
model = pickle.load(open("credit_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

fraud_sample = np.load("fraud_sample.npy")
legit_sample = np.load("legit_sample.npy")

st.write("Enter transaction details or load a sample transaction:")

n_features = 5  # Time + V1..V28 + Amount
inputs = [0.0] * n_features

# Load real fraud/legit samples
if st.button("🔄 Load Real Fraud Transaction"):
    inputs = fraud_sample.tolist()
    st.warning("Loaded real fraud transaction from dataset ⚠️")

elif st.button("🔄 Load Real Legit Transaction"):
    inputs = legit_sample.tolist()
    st.success("Loaded real legit transaction from dataset ✅")

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
