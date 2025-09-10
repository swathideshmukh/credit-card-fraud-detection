import streamlit as st
import pandas as pd
import pickle

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details to check fraud probability:")

# Load trained model & scaler
model = pickle.load(open("credit_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

inputs = []
for i in range(30):   # dataset has V1-V28 + Amount + Time
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    X_input = scaler.transform([inputs])
    pred = model.predict(X_input)[0]
    if pred == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
