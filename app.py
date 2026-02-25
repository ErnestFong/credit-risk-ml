import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

@st.cache_resource
def load_bundle(path="models/credit_risk_hgb_bundle.joblib"):
    return joblib.load(path)

bundle = load_bundle()
model = bundle["model"]
feature_names = bundle["feature_names"]

st.title("Credit Risk Predictor")
st.write("Enter borrower features to estimate probability of 2-year delinquency (dlq_2yrs).")

# Build a simple form so it doesn't predict on every small change
with st.form("risk_form"):
    inputs = {}
    for col in feature_names:
        inputs[col] = st.number_input(col, value=0.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    X_new = pd.DataFrame([inputs], columns=feature_names)  # enforce column order
    proba = model.predict_proba(X_new)[0][1]
    st.metric("Predicted probability of delinquency", f"{proba:.2%}")

    # Simple label (you can change thresholds later)
    if proba >= 0.7:
        st.error("Risk label: High")
    elif proba >= 0.4:
        st.warning("Risk label: Medium")
    else:
        st.success("Risk label: Low")
