import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("new_best_escalation_model.pkl")
scaler = joblib.load("escalation_scaler.pkl")

st.set_page_config(page_title="Complaint Escalation Predictor", layout="centered")

st.title("📢 Customer Complaint Escalation Prediction")
st.write("Predict whether a customer complaint will be escalated.")

# -----------------------------
# User Inputs
# -----------------------------

complaint_text = st.text_area(
    "Complaint Text",
    "The issue has not been resolved despite multiple calls"
)

sentiment_score = st.slider(
    "Sentiment Score (-1 = Very Negative, +1 = Very Positive)",
    -1.0, 1.0, -0.4
)

complaint_channel = st.selectbox(
    "Complaint Channel",
    ["Email", "Phone", "Chat", "Social Media"]
)

product = st.selectbox(
    "Product",
    ["Credit Card", "Loan", "Insurance", "Mobile App", "Internet Service"]
)

customer_tier = st.selectbox(
    "Customer Tier",
    ["Basic", "Silver", "Gold", "Platinum"]
)

past_complaints = st.slider(
    "Number of Past Complaints",
    0, 5, 2
)

resolution_attempts = st.slider(
    "Resolution Attempts",
    1, 5, 2
)

# -----------------------------
# Feature Engineering
# -----------------------------

complaint_length = len(complaint_text.split())

channel_map = {"Email": 0, "Phone": 1, "Chat": 2, "Social Media": 3}
product_map = {
    "Credit Card": 0,
    "Loan": 1,
    "Insurance": 2,
    "Mobile App": 3,
    "Internet Service": 4
}
tier_map = {"Basic": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

input_data = pd.DataFrame([[
    sentiment_score,
    channel_map[complaint_channel],
    product_map[product],
    tier_map[customer_tier],
    past_complaints,
    resolution_attempts,
    complaint_length
]], columns=[
    "sentiment_score",
    "complaint_channel",
    "product",
    "customer_tier",
    "past_complaints",
    "resolution_attempts",
    "complaint_length"
])

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Escalation"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ Complaint WILL be Escalated\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Complaint will NOT be Escalated\n\nProbability: {probability:.2%}")
