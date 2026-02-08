import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Page config
st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    layout="centered"
)

st.title("Wellness Tourism Package Prediction")

# Download model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="rvrjsingh548/wellness-tourism-model",
    filename="wellness_rf_model.pkl",
    repo_type="model"
)

# Load model
model = joblib.load(model_path)

st.write("Enter customer details below:")

# Collect inputs dynamically from model features
input_data = {}

for col in model.feature_names_in_:
    # SAFETY: skip accidental index column if present
    if "Unnamed" not in col:
        input_data[col] = st.number_input(col, value=0)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("Customer is likely to purchase the Wellness Package ✅")
    else:
        st.warning("Customer is unlikely to purchase the Wellness Package ❌")

