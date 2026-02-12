import streamlit as st
import os
import pandas as pd
import joblib
from time import sleep

from src.preprocessing import preprocess
from src.train_models import train_all_models

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection App")

status_placeholder = st.empty()

# ==============================
# STEP 1 — CHECK PROCESSED DATA
# ==============================
processed_file = "data/processed/fraud_processed.csv"
raw_file = "data/raw/fraudTest.csv"

if os.path.exists(processed_file):
    status_placeholder.text("✅ Processed CSV found, loading...")
    df_model = pd.read_csv(processed_file)
    sleep(1)
else:
    status_placeholder.text("⚡ Processed CSV not found, preprocessing raw data...")
    df_raw = pd.read_csv(raw_file)

    df_model = preprocess(df_raw, save_processed=True)

    sleep(1)
    status_placeholder.text("✅ Preprocessing complete and saved.")

# ==============================
# STEP 2 — CHECK MODELS
# ==============================
model_files = [
    "models/random_forest.pkl",
    "models/logistic.pkl",
    "models/scaler_log.pkl"
]

models_exist = all(os.path.exists(f) for f in model_files)

if models_exist:
    status_placeholder.text("✅ Models found, loading models...")
    rf_model = joblib.load("models/random_forest.pkl")
    log_model = joblib.load("models/logistic.pkl")
    scaler_log = joblib.load("models/scaler_log.pkl")
    sleep(1)
else:
    status_placeholder.text("⚡ Models not found, training models...")
    rf_model, log_model, scaler_log = train_all_models(df_model)
    status_placeholder.text("✅ Training complete and models saved.")

# ==============================
# SAVE TO SESSION STATE
# ==============================
st.session_state["df_model"] = df_model
st.session_state["rf_model"] = rf_model
st.session_state["log_model"] = log_model
st.session_state["scaler_log"] = scaler_log

status_placeholder.empty()
st.success("🚀 App ready! Models & data loaded.")
