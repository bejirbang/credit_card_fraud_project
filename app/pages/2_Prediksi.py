import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Prediction", layout="wide")
st.title("Prediksi Fraud")

rf_model = st.session_state.get("rf_model")
log_model = st.session_state.get("log_model")
scaler_log = st.session_state.get("scaler_log")

if rf_model is None:
    st.error("Model belum dimuat!")
    st.stop()

# Ambil feature names dari model
feature_names = rf_model.feature_names_in_

st.write("Model expects features:", feature_names)

with st.form("input_form"):

    input_data = {}

    for feature in feature_names:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    submitted = st.form_submit_button("Predict")

if submitted:

    X_input = pd.DataFrame([input_data])

    # Random Forest
    y_pred_rf = rf_model.predict(X_input)[0]
    y_proba_rf = rf_model.predict_proba(X_input)[0][1]

    # Logistic
    X_scaled = scaler_log.transform(X_input)
    y_pred_log = log_model.predict(X_scaled)[0]
    y_proba_log = log_model.predict_proba(X_scaled)[0][1]

    st.divider()

    st.subheader("🌲 Random Forest")
    st.success("Fraud ✅" if y_pred_rf == 1 else "Not Fraud ❌")
    st.write(f"Probability: {y_proba_rf:.2f}")

    st.subheader("📈 Logistic Regression")
    st.success("Fraud ✅" if y_pred_log == 1 else "Not Fraud ❌")
    st.write(f"Probability: {y_proba_log:.2f}")
