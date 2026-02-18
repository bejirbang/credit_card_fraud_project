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

    # =============================
    # RANDOM FOREST PREDICTION
    # =============================
    y_pred_rf = rf_model.predict(X_input)[0]
    y_proba_rf = rf_model.predict_proba(X_input)[0][1]

    # =============================
    # LOGISTIC REGRESSION PREDICTION
    # =============================
    X_scaled = scaler_log.transform(X_input)
    y_pred_log = log_model.predict(X_scaled)[0]
    y_proba_log = log_model.predict_proba(X_scaled)[0][1]

    st.divider()

    # =============================
    # RANDOM FOREST RESULT
    # =============================
    st.subheader("🌲 Random Forest")
    st.success("Fraud ✅" if y_pred_rf == 1 else "Not Fraud ❌")
    st.write(f"Probability: {y_proba_rf:.4f}")

    # Feature Importance
    st.markdown("### 🔎 Feature Importance (Top 10)")

    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(feat_imp_df.set_index("Feature"))

    # Explanation Text
    st.markdown("### 🧠 Explanation")
    top_features_rf = feat_imp_df["Feature"].values[:3]

    for feat in top_features_rf:
        st.write(f"- **{feat}** memiliki pengaruh besar terhadap keputusan model.")

    if y_pred_rf == 1:
        st.info(
            "Model Random Forest memprediksi FRAUD karena kombinasi fitur penting "
            "mendekati pola transaksi fraud yang dipelajari saat training."
        )
    else:
        st.info(
            "Model Random Forest memprediksi NOT FRAUD karena pola fitur "
            "lebih menyerupai transaksi normal."
        )

    st.divider()

    # =============================
    # LOGISTIC REGRESSION RESULT
    # =============================
    st.subheader("📈 Logistic Regression")
    st.success("Fraud ✅" if y_pred_log == 1 else "Not Fraud ❌")
    st.write(f"Probability: {y_proba_log:.4f}")

    st.markdown("### 🔎 Koefisien Paling Berpengaruh (Top 10)")

    coefficients = log_model.coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by="Abs", ascending=False).head(10)

    st.bar_chart(coef_df.set_index("Feature")["Coefficient"])

    # Explanation Text
    st.markdown("### 🧠 Interpretation")

    for index, row in coef_df.iterrows():
        if row["Coefficient"] > 0:
            st.write(f"- **{row['Feature']}** meningkatkan probabilitas fraud.")
        else:
            st.write(f"- **{row['Feature']}** menurunkan probabilitas fraud.")

    if y_pred_log == 1:
        st.info(
            "Logistic Regression memprediksi FRAUD karena kombinasi linear "
            "dari fitur menghasilkan probabilitas di atas threshold keputusan."
        )
    else:
        st.info(
            "Logistic Regression memprediksi NOT FRAUD karena kombinasi fitur "
            "tidak cukup kuat untuk melewati threshold klasifikasi."
        )

    st.divider()

    # =============================
    # PERBANDINGAN MODEL
    # =============================
    st.header("📊 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "Fraud Probability": [y_proba_rf, y_proba_log]
    })

    st.bar_chart(comparison_df.set_index("Model"))

    st.markdown("""
    ### 📌 Insight:
    - Random Forest menangkap pola non-linear & interaksi kompleks antar fitur.
    - Logistic Regression bekerja secara linear berdasarkan kombinasi koefisien.
    - Perbedaan probabilitas menunjukkan bagaimana tiap model memandang risiko transaksi ini.
    """)
