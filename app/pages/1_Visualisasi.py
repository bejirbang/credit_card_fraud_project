import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fraud Visualization", layout="wide")
st.title("Visualisasi Data Credit Card")

df = pd.read_csv("data/processed/fraud_processed.csv")

# Contoh: jumlah fraud per merchant
top_merchants = df.groupby("merchant_encoded")["is_fraud"].sum().sort_values(ascending=False).head(10)

st.subheader("Top 10 Merchant Terindikasi Fraud")
st.bar_chart(top_merchants)

# Contoh: fraud per jam
st.subheader("Fraud per Jam")
fraud_per_hour = df.groupby("trans_hour")["is_fraud"].sum()
st.line_chart(fraud_per_hour)
