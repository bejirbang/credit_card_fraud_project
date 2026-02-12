import pandas as pd
from geopy.distance import geodesic
import os

def preprocess(df: pd.DataFrame, save_processed=True) -> pd.DataFrame:
    # convert datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # fitur baru
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day"] = df["trans_date_trans_time"].dt.dayofweek
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # hitung jarak merchant
    df["distance_km"] = df.apply(
        lambda row: geodesic(
            (row["lat"], row["long"]),
            (row["merch_lat"], row["merch_long"])
        ).km,
        axis=1
    )

    # encode merchant & category
    merchant_fraud_rate = df.groupby("merchant")["is_fraud"].mean()
    category_fraud_rate = df.groupby("category")["is_fraud"].mean()
    df["merchant_encoded"] = df["merchant"].map(merchant_fraud_rate)
    df["category_encoded"] = df["category"].map(category_fraud_rate)

    # drop kolom yang tidak dipakai
    drop_cols = [
        "sn", "first", "last", "street", "trans_num",
        "cc_num", "zip", "lat", "long",
        "merch_lat", "merch_long",
        "dob", "trans_date_trans_time", "unix_time",
        "city", "state", "gender", "job",
        "merchant", "category"
    ]
    df = df.drop(columns=drop_cols)
    
    # simpan ke processed folder
    if save_processed:
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/fraud_processed.csv", index=False)
        print("✅ Processed CSV saved to 'data/processed/fraud_processed.csv'")
    
    return df
