import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.preprocessing import preprocess

def train_all_models(df_model=None):
    if df_model is None:
        processed_file = "data/processed/fraud_processed.csv"
        if os.path.exists(processed_file):
            print("Loading processed CSV...")
            df_model = pd.read_csv(processed_file)
        else:
            print("Processed CSV not found, preprocessing raw data...")
            df_raw = pd.read_csv("data/raw/fraudTest.csv")
            df_model = preprocess(df_raw, save_processed=True)
    
    X = df_model.drop("is_fraud", axis=1)
    y = df_model["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ===== RANDOM FOREST =====
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/random_forest.pkl")

    # ===== LOGISTIC REGRESSION =====
    scaler_log = StandardScaler()
    X_train_log = scaler_log.fit_transform(X_train)
    X_test_log = scaler_log.transform(X_test)

    log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    log_model.fit(X_train_log, y_train)

    joblib.dump(log_model, "models/logistic.pkl")
    joblib.dump(scaler_log, "models/scaler_log.pkl")

    print("Training complete, models saved in 'models/' folder.")

    return rf_model, log_model, scaler_log
