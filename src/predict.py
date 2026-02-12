import joblib
import numpy as np

# LOAD MODEL
rf_model = joblib.load("models/random_forest.pkl")
log_model = joblib.load("models/logistic.pkl")
scaler_log = joblib.load("models/scaler_log.pkl")

# PREDICT FUNCTIONS
def predict_rf(X):
    y_proba = rf_model.predict_proba(X)[:,1]
    y_pred = np.where(y_proba > 0.2, 1, 0)
    return y_pred, y_proba

def predict_log(X):
    X_scaled = scaler_log.transform(X)
    y_proba = log_model.predict_proba(X_scaled)[:,1]
    y_pred = np.where(y_proba > 0.2, 1, 0)
    return y_pred, y_proba
