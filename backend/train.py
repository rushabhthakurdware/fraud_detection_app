import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_and_save():
    print("Generating synthetic fraud dataset...")
    np.random.seed(42)
    n_samples = 10000
    n_fraud = 20  # ~0.2% fraud ratio matching real bank data

    # PCA features V1-V28 simulation
    legit_v = np.random.normal(loc=0, scale=1, size=(n_samples - n_fraud, 28))
    fraud_v = np.random.normal(loc=2, scale=2, size=(n_fraud, 28))

    legit_amt = np.random.exponential(
        scale=100, size=(n_samples - n_fraud, 1)
    )
    fraud_amt = np.random.exponential(scale=500, size=(n_fraud, 1))

    legit = np.hstack([legit_v, legit_amt, np.zeros((n_samples - n_fraud, 1))])
    fraud = np.hstack([fraud_v, fraud_amt, np.ones((n_fraud, 1))])

    data = np.vstack([legit, fraud])
    np.random.shuffle(data)

    cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = pd.DataFrame(data, columns=cols)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled["Amount"] = scaler.fit_transform(X_train[["Amount"]])

    # 1. Supervised XGBoost Model (Handling imbalance with scale_pos_weight)
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    model = XGBClassifier(
        scale_pos_weight=ratio,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # 2. Unsupervised Anomaly Detection (Isolation Forest)
    iso_model = IsolationForest(contamination=0.002, random_state=42)
    iso_model.fit(X_train_scaled)

    # Export Model Artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(iso_model, "iso_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print(
        "Artifacts successfully saved: model.pkl, iso_model.pkl, scaler.pkl"
    )


if __name__ == "__main__":
    train_and_save()