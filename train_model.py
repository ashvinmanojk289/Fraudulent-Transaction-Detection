import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
import gdown
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    """
    Downloads, loads, merges, and preprocesses the fraud detection dataset.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The preprocessed feature matrix (X).
            - pd.Series: The target variable (y).
            - StandardScaler: The scaler object used for normalization.
            - list: The list of columns used in the model.
    """
    print("Downloading datasets...")
    if not os.path.exists("train_identity.csv"):
        identity_file_id = "1w4Nn_XBouRWH21sz9vXUoX0Snw4bmQ6f"
        gdown.download(f"https://drive.google.com/uc?id={identity_file_id}", "train_identity.csv", quiet=False)
    if not os.path.exists("train_transaction.csv"):
        transaction_file_id = "170nqVVpvw9lCoOhPJiYXZ97jM9bU7f2V"
        gdown.download(f"https://drive.google.com/uc?id={transaction_file_id}", "train_transaction.csv", quiet=False)

    print("Loading and merging datasets...")
    identity = pd.read_csv("train_identity.csv")
    transaction = pd.read_csv("train_transaction.csv")
    df = pd.merge(transaction, identity, on='TransactionID', how='left')
    y = df['isFraud']
    X = df.drop('isFraud', axis=1)

    print("Preprocessing data...")
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].astype(np.float32)

    missing_vals = X.isnull().sum() / len(X)
    cols_to_drop = missing_vals[missing_vals > 0.5].index
    X = X.drop(columns=cols_to_drop)
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].fillna('missing')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X = X.fillna(-999)
    model_columns = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=model_columns), y, scaler, model_columns

def train_and_save_model(X, y, scaler, model_columns):
    print("Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    os.makedirs('model', exist_ok=True)

    print("Saving model, scaler, and columns...")
    joblib.dump(rf, 'model/fraud_detection_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(model_columns, 'model/model_columns.pkl')
    print("Model assets saved successfully in the 'model' directory.")


if __name__ == "__main__":
    X_processed, y_target, data_scaler, columns = load_and_preprocess_data()
    train_and_save_model(X_processed, y_target, data_scaler, columns)
