from flask import Flask, request, render_template, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Initialize the Flask application
app = Flask(__name__)
# Secure secret key for flashing messages
app.secret_key = os.urandom(24)

# --- Configuration ---
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_detection_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.pkl')

# --- Load Model Assets ---
def load_model_assets():
    """
    Loads the machine learning model, scaler, and column names from disk.
    This function is called at startup to ensure all assets are ready.
    """
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
        print("Model assets loaded successfully.")
        return model, scaler, model_columns
    except FileNotFoundError as e:
        print(f"Error loading model assets: {e}. Ensure the 'model' directory and its contents exist.")
        return None, None, None

# Load assets at application startup
model, scaler, model_columns = load_model_assets()

def preprocess_input_data(df: pd.DataFrame, scaler_obj, columns: list) -> pd.DataFrame:
    """
    Preprocesses the uploaded data to match the format of the training data.

    Args:
        df (pd.DataFrame): The input dataframe from the uploaded CSV.
        scaler_obj: The fitted scaler object from the training phase.
        columns (list): The list of column names from the training data.

    Returns:
        pd.DataFrame: The preprocessed dataframe ready for prediction.
    """
    df_processed = df.copy()

    # Optimize memory usage by converting data types
    for col in df_processed.select_dtypes(include=['float64', 'int64']).columns:
        df_processed[col] = df_processed[col].astype(np.float32)

    # Label encode categorical features, handling new or missing values
    for col in df_processed.select_dtypes(include='object').columns:
        df_processed[col] = df_processed[col].fillna('missing')
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    # Fill any remaining missing numerical values with a placeholder
    df_processed = df_processed.fillna(-999)

    # Align columns with the model's training columns to prevent errors
    for col in columns:
        if col not in df_processed.columns:
            df_processed[col] = -999  # Add missing columns with placeholder
    
    # Ensure the column order matches the training data exactly
    df_processed = df_processed[columns]

    # Scale the data using the pre-trained scaler
    df_scaled = scaler_obj.transform(df_processed)

    return pd.DataFrame(df_scaled, columns=columns)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page, including file uploads and fraud detection analysis.
    """
    # Check if model assets are loaded before proceeding
    if not all([model, scaler, model_columns]):
        flash("Model assets are not loaded. Please check the server configuration and ensure the 'model' directory is present.", "danger")
        return render_template('index.html', results_html=None)

    if request.method == 'POST':
        # Ensure a file was included in the request
        if 'file' not in request.files:
            flash("No file part in the request. Please select a file.", "danger")
            return redirect(request.url)

        file = request.files['file']
        # Ensure a file was selected by the user
        if file.filename == '':
            flash("No file selected. Please choose a CSV file to upload.", "warning")
            return redirect(request.url)

        # Validate file type
        if file and file.filename.endswith('.csv'):
            try:
                input_df = pd.read_csv(file)
                
                # Essential column check before processing
                if 'TransactionID' not in input_df.columns:
                     flash("The uploaded CSV file must contain a 'TransactionID' column.", "danger")
                     return render_template('index.html', results_html=None)

                # Preprocess the data and make predictions
                processed_df = preprocess_input_data(input_df.drop(columns=['TransactionID']), scaler, model_columns)
                
                predictions = model.predict(processed_df)
                prediction_proba = model.predict_proba(processed_df)[:, 1]
                
                # Add results to the original dataframe for display
                results_df = input_df[['TransactionID']].copy()
                results_df['Prediction'] = ['Fraud' if p == 1 else 'Not Fraud' for p in predictions]
                results_df['Fraud Confidence'] = [f"{p:.2%}" for p in prediction_proba]
                
                # Convert results to HTML for rendering in the template
                results_html = results_df.to_html(classes='table table-hover table-striped text-center', index=False)
                
                return render_template('index.html', results_html=results_html)
            except Exception as e:
                flash(f"An unexpected error occurred during analysis: {e}", "danger")
                return redirect(request.url)
        else:
            flash("Invalid file type. Please upload a valid CSV file.", "warning")
            return redirect(request.url)

    # Handle GET requests
    return render_template('index.html', results_html=None)

if __name__ == '__main__':
    # Create a 'templates' directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Run the Flask app
    # Note: In a production environment, use a WSGI server like Gunicorn or uWSGI
    app.run(host='0.0.0.0', port=5000, debug=True)