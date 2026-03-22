# Fraud Detection Web Application

This project is a Flask-based web application and machine learning pipeline for detecting fraudulent transactions. It includes scripts to download datasets, train a Random Forest classifier, and serve a web interface for uploading new transaction data and obtaining predictions.

## Project Structure

- `app.py`: The Flask web application for uploading CSV files and displaying fraud predictions.
- `train_model.py`: Script to intelligently download the dataset, preprocess it, train the machine learning model, and save model assets.
- `model/`: Directory containing the saved model (`fraud_detection_model.pkl`), standard scaler (`scaler.pkl`), and column definitions (`model_columns.pkl`) generated during training.
- `templates/`: Directory containing HTML templates for the Flask application.

## Prerequisites

Make sure you have Python installed. The required packages include `flask`, `pandas`, `numpy`, `scikit-learn`, `joblib`, and `gdown`. You can install them using:

```bash
pip install flask pandas numpy scikit-learn joblib gdown
```

## How to Run

### 1. Train the Model
Before running the web application, you must train the model. This script will automatically download the required datasets (`train_identity.csv` and `train_transaction.csv`) via Google Drive, preprocess the data, and save the model assets into the `model/` directory.

```bash
python train_model.py
```

### 2. Run the Web Application
After the model assets are successfully saved, start the Flask application:

```bash
python app.py
```

The application will start the development server and will be accessible at `http://localhost:5000` (or `http://0.0.0.0:5000`).

### 3. Usage
- Open your web browser and navigate to the application URL.
- Upload a CSV file containing transaction data to test. **Note: The uploaded CSV must contain a `TransactionID` column.**
- The application will preprocess your dataset, run predictions through the Random Forest model, and display a table showing if each transaction is predicted as 'Fraud' or 'Not Fraud', along with a 'Fraud Confidence' percentage.

## Built With
- **Python** - Core language
- **Flask** - Web framework serving the inference app
- **scikit-learn** - Machine learning library (RandomForestClassifier, StandardScaler, LabelEncoder)
- **Pandas & NumPy** - Data manipulation and numerical computations
