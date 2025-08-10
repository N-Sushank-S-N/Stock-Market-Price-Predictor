# app.py - The main Flask application for the Stock Forecasting Web App
# This file includes all routes and orchestrates the model-running logic.

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

from flask import Flask, render_template, request

# Import the model prediction functions.
# These files should contain a function that accepts a DataFrame and
# a number of days to forecast, and returns (actual, predicted, labels, rmse).
from train_arima import run_arima
from train_lstm import run_lstm
from train_xgboost import run_xgboost

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Helper Functions ---
def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def fetch_stock_data(ticker, period="1y"):
    """
    Fetches historical stock data using yfinance for a given ticker and period.
    Raises a ValueError if the data is invalid or empty.
    """
    df = yf.download(ticker, period=period)
    if df.empty or 'Close' not in df.columns:
        raise ValueError(f"Failed to fetch valid data for ticker: {ticker}")
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


# --- Main Routes ---
@app.route('/')
def index():
    """Renders the home page where the user can enter a ticker."""
    return render_template('index.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare_models():
    """
    Handles the model comparison logic.
    - Runs all models on the same data.
    - Compares their accuracy (RMSE) to find the best one.
    - Renders the comparison results page with plots and metrics.
    """
    if request.method == 'GET':
        return render_template('index.html')
        
    ticker = request.form['ticker'].upper()
    forecast_days = 30 # Number of days to forecast
    labels = [] # Initialize labels with an empty list

    try:
        df = fetch_stock_data(ticker)
    except ValueError as e:
        return render_template("error.html", message=f"Failed to fetch data for ticker: {ticker}. Please check the ticker symbol.")
    except Exception as e:
        return render_template("error.html", message=f"An unexpected error occurred: {str(e)}")

    # --- Run all models and collect results with error handling ---
    results = {}
    errors = {}

    try:
        # Note: We assume the model functions return (actual, predicted, labels, rmse)
        arima_actual, arima_predicted, labels, arima_rmse = run_arima(df, forecast_days)
        results['ARIMA'] = {'predicted': arima_predicted.tolist(), 'rmse': arima_rmse, 'actual': arima_actual.tolist()}
    except Exception as e:
        errors['ARIMA'] = f"Error: {str(e)}"
        results['ARIMA'] = {'predicted': [], 'rmse': float('inf'), 'actual': []}
    
    try:
        xgb_actual, xgb_predicted, _, xgb_rmse = run_xgboost(df, forecast_days)
        results['XGBoost'] = {'predicted': xgb_predicted.tolist(), 'rmse': xgb_rmse, 'actual': xgb_actual.tolist()}
    except Exception as e:
        errors['XGBoost'] = f"Error: {str(e)}"
        results['XGBoost'] = {'predicted': [], 'rmse': float('inf'), 'actual': []}

    try:
        lstm_actual, lstm_predicted, _, lstm_rmse = run_lstm(df, forecast_days)
        results['LSTM'] = {'predicted': lstm_predicted.tolist(), 'rmse': lstm_rmse, 'actual': lstm_actual.tolist()}
    except Exception as e:
        errors['LSTM'] = f"Error: {str(e)}"
        results['LSTM'] = {'predicted': [], 'rmse': float('inf'), 'actual': []}

    # Use the labels from one of the successful models or default to a safe empty list.
    if arima_actual is not None:
        labels = [d.strftime('%Y-%m-%d') for d in df.index[-len(arima_actual):]]
        actual_data = arima_actual.tolist()
    else:
        labels = []
        actual_data = []

    # --- Determine the best model based on RMSE (lower is better) ---
    best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
    
    return render_template(
        'compare_result.html',
        ticker=ticker,
        best_model=best_model,
        results=results,
        errors=errors,
        labels=labels,
        # Pass all data for the template to render
        arima_data=results['ARIMA']['predicted'],
        lstm_data=results['LSTM']['predicted'],
        xgb_data=results['XGBoost']['predicted'],
        actual_data=actual_data, 
        arima_rmse=results['ARIMA']['rmse'],
        lstm_rmse=results['LSTM']['rmse'],
        xgb_rmse=results['XGBoost']['rmse']
    )

# --- Individual Model Result Routes for "View Details" buttons ---
@app.route('/arima_result', methods=['POST'])
def arima_result():
    ticker = request.form['ticker'].upper()
    try:
        df = fetch_stock_data(ticker)
        actual, predicted, labels, rmse = run_arima(df)
        return render_template('arima_result.html', ticker=ticker, actual_data=actual.tolist(), predicted_data=predicted.tolist(), labels=labels, arima_rmse=rmse)
    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/lstm_result', methods=['POST'])
def lstm_result():
    ticker = request.form['ticker'].upper()
    try:
        df = fetch_stock_data(ticker)
        actual, predicted, labels, rmse = run_lstm(df)
        return render_template('lstm_result.html', ticker=ticker, actual_data=actual.tolist(), predicted_data=predicted.tolist(), labels=labels, lstm_rmse=rmse)
    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/xgboost_result', methods=['POST'])
def xgboost_result():
    ticker = request.form['ticker'].upper()
    try:
        df = fetch_stock_data(ticker)
        actual, predicted, labels, rmse = run_xgboost(df)
        return render_template('xgboost_result.html', ticker=ticker, actual_data=actual.tolist(), predicted_data=predicted.tolist(), labels=labels, xgb_rmse=rmse)
    except Exception as e:
        return render_template('error.html', message=str(e))


# --- Entry Point ---
if __name__ == "__main__":
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
