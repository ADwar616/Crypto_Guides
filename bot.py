import pandas as pd
import yfinance as yf
import joblib
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import streamlit as st

# Load the ARIMA model
arima_model = joblib.load('arima_model.joblib')

# Define functions for data fetching and prediction
def fetch_data(ticker):
    data = yf.download(ticker, period="1d", interval="1m")
    latest_data = data.iloc[-1]
    return latest_data

def predict_price(ticker):
    ticker = ticker.upper()
    latest_data = fetch_data(ticker)
    opening_price = latest_data['Open']
    high_price = latest_data['High']
    low_price = latest_data['Low']
    adj_closing_price = latest_data['Adj Close']
    vol = latest_data['Volume']

    user_data = {
        'Open': opening_price,
        'High': high_price,
        'Low': low_price,
        'Adj Close': adj_closing_price,
        'Volume': vol,
        'Year': latest_data.name.year,
        'Month': latest_data.name.month,
        'Day': latest_data.name.day
    }

    columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Year', 'Month', 'Day']
    input_data = pd.DataFrame([user_data], columns=columns)

    forecast = arima_model.get_forecast(steps=1)
    predicted_residuals = forecast.predicted_mean
    predicted_close = adj_closing_price + predicted_residuals.values[0]

    return predicted_close

# Streamlit UI
def main():
    st.title("Stock Price Prediction App")

    ticker = st.text_input("Enter the ticker symbol (e.g., BTC-USD, ETH-USD, LTC-USD):")

    if st.button("Predict"):
        if ticker:
            predicted_closing_price = predict_price(ticker)
            st.write(f"Current Price for {ticker}:", fetch_data(ticker)['Adj Close'])
            st.write(f"Predicted Closing Price for {ticker}:", predicted_closing_price)

# Execute the Streamlit app
if __name__ == '__main__':
    main()
