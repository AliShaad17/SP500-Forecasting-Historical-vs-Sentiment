"""
Module: final_historical_model.py
Purpose: Implements a hybrid historical forecasting model that combines ARIMA and BI-LSTM for weekly S&P 500 data.
         The script downloads data, computes technical indicators, performs a rolling forecast, and saves the final models.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout


def download_data(ticker: str = "^GSPC", start_date: str = "2000-01-01", end_date: str = "2020-06-11") -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance and resamples it to weekly close prices.
    :param ticker: Stock ticker symbol.
    :param start_date: Start date for download.
    :param end_date: End date for download.
    :return: DataFrame of weekly close prices.
    """
    print("Downloading data from yfinance...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    print(f"Total daily records downloaded: {len(data)}")
    data = data[['Close']]
    data = data.resample('W').last()
    print(f"Resampled to weekly frequency: {len(data)} records")
    return data


def explore_and_clean(data: pd.DataFrame) -> pd.DataFrame:
    """
    Explores and cleans the data by sorting the index, dropping duplicates, and displaying key statistics.
    Also plots the weekly close prices.
    :param data: Raw DataFrame.
    :return: Cleaned DataFrame.
    """
    print("\n--- Data Exploration & Cleaning ---")
    print("Data Info:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())
    print("\nMissing Values per Column:")
    print(data.isnull().sum())

    data.sort_index(inplace=True)
    data.drop_duplicates(inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price (Weekly)')
    plt.title("Weekly Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    return data


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators (SMA, RSI, MACD) on the weekly close prices.
    :param data: Cleaned DataFrame with weekly close prices.
    :return: DataFrame enriched with technical indicators.
    """
    print("\n--- Computing Technical Indicators ---")
    close_col = 'Close'
    close_series = data[close_col].values.ravel()
    close_series = pd.to_numeric(close_series, errors='coerce')
    df = pd.DataFrame({close_col: close_series}, index=data.index)

    df['SMA_20'] = SMAIndicator(close=df[close_col], window=20).sma_indicator()
    df['RSI_14'] = RSIIndicator(close=df[close_col], window=14).rsi()
    macd = MACD(close=df[close_col])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    df.dropna(inplace=True)

    selected_cols = [close_col, 'SMA_20', 'RSI_14', 'MACD', 'MACD_signal']
    corr_matrix = df[selected_cols].corr()
    print("\nCorrelation matrix:")
    print(corr_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix: Price & Technical Indicators")
    plt.show()

    print("Technical indicators computed successfully!")
    return df


def split_data(data: pd.DataFrame, train_ratio: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataset into training and testing sets.
    :param data: DataFrame to split.
    :param train_ratio: Ratio of data to use for training.
    :return: Tuple (train_data, test_data).
    """
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    return train_data, test_data


def prepare_lstm_data(series: pd.Series, look_back: int = 60) -> (np.ndarray, np.ndarray, MinMaxScaler):
    """
    Prepares time series data for the BI-LSTM model by scaling and creating sliding windows.
    :param series: Time series data (e.g., ARIMA residuals).
    :param look_back: Number of past observations per sample.
    :return: Tuple (X, y, scaler).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_bi_lstm_model(input_shape: tuple) -> Sequential:
    """
    Constructs and compiles a Bidirectional LSTM model.
    :param input_shape: Shape of the input data.
    :return: Compiled Keras model.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def rolling_hybrid_forecast(train_data: pd.DataFrame, test_data: pd.DataFrame, look_back: int = 60,
                            arima_order: tuple = (5, 1, 0)) -> (pd.DataFrame, object, object):
    """
    Implements a rolling forecast combining ARIMA and BI-LSTM.
    For each test point:
      - Fit ARIMA to current training data.
      - Compute residuals and train BI-LSTM.
      - Combine forecasts from both models.
    :param train_data: Training DataFrame.
    :param test_data: Testing DataFrame.
    :param look_back: Look-back window for BI-LSTM.
    :param arima_order: ARIMA order parameters.
    :return: Tuple (forecast DataFrame, final ARIMA model, final BI-LSTM model).
    """
    close_col = 'Close'
    train_data.index = pd.to_datetime(train_data.index).to_period('W')
    test_data.index = pd.to_datetime(test_data.index).to_period('W')
    rolling_train = train_data.copy()
    forecasts = []
    actuals = []
    final_arima_model = None
    final_lstm_model = None

    for i in range(len(test_data)):
        train_close = rolling_train[close_col]
        try:
            arima_model = ARIMA(train_close, order=arima_order)
            arima_fit = arima_model.fit()
        except Exception as e:
            print(f"ARIMA fitting error on iteration {i}: {e}")
            break

        arima_forecast = arima_fit.forecast(steps=1)
        arima_val = float(arima_forecast.iloc[-1])
        arima_fitted = arima_fit.fittedvalues
        residuals = train_close - arima_fitted

        if len(residuals) < look_back:
            print(f"Not enough residual data on iteration {i}. Skipping BI-LSTM forecast.")
            continue

        X_train_lstm, y_train_lstm, lstm_scaler = prepare_lstm_data(residuals, look_back)
        lstm_model = build_bi_lstm_model((X_train_lstm.shape[1], 1))
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=0)

        last_sequence = residuals[-look_back:]
        last_scaled = lstm_scaler.transform(last_sequence.values.reshape(-1, 1))
        X_last = np.array([last_scaled[:, 0]])
        X_last = np.reshape(X_last, (X_last.shape[0], X_last.shape[1], 1))
        lstm_pred = lstm_model.predict(X_last)
        lstm_pred_rescaled = lstm_scaler.inverse_transform(lstm_pred)
        lstm_val = float(lstm_pred_rescaled.flatten()[0])

        hybrid_pred = arima_val + lstm_val
        forecasts.append(hybrid_pred)
        actual_value = test_data[close_col].iloc[i]
        actuals.append(actual_value)
        rolling_train = pd.concat([rolling_train, test_data.iloc[[i]]])
        final_arima_model = arima_fit
        final_lstm_model = lstm_model

        print(f"Iteration {i+1}/{len(test_data)}: ARIMA={arima_val:.2f}, LSTM Residual={lstm_val:.2f}, Hybrid={hybrid_pred:.2f}, Actual={actual_value:.2f}")

    test_actuals = np.array(actuals)
    test_forecasts = np.array(forecasts)
    rmse = math.sqrt(mean_squared_error(test_actuals, test_forecasts))
    mae = mean_absolute_error(test_actuals, test_forecasts)
    mape = np.mean(np.abs((test_actuals - test_forecasts) / test_actuals)) * 100
    print("\nRolling Hybrid Forecast Evaluation:")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("MAPE:", mape, "%")

    test_index_ts = test_data.index[:len(forecasts)].to_timestamp()
    plt.figure(figsize=(12, 6))
    plt.plot(test_index_ts, test_actuals, label='Actual', marker='o')
    plt.plot(test_index_ts, test_forecasts, label='Hybrid Forecast', marker='x', linestyle='--')
    plt.title("Rolling Hybrid Forecast vs. Actual (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Weekly Close Price")
    plt.legend()
    plt.show()

    df_forecast = pd.DataFrame({
        "Date": test_index_ts,
        "Actual_Close": test_actuals,
        "Predicted_Close": test_forecasts
    })

    return df_forecast, final_arima_model, final_lstm_model


def save_final_models(final_arima_model: object, final_lstm_model: object) -> None:
    """
    Saves the final ARIMA model (using pickle) and BI-LSTM model (using Keras).
    :param final_arima_model: Final ARIMA model instance.
    :param final_lstm_model: Final BI-LSTM model instance.
    """
    print("\n--- Saving Final Models ---")
    if final_lstm_model is not None:
        final_lstm_model.save("models/final_lstm_model.keras", save_format='keras')
        print("Final BI-LSTM model saved to 'final_lstm_model.keras'.")
    else:
        print("No BI-LSTM model available to save.")

    if final_arima_model is not None:
        with open("models/final_arima_model.pkl", "wb") as f:
            pickle.dump(final_arima_model, f)
        print("Final ARIMA model saved to 'final_arima_model.pkl'.")
    else:
        print("No ARIMA model available to save.")


def historical_forecast(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Performs a one-step ARIMA forecast for historical data.
    :param ticker: Stock ticker symbol.
    :param start_date: Start date for data download.
    :param end_date: End date for data download.
    :return: DataFrame with Date, Actual, and Forecast columns.
    """
    start_date_clean = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_clean = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    data = download_data(ticker, start_date_clean, end_date_clean)
    data = data.rename(columns={"Close": "Actual"})
    data.reset_index(inplace=True)
    data.rename(columns={"index": "Date"}, inplace=True)

    try:
        model = ARIMA(data["Actual"], order=(5, 1, 0))
        model_fit = model.fit()
        forecast_series = model_fit.forecast(steps=1)
        forecast_value = float(forecast_series.iloc[0])
        print("One-step forecast value:", forecast_value)
        data["Forecast"] = forecast_value
    except Exception as e:
        print("Error in forecasting:", e)
        data["Forecast"] = None

    return data[["Date", "Actual", "Forecast"]]


def main() -> None:
    data = download_data(ticker="^GSPC", start_date="2000-01-01", end_date="2020-06-11")
    data = explore_and_clean(data)
    data = compute_indicators(data)
    train_data, test_data = split_data(data, train_ratio=0.8)
    df_forecast, final_arima_model, final_lstm_model = rolling_hybrid_forecast(train_data, test_data, look_back=60, arima_order=(5, 1, 0))
    df_forecast.to_csv("outputs/final_historical_prediction.csv", index=False)
    print("CSV saved to 'final_historical_prediction.csv'")
    save_final_models(final_arima_model, final_lstm_model)


if __name__ == "__main__":
    main()
