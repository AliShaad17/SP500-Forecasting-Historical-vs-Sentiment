"""
Module: final_sentiment_model.py
Purpose: Implements sentiment-based forecasting by aggregating weekly sentiment data and merging
         with S&P 500 price data. Trains an XGBoost model and performs evaluation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def load_sentiment_data(path: str = "scored_sentiment.csv") -> pd.DataFrame:
    """Load sentiment data from CSV and convert dates into weekly period start times."""
    print(f"Loading sentiment data from: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Date"].notna()]
    df["week"] = df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    return df


def aggregate_weekly_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment values on a weekly basis."""
    df["sentiment_value"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
    weekly = df.groupby("week").agg(
        avg_sentiment=("sentiment_value", "mean"),
        avg_sentiment_confidence=("sentiment_score", "mean"),
        article_count=("sentiment_value", "count")
    ).reset_index()
    return weekly


def load_price_data(start: str, end: str) -> pd.DataFrame:
    """Download S&P 500 weekly price data from Yahoo Finance for the specified period."""
    print("Downloading S&P 500 data from Yahoo Finance...")
    df = yf.download("^GSPC", start=start, end=end, interval="1wk", auto_adjust=True)
    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    if "Date_" in df.columns and "Close_^GSPC" in df.columns:
        df = df.rename(columns={"Date_": "week", "Close_^GSPC": "sp500_weekly_close"})
    elif "Date" in df.columns and "Close" in df.columns:
        df = df.rename(columns={"Date": "week", "Close": "sp500_weekly_close"})

    df["week"] = pd.to_datetime(df["week"])
    return df


def engineer_features(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment and price data and compute additional forecasting features."""
    sentiment_df["week"] = pd.to_datetime(sentiment_df["week"])
    price_df["week"] = pd.to_datetime(price_df["week"])

    try:
        data = pd.merge(sentiment_df, price_df, on="week", how="inner")
    except Exception as e:
        print("Merge failed:", e)
        print("Sentiment columns:", sentiment_df.columns)
        print("Price columns:", price_df.columns)
        raise

    data["prev_close"] = data["sp500_weekly_close"].shift(1).bfill()
    data["weekly_return"] = data["sp500_weekly_close"].pct_change().fillna(0)
    data["rolling_sentiment"] = data["avg_sentiment"].rolling(4).mean().bfill()
    return data


def sentiment_forecast(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Perform sentiment-based forecasting:
      1. Load and aggregate sentiment data.
      2. Download corresponding S&P 500 price data.
      3. Merge and engineer features.
      4. Predict using a pre-trained XGBoost model.
    Returns DataFrame with Date, Actual price, and Forecast.
    """
    start_date_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    price_df = load_price_data(start=start_date_str, end=end_date_str)
    sentiment_df = load_sentiment_data("scored_sentiment.csv")
    weekly_sentiment = aggregate_weekly_sentiment(sentiment_df)
    weekly_sentiment = weekly_sentiment[
        (weekly_sentiment["week"] >= pd.to_datetime(start_date)) &
        (weekly_sentiment["week"] <= pd.to_datetime(end_date))
    ]
    price_df = load_price_data(start=start_date_str, end=end_date_str)
    final_df = engineer_features(weekly_sentiment, price_df)
    final_df = final_df.sort_values(by="week")

    model = joblib.load("final_sentiment_model.joblib")
    features = [
        "avg_sentiment",
        "avg_sentiment_confidence",
        "article_count",
        "prev_close",
        "weekly_return",
        "rolling_sentiment"
    ]
    final_df["Forecast"] = model.predict(final_df[features])
    final_df = final_df.rename(columns={"week": "Date", "sp500_weekly_close": "Actual"})

    return final_df[["Date", "Actual", "Forecast"]]


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train and evaluate the XGBoost model using an 80/20 train-test split."""
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    features = [
        "avg_sentiment",
        "avg_sentiment_confidence",
        "article_count",
        "prev_close",
        "weekly_return",
        "rolling_sentiment"
    ]
    X_train = train[features]
    y_train = train["sp500_weekly_close"]
    X_test = test[features]
    y_test = test["sp500_weekly_close"]
    print(f"\nTraining model on {len(train)} rows, testing on {len(test)} rows...")
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)
    joblib.dump(model, "final_sentiment_model.joblib")
    print("Model saved to 'final_sentiment_model.joblib'")

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("\nModel Performance:")
    print(f"   • RMSE: {rmse:.2f}")
    print(f"   • R² Score: {r2:.2f}")

    plt.figure(figsize=(14, 6))
    plt.plot(test["week"], y_test.values, label="Actual S&P 500", color="blue")
    plt.plot(test["week"], predictions, label="Predicted (XGBoost)", color="orange")
    plt.title("S&P 500 Weekly Close: Actual vs Predicted (Sentiment-Based)")
    plt.xlabel("Week")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("final_sentiment_prediction.png")
    print("Plot saved to 'final_sentiment_prediction.png'")


def export_sentiment_forecast_csv(start_date: datetime, end_date: datetime, 
                                  filename: str = "outputs/final_sentiment_prediction.csv") -> pd.DataFrame:
    """Export the sentiment forecast for the specified period to CSV."""
    df = sentiment_forecast(start_date, end_date)
    df.to_csv(filename, index=False)
    print(f"Sentiment forecast CSV saved to '{filename}'")
    return df


def main() -> None:
    start_date = pd.to_datetime("2018-01-01")
    end_date = pd.to_datetime("2020-12-31")
    sentiment_df = load_sentiment_data("scored_sentiment.csv")
    weekly_sentiment = aggregate_weekly_sentiment(sentiment_df)
    price_df = load_price_data(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    final_df = engineer_features(weekly_sentiment, price_df)
    print(f"\nFinal dataset shape: {final_df.shape}")
    train_and_evaluate(final_df)
    export_sentiment_forecast_csv(start_date, end_date)


if __name__ == "__main__":
    main()
