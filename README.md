# SP500 Forecasting: Historical vs. Sentiment-Based Models

This project compares two distinct machine learning models to forecast the S&P 500 index:

1. Historical Model: A hybrid time-series model combining ARIMA (linear trend detection) and Bi-directional LSTM (nonlinear pattern learning).
2. Sentiment-Based Model: A model using FinBERT to score financial news sentiment, followed by an XGBoost regressor to forecast weekly index movements.


# Objective

To evaluate whether traditional price-based forecasting or sentiment-driven prediction provides better accuracy and directional insight for S&P 500 market trends.



# Dataset Access

Due to GitHub's file size limits, dataset is hosted externally.

Download Full Dataset: 
[Download from Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main/Stock_news)

Included:
- `All_external.csv` — Raw news and sentiment data (5 GB)

#Methodology Overview

# 🔹 Historical Model (ARIMA + BiLSTM)
- ARIMA captures short-term linear trends.
- Residuals passed to BiLSTM to model nonlinear relationships.
- Final forecast = ARIMA output + BiLSTM residual prediction.

# 🔹 Sentiment Model (FinBERT + XGBoost)
- News articles classified using FinBERT into sentiment scores.
- Weekly sentiment aggregates fed into an XGBoost regressor.
- Forecasts weekly S&P 500 close based on market tone.



# Tech Stack

- Languages: Python 3
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `yfinance`, `plotly`, `dash`, `statsmodels`
- NLP Model: FinBERT via Hugging Face Transformers
- Visualization: Dash web app + Matplotlib plots
- Storage: Git LFS for large `.csv` files


# Evaluation Metrics

- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (up/down prediction correctness)



# Key Features

- Time-series modeling using real-world market data.
- Financial news sentiment analysis using transformer models.
- Interactive Dash app with forecast visualizations.
- Side-by-side comparison of model performance across market phases.

# File Path Configuration

Some scripts in this project reference local files using relative or absolute paths (e.g., `All_external.csv`, `scored_sentiment.csv`, `final_sentiment_model.joblib`, etc.).

If you encounter `FileNotFoundError` or similar issues, please make sure to:

- Place the necessary CSV or model files in the correct working directory.
- Or, update file paths in the relevant scripts to point to the correct location on your machine.

Key scripts that may need path changes:
- `final_sentiment_model.py`
- `final_historical_model.py`
- `clean.py`
- `app.py`

> Example:
> Change this:
> ```python
> df = pd.read_csv("scored_sentiment.csv")
> ```
> To this (if your file is in a `data/` folder):
> ```python
> df = pd.read_csv("data/scored_sentiment.csv")
> ```

Ensure consistent folder structure if you're moving files.



# How to Run

Before running `main.py`, ensure you have downloaded `All_external.csv` and run `clean.py` to generate the cleaned dataset used by the models.


```bash
# Install dependencies
pip install -r requirements.txt

# Run the main file
python main.py
