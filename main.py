"""
Demo Entry Point: main_demo.py
Purpose: Demonstrate the project pipeline using preprocessed CSV files and pretrained models.
         This script loads the saved CSVs and then launches the Dash GUI without retraining models.
"""

import os
import sys
import pandas as pd

def main():
    print("Demo Mode: Using precomputed CSV files and pretrained models.")

    # Define file paths for the precomputed outputs
    historical_csv = "outputs/final_historical_prediction.csv"
    sentiment_csv = "outputs/final_sentiment_prediction.csv"
    
    # Check if required files exist
    if not os.path.exists(historical_csv):
        print(f"Error: '{historical_csv}' not found. Please run the historical forecasting module first.")
        sys.exit(1)
    if not os.path.exists(sentiment_csv):
        print(f"Error: '{sentiment_csv}' not found. Please run the sentiment forecasting module first.")
        sys.exit(1)

    # Load the CSV files (just to simulate that data is available for your GUI)
    hist_df = pd.read_csv(historical_csv)
    sentiment_df = pd.read_csv(sentiment_csv)
    print("Precomputed CSV files loaded successfully.")
    print(f"Historical Predictions: {hist_df.shape[0]} rows")
    print(f"Sentiment Predictions: {sentiment_df.shape[0]} rows")

    # Launch the Dash application
    print("Launching Dashboard...")
    from scripts.app import app  # Ensure your app.py uses the correct relative paths
    app.run(debug=True)

if __name__ == "__main__":
    main()
