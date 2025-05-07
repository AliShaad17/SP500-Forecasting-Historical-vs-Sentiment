"""
Module: app.py
Purpose: Implements a Dash application that combines historical and sentiment forecasting results,
         and displays interactive graphs along with sentiment article details.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime

# -------------------------------
# Global Data Loading
# -------------------------------
df_hist = pd.read_csv("outputs/final_historical_prediction.csv")
df_hist["Date"] = pd.to_datetime(df_hist["Date"])

df_sent = pd.read_csv("outputs/final_sentiment_prediction.csv")
df_sent["Date"] = pd.to_datetime(df_sent["Date"])

# Load scored sentiment data once globally
df_articles = pd.read_csv("data/processed/scored_sentiment.csv", parse_dates=["Date"])
df_articles["week"] = df_articles["Date"].dt.to_period("W").apply(lambda r: r.start_time)

MIN_DATE = pd.to_datetime("2018-01-01")
MAX_DATE = pd.to_datetime("2020-12-31")

# -------------------------------
# Initialize the Dash App
# -------------------------------
app = dash.Dash(__name__)
app.title = "Combined Forecast (Historical + Sentiment)"

# -------------------------------
# Helper Function: Aggregate Weekly Sentiment by Mode
# -------------------------------
def aggregate_weekly_sentiment_by_mode(df: pd.DataFrame) -> dict:
    """
    For articles in one week, determine the overall sentiment label (mode) and
    compute the average confidence score for articles with that label.
    Returns a dictionary with 'overall_label' and 'overall_confidence'.
    """
    mode_series = df['sentiment'].mode()
    overall_label = mode_series.iloc[0] if not mode_series.empty else "neutral"
    selected = df[df['sentiment'] == overall_label]
    overall_confidence = selected['sentiment_score'].mean() if not selected.empty else 0.0
    return {"overall_label": overall_label, "overall_confidence": overall_confidence}

# -------------------------------
# Layout
# -------------------------------
app.layout = html.Div([
    html.H1("Combined Forecast (Historical + Sentiment)"),
    dcc.DatePickerRange(
        id="date-picker-range",
        min_date_allowed=MIN_DATE,
        max_date_allowed=MAX_DATE,
        start_date=MIN_DATE,
        end_date=MAX_DATE
    ),
    html.Div([
        html.Label("Shortcut Start Date (YYYY-MM-DD):"),
        dcc.Input(id="shortcut-start-date", type="text", placeholder="YYYY-MM-DD")
    ], style={'marginTop': '10px'}),
    html.Div([
        html.Label("Shortcut End Date (YYYY-MM-DD):"),
        dcc.Input(id="shortcut-end-date", type="text", placeholder="YYYY-MM-DD")
    ], style={'marginTop': '10px'}),
    html.Button("Apply Shortcut", id="apply-shortcut-button", n_clicks=0),
    dcc.Graph(id="combined-forecast-graph"),
    html.Div(id="sentiment-details", style={
        "border": "1px solid #ccc", "padding": "10px", "marginTop": "20px"
    })
])

# -------------------------------
# Callback: Update Combined Forecast Graph
# -------------------------------
@app.callback(
    Output("combined-forecast-graph", "figure"),
    [Input("date-picker-range", "start_date"),
     Input("date-picker-range", "end_date"),
     Input("apply-shortcut-button", "n_clicks")],
    [State("shortcut-start-date", "value"),
     State("shortcut-end-date", "value")]
)
def update_combined_forecast(picker_start, picker_end, n_clicks, shortcut_start, shortcut_end):
    try:
        if n_clicks > 0 and shortcut_start and shortcut_end:
            final_start = pd.to_datetime(shortcut_start)
            final_end = pd.to_datetime(shortcut_end)
            if final_start < MIN_DATE:
                final_start = MIN_DATE
            if final_end > MAX_DATE:
                final_end = MAX_DATE
        else:
            final_start = pd.to_datetime(picker_start)
            final_end = pd.to_datetime(picker_end)
    except Exception as e:
        final_start = MIN_DATE
        final_end = MAX_DATE

    hist_filtered = df_hist[(df_hist["Date"] >= final_start) & (df_hist["Date"] <= final_end)]
    sent_filtered = df_sent[(df_sent["Date"] >= final_start) & (df_sent["Date"] <= final_end)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_filtered["Date"],
        y=hist_filtered["Actual_Close"],
        mode="lines+markers",
        name="Actual Close"
    ))
    fig.add_trace(go.Scatter(
        x=hist_filtered["Date"],
        y=hist_filtered["Predicted_Close"],
        mode="lines+markers",
        name="Historical Predicted Close"
    ))
    fig.add_trace(go.Scatter(
        x=sent_filtered["Date"],
        y=sent_filtered["Forecast"],
        mode="lines+markers",
        name="Sentiment Predicted Close"
    ))
    fig.update_layout(
        title="Combined Forecast (Historical & Sentiment)",
        xaxis_title="Date",
        yaxis_title="Close Price"
    )
    return fig

# -------------------------------
# Callback: Display Sentiment Article Details on Click
# -------------------------------
@app.callback(
    Output("sentiment-details", "children"),
    Input("combined-forecast-graph", "clickData")
)
def display_sentiment_details(clickData):
    if not clickData:
        return "Click on a point from the Sentiment Predicted line to see article details and overall sentiment."
    
    pt = clickData["points"][0]
    if pt["curveNumber"] != 2:
        return "Click on a point from the Sentiment Predicted line to see article details and overall sentiment."
    
    clicked_date = pd.to_datetime(pt["x"])
    clicked_week = clicked_date.to_period("W").start_time

    # Use the globally loaded df_articles instead of reading again
    articles_this_week = df_articles[df_articles["week"] == clicked_week]
    if articles_this_week.empty:
        return f"No articles found for the week starting {clicked_week.date()}."
    
    # Get overall sentiment (label and average confidence) using the helper function.
    agg_results = aggregate_weekly_sentiment_by_mode(articles_this_week)
    overall_label = agg_results["overall_label"]
    overall_conf = agg_results["overall_confidence"]

    articles_this_week = articles_this_week.copy()
    articles_this_week["abs_score"] = articles_this_week["sentiment_score"].abs()
    articles_this_week.sort_values("abs_score", ascending=False, inplace=True)
    top_articles = articles_this_week.head(5)
    
    bullet_list = []
    for _, row in top_articles.iterrows():
        title = row.get("Article_title", "No Title Provided")
        score = row.get("sentiment_score", "N/A")
        bullet_list.append(html.Li(f"{title} (score: {score})"))
    
    return html.Div([
        html.H4(f"Articles for week starting {clicked_week.date()}"),
        html.P(f"Overall Sentiment: {overall_label} | Confidence: {overall_conf:.2f}"),
        html.Ul(bullet_list)
    ], style={"border": "2px solid #444", "padding": "10px", "backgroundColor": "#f9f9f9"})

if __name__ == "__main__":
    app.run(debug=True)

