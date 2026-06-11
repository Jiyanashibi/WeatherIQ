import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# ── ANOMALY DETECTION ─────────────────────────────────────────────────────────

def detect_anomalies(df):
    """
    Takes the 30-day historical dataframe and flags days where
    temperature was unusually high or low using Isolation Forest.
    """

    # Prepare features for the model
    features = df[["temp_avg", "temp_max", "temp_min", "precipitation", "wind_speed"]].copy()

    # Scale the data — Isolation Forest works better when all features
    # are on the same scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Train Isolation Forest
    # contamination=0.1 means we expect ~10% of days to be anomalies
    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly"] = model.fit_predict(scaled)

    # Isolation Forest returns -1 for anomalies and 1 for normal
    # Convert to True/False for easier use
    df["is_anomaly"] = df["anomaly"] == -1

    # Calculate Z-score for temperature as a second signal
    df["temp_zscore"] = (df["temp_avg"] - df["temp_avg"].mean()) / df["temp_avg"].std()

    return df


# ── TEMPERATURE FORECASTING WITH PROPHET ─────────────────────────────────────

def forecast_temperature(df, days=7):
    """
    Takes the 30-day historical dataframe and forecasts
    the next 7 days of temperature using Facebook Prophet.
    """

    # Prophet requires columns named exactly 'ds' (date) and 'y' (value)
    prophet_df = df[["date", "temp_avg"]].rename(columns={
        "date": "ds",
        "temp_avg": "y"
    })

    # Train the model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95  # 95% confidence interval
    )
    model.fit(prophet_df)

    # Create future dates to predict
    future = model.make_future_dataframe(periods=days)

    # Generate forecast
    forecast = model.predict(future)

    # Keep only the columns we need
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
    forecast.columns = ["date", "predicted_temp", "lower_bound", "upper_bound"]

    return forecast


# ── PATTERN ANALYSIS ──────────────────────────────────────────────────────────

def analyse_patterns(df):
    """
    Extracts interesting patterns from historical data.
    Returns a dictionary of insights.
    """

    insights = {
        "avg_temp": round(df["temp_avg"].mean(), 1),
        "max_temp": round(df["temp_max"].max(), 1),
        "min_temp": round(df["temp_min"].min(), 1),
        "max_temp_date": df.loc[df["temp_max"].idxmax(), "date"].strftime("%b %d"),
        "min_temp_date": df.loc[df["temp_min"].idxmin(), "date"].strftime("%b %d"),
        "avg_precipitation": round(df["precipitation"].mean(), 1),
        "rainy_days": int((df["precipitation"] > 1).sum()),
        "anomaly_count": int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0,
    }

    return insights


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fetchers import get_coordinates, get_historical_weather

    city = "Bengaluru"
    print(f"Running ML analysis for {city}...\n")

    lat, lon = get_coordinates(city)
    df = get_historical_weather(lat, lon)

    print("Running anomaly detection...")
    df = detect_anomalies(df)
    print(df[["date", "temp_avg", "is_anomaly", "temp_zscore"]])

    print("\nRunning Prophet forecast...")
    forecast = forecast_temperature(df)
    print(forecast)

    print("\nPattern insights:")
    insights = analyse_patterns(df)
    for key, value in insights.items():
        print(f"  {key}: {value}")