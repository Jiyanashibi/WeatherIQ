import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── AI WEATHER SUMMARY ────────────────────────────────────────────────────────

def generate_weather_summary(current_weather, forecast_df, insights, anomalies_df):
    """
    Sends weather data to Groq (Llama 3) and gets back an intelligent
    natural language summary with insights and recommendations.
    """

    # Build anomaly description
    anomaly_days = anomalies_df[anomalies_df["is_anomaly"] == True]
    anomaly_text = ""
    if len(anomaly_days) > 0:
        anomaly_text = "Anomalous days detected:\n"
        for _, row in anomaly_days.iterrows():
            anomaly_text += f"  - {row['date'].strftime('%b %d')}: {row['temp_avg']}°C (zscore: {row['temp_zscore']:.2f})\n"
    else:
        anomaly_text = "No anomalies detected in the past 30 days."

    # Build forecast summary
    forecast_text = "7-day forecast:\n"
    for _, row in forecast_df.iterrows():
        forecast_text += f"  - {row['date'].strftime('%b %d')}: {row['predicted_temp']:.1f}°C (range: {row['lower_bound']:.1f} - {row['upper_bound']:.1f}°C)\n"

    # Build the prompt
    prompt = f"""You are a professional meteorologist and data analyst.
Analyse the following weather data for {current_weather['city']}, {current_weather['country']} and provide an intelligent summary.

CURRENT CONDITIONS:
- Temperature: {current_weather['temperature']}°C (feels like {current_weather['feels_like']}°C)
- Condition: {current_weather['condition']}
- Humidity: {current_weather['humidity']}%
- Wind Speed: {current_weather['wind_speed']} m/s

LAST 30 DAYS SUMMARY:
- Average Temperature: {insights['avg_temp']}°C
- Highest: {insights['max_temp']}°C on {insights['max_temp_date']}
- Lowest: {insights['min_temp']}°C on {insights['min_temp_date']}
- Rainy Days: {insights['rainy_days']} out of 30
- Average Precipitation: {insights['avg_precipitation']}mm/day

ANOMALY DETECTION RESULTS:
{anomaly_text}

ML FORECAST (next 7 days):
{forecast_text}

Please provide:
1. A 2-3 sentence intelligent weather summary for today
2. A key trend you notice from the historical data
3. One practical recommendation based on the forecast
4. A one-line anomaly insight if any were detected

Keep the tone professional but conversational. Total response under 150 words."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message.content


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fetchers import get_current_weather, get_coordinates, get_historical_weather
    from ml_engine import detect_anomalies, forecast_temperature, analyse_patterns

    city = "Bengaluru"
    print(f"Generating AI summary for {city}...\n")

    current = get_current_weather(city)
    lat, lon = get_coordinates(city)
    history = get_historical_weather(lat, lon)

    history = detect_anomalies(history)
    forecast = forecast_temperature(history)
    insights = analyse_patterns(history)

    summary = generate_weather_summary(current, forecast, insights, history)
    print("AI Weather Summary:")
    print("─" * 50)
    print(summary)