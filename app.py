import requests
import pandas as pd
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ── FETCHER 1: Current Weather + 5-Day Forecast ──────────────────────────────

def get_current_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        return None

    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "condition": data["weather"][0]["description"].title(),
        "icon": data["weather"][0]["icon"]
    }


def get_forecast(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        return None

    records = []
    for item in data["list"]:
        records.append({
            "datetime": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "feels_like": item["main"]["feels_like"],
            "humidity": item["main"]["humidity"],
            "wind_speed": item["wind"]["speed"],
            "condition": item["weather"][0]["description"].title()
        })

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ── FETCHER 2: Historical Data (Open-Meteo, no key needed) ───────────────────

def get_historical_weather(lat, lon, days=30):
    from datetime import datetime, timedelta

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "precipitation": data["daily"]["precipitation_sum"],
        "wind_speed": data["daily"]["windspeed_10m_max"]
    })

    df["date"] = pd.to_datetime(df["date"])
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    return df


# ── HELPER: Get lat/lon from city name ───────────────────────────────────────

def get_coordinates(city):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        "q": city,
        "limit": 1,
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if not data:
        return None, None

    return data[0]["lat"], data[0]["lon"]


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    city = "Bengaluru"

    print("Testing current weather...")
    weather = get_current_weather(city)
    print(weather)

    print("\nTesting forecast...")
    forecast = get_forecast(city)
    print(forecast.head())

    print("\nTesting historical...")
    lat, lon = get_coordinates(city)
    history = get_historical_weather(lat, lon)
    print(history.head())