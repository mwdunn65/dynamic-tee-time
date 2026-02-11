"""
Fetch Historical Atlanta Weather Data
======================================
Uses the free Open-Meteo Historical Weather API (no key needed).
Run this locally to pull real weather data for your project.

Usage:
    pip install requests pandas
    python fetch_weather.py

Output:
    data/weather_historical.csv  - Daily weather for Atlanta (2024-2025)
    data/weather_forecasts.csv   - Overwritten with real data for project date range
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Atlanta coordinates (Hartsfield-Jackson area)
LAT = 33.749
LON = -84.388

def fetch_open_meteo_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily historical weather from Open-Meteo Archive API.
    Free, no API key required for non-commercial use.
    Docs: https://open-meteo.com/en/docs/historical-weather-api
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "weather_code",
        ],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }

    print(f"Fetching weather data from {start_date} to {end_date}...")
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    df = pd.DataFrame({
        "date": daily["time"],
        "high_temp_f": daily["temperature_2m_max"],
        "low_temp_f": daily["temperature_2m_min"],
        "precipitation_inches": daily["precipitation_sum"],
        "wind_speed_mph": daily["wind_speed_10m_max"],
        "weather_code": daily["weather_code"],
    })

    # Map WMO weather codes to human-readable conditions
    # https://open-meteo.com/en/docs
    def map_weather_code(code):
        if code is None:
            return "Unknown"
        code = int(code)
        if code == 0:
            return "Sunny"
        elif code in (1, 2):
            return "Partly Cloudy"
        elif code == 3:
            return "Overcast"
        elif code in (45, 48):
            return "Foggy"
        elif code in (51, 53, 55, 56, 57):
            return "Drizzle"
        elif code in (61, 63, 65, 66, 67):
            return "Rainy"
        elif code in (71, 73, 75, 77):
            return "Snowy"
        elif code in (80, 81, 82):
            return "Rain Showers"
        elif code in (85, 86):
            return "Snow Showers"
        elif code in (95, 96, 99):
            return "Thunderstorm"
        else:
            return "Unknown"

    df["conditions"] = df["weather_code"].apply(map_weather_code)

    # Estimate precipitation chance from actual precipitation
    df["precip_chance_pct"] = df["precipitation_inches"].apply(
        lambda x: min(95, int(x * 200)) if x > 0 else 5
    )

    return df


def main():
    print("🌤️  Atlanta Historical Weather Fetcher")
    print("=" * 50)

    # Fetch 2 years of data for model training
    # This gives seasonal patterns across all months
    try:
        df_full = fetch_open_meteo_history("2024-01-01", "2025-12-31")
        df_full.to_csv(f"{OUTPUT_DIR}/weather_historical.csv", index=False)
        print(f"✅ Saved {len(df_full)} days of historical data → weather_historical.csv")

        # Also create the project-range weather file (Feb 11 - Mar 10, 2026)
        # Use 2025 same-dates as proxy for 2026
        df_proxy = fetch_open_meteo_history("2025-02-11", "2025-03-10")
        # Shift dates to 2026
        df_proxy["date"] = pd.to_datetime(df_proxy["date"]).apply(
            lambda d: d.replace(year=2026).strftime("%Y-%m-%d")
        )
        df_proxy.to_csv(f"{OUTPUT_DIR}/weather_forecasts.csv", index=False)
        print(f"✅ Saved project-range weather (2026 proxy) → weather_forecasts.csv")

        # Summary
        print(f"\n📊 Historical Data Summary:")
        print(f"   Date range: {df_full['date'].iloc[0]} to {df_full['date'].iloc[-1]}")
        print(f"   Avg high: {df_full['high_temp_f'].mean():.1f}°F")
        print(f"   Avg low: {df_full['low_temp_f'].mean():.1f}°F")
        print(f"   Rainy days: {(df_full['precipitation_inches'] > 0.01).sum()}")
        print(f"   Conditions breakdown:")
        print(df_full["conditions"].value_counts().to_string())

    except requests.exceptions.ConnectionError:
        print("❌ No internet connection. Run this script locally with internet access.")
        print("   pip install requests pandas")
        print("   python fetch_weather.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have internet access and the requests library installed.")


if __name__ == "__main__":
    main()
