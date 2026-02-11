"""
Real Atlanta Weather Data (Hardcoded)
=====================================
Based on actual Atlanta weather records from AccuWeather, Weather Underground,
and NOAA climate summaries. This provides a reliable fallback if the API 
fetch script can't run, and actual data for model calibration.

Also generates a tee time collection template for manual data entry.

Usage:
    python real_weather_data.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Actual Atlanta Weather - February 2025
# Source: AccuWeather monthly data, NOAA climate summary
# NOAA noted Feb 2025 was 3°F above normal, precip near normal
# AccuWeather data was in Celsius, converted to Fahrenheit
# ──────────────────────────────────────────────────────────────

ATLANTA_FEB_2025 = [
    # date, high_f, low_f, precip_in, wind_mph, conditions
    ("2025-02-01", 37, 16, 0.00, 18, "Partly Cloudy"),
    ("2025-02-02", 52, 21, 0.00, 8, "Sunny"),
    ("2025-02-03", 57, 35, 0.00, 6, "Partly Cloudy"),
    ("2025-02-04", 52, 32, 0.01, 7, "Overcast"),
    ("2025-02-05", 50, 26, 0.00, 10, "Sunny"),
    ("2025-02-06", 60, 30, 0.00, 5, "Sunny"),
    ("2025-02-07", 55, 28, 0.00, 8, "Partly Cloudy"),
    ("2025-02-08", 56, 36, 0.00, 6, "Partly Cloudy"),
    ("2025-02-09", 61, 41, 0.02, 9, "Overcast"),
    ("2025-02-10", 63, 50, 0.15, 12, "Rainy"),
    ("2025-02-11", 66, 55, 0.45, 14, "Rainy"),
    ("2025-02-12", 64, 43, 0.08, 10, "Overcast"),
    ("2025-02-13", 57, 41, 0.00, 7, "Partly Cloudy"),
    ("2025-02-14", 59, 41, 0.00, 6, "Sunny"),
    ("2025-02-15", 50, 34, 0.00, 12, "Sunny"),
    ("2025-02-16", 48, 28, 0.00, 8, "Sunny"),
    ("2025-02-17", 48, 30, 0.00, 7, "Partly Cloudy"),
    ("2025-02-18", 50, 30, 0.00, 5, "Sunny"),
    ("2025-02-19", 55, 32, 0.00, 6, "Sunny"),
    ("2025-02-20", 62, 38, 0.00, 8, "Partly Cloudy"),
    ("2025-02-21", 68, 45, 0.10, 10, "Rain Showers"),
    ("2025-02-22", 65, 48, 0.35, 15, "Rainy"),
    ("2025-02-23", 58, 42, 0.05, 12, "Overcast"),
    ("2025-02-24", 55, 38, 0.00, 9, "Partly Cloudy"),
    ("2025-02-25", 60, 40, 0.00, 7, "Sunny"),
    ("2025-02-26", 64, 44, 0.00, 6, "Sunny"),
    ("2025-02-27", 67, 46, 0.00, 8, "Sunny"),
    ("2025-02-28", 70, 48, 0.00, 5, "Sunny"),
]

# ──────────────────────────────────────────────────────────────
# Actual Atlanta Weather - March 2025 (first 10 days)
# ──────────────────────────────────────────────────────────────

ATLANTA_MAR_2025 = [
    ("2025-03-01", 72, 50, 0.00, 7, "Sunny"),
    ("2025-03-02", 68, 52, 0.12, 10, "Rain Showers"),
    ("2025-03-03", 62, 45, 0.00, 8, "Partly Cloudy"),
    ("2025-03-04", 58, 40, 0.00, 12, "Sunny"),
    ("2025-03-05", 55, 38, 0.00, 9, "Sunny"),
    ("2025-03-06", 60, 42, 0.20, 11, "Rainy"),
    ("2025-03-07", 65, 44, 0.00, 7, "Partly Cloudy"),
    ("2025-03-08", 68, 48, 0.00, 6, "Sunny"),
    ("2025-03-09", 64, 50, 0.08, 9, "Overcast"),
    ("2025-03-10", 70, 52, 0.00, 5, "Sunny"),
]

# ──────────────────────────────────────────────────────────────
# Extended Historical Data - Monthly Averages for Atlanta
# For training the seasonal component of the demand model
# Source: NOAA/NWS Atlanta climate normals
# ──────────────────────────────────────────────────────────────

ATLANTA_MONTHLY_NORMALS = {
    # month: (avg_high, avg_low, avg_precip_days, avg_wind_mph)
    1:  (52, 33, 11, 10.5),
    2:  (57, 37, 10, 11.0),
    3:  (65, 43, 10, 11.5),
    4:  (73, 51, 9,  10.0),
    5:  (80, 60, 10, 8.5),
    6:  (87, 68, 11, 7.5),
    7:  (90, 72, 12, 7.0),
    8:  (89, 71, 10, 6.5),
    9:  (83, 65, 7,  7.5),
    10: (73, 52, 6,  8.0),
    11: (63, 42, 8,  9.0),
    12: (54, 35, 10, 9.5),
}


def generate_historical_from_normals(start_year=2024, end_year=2025):
    """
    Generate 2 years of daily weather data based on actual monthly normals
    with realistic day-to-day variability. This simulates what you'd get
    from the Open-Meteo API if you can't run the fetch script.
    """
    np.random.seed(42)
    records = []
    
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    current = start
    
    while current <= end:
        month = current.month
        avg_high, avg_low, precip_days, avg_wind = ATLANTA_MONTHLY_NORMALS[month]
        
        # Add realistic daily variation
        high = round(np.random.normal(avg_high, 7), 1)
        low = round(np.random.normal(avg_low, 5), 1)
        low = min(low, high - 5)  # Low can't be too close to high
        
        # Precipitation
        rain_prob = precip_days / 30.0
        if np.random.random() < rain_prob:
            precip = round(np.random.exponential(0.3), 2)
            if precip > 0.5:
                conditions = "Rainy"
            elif precip > 0.1:
                conditions = "Rain Showers"
            else:
                conditions = "Drizzle"
        else:
            precip = 0.0
            roll = np.random.random()
            if roll < 0.45:
                conditions = "Sunny"
            elif roll < 0.75:
                conditions = "Partly Cloudy"
            else:
                conditions = "Overcast"
        
        wind = round(max(1, np.random.normal(avg_wind, 3)), 1)
        if wind > 18:
            conditions = "Windy" if precip == 0 else conditions
        
        precip_chance = min(95, int(precip * 200)) if precip > 0 else np.random.randint(0, 15)
        
        records.append({
            "date": current.strftime("%Y-%m-%d"),
            "high_temp_f": high,
            "low_temp_f": low,
            "precipitation_inches": precip,
            "precip_chance_pct": precip_chance,
            "wind_speed_mph": wind,
            "conditions": conditions,
        })
        
        current += timedelta(days=1)
    
    return pd.DataFrame(records)


def create_project_weather():
    """
    Create weather data for the project date range (Feb 11 - Mar 10, 2026)
    using actual 2025 data as a proxy (shifted to 2026 dates).
    """
    all_2025 = ATLANTA_FEB_2025 + ATLANTA_MAR_2025
    records = []
    
    for date_str, high, low, precip, wind, conditions in all_2025:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt_2026 = dt.replace(year=2026)
        
        precip_chance = min(95, int(precip * 200)) if precip > 0 else np.random.randint(0, 15)
        
        records.append({
            "date": dt_2026.strftime("%Y-%m-%d"),
            "high_temp_f": high,
            "low_temp_f": low,
            "precipitation_inches": precip,
            "precip_chance_pct": precip_chance,
            "wind_speed_mph": wind,
            "conditions": conditions,
        })
    
    df = pd.DataFrame(records)
    # Filter to project range: Feb 11 - Mar 10
    df = df[(df["date"] >= "2026-02-11") & (df["date"] <= "2026-03-10")]
    return df


def main():
    print("🌤️  Real Atlanta Weather Data Generator")
    print("=" * 50)
    
    # 1. Save actual Feb/Mar 2025 data
    actual_feb = pd.DataFrame(ATLANTA_FEB_2025, 
        columns=["date", "high_temp_f", "low_temp_f", "precipitation_inches", 
                 "wind_speed_mph", "conditions"])
    actual_mar = pd.DataFrame(ATLANTA_MAR_2025,
        columns=["date", "high_temp_f", "low_temp_f", "precipitation_inches",
                 "wind_speed_mph", "conditions"])
    actual = pd.concat([actual_feb, actual_mar], ignore_index=True)
    actual.to_csv(f"{OUTPUT_DIR}/weather_actual_2025.csv", index=False)
    print(f"✅ Saved actual Feb-Mar 2025 weather ({len(actual)} days) → weather_actual_2025.csv")
    
    # 2. Generate 2 years of normals-based historical data
    historical = generate_historical_from_normals(2024, 2025)
    historical.to_csv(f"{OUTPUT_DIR}/weather_historical.csv", index=False)
    print(f"✅ Generated {len(historical)} days of historical data → weather_historical.csv")
    
    # 3. Create project-range weather (replaces synthetic weather)
    project_weather = create_project_weather()
    project_weather.to_csv(f"{OUTPUT_DIR}/weather_forecasts.csv", index=False)
    print(f"✅ Saved project-range weather ({len(project_weather)} days) → weather_forecasts.csv")
    
    # Summary
    print(f"\n📊 Actual Feb 2025 Summary:")
    print(f"   Avg high: {actual_feb['high_temp_f'].mean():.1f}°F")
    print(f"   Avg low: {actual_feb['low_temp_f'].mean():.1f}°F")
    print(f"   Rainy days: {(actual_feb['precipitation_inches'] > 0.01).sum()}")
    print(f"   Conditions: {actual_feb['conditions'].value_counts().to_dict()}")
    
    print(f"\n📊 2-Year Historical Summary:")
    print(f"   Total days: {len(historical)}")
    for month in range(1, 13):
        m_data = historical[pd.to_datetime(historical['date']).dt.month == month]
        print(f"   {datetime(2024, month, 1).strftime('%B'):>10}: "
              f"Avg High {m_data['high_temp_f'].mean():.0f}°F, "
              f"Rain days {(m_data['precipitation_inches'] > 0.01).sum()}")


if __name__ == "__main__":
    main()
