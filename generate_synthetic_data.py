"""
Dynamic Tee Time Booking Assistant - Synthetic Data Generator
=============================================================
Generates realistic tee time pricing and availability data for Atlanta-area 
golf courses. Designed to reflect real-world patterns:

- Weekend prices > weekday prices
- Morning prime times (7-10am) cost more
- Prices rise as the tee date approaches (demand curve)
- Weather impacts demand and pricing
- Seasonal variation (spring/fall peak, summer heat discount, winter low)
- Course quality tiers affect base pricing

Usage:
    python generate_synthetic_data.py
    
Outputs:
    - tee_times.csv          : Main tee time listings with prices
    - courses.csv            : Course metadata
    - weather_forecasts.csv  : Simulated weather data
    - price_snapshots.csv    : Price changes over time (for demand forecasting)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date range for generated tee times
START_DATE = datetime(2026, 2, 11)  # Today
NUM_DAYS = 28                        # 4 weeks of data

# How many days before the tee date we start tracking prices
SNAPSHOT_DAYS_OUT = [14, 10, 7, 5, 3, 2, 1, 0]

# ──────────────────────────────────────────────────────────────
# Atlanta Golf Courses (3-5 as specified in business plan)
# ──────────────────────────────────────────────────────────────

COURSES = [
    {
        "course_id": "cherokee_run",
        "name": "Cherokee Run Golf Club",
        "location": "Conyers, GA",
        "region": "East Atlanta",
        "tier": "premium",        # Affects base pricing
        "base_price_18": 72,      # Base weekend 18-hole price
        "base_price_9": 40,
        "holes": 18,
        "par": 72,
        "rating": 4.3,
        "latitude": 33.6465,
        "longitude": -84.0077,
    },
    {
        "course_id": "mystery_valley",
        "name": "Mystery Valley Golf Club",
        "location": "Lithonia, GA",
        "region": "East Atlanta",
        "tier": "value",
        "base_price_18": 45,
        "base_price_9": 28,
        "holes": 18,
        "par": 72,
        "rating": 3.8,
        "latitude": 33.7100,
        "longitude": -84.1052,
    },
    {
        "course_id": "bobby_jones",
        "name": "Bobby Jones Golf Course",
        "location": "Atlanta, GA (Buckhead)",
        "region": "North Atlanta",
        "tier": "premium",
        "base_price_18": 65,      # 9-hole price (no 18-hole option)
        "base_price_9": 65,       # Weekend rate ~$55-65 per GolfPass reviews
        "holes": 9,
        "par": 36,
        "rating": 4.1,
        "latitude": 33.8115,
        "longitude": -84.4100,
    },

    {
        "course_id": "sugar_creek",
        "name": "Sugar Creek Golf & Tennis Club",
        "location": "Sugar Hill, GA",
        "region": "North Atlanta",
        "tier": "premium",
        "base_price_18": 68,
        "base_price_9": 38,
        "holes": 18,
        "par": 72,
        "rating": 4.2,
        "latitude": 34.1065,
        "longitude": -84.0357,
    },
    {
        "course_id": "chastain_park",
        "name": "Chastain Park Golf Course",
        "location": "Atlanta, GA (Buckhead)",
        "region": "North Atlanta",
        "tier": "premium",
        "base_price_18": 32,       # Non-resident weekend rate from website
        "base_price_9": 16,
        "holes": 18,
        "par": 72,
        "rating": 4.4,
        "latitude": 33.8792,
        "longitude": -84.3827,
        "city_of_atlanta": True,
    },
    {
        "course_id": "browns_mill",
        "name": "Browns Mill Golf Course",
        "location": "Atlanta, GA",
        "region": "South Atlanta",
        "tier": "value",
        "base_price_18": 32,       # Non-resident weekend rate (same tier as Chastain)
        "base_price_9": 16,
        "holes": 18,
        "par": 72,
        "rating": 3.8,
        "latitude": 33.6883,
        "longitude": -84.3460,
        "city_of_atlanta": True,
    },
    {
        "course_id": "candler_park",
        "name": "Candler Park Golf Course",
        "location": "Atlanta, GA",
        "region": "East Atlanta",
        "tier": "value",
        "base_price_18": 13,       # 9-hole course, non-resident weekend rate
        "base_price_9": 13,
        "holes": 9,
        "par": 33,
        "rating": 3.6,
        "latitude": 33.7578,
        "longitude": -84.3388,
        "city_of_atlanta": True,
    },
    {
        "course_id": "tup_holmes",
        "name": 'Alfred "Tup" Holmes Golf Course',
        "location": "Atlanta, GA",
        "region": "West Atlanta",
        "tier": "value",
        "base_price_18": 35,       # Non-resident weekend rate with cart
        "base_price_9": 25,
        "holes": 18,
        "par": 72,
        "rating": 3.7,
        "latitude": 33.7367,
        "longitude": -84.4700,
        "city_of_atlanta": True,
    },
]


# ──────────────────────────────────────────────────────────────
# Pricing Model Parameters
# ──────────────────────────────────────────────────────────────

# Day-of-week multipliers (Mon=0 ... Sun=6)
DAY_MULTIPLIERS = {
    0: 0.70,  # Monday - lowest
    1: 0.72,  # Tuesday
    2: 0.75,  # Wednesday
    3: 0.78,  # Thursday
    4: 0.88,  # Friday - ramps up
    5: 1.00,  # Saturday - peak
    6: 0.95,  # Sunday - slightly less than Saturday
}

# Time-of-day multipliers (hour buckets)
def get_time_multiplier(hour: int, is_weekend: bool) -> float:
    """Prime morning times cost more, especially on weekends."""
    if is_weekend:
        if 7 <= hour <= 9:
            return 1.15    # Prime morning
        elif 10 <= hour <= 11:
            return 1.05    # Late morning
        elif 12 <= hour <= 14:
            return 0.90    # Early afternoon
        elif 15 <= hour <= 16:
            return 0.75    # Twilight starts
        else:
            return 0.60    # Super twilight
    else:
        if 7 <= hour <= 9:
            return 1.05
        elif 10 <= hour <= 11:
            return 1.00
        elif 12 <= hour <= 14:
            return 0.88
        elif 15 <= hour <= 16:
            return 0.72
        else:
            return 0.55


# Season multipliers for Atlanta (month-based)
SEASON_MULTIPLIERS = {
    1: 0.75,   # January - cold
    2: 0.82,   # February - still cool
    3: 0.95,   # March - spring starts
    4: 1.10,   # April - peak spring (Masters month!)
    5: 1.05,   # May - warm and popular
    6: 0.90,   # June - getting hot
    7: 0.82,   # July - peak heat discount
    8: 0.80,   # August - hottest
    9: 0.92,   # September - cooling
    10: 1.08,  # October - peak fall
    11: 0.90,  # November - getting cold
    12: 0.72,  # December - winter
}


def get_demand_curve_multiplier(days_until: int) -> float:
    """
    Prices rise as tee date approaches (similar to airlines).
    Biggest jump in last 2-3 days for popular times.
    """
    if days_until >= 14:
        return 0.88
    elif days_until >= 10:
        return 0.92
    elif days_until >= 7:
        return 0.95
    elif days_until >= 5:
        return 0.98
    elif days_until >= 3:
        return 1.00
    elif days_until >= 1:
        return 1.05
    else:  # day-of
        return 1.08


# ──────────────────────────────────────────────────────────────
# Weather Generation
# ──────────────────────────────────────────────────────────────

def generate_weather(start_date: datetime, num_days: int) -> pd.DataFrame:
    """
    Generate realistic Atlanta February/March weather.
    Avg high: 55-65°F, chance of rain ~30%.
    """
    records = []
    for i in range(num_days):
        date = start_date + timedelta(days=i)
        month = date.month

        # Monthly temp baselines for Atlanta (°F)
        temp_baselines = {
            1: (43, 54), 2: (46, 58), 3: (52, 65), 4: (59, 73),
            5: (67, 81), 6: (74, 87), 7: (77, 90), 8: (76, 89),
            9: (70, 83), 10: (59, 73), 11: (50, 63), 12: (43, 54),
        }
        low_base, high_base = temp_baselines.get(month, (55, 70))

        high_temp = round(np.random.normal(high_base, 5), 1)
        low_temp = round(np.random.normal(low_base, 4), 1)

        # Rain probability varies by month
        rain_prob_by_month = {
            1: 0.30, 2: 0.30, 3: 0.32, 4: 0.28,
            5: 0.28, 6: 0.32, 7: 0.35, 8: 0.30,
            9: 0.25, 10: 0.22, 11: 0.28, 12: 0.30,
        }
        rain_prob = rain_prob_by_month.get(month, 0.30)
        precip_chance = round(np.random.beta(2, 5) * 100) if random.random() < rain_prob else round(np.random.beta(1, 8) * 100)

        wind_speed = round(max(0, np.random.normal(8, 4)), 1)

        conditions = "Sunny"
        if precip_chance > 60:
            conditions = "Rainy"
        elif precip_chance > 35:
            conditions = "Overcast"
        elif precip_chance > 15:
            conditions = "Partly Cloudy"
        elif wind_speed > 15:
            conditions = "Windy"

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "high_temp_f": high_temp,
            "low_temp_f": low_temp,
            "precip_chance_pct": precip_chance,
            "wind_speed_mph": wind_speed,
            "conditions": conditions,
        })

    return pd.DataFrame(records)


def get_weather_multiplier(weather_row: dict) -> float:
    """Weather impact on pricing/demand."""
    mult = 1.0
    precip = weather_row.get("precip_chance_pct", 0)
    temp = weather_row.get("high_temp_f", 70)
    wind = weather_row.get("wind_speed_mph", 5)

    # Rain suppresses demand → lower prices
    if precip > 60:
        mult *= 0.78
    elif precip > 35:
        mult *= 0.88
    elif precip > 15:
        mult *= 0.95

    # Temperature sweet spot is 65-80°F
    if temp < 45:
        mult *= 0.80
    elif temp < 55:
        mult *= 0.90
    elif temp > 95:
        mult *= 0.82
    elif temp > 88:
        mult *= 0.90

    # High wind
    if wind > 20:
        mult *= 0.88
    elif wind > 15:
        mult *= 0.94

    return round(mult, 3)


# ──────────────────────────────────────────────────────────────
# Tee Time Slot Generation
# ──────────────────────────────────────────────────────────────

def generate_tee_time_slots(course: dict, date: datetime) -> list:
    """Generate available tee time slots for one course on one day."""
    slots = []
    # Tee times from 7:00 AM to 5:00 PM, every 8-10 minutes
    current_time = datetime(date.year, date.month, date.day, 7, 0)
    end_time = datetime(date.year, date.month, date.day, 17, 0)

    while current_time < end_time:
        slots.append(current_time)
        # Intervals: 8 or 10 minutes (realistic)
        interval = random.choice([8, 8, 10, 10, 10])
        current_time += timedelta(minutes=interval)

    return slots


def calculate_price(
    course: dict,
    tee_datetime: datetime,
    weather_mult: float,
    days_until: int,
    holes: int = 18,
) -> float:
    """
    Calculate tee time price using all multipliers.
    Adds realistic noise for variability.
    """
    base = course["base_price_18"] if holes == 18 else course["base_price_9"]

    day_mult = DAY_MULTIPLIERS[tee_datetime.weekday()]
    is_weekend = tee_datetime.weekday() >= 5
    time_mult = get_time_multiplier(tee_datetime.hour, is_weekend)
    season_mult = SEASON_MULTIPLIERS[tee_datetime.month]
    demand_mult = get_demand_curve_multiplier(days_until)

    price = base * day_mult * time_mult * season_mult * weather_mult * demand_mult

    # Add noise (±5%)
    noise = np.random.normal(1.0, 0.025)
    price *= noise

    # Round to nearest dollar
    return max(15, round(price))


# ──────────────────────────────────────────────────────────────
# Availability Model
# ──────────────────────────────────────────────────────────────

def is_available(
    tee_datetime: datetime,
    days_until: int,
    weather_mult: float,
) -> bool:
    """
    Simulate whether a slot is still available.
    Popular times sell out faster.
    """
    is_weekend = tee_datetime.weekday() >= 5
    hour = tee_datetime.hour

    # Base sell-out probability
    if is_weekend and 7 <= hour <= 9:
        base_sellout = 0.45  # Prime weekend mornings
    elif is_weekend:
        base_sellout = 0.20
    elif 7 <= hour <= 9:
        base_sellout = 0.15
    else:
        base_sellout = 0.08

    # Closer dates → more sold out
    if days_until <= 1:
        base_sellout *= 1.8
    elif days_until <= 3:
        base_sellout *= 1.4

    # Bad weather reduces sellout (people cancel)
    base_sellout *= weather_mult

    return random.random() > min(base_sellout, 0.85)


# ──────────────────────────────────────────────────────────────
# Main Generation
# ──────────────────────────────────────────────────────────────

def generate_all_data():
    print("🏌️ Dynamic Tee Time - Synthetic Data Generator")
    print("=" * 50)

    # 1. Save course metadata
    courses_df = pd.DataFrame(COURSES)
    courses_df.to_csv(f"{OUTPUT_DIR}/courses.csv", index=False)
    print(f"✅ Generated {len(COURSES)} courses → courses.csv")

    # 2. Generate weather
    weather_df = generate_weather(START_DATE, NUM_DAYS)
    weather_df.to_csv(f"{OUTPUT_DIR}/weather_forecasts.csv", index=False)
    print(f"✅ Generated {NUM_DAYS} days of weather → weather_forecasts.csv")

    # 3. Generate tee times (current snapshot - as of today)
    tee_time_records = []
    weather_lookup = {row["date"]: row for _, row in weather_df.iterrows()}

    for course in COURSES:
        for day_offset in range(NUM_DAYS):
            date = START_DATE + timedelta(days=day_offset)
            date_str = date.strftime("%Y-%m-%d")
            weather_row = weather_lookup.get(date_str, {})
            weather_mult = get_weather_multiplier(weather_row)
            days_until = day_offset  # days from today

            slots = generate_tee_time_slots(course, date)

            for slot in slots:
                available = is_available(slot, days_until, weather_mult)
                price_18 = calculate_price(course, slot, weather_mult, days_until, 18)
                price_9 = calculate_price(course, slot, weather_mult, days_until, 9)

                # Available spots (1-4 golfers)
                if available:
                    spots = random.choices([1, 2, 3, 4], weights=[15, 25, 25, 35])[0]
                else:
                    spots = 0

                tee_time_records.append({
                    "course_id": course["course_id"],
                    "course_name": course["name"],
                    "region": course["region"],
                    "date": date_str,
                    "tee_time": slot.strftime("%H:%M"),
                    "day_of_week": date.strftime("%A"),
                    "is_weekend": date.weekday() >= 5,
                    "price_18_holes": price_18,
                    "price_9_holes": price_9,
                    "spots_available": spots,
                    "is_available": available,
                    "days_until": days_until,
                    "weather_conditions": weather_row.get("conditions", "Unknown"),
                    "high_temp_f": weather_row.get("high_temp_f", 65),
                    "precip_chance_pct": weather_row.get("precip_chance_pct", 0),
                })

    tee_times_df = pd.DataFrame(tee_time_records)
    tee_times_df.to_csv(f"{OUTPUT_DIR}/tee_times.csv", index=False)
    print(f"✅ Generated {len(tee_times_df):,} tee time slots → tee_times.csv")

    # 4. Generate price snapshots (how prices changed over time)
    #    This is KEY for training the demand forecasting model
    print("\n📈 Generating price snapshots for demand forecasting...")
    snapshot_records = []

    for course in COURSES:
        for day_offset in range(NUM_DAYS):
            date = START_DATE + timedelta(days=day_offset)
            date_str = date.strftime("%Y-%m-%d")
            weather_row = weather_lookup.get(date_str, {})
            weather_mult = get_weather_multiplier(weather_row)

            # Pick a sample of slots per day (not all, to keep data manageable)
            sample_hours = [7, 8, 9, 10, 12, 14, 16]
            for hour in sample_hours:
                slot = datetime(date.year, date.month, date.day, hour, 0)
                is_weekend = date.weekday() >= 5

                for days_out in SNAPSHOT_DAYS_OUT:
                    if days_out > day_offset:
                        continue  # Can't have a snapshot before today

                    snapshot_date = date - timedelta(days=days_out)
                    price = calculate_price(course, slot, weather_mult, days_out, 18)

                    snapshot_records.append({
                        "course_id": course["course_id"],
                        "tee_date": date_str,
                        "tee_time": slot.strftime("%H:%M"),
                        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                        "days_until_tee": days_out,
                        "price_18_holes": price,
                        "day_of_week": date.strftime("%A"),
                        "is_weekend": is_weekend,
                        "hour": hour,
                        "weather_conditions": weather_row.get("conditions", "Unknown"),
                        "high_temp_f": weather_row.get("high_temp_f", 65),
                        "precip_chance_pct": weather_row.get("precip_chance_pct", 0),
                    })

    snapshots_df = pd.DataFrame(snapshot_records)
    snapshots_df.to_csv(f"{OUTPUT_DIR}/price_snapshots.csv", index=False)
    print(f"✅ Generated {len(snapshots_df):,} price snapshots → price_snapshots.csv")

    # 5. Summary stats
    print("\n" + "=" * 50)
    print("📊 Data Summary")
    print("=" * 50)
    avail = tee_times_df[tee_times_df["is_available"]]
    print(f"Total tee time slots:     {len(tee_times_df):,}")
    print(f"Available slots:          {len(avail):,} ({len(avail)/len(tee_times_df)*100:.0f}%)")
    print(f"Sold out slots:           {len(tee_times_df) - len(avail):,}")
    print(f"Price range (18 holes):   ${tee_times_df['price_18_holes'].min()} - ${tee_times_df['price_18_holes'].max()}")
    print(f"Avg weekend price:        ${tee_times_df[tee_times_df['is_weekend']]['price_18_holes'].mean():.0f}")
    print(f"Avg weekday price:        ${tee_times_df[~tee_times_df['is_weekend']]['price_18_holes'].mean():.0f}")
    print(f"Price snapshots:          {len(snapshots_df):,}")
    print(f"\nDate range: {START_DATE.strftime('%Y-%m-%d')} to {(START_DATE + timedelta(days=NUM_DAYS-1)).strftime('%Y-%m-%d')}")
    print(f"Courses: {', '.join(c['name'] for c in COURSES)}")

    return tee_times_df, courses_df, weather_df, snapshots_df


if __name__ == "__main__":
    generate_all_data()
    print("\n✅ All data saved to ./data/ directory")
