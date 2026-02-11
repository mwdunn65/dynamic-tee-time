"""
Data Explorer - Quick look at the generated synthetic data
Run this to verify the data looks realistic before building models.
"""

import pandas as pd

DATA_DIR = "data"

print("=" * 60)
print("📋 TEE TIMES DATA")
print("=" * 60)
tee = pd.read_csv(f"{DATA_DIR}/tee_times.csv")
print(f"\nShape: {tee.shape}")
print(f"\nColumns: {list(tee.columns)}")
print(f"\nSample rows:")
print(tee[tee["is_available"] == True].head(10).to_string(index=False))

print(f"\n\n📊 Price by Course (18 holes, available only):")
avail = tee[tee["is_available"]]
print(avail.groupby("course_name")["price_18_holes"].agg(["mean", "min", "max", "count"]).round(1).to_string())

print(f"\n\n📊 Weekend vs Weekday Pricing:")
print(avail.groupby("is_weekend")["price_18_holes"].agg(["mean", "min", "max"]).round(1).to_string())

print(f"\n\n📊 Price by Day of Week:")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_prices = avail.groupby("day_of_week")["price_18_holes"].mean().reindex(day_order).round(1)
for day, price in day_prices.items():
    bar = "█" * int(price / 2)
    print(f"  {day:>10}: ${price:>5.1f}  {bar}")

print(f"\n\n📊 Availability by Day of Week:")
for day in day_order:
    subset = tee[tee["day_of_week"] == day]
    avail_pct = subset["is_available"].mean() * 100
    bar = "█" * int(avail_pct / 2)
    print(f"  {day:>10}: {avail_pct:>5.1f}%  {bar}")

print("\n\n" + "=" * 60)
print("🌤️  WEATHER DATA")
print("=" * 60)
weather = pd.read_csv(f"{DATA_DIR}/weather_forecasts.csv")
print(f"\nShape: {weather.shape}")
print(weather.to_string(index=False))

print("\n\n" + "=" * 60)
print("📈 PRICE SNAPSHOTS (for demand forecasting)")
print("=" * 60)
snap = pd.read_csv(f"{DATA_DIR}/price_snapshots.csv")
print(f"\nShape: {snap.shape}")
print(f"\nExample: How price changes as tee date approaches")
example = snap[(snap["course_id"] == "cherokee_run") & 
               (snap["tee_date"] == "2026-02-21") & 
               (snap["tee_time"] == "09:00")]
if not example.empty:
    print(example[["days_until_tee", "price_18_holes", "snapshot_date"]].to_string(index=False))
else:
    # Pick whatever is available
    sample_course = snap["course_id"].iloc[0]
    sample_date = snap["tee_date"].iloc[0]
    example = snap[(snap["course_id"] == sample_course) & 
                   (snap["tee_date"] == sample_date) & 
                   (snap["tee_time"] == "09:00")]
    print(example[["days_until_tee", "price_18_holes", "snapshot_date"]].to_string(index=False))

print("\n✅ Data looks good! Ready for model training.")
