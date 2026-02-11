"""
Dynamic Tee Time - Demand Forecasting Model
=============================================
Predicts tee time price movements and generates "demand scores" 
that indicate how likely a slot is to increase in price or sell out.

This is the predictive layer described in the business plan:
  - Input: day of week, time of day, days until tee time, weather, course
  - Output: predicted price, demand score, price change forecast

Models:
  1. Price Prediction Model (Random Forest Regressor)
     → Predicts what a tee time will cost at any point before the date
  2. Demand Score Model (Gradient Boosting Classifier)  
     → Classifies slots as Low / Medium / High demand
  3. Price Change Forecaster
     → Predicts how much a price will increase between now and tee date

Usage:
    python demand_model.py              # Train and evaluate all models
    python demand_model.py --predict    # Run sample predictions

Requires: pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Data Loading & Feature Engineering
# ──────────────────────────────────────────────────────────────

def load_and_prepare_data():
    """Load price snapshots and engineer features for model training."""
    
    print("📂 Loading data...")
    snapshots = pd.read_csv(f"{DATA_DIR}/price_snapshots.csv")
    courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
    
    print(f"   Price snapshots: {len(snapshots):,} rows")
    print(f"   Courses: {len(courses)} courses")
    
    # Merge course metadata
    df = snapshots.merge(courses[["course_id", "tier", "base_price_18", "rating", "holes"]], 
                         on="course_id", how="left")
    
    # ── Feature Engineering ──
    print("🔧 Engineering features...")
    
    # Time features
    df["hour"] = df["tee_time"].apply(lambda x: int(x.split(":")[0]))
    df["is_morning_prime"] = df["hour"].between(7, 9).astype(int)
    df["is_afternoon"] = df["hour"].between(12, 14).astype(int)
    df["is_twilight"] = (df["hour"] >= 15).astype(int)
    
    # Day features
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["day_num"] = df["day_of_week"].map(day_map)
    df["is_weekend"] = df["is_weekend"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).astype(int)
    df["is_friday"] = (df["day_num"] == 4).astype(int)
    
    # Booking window features
    df["days_until_tee"] = df["days_until_tee"].astype(int)
    df["is_last_minute"] = (df["days_until_tee"] <= 2).astype(int)
    df["is_advance"] = (df["days_until_tee"] >= 7).astype(int)
    df["booking_window_sq"] = df["days_until_tee"] ** 2  # Non-linear demand curve
    
    # Weather features
    df["high_temp_f"] = df["high_temp_f"].astype(float)
    df["precip_chance_pct"] = df["precip_chance_pct"].astype(float)
    df["is_rainy"] = (df["precip_chance_pct"] > 40).astype(int)
    df["is_cold"] = (df["high_temp_f"] < 50).astype(int)
    df["is_perfect_weather"] = ((df["high_temp_f"].between(65, 82)) & 
                                 (df["precip_chance_pct"] < 20)).astype(int)
    
    # Course features
    tier_map = {"value": 0, "mid": 1, "premium": 2}
    df["tier_num"] = df["tier"].map(tier_map).fillna(1)
    df["base_price_18"] = df["base_price_18"].astype(float)
    
    # Course encoding (one-hot, as int not bool)
    course_dummies = pd.get_dummies(df["course_id"], prefix="course").astype(int)
    df = pd.concat([df, course_dummies], axis=1)
    
    # Interaction features (these help the model a lot)
    df["weekend_x_morning"] = df["is_weekend"] * df["is_morning_prime"]
    df["weekend_x_lastmin"] = df["is_weekend"] * df["is_last_minute"]
    df["rain_x_weekend"] = df["is_rainy"] * df["is_weekend"]
    df["temp_x_weekend"] = df["high_temp_f"] * df["is_weekend"]
    
    print(f"   Total features engineered: {len([c for c in df.columns if c not in snapshots.columns])}")
    
    return df


def get_feature_columns(df):
    """Return the feature columns used for modeling."""
    feature_cols = [
        # Time
        "hour", "is_morning_prime", "is_afternoon", "is_twilight",
        # Day
        "day_num", "is_weekend", "is_friday",
        # Booking window
        "days_until_tee", "is_last_minute", "is_advance", "booking_window_sq",
        # Weather
        "high_temp_f", "precip_chance_pct", "is_rainy", "is_cold", "is_perfect_weather",
        # Course
        "tier_num", "base_price_18",
        # Interactions
        "weekend_x_morning", "weekend_x_lastmin", "rain_x_weekend", "temp_x_weekend",
    ]
    # Add course dummies (but not course_id itself)
    course_cols = [c for c in df.columns if c.startswith("course_") and c != "course_id"]
    feature_cols.extend(course_cols)
    
    return feature_cols


# ──────────────────────────────────────────────────────────────
# Model 1: Price Prediction
# ──────────────────────────────────────────────────────────────

def train_price_model(df):
    """
    Random Forest Regressor to predict tee time price.
    This answers: "What will this slot cost?"
    """
    print("\n" + "=" * 60)
    print("📈 MODEL 1: Price Prediction (Random Forest)")
    print("=" * 60)
    
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["price_18_holes"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()
    
    print(f"\n   Test MAE:  ${mae:.2f} (off by this much on average)")
    print(f"   Test R²:   {r2:.4f}")
    print(f"   CV MAE:    ${cv_mae:.2f} (5-fold cross-validation)")
    
    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(10)
    print(f"\n   Top 10 Most Important Features:")
    for feat, imp in top_features.items():
        bar = "█" * int(imp * 100)
        print(f"   {feat:>25}: {imp:.3f} {bar}")
    
    # Save model
    with open(f"{MODEL_DIR}/price_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\n   ✅ Model saved → models/price_model.pkl")
    
    return model


# ──────────────────────────────────────────────────────────────
# Model 2: Demand Score Classification
# ──────────────────────────────────────────────────────────────

def train_demand_model(df):
    """
    Gradient Boosting Classifier to predict demand level.
    This answers: "How quickly will this slot sell out?"
    
    Labels:
      - LOW:    bottom 33% of prices (good deals)
      - MEDIUM: middle 33%
      - HIGH:   top 33% (high demand, book now)
    """
    print("\n" + "=" * 60)
    print("🔥 MODEL 2: Demand Score (Gradient Boosting Classifier)")
    print("=" * 60)
    
    # Create demand labels based on price percentiles within each course
    df = df.copy()
    
    def assign_demand(group):
        low = group["price_18_holes"].quantile(0.33)
        high = group["price_18_holes"].quantile(0.67)
        conditions = [
            group["price_18_holes"] <= low,
            group["price_18_holes"] <= high,
            group["price_18_holes"] > high,
        ]
        labels = ["LOW", "MEDIUM", "HIGH"]
        group["demand_label"] = np.select(conditions, labels, default="MEDIUM")
        return group
    
    df = df.groupby("course_id", group_keys=False).apply(assign_demand)
    
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    
    le = LabelEncoder()
    y = le.fit_transform(df["demand_label"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    print(f"\n   Accuracy: {accuracy:.1%}")
    print(f"\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    for line in report.split("\n"):
        print(f"   {line}")
    
    # Save model and label encoder
    with open(f"{MODEL_DIR}/demand_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/demand_labels.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"\n   ✅ Model saved → models/demand_model.pkl")
    
    return model, le


# ──────────────────────────────────────────────────────────────
# Model 3: Price Change Forecaster
# ──────────────────────────────────────────────────────────────

def train_price_change_model(df):
    """
    Gradient Boosting Regressor to predict price change.
    This answers: "How much will this price increase by tee date?"
    
    Uses paired snapshots: compares early booking price vs day-of price.
    """
    print("\n" + "=" * 60)
    print("📊 MODEL 3: Price Change Forecast (Gradient Boosting)")
    print("=" * 60)
    
    # Create price change pairs: for each slot, compare advance price to day-of price
    df_sorted = df.sort_values(["course_id", "tee_date", "tee_time", "days_until_tee"])
    
    pairs = []
    for (course, tee_date, tee_time), group in df_sorted.groupby(
        ["course_id", "tee_date", "tee_time"]
    ):
        group = group.sort_values("days_until_tee", ascending=False)
        if len(group) < 2:
            continue
        
        day_of_price = group[group["days_until_tee"] == 0]["price_18_holes"]
        if day_of_price.empty:
            day_of_price = group["price_18_holes"].iloc[-1]  # Closest to day-of
        else:
            day_of_price = day_of_price.iloc[0]
        
        for _, row in group.iterrows():
            if row["days_until_tee"] > 0:
                pairs.append({
                    **row.to_dict(),
                    "price_at_booking": row["price_18_holes"],
                    "price_at_teetime": day_of_price,
                    "price_change": day_of_price - row["price_18_holes"],
                    "price_change_pct": (day_of_price - row["price_18_holes"]) / row["price_18_holes"] * 100,
                })
    
    pairs_df = pd.DataFrame(pairs)
    print(f"\n   Training pairs: {len(pairs_df):,}")
    print(f"   Avg price change: +${pairs_df['price_change'].mean():.2f}")
    print(f"   Max price change: +${pairs_df['price_change'].max():.2f}")
    
    feature_cols = get_feature_columns(pairs_df)
    X = pairs_df[feature_cols]
    y = pairs_df["price_change"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n   Test MAE:  ${mae:.2f}")
    print(f"   Test R²:   {r2:.4f}")
    
    with open(f"{MODEL_DIR}/price_change_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\n   ✅ Model saved → models/price_change_model.pkl")
    
    return model, pairs_df


# ──────────────────────────────────────────────────────────────
# Prediction Engine (used by the conversational interface)
# ──────────────────────────────────────────────────────────────

def predict_tee_time(course_id, tee_date, tee_time, days_until, 
                     weather_high, weather_precip, df_reference):
    """
    Make a complete prediction for a single tee time slot.
    Returns predicted price, demand score, and price change forecast.
    
    This is the function the conversational AI will call.
    """
    # Load models
    with open(f"{MODEL_DIR}/price_model.pkl", "rb") as f:
        price_model = pickle.load(f)
    with open(f"{MODEL_DIR}/demand_model.pkl", "rb") as f:
        demand_model = pickle.load(f)
    with open(f"{MODEL_DIR}/demand_labels.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(f"{MODEL_DIR}/price_change_model.pkl", "rb") as f:
        change_model = pickle.load(f)
    
    # Load course info
    courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
    course_info = courses[courses["course_id"] == course_id].iloc[0]
    
    # Parse inputs
    hour = int(tee_time.split(":")[0])
    tee_dt = pd.to_datetime(tee_date)
    day_num = tee_dt.dayofweek
    day_name = tee_dt.strftime("%A")
    is_weekend = int(day_num >= 5)
    
    tier_map = {"value": 0, "mid": 1, "premium": 2}
    
    # Build feature dict
    features = {
        "hour": hour,
        "is_morning_prime": int(7 <= hour <= 9),
        "is_afternoon": int(12 <= hour <= 14),
        "is_twilight": int(hour >= 15),
        "day_num": day_num,
        "is_weekend": is_weekend,
        "is_friday": int(day_num == 4),
        "days_until_tee": days_until,
        "is_last_minute": int(days_until <= 2),
        "is_advance": int(days_until >= 7),
        "booking_window_sq": days_until ** 2,
        "high_temp_f": weather_high,
        "precip_chance_pct": weather_precip,
        "is_rainy": int(weather_precip > 40),
        "is_cold": int(weather_high < 50),
        "is_perfect_weather": int(65 <= weather_high <= 82 and weather_precip < 20),
        "tier_num": tier_map.get(course_info["tier"], 1),
        "base_price_18": course_info["base_price_18"],
        "weekend_x_morning": is_weekend * int(7 <= hour <= 9),
        "weekend_x_lastmin": is_weekend * int(days_until <= 2),
        "rain_x_weekend": int(weather_precip > 40) * is_weekend,
        "temp_x_weekend": weather_high * is_weekend,
    }
    
    # Add course dummies
    all_courses = courses["course_id"].unique()
    for c in all_courses:
        features[f"course_{c}"] = int(c == course_id)
    
    # Create DataFrame for prediction
    feature_cols = get_feature_columns(df_reference)
    X = pd.DataFrame([features])
    
    # Ensure all columns exist
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]
    
    # Predict
    predicted_price = price_model.predict(X)[0]
    demand_code = demand_model.predict(X)[0]
    demand_label = label_encoder.inverse_transform([demand_code])[0]
    demand_proba = demand_model.predict_proba(X)[0]
    price_change = change_model.predict(X)[0]
    
    return {
        "course": course_info["name"],
        "date": tee_date,
        "tee_time": tee_time,
        "day": day_name,
        "days_until": days_until,
        "predicted_price": round(predicted_price),
        "demand_level": demand_label,
        "demand_confidence": f"{max(demand_proba) * 100:.0f}%",
        "predicted_price_change": round(price_change, 2),
        "recommendation": get_recommendation(demand_label, price_change, days_until),
        "weather": f"{weather_high}°F, {weather_precip}% rain",
    }


def get_recommendation(demand_label, price_change, days_until):
    """Generate a plain-English booking recommendation."""
    if demand_label == "HIGH" and price_change > 3:
        return "🔴 Book now — high demand, price likely to rise"
    elif demand_label == "HIGH":
        return "🟠 Book soon — popular time slot"
    elif demand_label == "MEDIUM" and days_until <= 2:
        return "🟡 Good price, but availability thinning"
    elif demand_label == "LOW" and price_change < 1:
        return "🟢 Great deal — price is stable, no rush"
    elif price_change > 5:
        return "🟠 Price trending up — consider booking soon"
    else:
        return "🟢 Good value — stable pricing"


# ──────────────────────────────────────────────────────────────
# Sample Predictions Demo
# ──────────────────────────────────────────────────────────────

def run_sample_predictions(df):
    """
    Demo predictions matching the business plan example:
    "Find me a Saturday morning twosome under $100 in North Atlanta"
    """
    print("\n" + "=" * 60)
    print("🏌️  SAMPLE PREDICTIONS")
    print("=" * 60)
    print('\nScenario: "Saturday morning tee time, Feb 21, 2026"')
    print("─" * 60)
    
    test_cases = [
        ("cherokee_run", "2026-02-21", "09:00", 10, 52.0, 60),
        ("chastain_park", "2026-02-21", "08:30", 10, 52.0, 60),
        ("mystery_valley", "2026-02-21", "09:20", 10, 52.0, 60),
        ("bobby_jones", "2026-02-21", "10:00", 10, 52.0, 60),
        ("browns_mill", "2026-02-21", "08:00", 10, 52.0, 60),
        ("tup_holmes", "2026-02-21", "09:00", 10, 52.0, 60),
    ]
    
    for course_id, date, time, days, temp, precip in test_cases:
        result = predict_tee_time(course_id, date, time, days, temp, precip, df)
        print(f"\n   ⛳ {result['course']}")
        print(f"      {result['day']} {result['date']} at {result['tee_time']}")
        print(f"      💰 Predicted price: ${result['predicted_price']}")
        print(f"      📊 Demand: {result['demand_level']} ({result['demand_confidence']})")
        print(f"      📈 Expected price change: +${result['predicted_price_change']:.2f}")
        print(f"      🌤️  Weather: {result['weather']}")
        print(f"      💡 {result['recommendation']}")
    
    # Show the business plan example narrative
    print("\n" + "─" * 60)
    print("📝 Example output for conversational AI:")
    print("─" * 60)
    
    r1 = predict_tee_time("cherokee_run", "2026-02-21", "09:20", 10, 52.0, 60, df)
    r2 = predict_tee_time("mystery_valley", "2026-02-21", "10:40", 10, 52.0, 60, df)
    
    print(f"""
   "The 9:20am slot at Cherokee Run is ${r1['predicted_price']} but forecasted 
   to rise ${r1['predicted_price_change']:.0f} by Saturday. The 10:40am slot at 
   Mystery Valley is ${r2['predicted_price']} and stable. Moving 80 minutes 
   later saves ${r1['predicted_price'] - r2['predicted_price']}."
   """)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("🏌️  Dynamic Tee Time - Demand Forecasting Model")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Train all three models
    price_model = train_price_model(df)
    demand_model, label_encoder = train_demand_model(df)
    change_model, pairs_df = train_price_change_model(df)
    
    # Save feature columns for later use
    feature_cols = get_feature_columns(df)
    with open(f"{MODEL_DIR}/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED AND SAVED")
    print("=" * 60)
    print(f"   models/price_model.pkl        - Price prediction")
    print(f"   models/demand_model.pkl       - Demand classification")
    print(f"   models/price_change_model.pkl - Price change forecast")
    print(f"   models/demand_labels.pkl      - Label encoder")
    print(f"   models/feature_cols.pkl       - Feature column list")
    
    # Run predictions if requested
    if "--predict" in sys.argv:
        run_sample_predictions(df)
    else:
        print(f"\n   Run with --predict to see sample predictions:")
        print(f"   python3 demand_model.py --predict")


if __name__ == "__main__":
    main()
