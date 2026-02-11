"""
Dynamic Tee Time Booking Assistant — Streamlit Interface
==========================================================
Conversational AI interface where users type natural language queries
like "Find me a Saturday morning twosome under $50 in North Atlanta"
and get ranked tee time recommendations with explanations.

Usage:
    pip3 install streamlit
    streamlit run app.py

Requires: models/ folder with trained .pkl files (run demand_model.py first)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta

DATA_DIR = "data"
MODEL_DIR = "models"

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dynamic Tee Time",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Chat messages */
    .user-msg {
        background: #1e3a5f;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e2e8f0;
    }
    .bot-msg {
        background: #1a1d27;
        border: 1px solid #262a36;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e2e8f0;
    }
    
    /* Tee time cards */
    .tee-card {
        background: #181b24;
        border: 1px solid #262a36;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }
    .tee-card:hover { border-color: #34d399; }
    .tee-card h4 { color: #34d399; margin: 0 0 8px 0; }
    .tee-card .price { font-size: 1.4em; font-weight: 700; color: #fbbf24; }
    .tee-card .detail { color: #94a3b8; font-size: 0.9em; }
    .tee-card .rec { 
        margin-top: 8px; padding: 6px 10px; 
        border-radius: 6px; font-size: 0.85em;
        display: inline-block;
    }
    .rec-high { background: #7f1d1d40; color: #f87171; border: 1px solid #7f1d1d; }
    .rec-med { background: #78350f40; color: #fbbf24; border: 1px solid #78350f; }
    .rec-low { background: #065f4640; color: #34d399; border: 1px solid #065f46; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #181b24; }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Load Data & Models
# ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    courses = pd.read_csv(f"{DATA_DIR}/courses.csv")
    tee_times = pd.read_csv(f"{DATA_DIR}/tee_times.csv")
    weather = pd.read_csv(f"{DATA_DIR}/weather_forecasts.csv")
    return courses, tee_times, weather

@st.cache_resource
def load_models():
    models = {}
    try:
        with open(f"{MODEL_DIR}/price_model.pkl", "rb") as f:
            models["price"] = pickle.load(f)
        with open(f"{MODEL_DIR}/demand_model.pkl", "rb") as f:
            models["demand"] = pickle.load(f)
        with open(f"{MODEL_DIR}/demand_labels.pkl", "rb") as f:
            models["labels"] = pickle.load(f)
        with open(f"{MODEL_DIR}/price_change_model.pkl", "rb") as f:
            models["change"] = pickle.load(f)
        with open(f"{MODEL_DIR}/feature_cols.pkl", "rb") as f:
            models["feature_cols"] = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Model files not found. Run `python3 demand_model.py` first.\n\n{e}")
        st.stop()
    return models


# ──────────────────────────────────────────────────────────────
# Query Parser (NLP-lite — no API key needed)
# ──────────────────────────────────────────────────────────────

def parse_query(query, courses_df):
    """
    Parse a natural language query into structured filters.
    This is a rule-based parser — works without an LLM API key.
    The LLM integration can be layered on top for more complex queries.
    """
    query_lower = query.lower()
    filters = {
        "max_price": None,
        "min_price": None,
        "day_pref": None,
        "time_pref": None,
        "region": None,
        "course": None,
        "party_size": None,
        "holes": None,
    }
    
    # ── Price ──
    import re
    price_match = re.search(r'under\s*\$?(\d+)', query_lower)
    if price_match:
        filters["max_price"] = int(price_match.group(1))
    price_match2 = re.search(r'less than\s*\$?(\d+)', query_lower)
    if price_match2:
        filters["max_price"] = int(price_match2.group(1))
    price_match3 = re.search(r'budget.*?\$?(\d+)', query_lower)
    if price_match3:
        filters["max_price"] = int(price_match3.group(1))
    cheap_words = ["cheap", "affordable", "budget", "value", "inexpensive"]
    if any(w in query_lower for w in cheap_words) and not filters["max_price"]:
        filters["max_price"] = 35
    
    # ── Day ──
    days = {
        "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
        "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
        "sunday": "Sunday"
    }
    for key, val in days.items():
        if key in query_lower:
            filters["day_pref"] = val
            break
    if "weekend" in query_lower:
        filters["day_pref"] = "weekend"
    if "weekday" in query_lower:
        filters["day_pref"] = "weekday"
    if "tomorrow" in query_lower:
        tomorrow = datetime.now() + timedelta(days=1)
        filters["day_pref"] = tomorrow.strftime("%A")
    if "today" in query_lower:
        filters["day_pref"] = datetime.now().strftime("%A")
    
    # ── Time ──
    if any(w in query_lower for w in ["morning", "early", "sunrise", "dawn"]):
        filters["time_pref"] = "morning"
    elif any(w in query_lower for w in ["afternoon", "midday", "lunch"]):
        filters["time_pref"] = "afternoon"
    elif any(w in query_lower for w in ["twilight", "evening", "late", "sunset"]):
        filters["time_pref"] = "twilight"
    
    time_match = re.search(r'(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)', query_lower)
    if time_match:
        hour = int(time_match.group(1))
        ampm = time_match.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if hour < 12:
            filters["time_pref"] = "morning"
        elif hour < 15:
            filters["time_pref"] = "afternoon"
        else:
            filters["time_pref"] = "twilight"
    
    # ── Region ──
    if any(w in query_lower for w in ["north atlanta", "north side", "buckhead"]):
        filters["region"] = "North Atlanta"
    elif any(w in query_lower for w in ["east atlanta", "east side"]):
        filters["region"] = "East Atlanta"
    elif any(w in query_lower for w in ["south atlanta", "south side"]):
        filters["region"] = "South Atlanta"
    elif any(w in query_lower for w in ["west atlanta", "west side"]):
        filters["region"] = "West Atlanta"
    
    # ── Specific course ──
    course_names = {
        "cherokee": "cherokee_run",
        "mystery valley": "mystery_valley",
        "bobby jones": "bobby_jones",
        "sugar creek": "sugar_creek",
        "chastain": "chastain_park",
        "browns mill": "browns_mill",
        "candler": "candler_park",
        "tup holmes": "tup_holmes",
        "holmes": "tup_holmes",
    }
    for key, val in course_names.items():
        if key in query_lower:
            filters["course"] = val
            break
    
    # ── Party size ──
    party_match = re.search(r'(\w+)some', query_lower)
    if party_match:
        word_to_num = {"two": 2, "three": 3, "four": 4, "one": 1}
        num = word_to_num.get(party_match.group(1), None)
        if num:
            filters["party_size"] = num
    party_match2 = re.search(r'(\d)\s*(?:player|golfer|people|person)', query_lower)
    if party_match2:
        filters["party_size"] = int(party_match2.group(1))
    
    # ── Holes ──
    if "9 hole" in query_lower or "nine hole" in query_lower:
        filters["holes"] = 9
    elif "18 hole" in query_lower or "eighteen hole" in query_lower:
        filters["holes"] = 18
    
    return filters


# ──────────────────────────────────────────────────────────────
# Recommendation Engine
# ──────────────────────────────────────────────────────────────

def get_recommendations(filters, courses_df, tee_times_df, weather_df, models):
    """
    Score and rank available tee times based on parsed filters.
    Returns top recommendations with predictions and explanations.
    """
    df = tee_times_df.copy()
    
    # Only available slots
    df = df[df["is_available"] == True]
    
    # Only future dates
    df = df[df["days_until"] > 0]
    
    # ── Apply filters ──
    if filters["max_price"]:
        df = df[df["price_18_holes"] <= filters["max_price"]]
    
    if filters["day_pref"]:
        if filters["day_pref"] == "weekend":
            df = df[df["is_weekend"] == True]
        elif filters["day_pref"] == "weekday":
            df = df[df["is_weekend"] == False]
        else:
            df = df[df["day_of_week"] == filters["day_pref"]]
    
    if filters["time_pref"]:
        hour = df["tee_time"].apply(lambda x: int(x.split(":")[0]))
        if filters["time_pref"] == "morning":
            df = df[hour.between(7, 10)]
        elif filters["time_pref"] == "afternoon":
            df = df[hour.between(11, 14)]
        elif filters["time_pref"] == "twilight":
            df = df[hour >= 15]
    
    if filters["region"]:
        df = df[df["region"] == filters["region"]]
    
    if filters["course"]:
        df = df[df["course_id"] == filters["course"]]
    
    if filters["party_size"]:
        df = df[df["spots_available"] >= filters["party_size"]]
    
    if filters["holes"]:
        course_holes = courses_df.set_index("course_id")["holes"].to_dict()
        df = df[df["course_id"].map(course_holes) == filters["holes"]]
    
    if len(df) == 0:
        return [], "No tee times found matching your criteria. Try adjusting your filters."
    
    # ── Score each slot ──
    # Merge course info
    df = df.merge(courses_df[["course_id", "tier", "base_price_18", "rating", "holes"]], 
                  on="course_id", how="left")
    
    # Build features for prediction
    tier_map = {"value": 0, "mid": 1, "premium": 2}
    df["hour"] = df["tee_time"].apply(lambda x: int(x.split(":")[0]))
    df["tier_num"] = df["tier"].map(tier_map).fillna(1)
    
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["day_num"] = df["day_of_week"].map(day_map)
    df["is_weekend_int"] = df["is_weekend"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).astype(int)
    
    results = []
    seen = set()  # Deduplicate similar slots
    
    # Sample diverse options (not all from same course/time)
    df_sorted = df.sort_values("price_18_holes")
    
    for _, row in df_sorted.iterrows():
        # Deduplicate: one slot per course per hour
        key = (row["course_id"], row["hour"], row["date"])
        if key in seen:
            continue
        seen.add(key)
        
        # Get weather for this date
        w = weather_df[weather_df["date"] == row["date"]]
        if not w.empty:
            w = w.iloc[0]
            temp = w["high_temp_f"]
            precip = w["precip_chance_pct"]
            conditions = w["conditions"]
        else:
            temp, precip, conditions = 60, 10, "Unknown"
        
        # Build prediction features
        is_weekend = int(row["day_num"] >= 5)
        is_morning = int(7 <= row["hour"] <= 9)
        
        features = {
            "hour": row["hour"],
            "is_morning_prime": is_morning,
            "is_afternoon": int(12 <= row["hour"] <= 14),
            "is_twilight": int(row["hour"] >= 15),
            "day_num": row["day_num"],
            "is_weekend": is_weekend,
            "is_friday": int(row["day_num"] == 4),
            "days_until_tee": row["days_until"],
            "is_last_minute": int(row["days_until"] <= 2),
            "is_advance": int(row["days_until"] >= 7),
            "booking_window_sq": row["days_until"] ** 2,
            "high_temp_f": float(temp),
            "precip_chance_pct": float(precip),
            "is_rainy": int(float(precip) > 40),
            "is_cold": int(float(temp) < 50),
            "is_perfect_weather": int(65 <= float(temp) <= 82 and float(precip) < 20),
            "tier_num": row["tier_num"],
            "base_price_18": float(row["base_price_18"]),
            "weekend_x_morning": is_weekend * is_morning,
            "weekend_x_lastmin": is_weekend * int(row["days_until"] <= 2),
            "rain_x_weekend": int(float(precip) > 40) * is_weekend,
            "temp_x_weekend": float(temp) * is_weekend,
        }
        
        # Course dummies
        for c in courses_df["course_id"].unique():
            features[f"course_{c}"] = int(c == row["course_id"])
        
        X = pd.DataFrame([features])
        feature_cols = models["feature_cols"]
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols]
        
        # Predict
        demand_code = models["demand"].predict(X)[0]
        demand_label = models["labels"].inverse_transform([demand_code])[0]
        demand_proba = models["demand"].predict_proba(X)[0]
        price_change = models["change"].predict(X)[0]
        
        results.append({
            "course_name": row["course_name"],
            "course_id": row["course_id"],
            "date": row["date"],
            "day": row["day_of_week"],
            "tee_time": row["tee_time"],
            "price": row["price_18_holes"],
            "spots": row["spots_available"],
            "demand": demand_label,
            "demand_conf": max(demand_proba) * 100,
            "price_change": round(price_change, 2),
            "weather_temp": temp,
            "weather_precip": precip,
            "weather_cond": conditions,
            "rating": row.get("rating", 0),
            "holes": row.get("holes", 18),
            "region": row["region"],
        })
        
        if len(results) >= 8:
            break
    
    # Sort by a composite score: low price + low demand + good weather
    for r in results:
        price_score = 100 - r["price"]  # Lower price = higher score
        demand_score = {"LOW": 30, "MEDIUM": 15, "HIGH": 0}[r["demand"]]
        weather_score = 20 if r["weather_precip"] < 20 else (10 if r["weather_precip"] < 40 else 0)
        r["score"] = price_score + demand_score + weather_score
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results[:5], None


def format_response(results, filters, error=None):
    """Format results into a conversational response."""
    if error:
        return error
    
    if not results:
        return "I couldn't find any tee times matching those criteria. Try widening your search — maybe a different day or a higher budget?"
    
    # Build natural language summary
    parts = []
    if filters["day_pref"]:
        parts.append(filters["day_pref"])
    if filters["time_pref"]:
        parts.append(filters["time_pref"])
    if filters["region"]:
        parts.append(f"in {filters['region']}")
    if filters["max_price"]:
        parts.append(f"under ${filters['max_price']}")
    
    summary = " ".join(parts) if parts else "upcoming"
    
    response = f"Here are the best {summary} tee times I found:\n\n"
    
    return response


# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────

def main():
    # Load everything
    courses_df, tee_times_df, weather_df = load_data()
    models = load_models()
    
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⛳ Dynamic Tee Time")
        st.markdown("*AI-Powered Booking Assistant*")
        st.markdown("---")
        
        st.markdown("### 📊 Coverage")
        st.metric("Courses", len(courses_df))
        st.metric("Available Slots", f"{tee_times_df[tee_times_df['is_available'] == True].shape[0]:,}")
        
        st.markdown("---")
        st.markdown("### 🏌️ Courses")
        for _, c in courses_df.iterrows():
            holes_label = f"{c['holes']}H"
            st.markdown(f"**{c['name']}**  \n{c.get('region', '')} · {holes_label} · ⭐ {c.get('rating', 'N/A')}")
        
        st.markdown("---")
        st.markdown("### 💡 Try asking:")
        st.markdown("""
        - *Saturday morning under $50*
        - *Cheapest weekday tee time*
        - *Bobby Jones this weekend*
        - *Twilight rate at Chastain*
        - *North Atlanta foursome*
        """)
    
    # ── Main area ──
    st.markdown("# ⛳ Dynamic Tee Time Booking Assistant")
    st.markdown("Tell me what you're looking for and I'll find the best tee times with demand forecasts.")
    st.markdown("---")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hey! I'm your tee time assistant for Atlanta golf courses. Tell me what you're looking for — day, time, budget, area — and I'll find the best options with price forecasts. 🏌️"
        })
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="⛳"):
                st.write(msg["content"])
                if "results" in msg:
                    display_results(msg["results"])
    
    # Chat input
    if prompt := st.chat_input("e.g. Find me a Saturday morning twosome under $50 in North Atlanta"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Parse and search
        filters = parse_query(prompt, courses_df)
        results, error = get_recommendations(filters, courses_df, tee_times_df, weather_df, models)
        
        # Build response
        response_text = format_response(results, filters, error)
        
        # Display assistant response
        with st.chat_message("assistant", avatar="⛳"):
            st.write(response_text)
            if results:
                display_results(results)
                
                # Add comparison insight
                if len(results) >= 2:
                    best = results[0]
                    alt = results[1]
                    if best["price"] < alt["price"]:
                        savings = alt["price"] - best["price"]
                        st.markdown(f"""
                        ---
                        💡 **Quick comparison:** {best['course_name']} at {best['tee_time']} is 
                        **${savings} cheaper** than {alt['course_name']}. 
                        {"The price at " + best['course_name'] + " is expected to rise $" + str(abs(best['price_change'])) + " — book soon!" if best['price_change'] > 3 else "Pricing is stable, so no rush."}
                        """)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "results": results
        })


def display_results(results):
    """Render tee time result cards."""
    for i, r in enumerate(results):
        # Demand badge styling
        if r["demand"] == "HIGH":
            badge_class = "rec-high"
            badge_text = "🔴 High Demand"
        elif r["demand"] == "MEDIUM":
            badge_class = "rec-med"
            badge_text = "🟡 Moderate Demand"
        else:
            badge_class = "rec-low"
            badge_text = "🟢 Low Demand"
        
        # Price change text
        if r["price_change"] > 5:
            change_text = f"⚠️ Expected to rise **+${r['price_change']:.0f}** by tee date"
        elif r["price_change"] > 2:
            change_text = f"📈 May increase **+${r['price_change']:.0f}** by tee date"
        else:
            change_text = "✅ Price is stable"
        
        # Recommendation
        if r["demand"] == "HIGH" and r["price_change"] > 3:
            rec = "Book now — high demand, price rising"
        elif r["demand"] == "LOW":
            rec = "Great deal — no rush to book"
        elif r["price_change"] > 5:
            rec = "Consider booking soon — price trending up"
        else:
            rec = "Good value"
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 2])
            
            with col1:
                st.markdown(f"**{'⭐ ' if i == 0 else ''}{r['course_name']}**")
                st.markdown(f"📅 {r['day']} {r['date']} at **{r['tee_time']}**")
                st.markdown(f"📍 {r['region']} · {r['holes']} holes · {r['spots']} spots open")
            
            with col2:
                st.markdown(f"### ${r['price']}")
                st.caption("per player")
            
            with col3:
                st.markdown(f"**{badge_text}**")
                st.markdown(change_text)
                st.markdown(f"🌤️ {r['weather_temp']}°F, {r['weather_cond']}")
            
            st.caption(f"💡 {rec}")
            st.markdown("---")


if __name__ == "__main__":
    main()
