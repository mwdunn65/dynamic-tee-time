"""
Tee Time Manual Data Collection Tool
======================================
Generates a CSV template for manually recording tee time prices from 
the City of Atlanta golf booking sites and other courses.

Also includes a web scraping starter (for the TeeItUp booking sites)
that you can try running locally.

Booking URLs:
  - Chastain Park:  https://chastain-park.book.teeitup.com/
  - Browns Mill:    https://browns-mill-golf-course.book.teeitup.com/
  - Candler Park:   https://candler-park-golf-course.book.teeitup.com/
  - Tup Holmes:     https://alfred-tup-holmes.book.teeitup.com/
  - GolfNow:        https://www.golfnow.com/tee-times/facility/1741-chastain-park-golf-course/search
  - GolfLink:       https://www.golflink.com/golf-courses/ga/atlanta/

Instructions:
  1. Run this script to generate the blank template
  2. Visit each booking site 2-3 times per week
  3. Record available tee times and prices in the CSV
  4. After 3-4 weeks, you'll have real data to calibrate the model

Usage:
    python collect_tee_times.py --template     # Generate blank template
    python collect_tee_times.py --validate     # Validate collected data
"""

import pandas as pd
import os
import sys
from datetime import datetime, timedelta

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# City of Atlanta courses with their booking URLs
COA_COURSES = {
    "chastain_park": {
        "name": "Chastain Park Golf Course",
        "booking_url": "https://chastain-park.book.teeitup.com/",
        "golfnow_url": "https://www.golfnow.com/tee-times/facility/1741-chastain-park-golf-course/search",
    },
    "browns_mill": {
        "name": "Browns Mill Golf Course",
        "booking_url": "https://browns-mill-golf-course.book.teeitup.com/",
    },
    "candler_park": {
        "name": "Candler Park Golf Course",
        "booking_url": "https://candler-park-golf-course.book.teeitup.com/",
    },
    "tup_holmes": {
        "name": 'Alfred "Tup" Holmes Golf Course',
        "booking_url": "https://alfred-tup-holmes.book.teeitup.com/",
    },
}

# Known rates from cityofatlantagolf.com/rates/
KNOWN_RATES = """
╔══════════════════════════════════════════════════════════════════╗
║              CITY OF ATLANTA GOLF - OFFICIAL RATES              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  BROWNS MILL & CHASTAIN (18 holes)                              ║
║  ────────────────────────────────────────────────────────────    ║
║  Mon-Thu:  Resident $23.50 | Non-Resident $28.00                ║
║           Senior(55+) Res $9.40 | Non-Res $11.00               ║
║           Junior Res $13.50 | Non-Res $14.50                   ║
║           Twilight Res $23.50 | Non-Res $29.50                 ║
║           Super Twilight $19.25                                 ║
║           Cart Fee $13.50/person                                ║
║                                                                  ║
║  Fri-Sun:  Resident $26.50 | Non-Resident $32.00               ║
║           Senior(55+) Res $15.00 | Non-Res $16.00              ║
║           Junior Res $15.50 | Non-Res $16.75                   ║
║           Twilight Res $28.00 | Non-Res $31.50                 ║
║           Super Twilight $19.25                                 ║
║           Cart Fee $13.50/person                                ║
║                                                                  ║
║  CANDLER PARK (9 holes, walking only)                           ║
║  ────────────────────────────────────────────────────────────    ║
║  Mon-Thu:  Resident $9.00 | Non-Resident $11.00                ║
║           Senior Res $6.75 | Non-Res $7.25                     ║
║           Junior Res $7.25 | Non-Res $8.25                     ║
║                                                                  ║
║  Fri-Sun:  Resident $11.00 | Non-Resident $13.00               ║
║           Senior Res $7.25 | Non-Res $8.25                     ║
║           Junior Res $7.75 | Non-Res $8.25                     ║
║           Pull Cart $4.00                                       ║
║                                                                  ║
║  ALFRED "TUP" HOLMES (18 holes)                                 ║
║  ────────────────────────────────────────────────────────────    ║
║  Mon-Thu:  Regular $20.50 walking | $28.00 with cart            ║
║           Twilight $19.25 (walking or riding)                   ║
║           9 Holes $18.00 (walking or riding)                    ║
║                                                                  ║
║  Fri-Sun:  Regular $27.50 walking | $35.00 with cart            ║
║           Twilight $19.25 (walking or riding)                   ║
║           9 Holes $25.00 (walking or riding)                    ║
║                                                                  ║
║  Source: https://www.cityofatlantagolf.com/rates/               ║
╚══════════════════════════════════════════════════════════════════╝
"""


def generate_template():
    """Generate a blank CSV template for manual data collection."""
    
    print("📋 Generating tee time collection template...")
    print(KNOWN_RATES)
    
    # Create template with example rows
    template_rows = [
        # Example filled rows to show the format
        {
            "collection_date": "2026-02-12",
            "collection_time": "10:30",
            "course_id": "chastain_park",
            "course_name": "Chastain Park Golf Course",
            "tee_date": "2026-02-15",
            "tee_time": "08:00",
            "price_displayed": 32.00,
            "price_type": "non_resident",
            "holes": 18,
            "includes_cart": False,
            "spots_available": 4,
            "is_hot_deal": False,
            "booking_platform": "teeitup",
            "notes": "Saturday morning, full price",
        },
        {
            "collection_date": "2026-02-12",
            "collection_time": "10:30",
            "course_id": "chastain_park",
            "course_name": "Chastain Park Golf Course",
            "tee_date": "2026-02-15",
            "tee_time": "08:10",
            "price_displayed": 32.00,
            "price_type": "non_resident",
            "holes": 18,
            "includes_cart": False,
            "spots_available": 2,
            "is_hot_deal": False,
            "booking_platform": "teeitup",
            "notes": "",
        },
        {
            "collection_date": "2026-02-12",
            "collection_time": "10:35",
            "course_id": "chastain_park",
            "course_name": "Chastain Park Golf Course",
            "tee_date": "2026-02-15",
            "tee_time": "14:00",
            "price_displayed": 28.00,
            "price_type": "non_resident",
            "holes": 18,
            "includes_cart": False,
            "spots_available": 4,
            "is_hot_deal": False,
            "booking_platform": "golfnow",
            "notes": "Twilight rate",
        },
        {
            "collection_date": "2026-02-12",
            "collection_time": "10:40",
            "course_id": "candler_park",
            "course_name": "Candler Park Golf Course",
            "tee_date": "2026-02-14",
            "tee_time": "09:00",
            "price_displayed": 11.00,
            "price_type": "non_resident",
            "holes": 9,
            "includes_cart": False,
            "spots_available": 3,
            "is_hot_deal": False,
            "booking_platform": "teeitup",
            "notes": "Friday, walking only",
        },
    ]
    
    df = pd.DataFrame(template_rows)
    
    # Save with example rows
    df.to_csv(f"{OUTPUT_DIR}/tee_times_collected_TEMPLATE.csv", index=False)
    
    # Also save a truly blank one (headers only)
    pd.DataFrame(columns=df.columns).to_csv(
        f"{OUTPUT_DIR}/tee_times_collected.csv", index=False
    )
    
    print(f"✅ Template with examples → tee_times_collected_TEMPLATE.csv")
    print(f"✅ Blank collection file  → tee_times_collected.csv")
    
    print(f"\n📝 DATA COLLECTION GUIDE:")
    print(f"{'─' * 50}")
    print(f"1. Visit these booking sites 2-3x per week:")
    for cid, info in COA_COURSES.items():
        print(f"   • {info['name']}")
        print(f"     {info['booking_url']}")
    print(f"\n2. For each visit, record:")
    print(f"   • When you're collecting (collection_date/time)")
    print(f"   • Each available tee time for the next 7-14 days")
    print(f"   • The displayed price and number of spots")
    print(f"   • Whether it's a hot deal / discounted")
    print(f"\n3. KEY: Collect the SAME future dates multiple times")
    print(f"   This shows how prices/availability change over time")
    print(f"   Example: Record Sat 2/22 times on Mon, Wed, and Fri")
    print(f"\n4. After 3-4 weeks you'll have ~500-1000 real data points")
    print(f"   to calibrate and validate the synthetic model")
    
    return df


def validate_collected_data():
    """Check collected data for common issues."""
    filepath = f"{OUTPUT_DIR}/tee_times_collected.csv"
    
    if not os.path.exists(filepath):
        print("❌ No collected data file found. Run --template first.")
        return
    
    df = pd.read_csv(filepath)
    
    if len(df) == 0:
        print("📭 Collection file is empty. Start collecting data!")
        return
    
    print(f"📊 Collected Data Validation")
    print(f"{'═' * 50}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['tee_date'].min()} to {df['tee_date'].max()}")
    print(f"Courses covered: {df['course_id'].nunique()}")
    print(f"Collection sessions: {df['collection_date'].nunique()}")
    
    # Check for issues
    issues = []
    if df['price_displayed'].isna().any():
        issues.append(f"⚠️  {df['price_displayed'].isna().sum()} rows missing price")
    if df['tee_time'].isna().any():
        issues.append(f"⚠️  {df['tee_time'].isna().sum()} rows missing tee time")
    if df['course_id'].nunique() < 3:
        issues.append(f"⚠️  Only {df['course_id'].nunique()} courses — try to cover at least 3")
    if df['collection_date'].nunique() < 3:
        issues.append(f"⚠️  Only {df['collection_date'].nunique()} collection sessions — need more for price tracking")
    
    if issues:
        print(f"\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n✅ Data looks good!")
    
    print(f"\n📊 Records by course:")
    print(df['course_name'].value_counts().to_string())
    print(f"\n📊 Records by collection date:")
    print(df['collection_date'].value_counts().to_string())


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate_collected_data()
    else:
        generate_template()
