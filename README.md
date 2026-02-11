# Dynamic Tee Time Demand Model

This project forecasts golf tee time demand using historical booking data and weather inputs.

The goal is to estimate demand patterns and improve dynamic pricing and scheduling decisions.

---

## Project Structure

- `app.py` — Main entry point  
- `collect_tee_times.py` — Tee time data collection  
- `fetch_weather.py` — Weather data integration  
- `real_weather_data.py` — Real weather ingestion logic  
- `generate_synthetic_data.py` — Synthetic dataset generator  
- `explore_data.py` — Data exploration utilities  
- `demand_model.py` — Machine learning demand forecasting logic  
- `/data` — Data storage  
- `/models` — Saved models  

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/mwdunn65/dynamic-tee-time.git
cd dynamic-tee-time
```
### 2. Create a virtual environment
Mac / Linux:

```
python3 -m venv venv
source venv/bin/activate
```
Windows:
```
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies
```
pip install pandas numpy requests scikit-learn matplotlib seaborn
```
### 4. Run the application
```
python app.py
```
### Requirements
- Python 3.10+
- pip
