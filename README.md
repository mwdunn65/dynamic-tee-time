Dynamic Tee Time – AI Pricing & Booking Assistant
Overview
This project builds an AI-powered golf tee time pricing and booking assistant for the Atlanta market. The system predicts demand, estimates dynamic pricing changes, and allows users to query tee times conversationally through a Streamlit interface.
Key Components
app.py – Streamlit conversational interface
demand_model.py – Demand prediction model
price_model.pkl – Trained pricing model
collect_tee_times.py – Data ingestion pipeline
fetch_weather.py – Weather integration
data/ – Tee time and weather datasets
models/ – Trained model artifacts

How to Run Locally
git clone git@github.com:mwdunn65/dynamic-tee-time.git
cd dynamic-tee-time
pip install -r requirements.txt
python3 -m streamlit run app.py
