# paste the full code from above here
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AQI Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.dropna(subset=['AQI','AQI_Bucket'])
    return df

df = load_data()

st.title("🌍 AQI Prediction & Forecasting Dashboard")

# Sidebar
cities = df['City'].unique()
city = st.sidebar.selectbox("Select City", sorted(cities))
horizon_days = st.sidebar.slider("Forecast horizon (days)", 30, 365, 180, step=30)

city_df = df[df['City']==city].copy()
city_df = city_df.fillna(0)

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3 = st.tabs(["📊 Predict AQI", "🔮 Forecast", "🌱 Recommendations"])

# --- Tab 1: Prediction ---
with tab1:
    st.subheader("📊 Predict AQI Category from Pollutants")
    features = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']
    inputs = []
    for f in features:
        val = st.number_input(f"Enter {f}", min_value=0.0, value=float(city_df[f].mean()))
        inputs.append(val)

    if st.button("Predict AQI Category"):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        X = city_df[features]
        y = city_df['AQI_Bucket']
        clf.fit(X, y)
        pred = clf.predict([inputs])[0]
        st.success(f"Predicted AQI Category: **{pred}**")

# --- Tab 2: Forecast ---
with tab2:
    st.subheader(f"🔮 AQI Forecast for {city} (next {horizon_days} days)")
    prophet_df = city_df[['Datetime','AQI']].dropna().rename(columns={'Datetime':'ds','AQI':'y'})

    if len(prophet_df) > 20:
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=horizon_days)
        forecast = m.predict(future)

        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.warning("Not enough data for forecasting.")

# --- Tab 3: Recommendations ---
with tab3:
    st.subheader("🌱 Recommended Steps for AQI Improvement")

    st.markdown("""
    ### 🚗 Vehicle Emission Control
    - Regular PUC checks for vehicles  
    - Encourage EV adoption & charging stations  
    - Carpooling, public transport, and cycling lanes  
    - Odd-even traffic rules during high pollution days  

    ### 🌳 Green Zone Promotion
    - Create low-emission green zones in cities  
    - Rooftop gardening & urban plantations  
    - Green belts near highways and industrial areas  

    ### 🎆 Festival-based Emission Warnings
    - Issue advance AQI alerts before festivals  
    - Promote eco-friendly 'green crackers'  
    - Encourage community celebrations with lights/laser shows  
    - Awareness campaigns during Diwali & Holi  

    ### 🏭 Industrial & Construction Regulation
    - Dust control at construction sites  
    - Install filters and scrubbers in industries  
    - Relocate highly polluting industries away from residential zones  

    ### 📡 Real-time Monitoring & Alerts
    - IoT-based AQI sensors across cities  
    - Mobile apps with AQI & health advisories  
    - Integration with health emergency systems  

    ### 🌍 Long-term Sustainable Measures
    - Shift towards renewable energy  
    - Waste segregation & reduced open burning  
    - Climate-resilient urban planning  
    """)
