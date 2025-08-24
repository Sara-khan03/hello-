import streamlit as st
import requests
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import random

st.set_page_config(page_title="Solar Mapping & Recommendation", layout="wide")

# ---------- Helper Functions ----------
def get_lat_lon(place_name):
    geolocator = Nominatim(user_agent="solar_app")
    location = geolocator.geocode(place_name + ", India")
    if location:
        return location.latitude, location.longitude
    return None, None

def fetch_ghi(lat, lon):
    """Fetch monthly average GHI from NASA POWER API"""
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=ALLSKY_KT,ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&format=JSON&start=2020&end=2020"
        r = requests.get(url).json()
        ghi_data = r["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        # return annual average
        annual_avg = sum(ghi_data.values()) / len(ghi_data)
        return annual_avg
    except:
        return 4.5  # fallback default

def calculate_solar(area_m2, lat, lon):
    panel_capacity_kw = 0.33
    panel_area = 1.7 * 1.0
    panels_fit = int(area_m2 // panel_area)
    system_capacity_kw = round(panels_fit * panel_capacity_kw, 2)

    ghi = fetch_ghi(lat, lon)  # <-- UPDATED HERE
    annual_output_kwh = round(system_capacity_kw * ghi * 365, 2)

    cost_per_kw = 50000
    install_cost = system_capacity_kw * cost_per_kw
    annual_savings = annual_output_kwh * 6
    payback_years = round(install_cost / annual_savings, 1)

    return {
        "Panels Fit": panels_fit,
        "System Capacity (kW)": system_capacity_kw,
        "Annual Solar Output (kWh)": annual_output_kwh,
        "Installation Cost (₹)": install_cost,
        "Annual Savings (₹)": annual_savings,
        "Payback Period (Years)": payback_years,
        "Suitability": "Good" if annual_output_kwh > 2000 else "Poor",
        "Avg GHI (kWh/m²/day)": round(ghi, 2)
    }

# ---------- UI ----------
st.title("☀️ Solar Mapping & Recommendation System with Chatbot")

# Location Input
st.subheader("📍 Enter Your Location")
city_list = ["Delhi", "Mumbai", "Bangalore", "Raipur", "Kolkata", "Chennai", "Hyderabad", "Jaipur", "Lucknow", "Pune"]

selected_city = st.selectbox("Select City", city_list)
custom_place = st.text_input("Enter Specific Area/Office (e.g., Collector Office, IIT Campus)", "")

if custom_place:
    place_query = custom_place + ", " + selected_city
else:
    place_query = selected_city

lat, lon = get_lat_lon(place_query)

if lat and lon:
    st.success(f"✅ Location found: {place_query} ({lat:.4f}, {lon:.4f})")

    # Show map
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], tooltip=place_query).add_to(m)
    st_map = st_folium(m, width=700, height=450)

    # Rooftop area input
    rooftop_area = st.number_input("Enter Rooftop Area (sq.m)", min_value=10, value=100, step=10)

    if st.button("Calculate Solar Potential"):
        results = calculate_solar(rooftop_area, lat, lon)

        st.subheader("📊 Solar Recommendation Results")
        for key, value in results.items():
            st.write(f"**{key}:** {value}")

else:
    st.error("⚠️ Could not find the location. Try another area/office.")

# Chatbot
st.subheader("💬 Solar Chatbot")
user_q = st.text_input("Ask me anything about solar installation:")
if user_q:
    responses = [
        "Solar energy reduces electricity bills by up to 80%.",
        "You can avail government subsidies on rooftop solar installations.",
        "The ideal rooftop should have at least 4 hours of direct sunlight.",
        "Payback period usually ranges from 4-6 years."
    ]
    st.info(random.choice(responses))
