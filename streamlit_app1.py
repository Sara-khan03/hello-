import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import numpy as np

# ----------------- Predefined States and Cities of India -----------------
india_states_cities = {
    "Delhi": ["New Delhi", "Dwarka", "Connaught Place"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangalore"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Durg"],
}

# ----------------- Helper Functions -----------------
def get_coordinates(place_name):
    """Fetch latitude and longitude of the given place using Geopy."""
    geolocator = Nominatim(user_agent="solar_app")
    location = geolocator.geocode(f"{place_name}, India")
    if location:
        return location.latitude, location.longitude
    return None, None

def forecast_monthly_output(system_capacity_kw):
    """Simple monthly forecast based on capacity."""
    base = np.array([80, 95, 110, 130, 150, 160, 170, 165, 140, 120, 100, 85])
    forecast = (system_capacity_kw / 5.0) * base
    return forecast

def calculate_recommendation(area_m2):
    """Calculate rooftop recommendation details."""
    panel_area = 1.7 * 1.0  # ~1.7 m² per panel
    panels_fit = int(area_m2 // panel_area)
    system_capacity_kw = round(panels_fit * 0.33, 2)
    annual_output_kwh = int(system_capacity_kw * 1500)
    installation_cost = int(system_capacity_kw * 60000)
    annual_savings = int(annual_output_kwh * 6)
    payback_years = round(installation_cost / annual_savings, 1)
    suitability = "Highly Suitable" if system_capacity_kw >= 2 else "Moderate Suitability"

    return {
        "Rooftop Area (m²)": area_m2,
        "Panels Fit": panels_fit,
        "System Capacity (kW)": system_capacity_kw,
        "Annual Solar Output (kWh)": annual_output_kwh,
        "Installation Cost (₹)": installation_cost,
        "Annual Savings (₹)": annual_savings,
        "Payback Period (years)": payback_years,
        "Suitability": suitability
    }

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Solar Mapping & Recommendation System", layout="wide")
st.title("🌞 Solar Mapping & Recommendation System (India)")

# -------- Location Selection --------
st.subheader("📍 Enter Your Location")

state = st.selectbox("Select State", list(india_states_cities.keys()))
city = st.selectbox("Select City", india_states_cities[state])

full_location = f"{city}, {state}, India"
lat, lon = get_coordinates(full_location)

if lat and lon:
    st.success(f"Location selected: {full_location} ({lat:.4f}, {lon:.4f})")

    # -------- Map Display --------
    st.subheader("🗺️ Location on Map (Satellite View)")
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="Esri.WorldImagery")  # Satellite view
    folium.Marker([lat, lon], tooltip=full_location).add_to(m)
    st_folium(m, width=700, height=500)

    # -------- Rooftop Area Input --------
    st.subheader("🏠 Rooftop Details")
    area_m2 = st.number_input("Enter Rooftop Area (m²)", min_value=10, max_value=1000, value=100)

    if st.button("🔎 Get Solar Recommendation"):
        results = calculate_recommendation(area_m2)

        # Display Recommendation
        st.subheader("📊 Recommendation Results")
        for key, value in results.items():
            st.write(f"**{key}:** {value}")

        # Forecast Chart
        st.subheader("📈 Monthly Solar Energy Output Forecast")
        forecast = forecast_monthly_output(results["System Capacity (kW)"])
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, ax = plt.subplots()
        ax.plot(months, forecast, marker="o")
        ax.set_title("Monthly Solar Output Forecast (kWh)")
        ax.set_ylabel("kWh")
        st.pyplot(fig)

else:
    st.error("⚠️ Could not fetch coordinates for the selected city. Please try another.")

# ----------------- Chatbot -----------------
st.subheader("💬 Ask SolarBot (Chatbot)")
user_q = st.text_input("Type your question here...")
if st.button("Ask"):
    if "cost" in user_q.lower():
        st.write("💡 Installation cost depends on system capacity (~₹60,000 per kW).")
    elif "payback" in user_q.lower():
        st.write("💡 Payback period is usually between 4-6 years depending on savings.")
    elif "savings" in user_q.lower():
        st.write("💡 Annual savings are calculated as output × ₹6 per unit.")
    else:
        st.write("🤖 I can help with cost, payback, savings, or suitability of solar panels.")
