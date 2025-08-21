import streamlit as st
from streamlit_option_menu import option_menu
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Helper function: NASA GHI Data
# ----------------------------
def get_solar_ghi(lat, lon):
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=ALLSKY_KT,ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&start=2020&end=2020&format=JSON"
        r = requests.get(url, timeout=10)
        data = r.json()
        ghi = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        return pd.DataFrame({"Month": list(ghi.keys()), "GHI": list(ghi.values())})
    except:
        # fallback values if NASA blocked
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        values = [4.5,5.0,5.5,6.0,6.5,6.2,5.8,5.5,5.0,4.8,4.6,4.4]
        return pd.DataFrame({"Month": months, "GHI": values})

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Solar Mapping System", layout="wide")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "Solar Navigator",
        ["Home", "Solar Forecast", "Panel Layout", "Chatbot", "Reports"],
        icons=["house", "sun", "grid", "chat", "file-earmark-text"],
        menu_icon="solar-panel",
        default_index=0,
    )

# ----------------------------
# Background + Header by Page
# ----------------------------
backgrounds = {
    "Home": "https://i.ibb.co/Jk1twXM/solar-home.jpg",
    "Solar Forecast": "https://i.ibb.co/2Ymx3dV/solar-forecast.jpg",
    "Panel Layout": "https://i.ibb.co/BrMxxMs/solar-layout.jpg",
    "Chatbot": "https://i.ibb.co/5xH6cY1/solar-chat.jpg",
    "Reports": "https://i.ibb.co/D5XbyVq/solar-report.jpg",
}

headers = {
    "Home": "☀️ Switch to Solar – Bright Future Ahead!",
    "Solar Forecast": "📈 Predict Your Sunshine Savings",
    "Panel Layout": "🔲 Plan Your Panels – Power Every Rooftop",
    "Chatbot": "💬 Ask Anything About Solar",
    "Reports": "📑 Your Solar Journey – Transparent & Simple",
}

# CSS for background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{backgrounds[selected]}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(255,255,255,0.85);
        padding: 2rem;
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title(headers[selected])

# ----------------------------
# Pages
# ----------------------------
if selected == "Home":
    st.markdown("### Welcome to the **Solar Mapping & Recommendation System** 🌍")
    st.write("Select a city on the map to see its solar potential.")

    # Simple city-labeled map
    city_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
    cities = {
        "Delhi": [28.7041, 77.1025],
        "Mumbai": [19.0760, 72.8777],
        "Raipur": [21.2514, 81.6296],
        "Chennai": [13.0827, 80.2707],
        "Kolkata": [22.5726, 88.3639],
    }
    for city, coords in cities.items():
        folium.Marker(location=coords, popup=city, tooltip=city).add_to(city_map)

    map_data = st_folium(city_map, width=800, height=500)
    if map_data and map_data["last_clicked"]:
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.success(f"📍 Selected Location: Lat {lat:.3f}, Lon {lon:.3f}")

elif selected == "Solar Forecast":
    st.write("Upload a location or pick one on Home page to see solar forecast.")
    lat = 21.25
    lon = 81.63
    df = get_solar_ghi(lat, lon)
    fig = px.bar(df, x="Month", y="GHI", title="Monthly Solar Irradiance (kWh/m²/day)", color="GHI")
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Panel Layout":
    st.write("Panel layout simulation 🚀")
    # Assume 1kW ≈ 3 panels, each 1.7 m²
    area = st.slider("Rooftop Area (m²)", 10, 200, 50)
    panel_area = 1.7
    panel_count = int(area // panel_area)
    st.success(f"You can fit about **{panel_count} panels** on this roof.")
    st.image("https://i.ibb.co/0MtzxQ2/panel-layout-demo.png", caption="Sample Panel Layout")

elif selected == "Chatbot":
    st.write("🤖 Solar Assistant at your service")
    user_q = st.text_input("Ask me about solar power:")
    if user_q:
        if "cost" in user_q.lower():
            st.info("💰 Solar panel installation costs ~₹40–60k per kW in India.")
        elif "saving" in user_q.lower():
            st.info("💵 Average savings: ₹10–15 per kWh depending on your state tariff.")
        elif "life" in user_q.lower():
            st.info("⚡ Solar panels last 25+ years with minimal maintenance.")
        else:
            st.info("🌞 Solar is clean, renewable, and reduces your electricity bill!")

elif selected == "Reports":
    st.write("📑 Download your solar assessment report (coming soon)")
    st.download_button("⬇ Download Report", "This is a demo solar report.", file_name="solar_report.txt")
