# app.py
# Solar Mapping & Recommendation System — Streamlit (with login, dashboard, save/load)
import math, sqlite3
import json, time
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solar Mapping & Recommendation System",
                   page_icon="☀️", layout="wide")

# ---------------------------- Database ---------------------------- #
conn = sqlite3.connect("solar_app.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS simulations (
    id INTEGER PRIMARY KEY, user_id INTEGER, timestamp TEXT,
    lat REAL, lon REAL, roof_w REAL, roof_h REAL, clearance REAL,
    panel_w REAL, panel_h REAL, panel_watt REAL, orientation TEXT,
    performance_ratio REAL, shading_factor REAL, tilt_deg REAL,
    cost_per_kw REAL, tariff REAL,
    system_kw REAL, rooftop_area REAL, annual_energy REAL,
    annual_savings REAL, payback_years REAL, suitability_score REAL
)''')
conn.commit()

# ---------------------------- Helpers ---------------------------- #

@st.cache_data(show_spinner=False)
def fetch_nasa_power_monthly(lat, lon):
    """Fetch monthly GHI (ALLSKY_SFC_SW_DWN) from NASA POWER"""
    urls = [
        f"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&format=JSON",
        f"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=ALLSKY_SFC_SW_DWN&community=SSE&longitude={lon}&latitude={lat}&format=JSON"
    ]
    month_map = {
        "01":"Jan","02":"Feb","03":"Mar","04":"Apr","05":"May","06":"Jun",
        "07":"Jul","08":"Aug","09":"Sep","10":"Oct","11":"Nov","12":"Dec"
    }
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            values = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
            monthly = {month_map[k]: float(v) for k,v in values.items()}
            return monthly
        except:
            continue
    return None

def days_in_months(year=2024):
    return {"Jan":31,"Feb":29 if (year%4==0 and (year%100!=0 or year%400==0)) else 28,
            "Mar":31,"Apr":30,"May":31,"Jun":30,"Jul":31,"Aug":31,"Sep":30,"Oct":31,"Nov":30,"Dec":31}

def deg2rad(d): return d*math.pi/180.0

def suitability_score(roof_area_m2, monthly_psh_dict, shading_factor, tilt_deg, lat):
    if not monthly_psh_dict:
        resource_score = 50
    else:
        avg_psh = np.mean(list(monthly_psh_dict.values()))
        resource_score = np.clip(20 + (avg_psh-3)*18, 20, 95)
    area_score = 30 + np.clip((roof_area_m2-10)*1.5, 0, 65)
    shade_score = np.clip(100 - shading_factor*100, 10, 100)
    ideal = abs(abs(lat) - tilt_deg)
    tilt_score = np.clip(100 - ideal*2.2, 40, 100)
    weights = [0.35,0.25,0.2,0.2]
    final = resource_score*weights[0] + area_score*weights[1] + shade_score*weights[2] + tilt_score*weights[3]
    return round(final,1)

def compute_panels_fit(roof_w, roof_h, panel_w, panel_h, clearance, orientation="Portrait"):
    if orientation=="Landscape": pw, ph = panel_h, panel_w
    else: pw, ph = panel_w, panel_h
    avail_w = max(roof_w-2*clearance,0)
    avail_h = max(roof_h-2*clearance,0)
    def count_axis(avail,p): return max(int((avail+clearance)//(p+clearance)),0) if p>0 else 0
    cols = count_axis(avail_w,pw)
    rows = count_axis(avail_h,ph)
    count = rows*cols
    return count, rows, cols

def monthly_energy_kwh(system_kw, monthly_psh, days, performance_ratio, shading_factor):
    return system_kw*monthly_psh*days*performance_ratio*(1-shading_factor)

FAQ = [
    ("what is psh", "PSH = Peak Sun Hours; approx. daily solar energy in kWh per kW of PV."),
    ("what is performance ratio", "Performance Ratio (PR) lumps system losses (temperature, wiring, inverter). Typical 0.72–0.82."),
    ("how many panels", "Panels Fit depends on roof size and panel size/orientation with required clearances."),
    ("best tilt", "Set tilt roughly equal to latitude for year-round balance."),
    ("payback", "Payback Period = Install Cost / Annual Savings.")
]

def chatbot_reply(msg):
    text = msg.lower().strip()
    if not text: return "Ask me anything about solar sizing, costs, tilt, or payback!"
    if "tilt" in text: return "Best tilt ≈ your latitude."
    if "cost" in text or "price" in text: return "Install cost ≈ ₹45–65k per kW (India)."
    if "payback" in text or "roi" in text: return "Payback = Install Cost / Annual Savings."
    if "panel" in text and ("how many" in text or "count" in text): return "Depends on roof & panel size, orientation, and spacing."
    for k,v in FAQ:
        if k in text: return v
    return "Try asking about tilt, PR, costs, or panels fit."

# ---------------------------- Login / Signup ---------------------------- #
with st.sidebar:
    st.header("🔐 User Login / Signup")
    choice = st.radio("Action", ["Login","Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Submit Login/Signup"):
        if choice=="Sign Up":
            try:
                c.execute("INSERT INTO users (username,password) VALUES (?,?)",(username,password))
                conn.commit()
                st.success("User created. Please login.")
            except:
                st.error("Username exists.")
        else:
            c.execute("SELECT id FROM users WHERE username=? AND password=?",(username,password))
            user=c.fetchone()
            if user: st.session_state.user_id=user[0]; st.success(f"Logged in as {username}")
            else: st.error("Invalid credentials")

# ---------------------------- UI ---------------------------- #
st.title("☀️ Solar Mapping & Recommendation System")

# Only allow simulation if logged in
if "user_id" not in st.session_state:
    st.info("Please login/signup to run simulations and save results.")
    st.stop()

# ---------------------------- Sidebar Inputs ---------------------------- #
with st.sidebar:
    st.header("Inputs")
    default_lat, default_lon = 21.2514, 81.6296
    lat = st.number_input("Latitude", value=float(default_lat), format="%.6f")
    lon = st.number_input("Longitude", value=float(default_lon), format="%.6f")
    st.markdown("---")
    st.markdown("**Rooftop Geometry**")
    roof_w = st.number_input("Width (m)", min_value=3.0,value=10.0,step=0.5)
    roof_h = st.number_input("Height (m)", min_value=3.0,value=8.0,step=0.5)
    clearance = st.number_input("Clearance (m)", min_value=0.0,value=0.4,step=0.1)
    st.markdown("---")
    st.markdown("**Panel Specs**")
    panel_w = st.number_input("Panel Width (m)", min_value=0.8,value=1.1,step=0.01)
    panel_h = st.number_input("Panel Height (m)", min_value=1.2,value=1.75,step=0.01)
    panel_watt = st.number_input("Panel Wattage (W)", min_value=200,value=400,step=10)
    orientation = st.selectbox("Orientation", ["Portrait","Landscape"])
    st.markdown("---")
    st.markdown("**Performance & Costs**")
    performance_ratio = st.slider("PR",0.6,0.9,0.75,0.01)
    shading_factor = st.slider("Shading",0.0,0.8,0.1,0.05)
    tilt_deg = st.slider("Tilt (°)",0,60,int(abs(lat)),1)
    cost_per_kw = st.number_input("Cost per kW (₹)", min_value=10000,value=55000,step=1000)
    tariff = st.number_input("Electricity Tariff (₹/kWh)", min_value=2.0,value=8.0,step=0.5)

# ---------------------------- Map Picker ---------------------------- #
st.subheader("📍 Pick Location on Map")
m = folium.Map(location=[lat, lon], zoom_start=12)
folium.Marker([lat, lon], tooltip="Selected Location").add_to(m)
returned = st_folium(m,width=700,height=400)
if returned and returned.get("last_clicked"):
    lat = float(returned["last_clicked"]["lat"])
    lon = float(returned["last_clicked"]["lng"])
    with st.sidebar: st.info(f"Map selected: Lat {lat:.5f}, Lon {lon:.5f}")

# ---------------------------- Fetch Resource ---------------------------- #
st.subheader("🔆 Solar Resource (Monthly PSH)")
monthly_psh = fetch_nasa_power_monthly(lat, lon)
if monthly_psh is None:
    st.warning("NASA POWER failed. Using fallback PSH (kWh/kW/day).")
    monthly_psh = {"Jan":4.5,"Feb":5.2,"Mar":6.0,"Apr":6.4,"May":6.5,"Jun":5.5,
                   "Jul":4.8,"Aug":5.0,"Sep":5.8,"Oct":5.7,"Nov":5.0,"Dec":4.6}
else:
    st.success("NASA POWER monthly averages loaded.")

# ---------------------------- Panel Fit & Layout ---------------------------- #
st.subheader("📐 Rooftop & Panel Layout")
roof_area = roof_w*roof_h
count, rows, cols = compute_panels_fit(roof_w, roof_h, panel_w, panel_h, clearance, orientation)
system_kw = round((count*panel_watt)/1000.0,2)

left,right = st.columns([1.2,1])
with left:
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    ax.set_aspect('equal'); ax.set_title(f"Layout Preview ({orientation}) — {rows}x{cols}={count}")
    ax.add_patch(plt.Rectangle((0,0), roof_w, roof_h, fill=False,lw=2))
    ax.add_patch(plt.Rectangle((clearance, clearance), max(roof_w-2*clearance,0), max(roof_h-2*clearance,0), fill=False,ls="--",lw=1))
    pw,ph = (panel_h,panel_w) if orientation=="Landscape" else (panel_w,panel_h)
    for r in range(rows):
        for c in range(cols):
            x=clearance+c*(pw+clearance); y=clearance+r*(ph+clearance)
            ax.add_patch(plt.Rectangle((x,y),pw,ph,fill=False))
    ax.set_xlim(-0.2,roof_w+0.2); ax.set_ylim(-0.2,roof_h+0.2)
    ax.set_xlabel("Width (m)"); ax.set_ylabel("Height (m)")
    st.pyplot(fig)

with right:
    st.metric("Rooftop Area", f"{roof_area:.1f} m²")
    st.metric("Panels Fit", f"{count} panels")
    st.metric("System Capacity", f"{system_kw:.2f} kW")

# ---------------------------- Monthly Forecast ---------------------------- #
st.subheader("📈 Monthly Energy Output Forecast")
dim = days_in_months()
months = list(monthly_psh.keys())
monthly_energy = [monthly_energy_kwh(system_kw, monthly_psh[m], dim[m], performance_ratio, shading_factor) for m in months]
df = pd.DataFrame({"Month":months, "PSH": [round(monthly_psh[m],2) for m in months], "Days":[dim[m] for m in months], "Energy":[round(v,1) for v in monthly_energy]})
annual_energy = round(float(np.sum(monthly_energy)),1)
install_cost = round(system_kw*cost_per_kw,0)
annual_savings = round(annual_energy*tariff,0)
payback_years = round(install_cost/max(annual_savings,1),1)
colA,colB,colC,colD=st.columns(4)
with colA: st.metric("Annual Solar Output",f"{annual_energy:,} kWh/yr")
with colB: st.metric("Installation Cost",f"₹{install_cost:,.0f}")
with colC: st.metric("Annual Savings",f"₹{annual_savings:,.0f}/yr")
with colD: st.metric("Payback Period",f"{payback_years} years")
st.dataframe(df,use_container_width=True)
fig2,ax2=plt.subplots(figsize=(7,3.5))
ax2.bar(df["Month"],df["Energy"])
ax2.set_ylabel("Monthly Energy (kWh)"); ax2.set_title("Monthly Solar Energy Output")
st.pyplot(fig2)

# ---------------------------- Suitability ---------------------------- #
st.subheader("✅ Suitability")
score = suitability_score(roof_area, monthly_psh, shading_factor, tilt_deg, lat)
explain=[]
if score>=80: explain.append("Excellent resource & roof size.")
elif score>=65: explain.append("Good potential — optimize tilt/spacing.")
else: explain.append("Moderate potential — shading or area limiting.")
if shading_factor<=0.1: explain.append("Low shading.")
elif shading_factor<=0.3: explain.append("Some shading present.")
else: explain.append("Significant shading.")
explain.append(f"Tilt set to {tilt_deg}°, latitude ≈ {abs(lat):.0f}°.")
st.progress(min(int(score),100))
st.write(f"**Suitability Score:** {score}/100 — "+" ".join(explain))

# ---------------------------- Chatbot ---------------------------- #
st.subheader("💬 Ask the Solar Chatbot")
if "chat" not in st.session_state: st.session_state.chat=[{"role":"assistant","content":"Hi! Ask me about panels, costs, tilt, PR, or payback."}]
for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.write(m["content"])
user_msg=st.chat_input("Type your question…")
if user_msg:
    st.session_state.chat.append({"role":"user","content":user_msg})
    reply=chatbot_reply(user_msg)
    st.session_state.chat.append({"role":"assistant","content":reply})
    with st.chat_message("user"): st.write(user_msg)
    with st.chat_message("assistant"): st.write(reply)

# ---------------------------- Save Simulation ---------------------------- #
c.execute("""INSERT INTO simulations (
    user_id, timestamp, lat, lon, roof_w, roof_h, clearance,
    panel_w, panel_h, panel_watt, orientation, performance_ratio,
    shading_factor, tilt_deg, cost_per_kw, tariff,
    system_kw, rooftop_area, annual_energy, annual_savings, payback_years, suitability_score
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
    st.session_state.user_id, datetime.now().isoformat(),
    lat, lon, roof_w, roof_h, clearance,
    panel_w, panel_h, panel_watt, orientation, performance_ratio,
    shading_factor, tilt_deg, cost_per_kw, tariff,
    system_kw, roof_area, annual_energy, annual_savings, payback_years, score
))
conn.commit()
st.success("Simulation saved to your dashboard!")

# ---------------------------- Dashboard ---------------------------- #
st.subheader("📊 Your Past Simulations")
c.execute("SELECT * FROM simulations WHERE user_id=? ORDER BY timestamp DESC",(st.session_state.user_id,))
rows=c.fetchall()
if rows:
    df_hist=pd.DataFrame(rows, columns=[desc[0] for desc in c.description])
    st.dataframe(df_hist[['timestamp','system_kw','annual_energy','annual_savings','payback_years','suitability_score']])
else:
    st.info("No past simulations yet.")
