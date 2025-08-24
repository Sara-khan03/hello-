# app.py
# Solar Mapping & Recommendation System — Full single-file Streamlit app
# Features:
# - Categorized, searchable Indian city/landmark selector + free-text geocoding (Nominatim if available)
# - Satellite map (Esri World Imagery) with marker that auto-centers and is draggable / clickable to refine location
# - NASA POWER monthly GHI fetch (climatology) with fallback profiles
# - Rooftop geometry inputs, panel specs, layout preview
# - Monthly forecast, annual output, installation cost, annual savings, payback
# - Suitability score + short explanation
# - Built-in chatbot for FAQs
# - PDF export (ReportLab if available) or text download fallback
#
# To run:
#   pip install streamlit folium streamlit-folium requests numpy pandas matplotlib reportlab geopy
# then:
#   streamlit run app.py

import io
import time
import math
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Solar Mapping & Recommendation System",
                   page_icon="☀️", layout="wide")

# Optional imports (handle missing gracefully)
try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ----------------------------
# Helper functions
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_nasa_power_monthly(lat, lon):
    """
    Fetch monthly average GHI (ALLSKY_SFC_SW_DWN, kWh/m2/day) from NASA POWER Climatology (monthly climatology)
    Returns: dict Jan..Dec -> kWh/m2/day
    Falls back to predefined profiles on failure.
    """
    fallback_india = {
        "Jan":4.5,"Feb":5.2,"Mar":6.0,"Apr":6.4,"May":6.5,"Jun":5.5,
        "Jul":4.8,"Aug":5.0,"Sep":5.8,"Oct":5.7,"Nov":5.0,"Dec":4.6
    }
    fallback_global = {
        "Jan":3.8,"Feb":4.5,"Mar":5.5,"Apr":5.8,"May":6.0,"Jun":5.0,
        "Jul":4.2,"Aug":4.5,"Sep":5.0,"Oct":5.2,"Nov":4.5,"Dec":4.0
    }

    base = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON"
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        values = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        # ensure order Jan..Dec
        ordered = {k: float(values[k]) for k in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]}
        mapping = {"Jan":ordered["JAN"], "Feb":ordered["FEB"], "Mar":ordered["MAR"], "Apr":ordered["APR"],
                   "May":ordered["MAY"], "Jun":ordered["JUN"], "Jul":ordered["JUL"], "Aug":ordered["AUG"],
                   "Sep":ordered["SEP"], "Oct":ordered["OCT"], "Nov":ordered["NOV"], "Dec":ordered["DEC"]}
        return mapping
    except Exception:
        # fallback chosen by latitude
        if abs(lat) < 30:
            return fallback_india
        else:
            return fallback_global

def days_in_months(year=None):
    # use non-leap default for simplicity, but if year provided compute feb
    if year is None:
        year = datetime.now().year
    feb = 29 if (year%4==0 and (year%100!=0 or year%400==0)) else 28
    return {"Jan":31,"Feb":feb,"Mar":31,"Apr":30,"May":31,"Jun":30,"Jul":31,"Aug":31,"Sep":30,"Oct":31,"Nov":30,"Dec":31}

def suitability_score(roof_area_m2, monthly_psh_dict, shading_factor, tilt_deg, lat):
    # resource score from mean PSH
    if not monthly_psh_dict:
        resource_score = 50
    else:
        avg_psh = np.mean(list(monthly_psh_dict.values()))
        resource_score = np.clip(20 + (avg_psh-3)*18, 20, 95)
    area_score = 30 + np.clip((roof_area_m2-10)*1.5, 0, 65)
    shade_score = np.clip(100 - shading_factor*100, 10, 100)
    ideal = abs(abs(lat) - tilt_deg)
    tilt_score = np.clip(100 - ideal*2.2, 40, 100)
    weights = [0.35, 0.25, 0.2, 0.2]
    final = resource_score*weights[0] + area_score*weights[1] + shade_score*weights[2] + tilt_score*weights[3]
    return round(float(final), 1)

def compute_panels_fit(roof_w, roof_h, panel_w, panel_h, clearance, orientation="Portrait"):
    if orientation == "Landscape":
        pw, ph = panel_h, panel_w
    else:
        pw, ph = panel_w, panel_h
    avail_w = max(roof_w - 2*clearance, 0)
    avail_h = max(roof_h - 2*clearance, 0)
    def count_axis(avail, p):
        if p <= 0: return 0
        return max(int((avail + clearance*0.5) // (p + clearance)), 0)
    cols = count_axis(avail_w, pw)
    rows = count_axis(avail_h, ph)
    count = rows * cols
    return count, rows, cols

def monthly_energy_kwh(system_kw, monthly_psh, days, performance_ratio, shading_factor):
    return system_kw * monthly_psh * days * performance_ratio * (1 - shading_factor)

def draw_layout_preview(roof_w, roof_h, rows, cols, panel_w, panel_h, clearance, orientation):
    # returns matplotlib figure
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    ax.set_aspect('equal')
    ax.set_title(f"Layout Preview ({orientation}) — {rows}×{cols} = {rows*cols} panels")
    ax.add_patch(plt.Rectangle((0,0), roof_w, roof_h, fill=False, lw=2))
    ax.add_patch(plt.Rectangle((clearance, clearance), max(roof_w-2*clearance,0), max(roof_h-2*clearance,0), fill=False, ls="--", lw=1))
    if orientation == "Landscape":
        pw, ph = panel_h, panel_w
    else:
        pw, ph = panel_w, panel_h
    x0 = clearance
    y0 = clearance
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * (pw + clearance)
            y = y0 + r * (ph + clearance)
            ax.add_patch(plt.Rectangle((x,y), pw, ph, fill=False))
    ax.set_xlim(-0.2, roof_w+0.2)
    ax.set_ylim(-0.2, roof_h+0.2)
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Height (m)")
    plt.tight_layout()
    return fig

def make_pdf_report(path, summary_lines, monthly_fig_bytes=None, mask_bytes=None):
    # If reportlab is available, produce PDF; otherwise raise
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab not installed")
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-50, "Solar Mapping & Recommendation Report")
    c.setFont("Helvetica", 9)
    c.drawString(40, h-70, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y = h-100
    for line in summary_lines:
        c.drawString(40, y, line)
        y -= 12
        if y < 140:
            c.showPage()
            y = h-80
    if monthly_fig_bytes:
        c.showPage()
        img = ImageReader(io.BytesIO(monthly_fig_bytes))
        c.drawImage(img, 40, h-360, width=520, height=320, preserveAspectRatio=True)
    if mask_bytes:
        c.showPage()
        img2 = ImageReader(io.BytesIO(mask_bytes))
        c.drawImage(img2, 40, h-360, width=520, height=320, preserveAspectRatio=True)
    c.save()

# ----------------------------
# Predefined categorized Indian places (short list; expand if needed)
# ----------------------------
PLACES = {
    "--- Central Govt ---": None,
    "Rashtrapati Bhavan (Delhi)": (28.6143, 77.1995),
    "Parliament House (Delhi)": (28.6172, 77.2080),
    "Supreme Court (Delhi)": (28.6260, 77.2410),

    "--- State Capitals ---": None,
    "Raipur Collectorate": (21.2514, 81.6296),
    "Bengaluru Vidhana Soudha": (12.9797, 77.5907),
    "Mumbai Mantralaya": (18.9430, 72.8238),
    "Chennai Fort St George": (13.0827, 80.2750),
    "Kolkata Writers' Building": (22.5726, 88.3639),
    "Hyderabad Secretariat": (17.3850, 78.4867),
    "Bhopal Mantralaya": (23.2599, 77.4126),
    "Patna Secretariat": (25.5941, 85.1376),
    "Jaipur Secretariat": (26.9124, 75.7873),
    "Lucknow Vidhan Sabha": (26.8467, 80.9462),
    "Gandhinagar Sachivalaya": (23.2156, 72.6369),
    "Thiruvananthapuram Secretariat": (8.5241, 76.9366),

    "--- Union Territories ---": None,
    "Chandigarh Secretariat": (30.7333, 76.7794),
    "Puducherry Raj Nivas": (11.9416, 79.8083),
    "Port Blair Secretariat": (11.6234, 92.7265)
}

# ----------------------------
# UI: Sidebar inputs
# ----------------------------
st.title("☀️ Solar Mapping & Recommendation System")
with st.sidebar:
    st.header("Inputs")

    # Location: search + categorized dropdown
    st.subheader("1) Location (City / Office)")
    search_filter = st.text_input("🔍 Search cities / offices (type to filter)", "")
    # Filter while keeping category separators
    filtered_keys = [k for k in PLACES.keys() if (search_filter.lower() in k.lower()) or k.startswith("---")]
    # Provide selectbox
    place_choice = st.selectbox("Choose from list", [""] + filtered_keys,
                                format_func=lambda x: x if not x.startswith("---") else x)

    # Free-text geocoding box
    free_text = st.text_input("Or type an address / office (will geocode)", "")

    # Manual lat/lon override fields
    manual_lat = st.number_input("Latitude (override)", value=21.2514, format="%.6f")
    manual_lon = st.number_input("Longitude (override)", value=81.6296, format="%.6f")

    st.markdown("---")
    st.subheader("2) Rooftop Geometry (Rectangular approximation)")
    roof_w = st.number_input("Rooftop Width (m)", min_value=2.0, value=10.0, step=0.5)
    roof_h = st.number_input("Rooftop Height (m)", min_value=2.0, value=8.0, step=0.5)
    clearance = st.number_input("Clearance / Spacing (m)", min_value=0.0, value=0.4, step=0.05)

    st.markdown("---")
    st.subheader("3) Panel Specs")
    panel_w = st.number_input("Panel Width (m)", min_value=0.5, value=1.1, step=0.01)
    panel_h = st.number_input("Panel Height (m)", min_value=1.0, value=1.75, step=0.01)
    panel_watt = st.number_input("Panel Wattage (W)", min_value=150, value=400, step=10)
    orientation = st.selectbox("Panel Orientation", ["Portrait", "Landscape"])

    st.markdown("---")
    st.subheader("4) Performance & Costs")
    performance_ratio = st.slider("Performance Ratio (PR)", 0.6, 0.9, 0.75, 0.01)
    shading_factor = st.slider("Shading Factor (0=None, 0.3=Medium, 0.6=High)", 0.0, 0.8, 0.1, 0.05)
    tilt_deg = st.slider("Tilt (°)", 0, 60,  int(abs(manual_lat)), 1)
    cost_per_kw = st.number_input("Installation Cost per kW (₹/kW)", min_value=10000, value=55000, step=1000)
    tariff = st.number_input("Electricity Tariff (₹/kWh)", min_value=1.0, value=8.0, step=0.5)

    st.markdown("---")
    st.caption("Tip: select a place, or type an address; the map (satellite) will center on it. You may also click the map or drag the marker to refine the exact rooftop spot.")

# ----------------------------
# Resolve location: priority - place_choice -> free_text geocode -> manual lat/lon
# ----------------------------
selected_lat = manual_lat
selected_lon = manual_lon
selected_name = ""

# If user picked a predefined place:
if place_choice and place_choice in PLACES and PLACES[place_choice] is not None:
    selected_lat, selected_lon = PLACES[place_choice]
    selected_name = place_choice

# If free_text provided, attempt geocoding (only if geopy available)
if free_text and GEOPY_AVAILABLE:
    try:
        geoloc = Nominatim(user_agent="solar_mapper_app")
        loc = geoloc.geocode(free_text + ", India", timeout=10)
        if loc:
            selected_lat, selected_lon = loc.latitude, loc.longitude
            selected_name = free_text
    except Exception:
        # ignore geocode failures — keep previous
        pass
elif free_text and not GEOPY_AVAILABLE:
    # No geopy installed; do not fail — fallback unchanged
    selected_name = free_text

# ----------------------------
# Map: Satellite view, center on selected coords, draggable marker
# ----------------------------
st.subheader("📍 Satellite Map — pick / refine rooftop location")
map_center = [selected_lat, selected_lon]
m = folium.Map(location=map_center, zoom_start=16, tiles=None)
# Esri World Imagery tiles
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite"
).add_to(m)
folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
# add draggable marker
marker = folium.Marker(
    location=map_center,
    draggable=True,
    popup=selected_name or "Selected location",
    tooltip="Drag to refine rooftop position"
)
marker.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

map_data = st_folium(m, width=900, height=500)

# map_data may contain last_clicked (click) or last_object_clicked keys depending on interaction
if map_data:
    # If user clicked the map (lat/lng)
    if map_data.get("last_clicked"):
        lc = map_data["last_clicked"]
        selected_lat = float(lc["lat"]); selected_lon = float(lc["lng"])
        selected_name = f"Clicked location ({selected_lat:.5f},{selected_lon:.5f})"
    # If a draggable marker was moved: some versions return "last_object_clicked" or "last_dragged_marker"
    # We'll check multiple possible keys
    if map_data.get("last_object_clicked"):
        obj = map_data["last_object_clicked"]
        if isinstance(obj, dict) and "lat" in obj and "lng" in obj:
            selected_lat = float(obj["lat"]); selected_lon = float(obj["lng"])
    if map_data.get("last_dragged"):
        # streamlit-folium sometimes returns last_dragged
        ld = map_data["last_dragged"]
        if isinstance(ld, dict) and "lat" in ld and "lng" in ld:
            selected_lat = float(ld["lat"]); selected_lon = float(ld["lng"])

# show selected coordinates
st.success(f"Selected coordinates: {selected_lat:.6f}, {selected_lon:.6f}")

# ----------------------------
# Solar Resource: NASA POWER monthly PSH (GHI)
# ----------------------------
st.subheader("🔆 Solar Resource (Monthly GHI / PSH)")
monthly_psh = fetch_nasa_power_monthly(selected_lat, selected_lon)
st.caption("Source: NASA POWER (ALLSKY_SFC_SW_DWN climatology).")
# Plot monthly GHI
fig_ghi, ax_ghi = plt.subplots(figsize=(8,3))
months = list(monthly_psh.keys())
vals = [monthly_psh[m] for m in months]
ax_ghi.bar(months, vals, color="orange")
ax_ghi.set_ylabel("kWh/m²/day")
ax_ghi.set_title(f"Monthly GHI @ {selected_lat:.4f}, {selected_lon:.4f}")
st.pyplot(fig_ghi)

# ----------------------------
# Panel layout, sizing, KPIs
# ----------------------------
st.subheader("📐 Rooftop & Panel Layout")

# compute panels that fit
panels_fit, rows, cols = compute_panels_fit(roof_w, roof_h, panel_w, panel_h, clearance, orientation)
system_kw = round((panels_fit * panel_watt) / 1000.0, 3)
roof_area = roof_w * roof_h

# layout preview
fig_layout = draw_layout_preview(roof_w, roof_h, rows, cols, panel_w, panel_h, clearance, orientation)
st.pyplot(fig_layout)

# KPIs
left, right = st.columns([1.3, 1])
with left:
    st.metric("Rooftop Area", f"{roof_area:.1f} m²")
    st.metric("Panels Fit", f"{panels_fit} panels")
with right:
    st.metric("System Capacity", f"{system_kw:.2f} kW")
    st.metric("Selected Location", f"{selected_lat:.5f}, {selected_lon:.5f}")

# Monthly forecast calculation
st.subheader("📈 Monthly Energy Output Forecast")
dim = days_in_months()
monthly_energy = [monthly_energy_kwh(system_kw, monthly_psh[m], dim[m], performance_ratio, shading_factor) for m in months]
annual_kwh = sum(monthly_energy)
df_month = pd.DataFrame({
    "Month": months,
    "PSH (kWh/m²/day)": [round(monthly_psh[m],3) for m in months],
    "Days": [dim[m] for m in months],
    "Energy (kWh)": [round(v,1) for v in monthly_energy]
})
st.dataframe(df_month, use_container_width=True)

# Monthly chart
fig_month, axm = plt.subplots(figsize=(8,3))
axm.bar(months, df_month["Energy (kWh)"])
axm.set_ylabel("kWh")
axm.set_title("Monthly Generation Forecast (kWh)")
st.pyplot(fig_month)

# Financials
install_cost = round(system_kw * cost_per_kw, 0)
annual_savings = round(annual_kwh * tariff, 0)
payback_years = round(install_cost / max(annual_savings, 1), 1)

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Annual Solar Output", f"{annual_kwh:,.0f} kWh/yr")
with col2: st.metric("Installation Cost", f"₹{install_cost:,.0f}")
with col3: st.metric("Annual Savings", f"₹{annual_savings:,.0f}")
with col4: st.metric("Payback Period", f"{payback_years} years")

# ----------------------------
# Suitability + suggestions
# ----------------------------
st.subheader("✅ Suitability & Suggestions")
score = suitability_score(roof_area, monthly_psh, shading_factor, tilt_deg, selected_lat)
st.progress(min(int(score), 100))
st.write(f"**Suitability Score:** {score}/100")

suggestions = []
if roof_area < 5:
    suggestions.append("Rooftop area is very small — installation may be uneconomical.")
if shading_factor > 0.4:
    suggestions.append("High shading detected — consider trimming trees or selecting other areas.")
if payback_years > 12:
    suggestions.append("Consider increasing self-consumption or using battery/storage subsidies to improve ROI.")
if not suggestions:
    suggestions.append("Good site. Consider optimizing tilt and orientation for marginal gains.")

for s in suggestions:
    st.info(s)

# ----------------------------
# Chatbot (simple rule-based)
# ----------------------------
st.subheader("💬 Solar Chatbot")
if "chat" not in st.session_state:
    st.session_state.chat = [{"role":"assistant","content":"Hi! Ask me about panels, costs, tilt, payback, or rooftop suitability."}]
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_q = st.chat_input("Type your question…")
if user_q:
    st.session_state.chat.append({"role":"user","content":user_q})
    # simple rule-based replies
    q = user_q.lower()
    if "tilt" in q:
        ans = f"Set tilt roughly equal to latitude ({abs(selected_lat):.1f}°). For seasonal bias, adjust ±10°."
    elif "cost" in q or "price" in q:
        ans = f"Typical installation cost used here: ₹{cost_per_kw:,.0f}/kW. Actual quotes vary by vendor."
    elif "payback" in q or "roi" in q:
        ans = f"Payback = Install Cost / Annual Savings = {payback_years} years (estimate)."
    elif "how many" in q and "panel" in q:
        ans = f"Based on roof geometry and panel size, {panels_fit} panels fit in the selected area."
    elif "ghi" in q or "sun" in q:
        mean_ghi = round(np.mean(list(monthly_psh.values())),2)
        ans = f"Mean GHI at selected location is ~{mean_ghi} kWh/m²/day (NASA POWER climatology)."
    else:
        ans = "I can help with tilt, cost, payback, panels fit, and GHI. Try questions like 'What's the payback?' or 'How many panels can fit?'"
    st.session_state.chat.append({"role":"assistant","content":ans})
    with st.chat_message("user"): st.write(user_q)
    with st.chat_message("assistant"): st.write(ans)

# ----------------------------
# PDF Report export (or TXT fallback)
# ----------------------------
st.subheader("📄 Export Report")
if st.button("Generate & Download Report"):
    summary_lines = [
        f"Location: {selected_name or 'Custom'} ({selected_lat:.6f}, {selected_lon:.6f})",
        f"Rooftop (WxH): {roof_w:.1f} x {roof_h:.1f} m  → Area: {roof_area:.1f} m²",
        f"Panels fit: {panels_fit} (Panel: {panel_w} x {panel_h} m, {panel_watt} W)",
        f"System size: {system_kw:.2f} kW",
        f"Annual generation: {annual_kwh:.0f} kWh",
        f"Installation cost: ₹{install_cost:,.0f}",
        f"Annual savings (@₹{tariff}/kWh): ₹{annual_savings:,.0f}",
        f"Payback period: {payback_years} years",
        f"Suitability score: {score}/100",
        f"Mean GHI: {round(np.mean(list(monthly_psh.values())),3)} kWh/m²/day"
    ]
    # create monthly chart image bytes
    buf = io.BytesIO()
    fig_month.savefig(buf, format='png', bbox_inches='tight')
    monthly_bytes = buf.getvalue()
    buf.close()
    # create layout image bytes
    buf2 = io.BytesIO()
    fig_layout.savefig(buf2, format='png', bbox_inches='tight')
    layout_bytes = buf2.getvalue()
    buf2.close()

    if REPORTLAB_AVAILABLE:
        pdf_path = "/tmp/solar_report.pdf"
        try:
            make_pdf_report(pdf_path, summary_lines, monthly_fig_bytes=monthly_bytes, mask_bytes=layout_bytes)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Report", data=f, file_name="solar_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}. Falling back to text report.")
            txt = "\n".join(summary_lines)
            st.download_button("Download TXT Report", data=txt, file_name="solar_report.txt", mime="text/plain")
    else:
        # fallback: provide a plain text summary download
        txt = "\n".join(summary_lines)
        st.download_button("Download TXT Report", data=txt, file_name="solar_report.txt", mime="text/plain")
        st.info("Install reportlab to enable PDF export: pip install reportlab")

# ----------------------------
# Final Summary area (same details)
# ----------------------------
st.markdown("---")
st.subheader("📋 Summary")
st.write(f"""
- **Location:** {selected_name or 'Custom'} — {selected_lat:.6f}, {selected_lon:.6f}
- **Rooftop:** {roof_w:.1f} × {roof_h:.1f} m (Area {roof_area:.1f} m²)
- **Panels:** {panels_fit} × {panel_watt} W ({orientation}) → **System = {system_kw:.2f} kW**
- **Annual Output:** **{annual_kwh:,.0f} kWh**
- **Cost:** ₹{install_cost:,.0f} | **Savings:** ₹{annual_savings:,.0f}/yr | **Payback:** {payback_years} yrs
- **Suitability:** **{score}/100**
""")
st.caption("Estimates are indicative. For installation, perform on-site structural/shading checks and get vendor quotes.")

