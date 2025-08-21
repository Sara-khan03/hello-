# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_folium import st_folium
import folium
from folium import plugins
import numpy as np
import math
import json
import requests
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO
import tempfile
from datetime import datetime

# -------------------------
# App config + basic style
# -------------------------
st.set_page_config(page_title="Solar Suite", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .stApp { background-image: url('https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1920&q=80');
              background-size: cover; background-attachment: fixed; }
      .card { background: rgba(255,255,255,0.95); padding:16px; border-radius:12px; }
      .title { font-size: 28px; font-weight:700; color:#083d33; }
      .slogan { color:#d97706; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<div class="title">☀️ Solar Suite — Mapping · Forecast · Layout · Reports</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan">Draw your rooftop, see forecasts, design panel layout, and export a report.</div>', unsafe_allow_html=True)

# -------------------------
# Navigation menu (icons)
# -------------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Map", "Forecast", "Panel Layout", "Chatbot", "Reports"],
        icons=["geo-alt", "bar-chart", "grid-1x2", "robot", "file-earmark-text"],
        menu_icon="solar-panel",
        default_index=0,
        orientation="vertical",
    )

# -------------------------
# Shared app state (session)
# -------------------------
if "polygon" not in st.session_state:
    st.session_state.polygon = None  # list of [lon,lat] pairs (outer ring)
if "marker" not in st.session_state:
    st.session_state.marker = None   # (lat, lon)
if "area_m2" not in st.session_state:
    st.session_state.area_m2 = None
if "ghi_monthly" not in st.session_state:
    st.session_state.ghi_monthly = None
if "monthly_kwh" not in st.session_state:
    st.session_state.monthly_kwh = None
if "panels_fit" not in st.session_state:
    st.session_state.panels_fit = None

# -------------------------
# Shared controls (sidebar compact)
# -------------------------
with st.sidebar:
    st.markdown("---")
    st.header("Quick Settings")
    panel_watt = st.number_input("Panel watt (W)", value=400, step=50)
    panel_eff = st.slider("Panel eff (%)", 10, 23, 18) / 100.0
    coverage = st.slider("Coverage fraction", 40, 95, 80) / 100.0
    tilt_deg = st.slider("Tilt (deg)", 0, 45, 20)
    derate = st.slider("Derate (losses)", 60, 90, 75) / 100.0
    shading_pct = st.slider("Shading loss (%)", 0, 60, 10)
    st.markdown("---")
    st.caption("Use the side menu to switch pages. Draw polygon on Map page. Click map to place a marker (single click).")

# -------------------------
# Helper functions
# -------------------------
def polygon_area_m2(coords):
    """Approximate polygon area in m². coords: [[lon,lat],...] outer ring"""
    if not coords or len(coords) < 3:
        return 0.0
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    lat_mean = np.mean(ys)
    lon_to_m = 111320 * math.cos(math.radians(lat_mean))
    lat_to_m = 110540
    xm = (xs - xs[0]) * lon_to_m
    ym = (ys - ys[0]) * lat_to_m
    area = 0.5 * abs(np.dot(xm, np.roll(ym, 1)) - np.dot(ym, np.roll(xm, 1)))
    return float(area)

def fetch_solar_ghi(lat, lon):
    """Try NASA POWER, then PVGIS, else fallback 5.0"""
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&format=JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        ghi_m = {k: float(v) for k, v in data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].items()}
        return float(np.mean(list(ghi_m.values()))), ghi_m, "NASA POWER"
    except Exception:
        pass
    try:
        url = f"https://re.jrc.ec.europa.eu/api/DRcalc?lat={lat}&lon={lon}&outputformat=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        monthly = data["outputs"]["monthly"]["fixed"]["E_d"]
        days = [31,28,31,30,31,30,31,31,30,31,30,31]
        ghi_m = {f"{i+1:02d}": float(monthly[i]/days[i]) for i in range(12)}
        return float(np.mean(list(ghi_m.values()))), ghi_m, "PVGIS"
    except Exception:
        pass
    ghi_m = {f"{i:02d}": 5.0 for i in range(1,13)}
    return 5.0, ghi_m, "Fallback (5.0)"

def monthly_output_kwh(area_m2, ghi_monthly, panel_eff, coverage, tilt_deg, derate, shading_pct, latitude):
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    active_area = area_m2 * coverage
    tilt_factor = max(0.5, math.cos(math.radians(tilt_deg - abs(latitude))))
    shading = max(0.0, 1 - shading_pct/100.0)
    out = []
    for i in range(12):
        ghi = float(ghi_monthly.get(f"{i+1:02d}", 5.0))
        kwh = active_area * ghi * panel_eff * tilt_factor * days_in_month[i] * derate * shading
        out.append(kwh)
    return out

def compute_layout(area_m2, coverage, panel_watt):
    panel_area = 1.9  # m2 incl spacing
    usable = area_m2 * coverage
    panels = int(max(0, usable // panel_area))
    system_kw = round(panels * panel_watt / 1000.0, 2)
    return panels, system_kw

def make_pdf(context):
    # small PDF builder: summary + charts embedded
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    # monthly chart
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(months, context["monthly_kwh"], marker="o")
    ax.set_title("Monthly Solar Output (kWh)")
    ax.set_ylabel("kWh")
    ax.grid(True, alpha=0.3)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    # compose PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Solar Report", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.ln(2)
    pdf.multi_cell(0, 6, f"Location (approx): {context['lat']:.5f}, {context['lon']:.5f}")
    pdf.multi_cell(0, 6, f"Estimated rooftop area: {context['area_m2']:.1f} m²")
    pdf.multi_cell(0, 6, f"Panels fit: {context['panels_fit']}  |  System size: {context['system_kw']:.2f} kW")
    pdf.ln(4)
    pdf.image(tmp.name, w=180)
    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# -------------------------
# Page: Map
# -------------------------
def page_map():
    st.header("🌍 Map — pick point / draw rooftop")
    st.markdown("**Click** on the map to place a marker (single click). Use the **polygon tool** to draw rooftop boundary.")
    center = [21.0, 78.0]  # India-ish
    m = folium.Map(location=center, zoom_start=6, control_scale=True)
    # add tile layer (Esri satellite optional) - Esri imagery (no key)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    # Add draw plugin (polygon + rectangle)
    draw = plugins.Draw(export=True, filename="rooftop.geojson",
                        draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False, "polygon": True, "rectangle": True},
                        edit_options={"edit": True})
    draw.add_to(m)

    # show existing polygon or marker (if in session)
    if st.session_state.polygon:
        folium.Polygon(locations=[(pt[1], pt[0]) for pt in st.session_state.polygon], color="cyan", fill=True, fill_opacity=0.25).add_to(m)
    if st.session_state.marker:
        folium.Marker(location=st.session_state.marker, popup="Selected point").add_to(m)

    map_data = st_folium(m, width=900, height=550)

    # handle last click (point selection)
    if map_data and map_data.get("last_clicked"):
        last = map_data["last_clicked"]
        lat, lon = last["lat"], last["lng"]
        st.session_state.marker = (lat, lon)
        st.success(f"Marker set at: {lat:.6f}, {lon:.6f}")

    # handle drawings
    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"].get("geometry")
        if geom and geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]  # lon,lat pairs
            st.session_state.polygon = coords
            st.session_state.area_m2 = polygon_area_m2(coords)
            st.info(f"Polygon captured. Area ≈ {st.session_state.area_m2:.1f} m²")
    # also show all drawings if user exported & pasted
    if map_data and map_data.get("all_drawings"):
        # display count
        st.write(f"Drawings on map: {len(map_data['all_drawings'])}")

# -------------------------
# Page: Forecast
# -------------------------
def page_forecast():
    st.header("📈 Forecast")
    st.markdown("This page computes solar resource (tries NASA → PVGIS → fallback) and monthly output.")
    if not st.session_state.polygon and not st.session_state.marker:
        st.info("Please draw a rooftop polygon on the **Map** page or click to set a point.")
        return
    # determine coordinate to query (centroid of polygon or marker)
    if st.session_state.polygon:
        coords = st.session_state.polygon
        lat = float(np.mean([p[1] for p in coords])); lon = float(np.mean([p[0] for p in coords]))
    else:
        lat, lon = st.session_state.marker

    ghi_avg, ghi_monthly, src = fetch_solar_ghi(lat, lon)
    st.info(f"Solar resource source: {src} (avg {ghi_avg:.2f} kWh/m²/day)")
    st.session_state.ghi_monthly = ghi_monthly

    monthly_kwh = monthly_output_kwh(
        area_m2=st.session_state.area_m2 or 50.0,
        ghi_monthly=ghi_monthly,
        panel_eff=panel_eff,
        coverage=coverage,
        tilt_deg=tilt_deg,
        derate=derate,
        shading_pct=shading_pct,
        latitude=lat
    )
    st.session_state.monthly_kwh = monthly_kwh
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=monthly_kwh, name="kWh"))
    fig.update_layout(title="Monthly Output (kWh)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    annual = sum(monthly_kwh)
    st.metric("Estimated annual energy (kWh)", f"{annual:,.0f}")

# -------------------------
# Page: Panel Layout
# -------------------------
def page_layout():
    st.header("🔆 Panel Layout")
    st.markdown("Visual, interactive panel grid. Hover to see per-panel kWh (conceptual).")
    if not st.session_state.monthly_kwh:
        st.info("Run Forecast (or draw polygon) first.")
        return
    area = st.session_state.area_m2 or 50.0
    panels, system_kw = compute_layout(area, coverage, panel_watt)
    st.session_state.panels_fit = panels

    st.write(f"Panels that fit (estimate): **{panels}** — System size ≈ **{system_kw:.2f} kW**")

    if panels <= 0:
        st.warning("No panels fit with the current coverage/area. Increase coverage or use a larger roof.")
        return

    cols_guess = max(4, min(14, int(math.sqrt(panels) * 1.2)))
    rows_guess = math.ceil(panels / cols_guess)
    per_panel_annual = sum(st.session_state.monthly_kwh) / panels if panels else 0.0

    xs, ys, hover = [], [], []
    idx = 0
    for r in range(rows_guess):
        for c in range(cols_guess):
            if idx >= panels:
                break
            xs.append(c); ys.append(-r)
            hover.append(f"Panel #{idx+1}<br>{panel_watt} W<br>~{per_panel_annual:.1f} kWh/yr")
            idx += 1

    fig = go.Figure(data=go.Scatter(x=xs, y=ys, mode="markers",
                                    marker=dict(size=28, symbol="square", color="goldenrod"),
                                    hovertext=hover, hoverinfo="text"))
    fig.update_layout(title="Hover panels to see per-panel energy", xaxis=dict(visible=False), yaxis=dict(visible=False), height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Page: Chatbot
# -------------------------
def page_chatbot():
    st.header("🤖 Solar Assistant")
    st.markdown("Ask about tilt, costs, payback, battery, shading, or grid rules.")
    q = st.text_input("Ask a question:")
    if q:
        ql = q.lower()
        if "tilt" in ql:
            st.write("Rule of thumb: tilt ≈ latitude. Seasonal tilt or trackers improve yield but cost more.")
        elif "cost" in ql or "price" in ql:
            st.write(f"Rough PV installed cost used here: ₹{bos_cost_per_kw}/kW (adjust in sidebar). Panel + inverter + structure vary regionally.")
        elif "battery" in ql:
            st.write("Battery sizing: daily_load × autonomy_days -> usable kWh. Nameplate = usable / DoD.")
        elif "payback" in ql:
            st.write("Payback depends on tariffs, metering model, subsidies, and self-consumption. Use Forecast + Reports pages to compute.")
        else:
            st.write("Great question! If you include keywords like 'tilt', 'cost', 'battery', 'payback', I'll give tailored answers.")

# -------------------------
# Page: Reports
# -------------------------
def page_reports():
    st.header("📄 Reports & Export")
    st.markdown("Generate a PDF summary of your rooftop analysis (based on the polygon & forecast).")
    if not st.session_state.monthly_kwh:
        st.info("Run Forecast first to generate data.")
        return
    # build context
    if st.session_state.polygon:
        lat = float(np.mean([p[1] for p in st.session_state.polygon])); lon = float(np.mean([p[0] for p in st.session_state.polygon]))
    else:
        lat, lon = st.session_state.marker or (0.0, 0.0)
    area = st.session_state.area_m2 or 0.0
    panels, system_kw = compute_layout(area, coverage, panel_watt)
    annual_kwh = sum(st.session_state.monthly_kwh)
    install_cost_pv = system_kw * bos_cost_per_kw
    ctx = {
        "lat": lat, "lon": lon, "area_m2": area, "panels_fit": panels,
        "system_kw": system_kw, "monthly_kwh": st.session_state.monthly_kwh,
        "annual_kwh": annual_kwh, "panel_watt": panel_watt,
        "ghi_monthly": st.session_state.ghi_monthly or {f"{i:02d}":5.0 for i in range(1,13)}
    }
    st.write("Quick summary:")
    st.metric("Area (m²)", f"{area:.1f}")
    st.metric("Panels", panels)
    st.metric("System (kW)", f"{system_kw:.2f}")
    if st.button("Generate & Download PDF"):
        pdf_bytes = make_pdf(ctx)
        st.download_button("Download report (PDF)", pdf_bytes, file_name="solar_report.pdf", mime="application/pdf")

# -------------------------
# Router: show selected page
# -------------------------
if selected == "Map":
    page_map()
elif selected == "Forecast":
    page_forecast()
elif selected == "Panel Layout":
    page_layout()
elif selected == "Chatbot":
    page_chatbot()
elif selected == "Reports":
    page_reports()

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Tip: draw rooftop polygon on Map page, then go to Forecast → Panel Layout → Reports.")

