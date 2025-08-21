# app.py
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_folium import st_folium
import folium
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math
from fpdf import FPDF
from io import BytesIO
from datetime import datetime

# -----------------------
# Page config & staging
# -----------------------
st.set_page_config(page_title="Solar Suite", layout="wide", initial_sidebar_state="collapsed")

# Ensure basic session state keys
if "picked_point" not in st.session_state:
    st.session_state.picked_point = None  # (lat, lon)
if "ghi_monthly" not in st.session_state:
    st.session_state.ghi_monthly = None
if "area_m2" not in st.session_state:
    st.session_state.area_m2 = None
if "monthly_kwh" not in st.session_state:
    st.session_state.monthly_kwh = None
if "panels_fit" not in st.session_state:
    st.session_state.panels_fit = None

# -----------------------
# Helper utilities
# -----------------------
def reverse_geocode(lat, lon):
    """Return a place name using Nominatim (OpenStreetMap). Return None on failure."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10, "addressdetails": 0}
        r = requests.get(url, params=params, timeout=6, headers={"User-Agent": "solar-suite-app"})
        r.raise_for_status()
        data = r.json()
        name = data.get("display_name") or data.get("name")
        return name
    except Exception:
        return None

def fetch_solar_ghi(lat, lon):
    """Try NASA POWER, fallback to PVGIS, else typical 5.0. Returns (avg, monthly_dict, source)."""
    # NASA POWER (monthly climatology)
    try:
        url = (
            "https://power.larc.nasa.gov/api/temporal/climatology/point"
            f"?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&format=JSON"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        dd = r.json()
        ghi = {k: float(v) for k, v in dd["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].items()}
        return float(np.mean(list(ghi.values()))), ghi, "NASA POWER"
    except Exception:
        pass

    # PVGIS fallback
    try:
        url = f"https://re.jrc.ec.europa.eu/api/DRcalc?lat={lat}&lon={lon}&outputformat=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        dd = r.json()
        monthly = dd["outputs"]["monthly"]["fixed"]["E_d"]  # monthly kWh/m^2
        days = [31,28,31,30,31,30,31,31,30,31,30,31]
        ghi = {f"{i+1:02d}": float(monthly[i] / days[i]) for i in range(12)}
        return float(np.mean(list(ghi.values()))), ghi, "PVGIS"
    except Exception:
        pass

    # Final fallback
    ghi = {f"{i:02d}": 5.0 for i in range(1,13)}
    return 5.0, ghi, "Fallback (5.0 typical)"

def monthly_output_kwh(area_m2, ghi_monthly, panel_eff, coverage, tilt_deg, derate=0.75, shading_pct=10, latitude=20.0):
    """Return list of 12 monthly kWh estimates using simple model."""
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    active_area = area_m2 * coverage
    tilt_factor = max(0.5, math.cos(math.radians(tilt_deg - abs(latitude))))
    shading_factor = max(0.0, 1 - shading_pct/100.0)
    out = []
    for m in range(1,13):
        ghi = float(ghi_monthly.get(f"{m:02d}", 5.0))
        kwh = active_area * ghi * panel_eff * tilt_factor * days_in_month[m-1] * derate * shading_factor
        out.append(kwh)
    return out

def compute_panel_fit(area_m2, coverage, panel_area_m2=1.9):
    usable = area_m2 * coverage
    panels = int(usable // panel_area_m2)
    return max(0, panels)

def make_pdf_report(context):
    """Create PDF bytes for download. Simple formatted PDF with a small chart image."""
    from matplotlib import pyplot as plt
    # monthly chart
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(8,2.8))
    ax.plot(months, context["monthly_kwh"], marker="o")
    ax.set_title("Monthly Solar Output (kWh)")
    ax.set_ylabel("kWh")
    ax.grid(True, alpha=0.3)
    tmp_chart = "/tmp/solar_chart.png"
    fig.savefig(tmp_chart, bbox_inches="tight", dpi=150)
    plt.close(fig)
    # Compose PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 8, "Solar Analysis Report", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.ln(4)
    pdf.multi_cell(0, 6, f"Location (approx): {context['place']}")
    pdf.multi_cell(0, 6, f"Coordinates: {context['lat']:.6f}, {context['lon']:.6f}")
    pdf.multi_cell(0, 6, f"Rooftop area estimate: {context['area_m2']:.1f} m²")
    pdf.multi_cell(0, 6, f"Panels fit (est): {context['panels_fit']}")
    pdf.multi_cell(0, 6, f"Estimated annual output: {sum(context['monthly_kwh']):,.0f} kWh")
    pdf.ln(4)
    pdf.image(tmp_chart, w=180)
    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# -----------------------
# Navigation menu
# -----------------------
with st.sidebar:
    selection = option_menu(
        menu_title="Navigation",
        options=["Home", "Forecast", "Layout", "Chatbot", "Reports"],
        icons=["house", "bar-chart", "grid-1x2", "chat-quote", "file-earmark-text"],
        menu_icon="sun",
        default_index=0,
    )

# -----------------------
# Shared header function
# -----------------------
def page_header(image_url, title_line, subtitle_line=None):
    """Show a large header image and encouraging lines at top of page."""
    st.image(image_url, use_column_width=True)
    st.markdown(f"### {title_line}")
    if subtitle_line:
        st.markdown(f"**{subtitle_line}**")
    st.markdown("---")

# -----------------------
# Page: Home
# -----------------------
if selection == "Home":
    # header image + lines
    page_header(
        image_url="https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
        title_line="☀️ Switch to Solar — Power your future with clean energy!",
        subtitle_line="Draw inspiration — pick a place on the map to explore its solar potential."
    )

    # main content area with map on right
    left_col, right_col = st.columns([1, 1.2])
    with left_col:
        st.write("Welcome — click any point on the map at right to select a location. We will show latitude & longitude and try to resolve the city/place name.")
        st.markdown("**Quick tips:**")
        st.write("- Zoom into the area and click on the rooftop or nearby to pick a point.")
        st.write("- After selecting, go to **Forecast** page for resource and energy estimates.")
    with right_col:
        # folium map with Esri satellite tiles and city markers
        m = folium.Map(location=[21.0,78.0], zoom_start=5, control_scale=True)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Esri Satellite",
            overlay=False,
            control=False
        ).add_to(m)
        # a few sample city markers (visible)
        sample_cities = {
            "Delhi": (28.7041, 77.1025),
            "Mumbai": (19.0760, 72.8777),
            "Chennai": (13.0827, 80.2707),
            "Kolkata": (22.5726, 88.3639),
            "Bengaluru": (12.9716, 77.5946),
        }
        for city, (latc, lonc) in sample_cities.items():
            folium.CircleMarker(location=(latc, lonc), radius=4, tooltip=city, color="orange", fill=True).add_to(m)
        map_result = st_folium(m, width=700, height=520)
        # handle click
        if map_result and map_result.get("last_clicked"):
            pt = map_result["last_clicked"]
            lat, lon = float(pt["lat"]), float(pt["lng"])
            st.session_state.picked_point = (lat, lon)
            place = reverse_geocode(lat, lon) or "Unknown place"
            st.success(f"Selected: {place} — Lat {lat:.6f}, Lon {lon:.6f}")

    st.markdown("---")
    st.info("Disclaimer: This tool provides simplified estimates for educational purposes only. Always consult a qualified installer for a final site survey and quotation.")

# -----------------------
# Page: Forecast
# -----------------------
elif selection == "Forecast":
    page_header(
        image_url="https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
        title_line="📈 Solar Forecast — Know your sunshine",
        subtitle_line="We try NASA → PVGIS → fallback to estimate monthly PSH and energy output."
    )
    if not st.session_state.picked_point:
        st.warning("Please pick a point on the Home map first.")
        st.info("Tip: go to Home, click a point on the map at right, then return here.")
    else:
        lat, lon = st.session_state.picked_point
        place = reverse_geocode(lat, lon) or "Selected location"
        st.markdown(f"**Location:** {place} — `Lat: {lat:.6f}`, `Lon: {lon:.6f}`")
        # input: rooftop area estimate (user-provided, in m^2)
        area_m2 = st.number_input("Estimated rooftop area (m²)", min_value=5.0, value=50.0, step=1.0)
        st.session_state.area_m2 = area_m2
        panel_eff = st.slider("Panel efficiency (%)", 10, 22, 18) / 100.0
        coverage = st.slider("Coverage fraction (%)", 40, 95, 80) / 100.0
        tilt_deg = st.slider("Tilt angle (deg)", 0, 45, 20)
        shading_pct = st.slider("Shading loss (%)", 0, 60, 10)
        derate = st.slider("System derate (losses fraction)", 60, 90, 75) / 100.0

        ghi_avg, ghi_monthly, src = fetch_solar_ghi(lat, lon)
        if src.startswith("Fallback"):
            st.warning("Could not fetch NASA / PVGIS — using typical fallback PSH (5.0 kWh/m²/day).")
        else:
            st.success(f"Data source: {src} (avg PSH = {ghi_avg:.2f} kWh/m²/day)")

        st.session_state.ghi_monthly = ghi_monthly
        monthly_kwh = monthly_output_kwh(area_m2, ghi_monthly, panel_eff, coverage, tilt_deg, derate, shading_pct, lat)
        st.session_state.monthly_kwh = monthly_kwh

        # Plot monthly output
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=monthly_kwh, name="kWh"))
        fig.update_layout(title="Estimated Monthly Energy (kWh)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Estimated annual production (kWh)", f"{sum(monthly_kwh):,.0f}")
        st.markdown("---")
        st.info("Disclaimer: These are simplified estimates. A professional site survey is needed for final design and quotes.")

# -----------------------
# Page: Layout
# -----------------------
elif selection == "Layout":
    page_header(
        image_url="https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
        title_line="🔲 Panel Layout — visualize your rooftop",
        subtitle_line="Conceptual grid layout and per-panel energy estimate (interactive)."
    )
    if not st.session_state.monthly_kwh or not st.session_state.area_m2:
        st.warning("Run Forecast first (pick a location and set rooftop area).")
    else:
        area = st.session_state.area_m2
        coverage = st.slider("Coverage fraction (again)", 40, 95, int(coverage*100) if 'coverage' in locals() else 80) / 100.0
        panel_area = st.number_input("Panel area (m²) — typical ~1.7–2.0", value=1.9)
        panels = compute_panel_fit(area, coverage, panel_area)
        st.session_state.panels_fit = panels
        st.write(f"Estimated panels that can fit: **{panels}**")
        if panels <= 0:
            st.error("Rooftop too small given coverage/panel area. Increase area or reduce panel spacing.")
        else:
            per_panel_kwh = sum(st.session_state.monthly_kwh) / panels
            st.write(f"Approx annual per-panel energy: **{per_panel_kwh:.1f} kWh/yr**")
            # simple grid visualization with Plotly markers (hover tooltips)
            cols_guess = max(4, min(14, int(math.sqrt(panels) * 1.25)))
            rows_guess = math.ceil(panels / cols_guess)
            xs, ys, texts = [], [], []
            idx = 0
            for r in range(rows_guess):
                for c in range(cols_guess):
                    if idx >= panels:
                        break
                    xs.append(c); ys.append(-r)
                    texts.append(f"Panel #{idx+1}<br>{per_panel_kwh:.1f} kWh/yr")
                    idx += 1
            layout_fig = go.Figure(data=go.Scatter(x=xs, y=ys, mode="markers",
                                                   marker=dict(size=26, symbol="square", color="gold"),
                                                   hovertext=texts, hoverinfo="text"))
            layout_fig.update_layout(title="Hover panels to see per-panel annual kWh",
                                     xaxis=dict(visible=False), yaxis=dict(visible=False),
                                     template="plotly_white", height=420)
            st.plotly_chart(layout_fig, use_container_width=True)
    st.markdown("---")
    st.info("Disclaimer: This layout is conceptual. Final mounting, spacing, tilts, and roof obstructions must be surveyed by an installer.")

# -----------------------
# Page: Chatbot
# -----------------------
elif selection == "Chatbot":
    page_header(
        image_url="https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
        title_line="💬 Solar Assistant — ask anything",
        subtitle_line="Type a question about solar: cost, tilt, batteries, payback, subsidies."
    )
    question = st.text_input("Ask your question:")
    if question:
        q = question.lower()
        if "tilt" in q:
            st.success("Rule of thumb: set tilt ≈ your latitude for year-round output. Seasonal tweaks can improve yield.")
        elif "cost" in q or "price" in q:
            st.success("Typical residential rooftop PV installed cost (India) ranges ~₹40,000–80,000 per kW depending on components & region.")
        elif "battery" in q:
            st.success("Battery sizing: daily load × desired autonomy. Nameplate = usable / DoD. Lithium systems: DoD 80–90%, RTE ~90–95%.")
        elif "payback" in q or "savings" in q:
            st.success("Payback depends on electricity tariff, subsidies, metering, and self-consumption. Use Forecast + Reports pages for estimates.")
        else:
            st.info("Good question — include keywords like 'tilt', 'cost', 'battery', or 'payback' for targeted answers.")
    st.markdown("---")
    st.info("Disclaimer: Chatbot answers are general guidance only and not a substitute for professional advice.")

# -----------------------
# Page: Reports
# -----------------------
elif selection == "Reports":
    page_header(
        image_url="https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
        title_line="📄 Reports — export your rooftop summary",
        subtitle_line="Create a compact downloadable PDF with your picked-point & estimates."
    )
    if not st.session_state.monthly_kwh or not st.session_state.picked_point:
        st.warning("Pick a point on Home then run Forecast to generate report data.")
    else:
        lat, lon = st.session_state.picked_point
        place = reverse_geocode(lat, lon) or f"{lat:.6f}, {lon:.6f}"
        area = st.session_state.area_m2 or 50.0
        panels = st.session_state.panels_fit or compute_panel_fit(area, 0.8)
        ctx = {
            "place": place, "lat": lat, "lon": lon, "area_m2": area,
            "panels_fit": panels, "monthly_kwh": st.session_state.monthly_kwh
        }
        st.write("Quick preview:")
        st.metric("Location", place)
        st.metric("Area (m²)", f"{area:.1f}")
        st.metric("Panels (est)", panels)
        st.metric("Annual energy (kWh)", f"{sum(st.session_state.monthly_kwh):,.0f}")
        if st.button("Generate PDF Report"):
            pdf_bytes = make_pdf_report(ctx)
            st.download_button("Download PDF", data=pdf_bytes, file_name="solar_report.pdf", mime="application/pdf")
    st.markdown("---")
    st.info("Disclaimer: PDF is an aid. Final technical design and invoice require a site visit and professional engineering.")

# -----------------------
# Footer note (always visible)
# -----------------------
st.markdown("<br><hr><p style='font-size:12px;color:gray'>© Solar Suite — estimates are indicative only. This tool does not replace professional site assessment.</p>", unsafe_allow_html=True)

