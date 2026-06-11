import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from fetchers import get_current_weather, get_forecast, get_coordinates, get_historical_weather
from ml_engine import detect_anomalies, forecast_temperature, analyse_patterns
from ai_summary import generate_weather_summary

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="WeatherIQ", page_icon="https://openweathermap.org/img/wn/01d@2x.png", layout="wide")

TODAY = datetime.now().strftime("%A, %d %B %Y")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.5rem 2.5rem; }

[data-testid="metric-container"] {
    background: #161b2e;
    border: 1px solid #1f2d4a;
    border-radius: 12px;
    padding: 18px 20px;
}
[data-testid="metric-container"] label {
    color: #6b7fa3 !important;
    font-size: 11px !important;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700;
    color: #e8edf5 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

.insight-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 4px; }
.insight-card {
    background: #161b2e;
    border: 1px solid #1f2d4a;
    border-radius: 10px;
    padding: 16px 18px;
}
.insight-card .label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f8bf9;
    margin-bottom: 6px;
}
.insight-card .value { font-size: 13px; color: #c8d3e8; line-height: 1.6; }

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4f8bf9;
    margin-bottom: 12px;
    margin-top: 8px;
}

[data-testid="stSidebar"] { background: #0d1120; border-right: 1px solid #1a2240; }
[data-testid="stSidebar"] .stButton button {
    background: #4f8bf9;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px;
    width: 100%;
    margin-top: 8px;
    font-size: 13px;
}
[data-testid="stSidebar"] .stButton button:hover { background: #3a74e0; }
[data-testid="stTabs"] button { font-size: 13px; font-weight: 500; letter-spacing: 0.02em; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    plot_bgcolor="#0d1120",
    paper_bgcolor="#161b2e",
    font=dict(color="#c8d3e8", family="Inter"),
    margin=dict(t=40, b=40, l=50, r=20),
    xaxis=dict(gridcolor="#1f2d4a", showline=False, tickformat="%b %d"),
    yaxis=dict(gridcolor="#1f2d4a", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hoverlabel=dict(bgcolor="#1f2d4a", bordercolor="#4f8bf9", font_color="white")
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## WeatherIQ")
    st.markdown(f"<div style='color:#6b7fa3;font-size:12px;margin-bottom:20px'>{TODAY}</div>", unsafe_allow_html=True)
    city = st.text_input("City", value="Bengaluru", placeholder="Enter any city...")
    days_history = st.slider("Historical window (days)", 7, 30, 30)
    forecast_days = st.slider("Forecast horizon (days)", 3, 10, 7)
    run_button = st.button("Analyse Weather", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style='color:#6b7fa3;font-size:11px;line-height:2'>
        Data Sources<br>
        <span style='color:#4f8bf9'>—</span> OpenWeatherMap API<br>
        <span style='color:#4f8bf9'>—</span> Open-Meteo Historical<br>
        <span style='color:#4f8bf9'>—</span> Groq Llama 3.3<br>
        <span style='color:#4f8bf9'>—</span> Facebook Prophet ML
    </div>""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
if run_button:
    st.session_state.pop("weather_data", None)

if run_button or "weather_data" not in st.session_state:
    with st.spinner(f"Analysing {city}..."):
        current = get_current_weather(city)
        if current is None:
            st.error("City not found. Check the spelling and try again.")
            st.stop()
        lat, lon = get_coordinates(city)
        history = get_historical_weather(lat, lon, days=days_history)
        forecast_raw = get_forecast(city)
        history = detect_anomalies(history)
        ml_forecast = forecast_temperature(history, days=forecast_days)
        insights = analyse_patterns(history)
        ai_summary = generate_weather_summary(current, ml_forecast, insights, history)
        st.session_state.weather_data = dict(
            current=current, history=history, forecast_raw=forecast_raw,
            ml_forecast=ml_forecast, insights=insights, ai_summary=ai_summary,
            city=city
        )

d = st.session_state.weather_data
current = d["current"]; history = d["history"]
forecast_raw = d["forecast_raw"]; ml_forecast = d["ml_forecast"]
insights = d["insights"]; ai_summary = d["ai_summary"]

# ── WEATHER ICON ──────────────────────────────────────────────────────────────
icon_url = f"https://openweathermap.org/img/wn/{current['icon']}@4x.png"

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;gap:24px;background:#161b2e;border:1px solid #1f2d4a;border-radius:16px;padding:24px 28px;margin-bottom:20px'>
    <img src='{icon_url}' width='110' style='flex-shrink:0'>
    <div>
        <div style='font-size:12px;color:#6b7fa3;margin-bottom:6px;letter-spacing:0.05em'>{TODAY}</div>
        <div style='font-size:20px;font-weight:600;color:#e8edf5;margin-bottom:2px'>{current['city']}, {current['country']}</div>
        <div style='font-size:52px;font-weight:800;color:#e8edf5;line-height:1.05'>{current['temperature']}°C</div>
        <div style='font-size:14px;color:#a0b0cc;margin-top:6px'>{current['condition']}</div>
        <div style='font-size:12px;color:#6b7fa3;margin-top:4px'>Feels like {current['feels_like']}°C &nbsp;·&nbsp; Humidity {current['humidity']}% &nbsp;·&nbsp; Wind {current['wind_speed']} m/s</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Now", "Forecast", "History", "Anomalies", "Compare"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — NOW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{current['temperature']}°C", f"Feels {current['feels_like']}°C")
    c2.metric("Humidity", f"{current['humidity']}%")
    c3.metric("Wind Speed", f"{current['wind_speed']} m/s")
    c4.markdown(
    f"""
    <div style="padding-top:8px;">
        <div style="font-size:14px;color:#6b7fa3;font-weight:600;margin-bottom:6px;">
            Condition
        </div>
        <div style="font-size:1.3rem;font-weight:700;color:#e8edf5;line-height:1.2;">
            {current['condition']}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>AI Weather Insight</div>", unsafe_allow_html=True)

    lines = [l.strip() for l in ai_summary.strip().split('\n') if l.strip()]
    labels = ["Today", "Trend", "Recommendation", "Anomaly Alert"]

    cards_html = "<div class='insight-grid'>"
    for i, line in enumerate(lines[:4]):
        clean = line.lstrip("1234567890.-) ").strip()
        label = labels[i] if i < len(labels) else f"Insight {i+1}"
        cards_html += f"""
        <div class='insight-card'>
            <div class='label'>{label}</div>
            <div class='value'>{clean}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>30-Day Summary</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg Temperature", f"{insights['avg_temp']}°C")
    s2.metric("Hottest Day", f"{insights['max_temp']}°C", insights['max_temp_date'])
    s3.metric("Coolest Day", f"{insights['min_temp']}°C", insights['min_temp_date'])
    s4.metric("Rainy Days", f"{insights['rainy_days']} / {days_history}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-label'>OpenWeatherMap — 5-Day</div>", unsafe_allow_html=True)
        forecast_raw["date"] = forecast_raw["datetime"].dt.date
        daily = forecast_raw.groupby("date").agg({"temperature": "mean", "humidity": "mean"}).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        fig1 = px.area(daily, x="date", y="temperature", color_discrete_sequence=["#4f8bf9"])
        fig1.update_traces(fill='tozeroy', fillcolor='rgba(79,139,249,0.1)', line_width=2)
        fig1.update_layout(**CHART_LAYOUT, title="Temperature Forecast (°C)", xaxis_title="", yaxis_title="°C")
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.markdown(f"<div class='section-label'>Prophet ML Model — {forecast_days}-Day</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ml_forecast["date"], y=ml_forecast["upper_bound"], line=dict(width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=ml_forecast["date"], y=ml_forecast["lower_bound"], fill='tonexty', fillcolor='rgba(79,139,249,0.12)', line=dict(width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=ml_forecast["date"], y=ml_forecast["predicted_temp"], line=dict(color="#4f8bf9", width=2.5), mode="lines+markers", marker=dict(size=6), name="Predicted"))
        fig2.update_layout(**CHART_LAYOUT, title="ML Predicted Temp with 95% Confidence (°C)", xaxis_title="", yaxis_title="°C")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-label'>Forecast Table</div>", unsafe_allow_html=True)
    table_df = ml_forecast.copy()
    table_df["date"] = table_df["date"].dt.strftime("%b %d, %Y")
    table_df.columns = ["Date", "Predicted °C", "Low °C", "High °C"]
    st.dataframe(table_df.round(1), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-label'>Temperature Range</div>", unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=history["date"], y=history["temp_max"], name="Max", line=dict(color="#ff6b6b", width=1.5)))
    fig3.add_trace(go.Scatter(x=history["date"], y=history["temp_avg"], name="Avg", line=dict(color="#4f8bf9", width=2), fill='tonexty', fillcolor='rgba(79,139,249,0.08)'))
    fig3.add_trace(go.Scatter(x=history["date"], y=history["temp_min"], name="Min", line=dict(color="#74c0fc", width=1.5), fill='tonexty', fillcolor='rgba(116,192,252,0.05)'))
    fig3.update_layout(**CHART_LAYOUT, yaxis_title="°C", xaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-label'>Daily Precipitation (mm)</div>", unsafe_allow_html=True)
        fig4 = px.bar(history, x="date", y="precipitation", color_discrete_sequence=["#4f8bf9"])
        fig4.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="mm")
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        st.markdown("<div class='section-label'>Wind Speed (m/s)</div>", unsafe_allow_html=True)
        fig5 = px.line(history, x="date", y="wind_speed", color_discrete_sequence=["#74c0fc"])
        fig5.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="m/s")
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    anomalies = history[history["is_anomaly"] == True]
    normal = history[history["is_anomaly"] == False]

    a1, a2, a3 = st.columns(3)
    a1.metric("Days Analysed", len(history))
    a2.metric("Anomalies Found", len(anomalies))
    a3.metric("Detection Model", "Isolation Forest")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Anomaly Map — Red markers are statistically unusual days</div>", unsafe_allow_html=True)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=normal["date"], y=normal["temp_avg"], mode="lines+markers", name="Normal", line=dict(color="#4f8bf9", width=2), marker=dict(size=5)))
    fig6.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["temp_avg"], mode="markers", name="Anomaly", marker=dict(color="#ff4b4b", size=14, symbol="x", line=dict(width=2, color="#ff4b4b"))))
    fig6.update_layout(**CHART_LAYOUT, yaxis_title="Avg Temp °C", xaxis_title="")
    st.plotly_chart(fig6, use_container_width=True)

    if len(anomalies) > 0:
        st.markdown("<div class='section-label'>Flagged Days</div>", unsafe_allow_html=True)
        disp = anomalies[["date", "temp_avg", "temp_max", "temp_min", "precipitation", "temp_zscore"]].copy()
        disp["date"] = disp["date"].dt.strftime("%b %d, %Y")
        disp.columns = ["Date", "Avg °C", "Max °C", "Min °C", "Rain mm", "Z-Score"]
        st.dataframe(disp.round(2), use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected in the selected period.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-label'>Compare up to 3 cities side by side</div>", unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    city1_label = cc1.text_input("City 1", value=current['city'], placeholder="Enter city...")
    city2 = cc2.text_input("City 2", value="Mumbai", placeholder="Enter city...")
    city3 = cc3.text_input("City 3", value="Chennai", placeholder="Enter city...")
    compare_btn = st.button("Compare", use_container_width=True)

    if compare_btn:
        with st.spinner("Fetching city data..."):
            cities = [city1_label, city2, city3]
            rows = []
            icons_compare = []
            for c in cities:
                w = get_current_weather(c)
                if w:
                    rows.append({
                        "City": w["city"],
                        "Temp °C": w["temperature"],
                        "Feels Like °C": w["feels_like"],
                        "Humidity %": w["humidity"],
                        "Wind m/s": w["wind_speed"],
                        "Condition": w["condition"]
                    })
                    icons_compare.append(f"https://openweathermap.org/img/wn/{w['icon']}@2x.png")

            df_c = pd.DataFrame(rows)
            city_labels = [f"City {i+1} — {row['City']}" for i, (_, row) in enumerate(df_c.iterrows())]

            card_cols = st.columns(len(df_c))
            for i, (_, row) in enumerate(df_c.iterrows()):
                with card_cols[i]:
                    st.image(icons_compare[i], width=60)
                    st.metric(city_labels[i], f"{row['Temp °C']}°C", row["Condition"])

            st.markdown("<br>", unsafe_allow_html=True)

            fig7 = go.Figure()
            colors = ["#4f8bf9", "#ff6b6b", "#74c0fc"]
            metrics = ["Temp °C", "Humidity %", "Wind m/s"]
            for i, (_, row) in enumerate(df_c.iterrows()):
                fig7.add_trace(go.Bar(
                    name=row["City"],
                    x=metrics,
                    y=[row["Temp °C"], row["Humidity %"], row["Wind m/s"]],
                    marker_color=colors[i % len(colors)]
                ))
            fig7.update_layout(**CHART_LAYOUT, barmode="group", xaxis_title="", yaxis_title="Value", title="Side-by-Side Comparison")
            st.plotly_chart(fig7, use_container_width=True)
            st.dataframe(df_c, use_container_width=True, hide_index=True)