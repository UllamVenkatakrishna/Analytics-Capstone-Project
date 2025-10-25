# app.py — NYC Citi Bike Dashboard (Fixed Anomaly Merge)
from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from meteostat import Stations, Hourly
from scipy import stats
from datetime import datetime, timedelta

st.set_page_config(page_title="NYC Citi Bike Dashboard", layout="wide", initial_sidebar_state="expanded")

MAPBOX_TOKEN = st.secrets.get("MAPBOX_API_KEY", os.getenv("MAPBOX_API_KEY", ""))
if not MAPBOX_TOKEN: st.error("Mapbox API Key not found. Please set secrets.toml or env var.")
MAP_STYLE = "mapbox://styles/mapbox/dark-v11"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

@st.cache_data(show_spinner="Loading summary data...")
def load_summary_data():
    BASE_DIR = Path(".")
    def _resolve_path(name: str) -> Path:
        paths_to_check = [BASE_DIR / name, Path("/Users/ashishb/Documents/STUDY/Capstone project/citi-bike dashboard/") / name]
        for p in paths_to_check:
            if p.exists(): return p
        raise FileNotFoundError(f"Missing required data file: {name}")
    hourly_ts = pd.read_csv(_resolve_path("hourly_demand_timekey.csv"), parse_dates=["hour"])
    daily_demand = pd.read_csv(_resolve_path("daily_demand.csv"))
    top_stations = pd.read_csv(_resolve_path("top_stations.csv"))
    station_hourly = pd.read_csv(_resolve_path("station_hourly_demand.csv"))
    daily_demand['day_of_week'] = pd.Categorical(daily_demand['day_of_week'], categories=DAY_ORDER, ordered=True)
    station_hourly['day_of_week'] = pd.Categorical(station_hourly['day_of_week'], categories=DAY_ORDER, ordered=True)
    station_hourly['hour'] = pd.to_numeric(station_hourly['hour'], errors='coerce')
    station_hourly.dropna(subset=['hour'], inplace=True)
    station_hourly['hour'] = station_hourly['hour'].astype(int)
    routes_file = None
    top_routes = pd.DataFrame()
    for c in ("top_200_routes_map.csv", "top_200_routes_map-2.csv"):
        try:
            routes_file = _resolve_path(c)
            break
        except FileNotFoundError:
            continue
    if routes_file:
        top_routes = pd.read_csv(routes_file)
    else:
        st.warning("Missing routes data file.")
    if not top_routes.empty:
        ren = {}
        req_cols = {'s_lat', 's_lng', 'e_lat', 'e_lng'}
        if "start_lat" in top_routes.columns: ren["start_lat"] = "s_lat"
        if "start_lng" in top_routes.columns: ren["start_lng"] = "s_lng"
        if "end_lat" in top_routes.columns: ren["end_lat"] = "e_lat"
        if "end_lng" in top_routes.columns: ren["end_lng"] = "e_lng"
        if ren: top_routes = top_routes.rename(columns=ren)
        if not req_cols.issubset(top_routes.columns): st.warning(f"Routes file missing coords.")
    station_locations = top_stations[['start_station_name', 'start_lat', 'start_lng']].drop_duplicates('start_station_name').set_index('start_station_name')
    return hourly_ts, daily_demand, station_locations, station_hourly, top_routes

try:
    hourly_ts, daily_demand, station_locations, station_hourly, top_routes = load_summary_data()
except FileNotFoundError as e:
    st.error(f"Data Loading Error: {e}.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading data: {e}")
    st.stop()

@st.cache_data(show_spinner="Fetching weather data...")
def load_weather_hourly(start_dt, end_dt):
    if pd.isna(start_dt) or pd.isna(end_dt): return pd.DataFrame(columns=["hour", "temp_c", "prcp_mm"])
    nyc_lat, nyc_lng = 40.7833, -73.9667
    stns = Stations().nearby(nyc_lat, nyc_lng).fetch(1)
    if stns.empty: return pd.DataFrame(columns=["hour", "temp_c", "prcp_mm"])
    station_id = stns.index[0]
    try:
        wx = Hourly(station_id, start_dt - timedelta(days=1), end_dt + timedelta(days=1), timezone="America/New_York").fetch()
        df = wx.reset_index().rename(columns={"time": "hour", "temp": "temp_c", "prcp": "prcp_mm"})
        df['hour'] = pd.to_datetime(df['hour'])
        return df[["hour", "temp_c", "prcp_mm"]]
    except Exception as e:
        st.warning(f"Weather fetch failed: {e}")
        return pd.DataFrame(columns=["hour", "temp_c", "prcp_mm"])

def tz_harmonize(series: pd.Series, tz="America/New_York") -> pd.Series:
    if series.empty: return series
    if not pd.api.types.is_datetime64_any_dtype(series): series = pd.to_datetime(series, errors='coerce')
    if series.dt.tz is None: return series.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    else: return series.dt.tz_convert(tz)

def resample_data(df, index_col, value_col, scale, agg_func='sum'):
    if df.empty or index_col not in df.columns or value_col not in df.columns:
        output_value_col_default = value_col.split('_')[0].capitalize()
        return pd.DataFrame(columns=["ts", output_value_col_default])
    d = df.set_index(index_col)[value_col]
    if scale == "Hourly": out = d.asfreq("H").fillna(0 if agg_func == 'sum' else np.nan)
    elif scale == "Daily": out = d.resample("D").agg(agg_func)
    else: out = d.resample("W-MON").agg(agg_func)
    output_value_col = value_col.split('_')[0].capitalize()
    if value_col == 'temp_c': output_value_col = 'Temp'
    if value_col == 'prcp_mm': output_value_col = 'Prcp'
    if value_col == 'trip_count': output_value_col = 'Trips'
    return out.reset_index().rename(columns={index_col: "ts", value_col: output_value_col})

def weekend_weekday_effect(ts_df):
    if ts_df.empty or 'hour' not in ts_df.columns or 'trip_count' not in ts_df.columns: return 0.0, 0.0
    df = ts_df.copy()
    df["dow"] = df["hour"].dt.dayofweek
    wkdy_d = df[df["dow"]<5].set_index("hour")["trip_count"].resample("D").sum()
    wknd_d = df[df["dow"]>=5].set_index("hour")["trip_count"].resample("D").sum()
    n_wkdy = len(wkdy_d.dropna())
    n_wknd = len(wknd_d.dropna())
    mean_wkdy = wkdy_d.mean() if n_wkdy > 0 else 0.0
    mean_wknd = wknd_d.mean() if n_wknd > 0 else 0.0
    m_diff = mean_wknd - mean_wkdy
    if n_wkdy < 2 or n_wknd < 2: return m_diff, 0.0
    var_wkdy = wkdy_d.var(ddof=1)
    var_wknd = wknd_d.var(ddof=1)
    if pd.isna(var_wkdy) or pd.isna(var_wknd): return m_diff, 0.0
    pooled_sd_numerator = ((n_wkdy - 1) * var_wkdy + (n_wknd - 1) * var_wknd)
    pooled_sd_denominator = max(1, (n_wkdy + n_wknd - 2))
    pooled_sd = np.sqrt(pooled_sd_numerator / pooled_sd_denominator)
    d = 0.0 if pooled_sd == 0 or pd.isna(pooled_sd) else m_diff / pooled_sd
    return m_diff, d

def anomaly_table(daily_df_input, wx_daily=None, z_thresh=2.5, top_k=10):
    if daily_df_input is None or daily_df_input.empty or 'Trips' not in daily_df_input.columns:
        return pd.DataFrame(columns=["date","Trips","z","type"])
    df = daily_df_input.copy()
    df['Trips'] = pd.to_numeric(df['Trips'], errors='coerce')
    df.dropna(subset=['Trips'], inplace=True)
    if len(df) < 2:
        return pd.DataFrame(columns=["date","Trips","z","type"])
    if 'date' not in df.columns:
        first_col = df.columns[0]
        if first_col != 'Trips':
            df = df.rename(columns={first_col: 'date'})
        else:
            df = df.reset_index().rename(columns={'index': 'date'})
    df["z"] = stats.zscore(df["Trips"].values)
    df["type"] = np.where(df["z"] >= z_thresh, "High", np.where(df["z"] <= -z_thresh, "Low", "Normal"))
    out = df[df["type"] != "Normal"].copy().sort_values(by="z", key=abs, ascending=False)
    if wx_daily is not None and not wx_daily.empty and not out.empty:
        try:
            out['date_merge_key'] = pd.to_datetime(out['date']).dt.date
            wx_daily['date_merge_key'] = pd.to_datetime(wx_daily['ts']).dt.date
            out = out.merge(wx_daily.rename(columns={"ts": "wx_date"}), on="date_merge_key", how="left").drop(columns=['date_merge_key'], errors='ignore')
        except Exception as e:
            st.warning(f"Could not merge anomalies with weather: {e}")
    cols = ["date", "Trips", "z", "type"]
    cols.extend([c for c in ["Temp", "Prcp"] if c in out.columns])
    return out[[c for c in cols if c in out.columns]].head(top_k)

st.sidebar.header("Master Filters")
start_dt, end_dt = None, None
if not hourly_ts.empty:
    min_dt_data, max_dt_data = hourly_ts["hour"].min(), hourly_ts["hour"].max()
    dr = st.sidebar.date_input("Date range", [min_dt_data.date(), max_dt_data.date()], min_value=min_dt_data.date(), max_value=max_dt_data.date(), key="date_range_filter")
    if len(dr) == 2:
        start_dt, end_dt = pd.to_datetime(dr[0]), pd.to_datetime(dr[1]) + pd.Timedelta(days=1)
else:
    st.sidebar.warning("Hourly time series data not found.")

sel_dow = st.sidebar.multiselect("Day(s) of week", options=DAY_ORDER, default=DAY_ORDER, key="dow_filter")
sel_hours = st.sidebar.slider("Hour range", 0, 23, (0, 23), key="hour_filter")
st.sidebar.header("Map Display Options")
map_layer_type = st.sidebar.radio("Station Layer Type", ['Column (3D)', 'Hexagon (3D Density)', 'Scatter + Heatmap (2D)'], key="map_layer_selector")
show_routes = st.sidebar.toggle("Show Top Routes Layer", value=True, key="show_routes_toggle")
topN_routes = st.sidebar.slider("Number of Top Routes", 20, 200, 100, step=10, disabled=not show_routes, key="top_routes_slider")
map_pitch = st.sidebar.slider("Map Pitch (Tilt)", 0, 60, 45, step=5, key="map_pitch_slider")
radius = 100
elevation_scale = 4
max_height_m = 4000
if map_layer_type == 'Column (3D)':
    radius = st.sidebar.slider("Column Radius (m)", 20, 150, 80, step=10, key="col_radius_slider")
    max_height_m = st.sidebar.slider("Target max bar height (m)", 1000, 8000, 4000, 250, key="col_height_slider")
elif map_layer_type == 'Hexagon (3D Density)':
    radius = st.sidebar.slider("Hexagon Radius (m)", 50, 250, 150, step=10, key="hex_radius_slider")
    elevation_scale = st.sidebar.slider("Elevation Scale", 1, 50, 10, step=2, key="hex_elev_slider")

filtered_daily_demand = pd.DataFrame()
if not daily_demand.empty:
    filtered_daily_demand = daily_demand[daily_demand['day_of_week'].isin(sel_dow)]

tab_overview, tab_map_explorer, tab_route_explorer, tab_station = st.tabs(["📊 Overview", "🗺️ Map Explorer (Stations)", "〰️ Route Explorer", "📍 Station Drilldown"])

with tab_overview:
    st.header("System-Wide Performance Overview")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    sel_ts = pd.DataFrame()
    if not hourly_ts.empty and start_dt and end_dt:
        tz = "America/New_York"
        sel_ts = hourly_ts[(hourly_ts["hour"] >= start_dt) & (hourly_ts["hour"] < end_dt)].copy()
        if not sel_ts.empty:
            sel_ts["hour"] = tz_harmonize(sel_ts["hour"], tz)
            total_trips = int(sel_ts["trip_count"].sum())
            sel_ts["dow"] = sel_ts["hour"].dt.dayofweek
            weekend_trips = sel_ts[sel_ts["dow"] >= 5]["trip_count"].sum()
            total_valid_trips = max(sel_ts["trip_count"].sum(), 1)
            weekend_share = 100.0 * weekend_trips / total_valid_trips
            peak_row = sel_ts.loc[[sel_ts["trip_count"].idxmax()]]
            peak_hour_dt = peak_row["hour"].iloc[0] if not peak_row.empty else None
            delta_avg, cohen_d = weekend_weekday_effect(sel_ts)
            kpi_col1.metric("Trips (selected)", f"{total_trips:,}")
            kpi_col2.metric("Weekend share", f"{weekend_share:.1f}%")
            kpi_col3.metric("Peak hour (overall)", peak_hour_dt.strftime("%H:00") if peak_hour_dt else "—")
            kpi_col4.metric("Weekend effect (d)", f"{cohen_d:.2f}")
        else:
            kpi_col1.metric("Trips (selected)", "0")
            kpi_col2.metric("Weekend share", "N/A")
            kpi_col3.metric("Peak hour (overall)", "N/A")
            kpi_col4.metric("Weekend effect (d)", "N/A")
    else:
        kpi_col1.metric("Trips (selected)", "Error")
        kpi_col2.metric("Weekend share", "Error")
        kpi_col3.metric("Peak hour (overall)", "Error")
        kpi_col4.metric("Weekend effect (d)", "Error")
    
    st.subheader("System Demand")
    ts_scale = st.selectbox("Time Scale", ["Hourly", "Daily", "Weekly"], index=1, key="ts_scale_overview")
    ts_view = st.selectbox("Chart Layers", ["Trips only", "Trips + Temp", "Trips + Precip", "Trips + Temp + Precip"], index=3, key="ts_view_overview")
    max_smooth_win = 24 if ts_scale == "Hourly" else (14 if ts_scale == "Daily" else 8)
    ts_smooth = st.slider("Smoothing (rolling periods, 0=off)", 0, max_smooth_win, 0, key="ts_smooth_overview")
    if not sel_ts.empty:
        trips_resampled = resample_data(sel_ts, "hour", "trip_count", ts_scale, 'sum')
        if ts_smooth > 0:
            trips_resampled["Trips"] = trips_resampled["Trips"].rolling(ts_smooth, center=True, min_periods=1).mean()
        wx_hourly = load_weather_hourly(start_dt, end_dt)
        wx_temp_resampled, wx_prcp_resampled = pd.DataFrame(), pd.DataFrame()
        if not wx_hourly.empty:
            wx_hourly["hour"] = tz_harmonize(wx_hourly["hour"])
        if "Temp" in ts_view:
            wx_temp_resampled = resample_data(wx_hourly, "hour", "temp_c", ts_scale, 'mean')
            if ts_smooth > 0 and 'Temp' in wx_temp_resampled.columns:
                wx_temp_resampled["Temp"] = wx_temp_resampled["Temp"].rolling(ts_smooth, center=True, min_periods=1).mean()
        if "Precip" in ts_view:
            wx_prcp_resampled = resample_data(wx_hourly, "hour", "prcp_mm", ts_scale, 'sum')
            if ts_smooth > 0 and 'Prcp' in wx_prcp_resampled.columns:
                wx_prcp_resampled["Prcp"] = wx_prcp_resampled["Prcp"].rolling(ts_smooth, center=True, min_periods=1).mean()
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(go.Scatter(x=trips_resampled["ts"], y=trips_resampled["Trips"], mode="lines", line=dict(color="#6EB5FF", width=2), name="Trips"), secondary_y=False)
        if not wx_temp_resampled.empty and "Temp" in ts_view:
            fig_ts.add_trace(go.Scatter(x=wx_temp_resampled["ts"], y=wx_temp_resampled["Temp"], mode="lines", line=dict(color="#FF7F50", width=1.5), name="Avg Temp (°C)"), secondary_y=True)
        if not wx_prcp_resampled.empty and "Precip" in ts_view:
            fig_ts.add_trace(go.Scatter(x=wx_prcp_resampled["ts"], y=wx_prcp_resampled["Prcp"], mode="lines", line=dict(color="#ADD8E6", width=1.5, dash='dot'), name="Total Precip (mm)"), secondary_y=True)
        fig_ts.update_yaxes(title_text="Trips", secondary_y=False)
        y2_title = [t for t in ["Avg Temp (°C)", "Total Precip (mm)"] if t.split()[0] in ts_view]
        fig_ts.update_yaxes(title_text=" / ".join(y2_title), secondary_y=True, showgrid=False)
        fig_ts.update_layout(margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", xaxis_title=None)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    st.subheader("Trend (7-day moving average)")
    if not sel_ts.empty:
        d_daily = sel_ts.set_index("hour")["trip_count"].resample("D").sum().rename("Trips").to_frame()
        if not d_daily.empty and len(d_daily) > 1:
            d_daily["MA7"] = d_daily["Trips"].rolling(7, center=True, min_periods=1).mean()
            sd14 = d_daily["Trips"].rolling(14, center=True, min_periods=1).std()
            d_daily["MA7_Upper"] = d_daily["MA7"] + 1.96 * sd14
            d_daily["MA7_Lower"] = d_daily["MA7"] - 1.96 * sd14
            fig_tr = go.Figure()
            fig_tr.add_trace(go.Scatter(x=d_daily.index.tolist() + d_daily.index.tolist()[::-1], y=d_daily["MA7_Upper"].tolist() + d_daily["MA7_Lower"].tolist()[::-1], fill='toself', fillcolor='rgba(0,176,246,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            fig_tr.add_trace(go.Scatter(x=d_daily.index, y=d_daily["MA7"], mode="lines", line=dict(color="#00B0F6", width=2.5), name="MA(7) Trend"))
            fig_tr.update_layout(margin=dict(l=0,r=0,t=5,b=0), yaxis_title="Trips (trend)", xaxis_title=None)
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info("Not enough daily data for trend.")
    
    st.subheader("Anomalies (daily z-score ≥ 2.5)")
    if not sel_ts.empty:
        if 'd_daily' in locals() and not d_daily.empty:
            daily_for_anom = d_daily.reset_index().rename(columns={"index":"date"})
        else:
            daily_for_anom = sel_ts.set_index("hour")["trip_count"].resample("D").sum().reset_index().rename(columns={"hour":"date", "trip_count":"Trips"})
        if not daily_for_anom.empty and len(daily_for_anom) > 1:
            wx_hourly_anom = load_weather_hourly(start_dt, end_dt)
            wx_daily_anom = pd.DataFrame()
            if not wx_hourly_anom.empty:
                wx_hourly_anom["hour"] = tz_harmonize(wx_hourly_anom["hour"])
                tmp_anom = wx_hourly_anom.rename(columns={"hour":"ts", "temp_c":"Temp", "prcp_mm":"Prcp"})
                wx_daily_anom = tmp_anom.set_index("ts")[["Temp", "Prcp"]].resample("D").agg({"Temp":"mean", "Prcp":"sum"}).reset_index()
            anom_df = anomaly_table(daily_for_anom, wx_daily_anom, z_thresh=2.5, top_k=12)
            if not anom_df.empty:
                st.dataframe(anom_df, use_container_width=True, hide_index=True)
                st.download_button("Download anomalies CSV", data=anom_df.to_csv(index=False).encode("utf-8"), file_name="citibike_anomalies.csv", mime="text/csv")
            else:
                st.info("No significant anomalies detected.")
        else:
            st.info("Not enough daily data for anomaly detection.")
    
    st.subheader("Day-of-Week × Rider Type")
    if not filtered_daily_demand.empty:
        pivot = filtered_daily_demand.pivot_table(index="day_of_week", columns="member_casual", values="trip_count", aggfunc='sum', observed=False).fillna(0)
        pivot = pivot.reindex(DAY_ORDER)
        fig_heat = px.imshow(pivot, text_auto=".3s", color_continuous_scale="Viridis", aspect="auto", labels=dict(color="Trips", x="Rider Type", y="Day of Week"))
        fig_heat.update_layout(margin=dict(l=0,r=0,t=20,b=0), title="Trip Distribution by Day and Rider Type (Filtered)")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data for heatmap based on filters.")

with tab_map_explorer:
    st.header("Map Explorer — Station Hotspots")
    st.markdown("Visualize station activity based on selected filters.")
    map_data_filtered = station_hourly.copy()
    if sel_dow:
        map_data_filtered = map_data_filtered[map_data_filtered['day_of_week'].isin(sel_dow)]
    map_data_filtered = map_data_filtered[(map_data_filtered['hour'] >= sel_hours[0]) & (map_data_filtered['hour'] <= sel_hours[1])]
    if not map_data_filtered.empty:
        station_summary = map_data_filtered.groupby('start_station_name', observed=False)['trip_count'].sum().reset_index()
        map_display_data = pd.merge(station_summary, station_locations, left_on='start_station_name', right_index=True, how='left').dropna(subset=['start_lat', 'start_lng'])
    else:
        map_display_data = pd.DataFrame(columns=['start_station_name', 'trip_count', 'start_lat', 'start_lng'])
    layers = []
    tooltip_content = {}
    current_pitch = map_pitch
    if map_display_data.empty:
        st.warning("No station data available.")
    else:
        min_count = map_display_data['trip_count'].min()
        max_count = map_display_data['trip_count'].max()
        map_display_data['norm_count'] = (map_display_data['trip_count'] - min_count) / max(1.0, max_count - min_count)
        if map_layer_type == 'Column (3D)':
            max_filtered_count = max(1.0, map_display_data["trip_count"].max())
            elevation_scale_dynamic = (max_height_m / max_filtered_count) if max_filtered_count > 0 else 0
            layers.append(pdk.Layer("ColumnLayer", data=map_display_data, get_position="[start_lng, start_lat]", get_elevation="trip_count", elevation_scale=elevation_scale_dynamic, radius=radius, get_fill_color="[255, (1 - norm_count) * 165, 0, 150 + norm_count * 105]", pickable=True, extruded=True, auto_highlight=True))
            tooltip_content = {"html": "<b>Station:</b> {start_station_name}<br/><b>Trips:</b> {trip_count}"}
        elif map_layer_type == 'Hexagon (3D Density)':
            layers.append(pdk.Layer("HexagonLayer", data=map_display_data, get_position="[start_lng, start_lat]", get_weight="trip_count", radius=radius, elevation_scale=elevation_scale, color_range=[[255, 255, 178], [254, 217, 118], [254, 178, 76], [253, 141, 60], [240, 59, 32], [189, 0, 38]], elevation_range=[0, 3000], extruded=True, coverage=0.9, pickable=True, auto_highlight=True))
            tooltip_content = {"html": "<b>Hotspot</b><br/>Agg. Trips: {elevationValue}"}
        elif map_layer_type == 'Scatter + Heatmap (2D)':
            current_pitch = 0
            layers.append(pdk.Layer('HeatmapLayer', data=map_display_data, get_position='[start_lng, start_lat]', opacity=0.9, get_weight='trip_count'))
            layers.append(pdk.Layer('ScatterplotLayer', data=map_display_data, get_position='[start_lng, start_lat]', get_radius=50, get_fill_color="[200 + norm_count * 55, 30 + (1-norm_count)*135 , 0, 160 + norm_count * 95]", radius_min_pixels=1, radius_max_pixels=15, pickable=True, auto_highlight=True))
            tooltip_content = {"html": "<b>Station:</b> {start_station_name}<br/><b>Trips:</b> {trip_count}"}
    if not layers:
        st.warning("No map layers.")
    else:
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.99, zoom=11.5, pitch=current_pitch, bearing=-20), map_provider="mapbox", map_style=MAP_STYLE, api_keys={"mapbox": MAPBOX_TOKEN}, tooltip=tooltip_content))

with tab_route_explorer:
    st.header("Route Explorer — Top Origin-Destination Flows")
    st.markdown("Visualize the most popular station-to-station routes.")
    routes_to_display = top_routes.nlargest(topN_routes, "trip_count")
    if routes_to_display.empty or not {'s_lat', 's_lng', 'e_lat', 'e_lng', 'trip_count'}.issubset(routes_to_display.columns):
        st.warning("Route data missing.")
    else:
        routes_to_display.dropna(subset=['s_lat', 's_lng', 'e_lat', 'e_lng'], inplace=True)
        max_count = routes_to_display['trip_count'].max()
        routes_to_display['width_px'] = np.clip(1 + 5 * (routes_to_display['trip_count'] / max(max_count, 1)), 1, 6)
        route_layer = pdk.Layer("ArcLayer", data=routes_to_display, get_source_position=["s_lng", "s_lat"], get_target_position=["e_lng", "e_lat"], get_width="width_px", get_source_color=[0, 255, 200, 160], get_target_color=[0, 128, 255, 160], pickable=True, auto_highlight=True)
        route_tooltip = {"html": "<b>Route:</b> {start_station_name} <br/><b>➜</b> {end_station_name}<br/><b>Total Trips:</b> {trip_count}", "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white", "border-radius": "5px", "padding": "5px"}}
        st.pydeck_chart(pdk.Deck(layers=[route_layer], initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.99, zoom=11, pitch=map_pitch, bearing=-20), map_provider="mapbox", map_style=MAP_STYLE, api_keys={"mapbox": MAPBOX_TOKEN}, tooltip=route_tooltip))
        st.caption(f"Showing Top {topN_routes} routes. Arc width indicates relative volume.")

with tab_station:
    st.header("Station Drilldown")
    station_list = sorted(station_hourly["start_station_name"].dropna().unique().tolist())
    sel_station = st.selectbox("Select station", station_list, index=station_list.index('W 21 St & 6 Ave') if 'W 21 St & 6 Ave' in station_list else 0, key="station_select")
    sd = station_hourly[station_hourly["start_station_name"] == sel_station].copy()
    if sd.empty:
        st.info("No data.")
    else:
        total_trips_station = int(sd["trip_count"].sum())
        avg_per_day = sd.groupby('day_of_week', observed=False)['trip_count'].sum().mean()
        peak_hr_row = sd.groupby("hour", as_index=False)["trip_count"].sum().sort_values("trip_count", ascending=False).head(1)
        peak_hr = int(peak_hr_row["hour"].iloc[0]) if not peak_hr_row.empty else None
        system_total_all_stations = int(station_hourly["trip_count"].sum())
        share_pct = (100.0 * total_trips_station / max(system_total_all_stations, 1))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Trips (Station, All Time)", f"{total_trips_station:,}")
        k2.metric("Est. Avg Trips/Day", f"{avg_per_day:,.0f}")
        k3.metric("Peak Hour (Typical)", f"{peak_hr}:00" if peak_hr is not None else "—")
        k4.metric("Share of System (All Time)", f"{share_pct:.2f}%")
        left, right = st.columns([2.2, 1.0])
        with left:
            st.subheader("Hourly Demand Profile")
            prof_h = sd.groupby("hour", as_index=False)["trip_count"].sum().sort_values("hour")
            fig1 = px.line(prof_h, x="hour", y="trip_count", labels={"hour":"Hour","trip_count":"Avg Trips/Hour (approx)"}, template="plotly_dark")
            fig1.update_traces(line_color="#6EB5FF", hovertemplate="Hour %{x}: %{y:,} trips (avg)")
            fig1.update_layout(margin=dict(l=0,r=8,t=10,b=0), xaxis=dict(showgrid=False, range=[-0.5, 23.5]), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"))
            st.plotly_chart(fig1, use_container_width=True)
            st.subheader("Daily Demand Profile")
            prof_d = sd.groupby("day_of_week", observed=False, as_index=False)["trip_count"].sum()
            prof_d['day_of_week'] = pd.Categorical(prof_d['day_of_week'], categories=DAY_ORDER, ordered=True)
            prof_d = prof_d.sort_values('day_of_week')
            fig2 = px.bar(prof_d, x="day_of_week", y="trip_count", text="trip_count", template="plotly_dark", color_discrete_sequence=["#FDB366"])
            fig2.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
            fig2.update_layout(margin=dict(l=0,r=8,t=10,b=0), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"))
            st.plotly_chart(fig2, use_container_width=True)
        with right:
            st.subheader("Location")
            row = station_locations.loc[[sel_station]] if sel_station in station_locations.index else pd.DataFrame()
            if not row.empty:
                lat0, lng0 = float(row["start_lat"].iloc[0]), float(row["start_lng"].iloc[0])
            else:
                st.warning("Station location missing.")
                lat0, lng0 = 40.73, -73.99
            marker_df = pd.DataFrame([{"lat": lat0, "lng": lng0, "name": sel_station}])
            st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", data=marker_df, get_position=["lng","lat"], get_radius=200, get_fill_color=[255,140,0,220], pickable=True)], initial_view_state=pdk.ViewState(latitude=lat0, longitude=lng0, zoom=14.5, pitch=40), map_provider="mapbox", map_style=MAP_STYLE, api_keys={"mapbox": MAPBOX_TOKEN}, tooltip={"text":"{name}"}))
            st.caption("Orange dot = station centroid.")

st.markdown("---")
st.caption("Data: Citi Bike System Data (processed via Colab). Weather: Meteostat. Tiles: Mapbox. Analysis includes MA trend, anomalies, DST-safe merging.")
