from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"


def build_aqi_figures(pred_df: pd.DataFrame, station_df: pd.DataFrame, selected_date, metrics: Dict[str, float]):
    title_date = pd.to_datetime(selected_date).strftime("%Y-%m-%d")
    color_scale = [
        [0.0, "#1a9850"],
        [0.2, "#91cf60"],
        [0.4, "#d9ef8b"],
        [0.6, "#fee08b"],
        [0.8, "#fc8d59"],
        [1.0, "#d73027"],
    ]
    map_fig = px.scatter_geo(
        pred_df,
        lat="lat",
        lon="lon",
        color="aqi_pred",
        size="uncertainty",
        color_continuous_scale=color_scale,
        projection="natural earth",
        title=f"Predicted Surface AQI over India - {title_date}",
    )
    map_fig.update_geos(
        lataxis_range=[5, 38],
        lonaxis_range=[67, 99],
        showland=True,
        landcolor="#1e293b",
        countrycolor="#475569",
        showocean=True,
        oceancolor="#0f172a",
        showcoastlines=False,
    )
    map_fig.update_layout(
        height=650, 
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    scatter_fig = px.scatter(
        station_df.sample(min(len(station_df), 300), random_state=3),
        x="aqi",
        y="pm25",
        title=f"CPCB Validation Snapshot - RMSE {metrics['rmse']:.2f}",
        labels={"aqi": "Observed AQI", "pm25": "PM2.5 proxy"},
        color_discrete_sequence=["#38bdf8"],
    )
    scatter_fig.update_layout(
        height=420, 
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fi = pd.Series(metrics.get("feature_importance", {}))
    if len(fi) == 0:
        fi = pd.Series({"aod": 1.0})
    feature_importance_fig = px.bar(
        x=fi.sort_values(ascending=True).values,
        y=fi.sort_values(ascending=True).index,
        orientation="h",
        title="Model Feature Importance",
        labels={"x": "Importance", "y": "Feature"},
        color_discrete_sequence=["#818cf8"],
    )
    feature_importance_fig.update_layout(
        height=420, 
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return {"map": map_fig, "scatter": scatter_fig, "feature_importance": feature_importance_fig}


def build_hcho_figures(hotspot_df: pd.DataFrame, summary: Dict[str, float], selected_date):
    title_date = pd.to_datetime(selected_date).strftime("%Y-%m-%d")
    map_fig = px.scatter_geo(
        hotspot_df,
        lat="lat",
        lon="lon",
        color="hotspot_score",
        size="fire_count",
        color_continuous_scale="Turbo",
        projection="natural earth",
        title=f"HCHO Hotspot Map - {title_date}",
    )
    map_fig.update_geos(
        lataxis_range=[5, 38],
        lonaxis_range=[67, 99],
        showland=True,
        landcolor="#1e293b",
        countrycolor="#475569",
        showocean=True,
        oceancolor="#0f172a",
        showcoastlines=False,
    )
    map_fig.update_layout(
        height=650, 
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    corr_fig = px.scatter(
        hotspot_df,
        x="fire_count",
        y="hcho_anomaly",
        color="hotspot_score",
        color_continuous_scale="Turbo",
        title="Fire Count vs HCHO Anomaly",
        labels={"fire_count": "Fire count", "hcho_anomaly": "HCHO anomaly"},
    )
    corr_fig.update_layout(
        height=420, 
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    wind_fig = px.scatter(
        hotspot_df,
        x="wind_u",
        y="wind_v",
        color="hotspot_score",
        color_continuous_scale="Turbo",
        title="Wind Field Context",
        labels={"wind_u": "Wind U", "wind_v": "Wind V"},
    )
    wind_fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=10, color="white"),
            name="Origin",
        )
    )
    wind_fig.update_layout(
        height=420, 
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return {"map": map_fig, "correlation": corr_fig, "wind": wind_fig}
