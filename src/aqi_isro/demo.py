from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data/demo")


def _india_grid():
    lats = np.linspace(8.0, 36.5, 18)
    lons = np.linspace(68.0, 97.5, 20)
    return lats, lons


def _seasonal_factor(day_of_year: int) -> float:
    return 1.0 + 0.25 * np.sin((2 * np.pi * day_of_year) / 365.25)


def _hotspot_centers_for_day(day_idx: int):
    # Two burning regimes that move over time. This creates distinct hotspot behavior.
    if day_idx < 20:
        return [(29.5, 76.0, 1.8), (26.5, 82.0, 1.2)]
    if day_idx < 45:
        return [(28.0, 78.5, 2.0), (23.0, 85.0, 1.4)]
    return [(27.0, 80.0, 1.6), (24.0, 86.0, 1.1)]


def _gaussian_bump(lat, lon, center_lat, center_lon, scale):
    d2 = ((lat - center_lat) ** 2) / 8.0 + ((lon - center_lon) ** 2) / 10.0
    return scale * np.exp(-d2)


def generate_demo_data(out_dir: Path = DEFAULT_DATA_DIR, n_days: int = 60) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    lats, lons = _india_grid()
    dates = pd.date_range("2024-10-01", periods=n_days, freq="D")

    rows = []
    for day_idx, date in enumerate(dates):
        seasonal = _seasonal_factor(date.dayofyear)
        fire_centers = _hotspot_centers_for_day(day_idx)
        for lat in lats:
            for lon in lons:
                fire_bump = sum(_gaussian_bump(lat, lon, clat, clon, scale) for clat, clon, scale in fire_centers)
                hcho = 1.2 + 0.35 * seasonal + 0.8 * fire_bump + rng.normal(0, 0.05)
                aod = 0.35 + 0.18 * seasonal + 0.22 * fire_bump + rng.normal(0, 0.03)
                no2 = 12 + 2.0 * seasonal + 3.2 * fire_bump + 0.04 * (97.5 - lon) + rng.normal(0, 0.8)
                so2 = 5 + 0.9 * fire_bump + 0.15 * seasonal + rng.normal(0, 0.3)
                co = 180 + 22 * fire_bump + 5 * seasonal + rng.normal(0, 2.5)
                o3 = 28 + 1.8 * seasonal - 0.8 * fire_bump + rng.normal(0, 1.0)
                temp = 24 + 6 * np.sin((date.dayofyear / 365.25) * 2 * np.pi) - 0.08 * (lat - 20) + rng.normal(0, 0.6)
                humidity = 60 - 0.4 * temp + rng.normal(0, 1.5)
                wind_u = 1.2 * np.sin((lat / 12.0) + day_idx / 7.0) + rng.normal(0, 0.25)
                wind_v = 1.0 * np.cos((lon / 14.0) + day_idx / 9.0) + rng.normal(0, 0.25)
                fire_count = max(0, int(round(1.5 + 5.5 * fire_bump + rng.normal(0, 0.6))))
                pm25 = (
                    18
                    + 42 * aod
                    + 0.55 * no2
                    + 0.03 * co
                    + 0.65 * fire_count
                    - 0.22 * wind_u
                    - 0.15 * wind_v
                    + rng.normal(0, 2.0)
                )
                aqi = np.clip(0.9 * pm25 + 0.35 * no2 + 0.2 * so2 + rng.normal(0, 2.0), 0, 500)
                rows.append(
                    {
                        "date": date,
                        "lat": lat,
                        "lon": lon,
                        "aod": aod,
                        "no2": no2,
                        "so2": so2,
                        "co": co,
                        "o3": o3,
                        "hcho": hcho,
                        "temp": temp,
                        "humidity": humidity,
                        "wind_u": wind_u,
                        "wind_v": wind_v,
                        "fire_count": fire_count,
                        "pm25": pm25,
                        "aqi": aqi,
                    }
                )

    spatial_df = pd.DataFrame(rows)
    spatial_df.to_csv(out_dir / "demo_spatial_data.csv", index=False)

    station_df = (
        spatial_df.sample(800, random_state=7)
        .loc[:, ["date", "lat", "lon", "aod", "no2", "so2", "co", "o3", "hcho", "temp", "humidity", "wind_u", "wind_v", "fire_count", "pm25", "aqi"]]
        .reset_index(drop=True)
    )
    station_df["station_id"] = [f"ST{idx:04d}" for idx in range(len(station_df))]
    cols = ["station_id"] + [c for c in station_df.columns if c != "station_id"]
    station_df = station_df[cols]
    station_df.to_csv(out_dir / "station_data.csv", index=False)


if __name__ == "__main__":
    generate_demo_data()
