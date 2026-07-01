# Methodology

## Objective 1: Surface AQI

The AQI module fuses satellite pollutant columns, AOD, meteorology, and ground-based CPCB observations. The working demo uses synthetic India-like data so the complete system can be tested without waiting for external data access.

The production pipeline should use the same schema:

- `date`
- `lat`
- `lon`
- `aod`
- `no2`
- `so2`
- `co`
- `o3`
- `hcho`
- `temp`
- `humidity`
- `wind_u`
- `wind_v`
- `fire_count`
- `pm25`
- `aqi`

The baseline model is a pure NumPy ridge regression. It is intentionally dependency-light for hackathon reliability. The next upgrade should be XGBoost or LightGBM, followed by CNN-LSTM or ConvLSTM after the real gridded data is stable.

## Objective 2: HCHO hotspots

The HCHO module computes:

- local HCHO anomaly against the daily spatial baseline
- fire-linked hotspot score
- wind-speed contribution
- likely source-region attribution

This creates a stronger story than plain thresholding because the hotspot output is connected to fire activity and transport context.

## Differentiators

- AQI prediction and HCHO hotspot detection are connected in one platform.
- Outputs include uncertainty/confidence for AQI maps.
- HCHO hotspots include source-region attribution.
- The demo runs end to end with synthetic data and can later be swapped to real satellite products.
