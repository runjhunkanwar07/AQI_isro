# AQI ISRO Hackathon Project

This project is a hackathon-ready prototype for the ISRO problem statement:

- surface AQI estimation over India using satellite, ground, and meteorological data
- HCHO hotspot detection during biomass burning seasons

The repo is designed to be **working end to end** even before the real datasets are connected.
It includes:

- a synthetic demo data generator
- a trainable AQI regression model
- HCHO hotspot detection with fire and wind influence scoring
- a Streamlit dashboard for presentation

## Why this approach is strong for a hackathon

- It is realistic: the pipeline can run on demo data and later swap to real data.
- It is differentiable: it combines AQI prediction plus transport-aware HCHO attribution.
- It is scientifically defensible: AQI is predicted from fused satellite and meteorological features, then interpreted with station validation.
- It is presentation-friendly: outputs include maps, metrics, and time-series analysis.

## Project structure

- `app.py` - Streamlit dashboard
- `src/aqi_isro/` - reusable pipeline code
- `scripts/generate_demo_data.py` - creates synthetic India-like data for demo/testing
- `scripts/train_demo.py` - trains the AQI model and saves artifacts
- `data/` - generated demo data and model outputs

## Quick start

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Generate demo data:

```bash
python scripts/generate_demo_data.py
```

4. Train the demo model:

```bash
python scripts/train_demo.py
```

5. Or run the complete demo pipeline in one command:

```bash
python scripts/run_demo_pipeline.py
```

6. Export CSV outputs for reports or slides:

```bash
python scripts/export_demo_outputs.py
```

7. Build a static dashboard that opens directly in a browser:

```bash
python scripts/build_static_dashboard.py
```

8. Launch the optional Streamlit dashboard:

```bash
streamlit run app.py
```

## Real data integration plan

Replace demo CSVs with:

- INSAT-3D AOD
- Sentinel-5P HCHO / NO2 / SO2 / CO / O3
- CPCB station observations
- MODIS / VIIRS fire counts
- ERA5 / IMDAA / MERRA-2 meteorology

The rest of the pipeline stays the same.

## Hackathon message

Your story should be:

> Satellite data gives the pollution field, CPCB validates the ground truth, meteorology explains transport, and fire activity explains biomass-burning-driven HCHO spikes.

That makes the project more than a map: it becomes an interpretable air-quality intelligence system.
