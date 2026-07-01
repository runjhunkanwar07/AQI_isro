"""
Real Data Ingestion Module for ISRO AQI Hackathon
Downloads actual satellite and ground-truth data from official sources.

Data Sources:
1. NASA FIRMS   — MODIS/VIIRS Active Fire Data (direct CSV, no API key for South Asia)
2. Sentinel-5P  — TROPOMI NO2, SO2, CO, O3, HCHO (DLR direct download or GEE)
3. CPCB         — Ground-based air quality data (via OpenAQ API)
4. ERA5         — Reanalysis meteorological data (via CDS API)
5. INSAT-3D     — AOD data (via MOSDAC portal)
"""

from __future__ import annotations

import logging
import io
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDIA_BBOX = {"west": 68, "south": 8, "east": 98, "north": 38}
REAL_DATA_DIR = Path("data/real")


# ============================================================
# 1. NASA FIRMS — MODIS/VIIRS Fire Count Data
# ============================================================

FIRMS_DIRECT_URLS = {
    "MODIS_24h": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_24h.csv",
    "MODIS_48h": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_48h.csv",
    "MODIS_7d": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_7d.csv",
    "VIIRS_SNPP_24h": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_24h.csv",
    "VIIRS_SNPP_7d": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_7d.csv",
    "VIIRS_NOAA20_24h": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_Asia_24h.csv",
    "VIIRS_NOAA20_7d": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_Asia_7d.csv",
}


def download_firms_fire_data(time_range: str = "7d", sensor: str = "VIIRS_SNPP", out_dir: Path = REAL_DATA_DIR) -> pd.DataFrame:
    """
    Download MODIS/VIIRS active fire data for South Asia from NASA FIRMS.
    No API key required for these direct downloads.

    Args:
        time_range: "24h", "48h", or "7d"
        sensor: "MODIS", "VIIRS_SNPP", or "VIIRS_NOAA20"
        out_dir: Directory to save the CSV
    Returns:
        pd.DataFrame of fire detections within India
    """
    if requests is None:
        raise ImportError("Please install requests: pip install requests")

    key = f"{sensor}_{time_range}"
    url = FIRMS_DIRECT_URLS.get(key)
    if url is None:
        raise ValueError(f"Invalid combination: {key}. Valid keys: {list(FIRMS_DIRECT_URLS.keys())}")

    logger.info(f"Downloading fire data from NASA FIRMS: {key}")
    logger.info(f"URL: {url}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    logger.info(f"Downloaded {len(df)} fire detections for South Asia")

    # Filter to India bounding box
    df = df[
        (df["latitude"] >= INDIA_BBOX["south"])
        & (df["latitude"] <= INDIA_BBOX["north"])
        & (df["longitude"] >= INDIA_BBOX["west"])
        & (df["longitude"] <= INDIA_BBOX["east"])
    ].copy()
    logger.info(f"Filtered to {len(df)} fire detections within India")

    # Save to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"firms_fire_{sensor}_{time_range}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    return df


def download_firms_api(map_key: str, source: str = "VIIRS_SNPP_NRT", days: int = 30, out_dir: Path = REAL_DATA_DIR) -> pd.DataFrame:
    """
    Download fire data using the FIRMS REST API (requires free MAP_KEY).
    Get your key at: https://firms.modaps.eosdis.nasa.gov/api/map_key

    Args:
        map_key: Your FIRMS API key
        source: Sensor identifier (e.g., "VIIRS_SNPP_NRT", "MODIS_NRT")
        days: Number of days of data (will be fetched in 5-day chunks)
        out_dir: Output directory
    """
    if requests is None:
        raise ImportError("Please install requests: pip install requests")

    bbox = f"{INDIA_BBOX['west']},{INDIA_BBOX['south']},{INDIA_BBOX['east']},{INDIA_BBOX['north']}"
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    all_data = []
    end_date = datetime.now()

    chunks = (days + 4) // 5  # ceil division
    for i in range(chunks):
        date_str = (end_date - timedelta(days=i * 5)).strftime("%Y-%m-%d")
        url = f"{base_url}/{map_key}/{source}/{bbox}/5/{date_str}"
        logger.info(f"Fetching chunk {i+1}/{chunks}: {date_str}")

        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and resp.text.strip():
            chunk_df = pd.read_csv(io.StringIO(resp.text))
            all_data.append(chunk_df)
        else:
            logger.warning(f"Failed for {date_str}: HTTP {resp.status_code}")

    if not all_data:
        logger.error("No data retrieved from FIRMS API")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True).drop_duplicates()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"firms_fire_{source}_{days}d.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} fire detections to {out_path}")
    return df


# ============================================================
# 2. Sentinel-5P TROPOMI — NO2, SO2, CO, O3, HCHO
# ============================================================

# DLR direct download (NetCDF, global files)
DLR_STAC_COLLECTIONS = {
    "NO2": "S5P_TROPOMI_L3_P1D_NO2",
    "SO2": "S5P_TROPOMI_L3_P1D_SO2",
    "CO": "S5P_TROPOMI_L3_P1D_CO",
    "O3": "S5P_TROPOMI_L3_P1D_O3",
    "HCHO": "S5P_TROPOMI_L3_P1D_HCHO",
}

# GEE collection IDs (recommended for India-specific analysis)
GEE_COLLECTIONS = {
    "NO2": {"collection": "COPERNICUS/S5P/OFFL/L3_NO2", "band": "tropospheric_NO2_column_number_density"},
    "SO2": {"collection": "COPERNICUS/S5P/OFFL/L3_SO2", "band": "SO2_column_number_density"},
    "CO": {"collection": "COPERNICUS/S5P/OFFL/L3_CO", "band": "CO_column_number_density"},
    "O3": {"collection": "COPERNICUS/S5P/OFFL/L3_O3", "band": "O3_column_number_density"},
    "HCHO": {"collection": "COPERNICUS/S5P/OFFL/L3_HCHO", "band": "tropospheric_HCHO_column_number_density"},
}


def download_sentinel5p_dlr(pollutant: str, date_str: str, out_dir: Path = REAL_DATA_DIR) -> Path:
    """
    Download a Sentinel-5P L3 daily composite from DLR Geoservice.
    The file is global NetCDF; subset to India using xarray after download.

    Args:
        pollutant: One of "NO2", "SO2", "CO", "O3", "HCHO"
        date_str: "YYYY-MM-DD"
        out_dir: Output directory
    Returns:
        Path to downloaded NetCDF file
    """
    if requests is None:
        raise ImportError("Please install requests: pip install requests")

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

    # Browse the DLR directory for the specific date
    base_url = f"https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/{year}/{month}/{day}/"
    logger.info(f"Browsing DLR directory: {base_url}")

    resp = requests.get(base_url, timeout=30)
    if resp.status_code != 200:
        logger.error(f"Could not access DLR directory for {date_str}: HTTP {resp.status_code}")
        return None

    # Parse HTML to find relevant product folder
    import re
    folders = re.findall(r'href="([^"]*' + pollutant.upper() + r'[^"]*)"', resp.text)
    if not folders:
        # Try lowercase
        folders = re.findall(r'href="([^"]*' + pollutant.lower() + r'[^"]*)"', resp.text)

    if not folders:
        logger.error(f"No {pollutant} product found for {date_str}")
        return None

    folder_url = base_url + folders[0]
    logger.info(f"Found product folder: {folder_url}")

    # List files in product folder
    resp2 = requests.get(folder_url, timeout=30)
    nc_files = re.findall(r'href="([^"]*\.nc)"', resp2.text)

    if not nc_files:
        logger.error(f"No NetCDF files found in {folder_url}")
        return None

    file_url = folder_url + nc_files[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"S5P_{pollutant}_{date_str}.nc"

    logger.info(f"Downloading {file_url} ...")
    resp3 = requests.get(file_url, stream=True, timeout=120)
    resp3.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp3.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Saved to {out_path}")
    return out_path


def fetch_sentinel5p_gee(pollutant: str, start_date: str, end_date: str, out_dir: Path = REAL_DATA_DIR):
    """
    Fetch Sentinel-5P data via Google Earth Engine and export to Drive.
    Requires: pip install earthengine-api && earthengine authenticate

    Args:
        pollutant: One of "NO2", "SO2", "CO", "O3", "HCHO"
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
    """
    try:
        import ee
    except ImportError:
        raise ImportError("Install earthengine-api: pip install earthengine-api")

    ee.Initialize()
    config = GEE_COLLECTIONS[pollutant]
    roi = ee.Geometry.Rectangle([68.0, 7.0, 98.0, 36.0])

    collection = (
        ee.ImageCollection(config["collection"])
        .select(config["band"])
        .filterDate(start_date, end_date)
        .filterBounds(roi)
    )

    image = collection.mean().clip(roi)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"S5P_{pollutant}_India_{start_date}_to_{end_date}",
        folder="ISRO_AQI_Data",
        region=roi,
        scale=11132,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    logger.info(f"GEE export task started for {pollutant}. Check Google Drive folder 'ISRO_AQI_Data'.")
    return task


# ============================================================
# 3. CPCB Ground-Based AQI Data (via OpenAQ)
# ============================================================

OPENAQ_BASE = "https://api.openaq.org/v3"


def download_cpcb_openaq(api_key: str = None, limit: int = 1000, out_dir: Path = REAL_DATA_DIR) -> pd.DataFrame:
    """
    Download Indian ground-station air quality data via OpenAQ API.
    Register for a free API key at: https://explore.openaq.org/register

    Args:
        api_key: OpenAQ API key (optional but recommended for higher rate limits)
        limit: Number of locations to fetch
        out_dir: Output directory
    Returns:
        pd.DataFrame of station locations with latest measurements
    """
    if requests is None:
        raise ImportError("Please install requests: pip install requests")

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # Fetch Indian monitoring locations
    logger.info("Fetching Indian air quality stations from OpenAQ (v2 API)...")
    url = f"https://api.openaq.org/v2/locations?country=IN&limit={limit}"
    resp = requests.get(url, headers=headers, timeout=30)

    if resp.status_code != 200:
        logger.error(f"OpenAQ API error: {resp.status_code} - {resp.text[:200]}")
        return pd.DataFrame()

    data = resp.json()
    locations = data.get("results", [])
    logger.info(f"Found {len(locations)} monitoring stations in India")

    rows = []
    for loc in locations:
        coords = loc.get("coordinates", {})
        for param in loc.get("parameters", []):
            rows.append({
                "station_id": loc.get("id"),
                "station_name": loc.get("name"),
                "city": loc.get("city"),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "parameter": param.get("displayName", param.get("parameter")),
                "last_value": param.get("lastValue"),
                "unit": param.get("unit"),
                "last_updated": param.get("lastUpdated"),
            })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cpcb_openaq_stations.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} records to {out_path}")
    return df


# ============================================================
# 4. ERA5 Reanalysis Meteorological Data
# ============================================================

def download_era5(year: str = "2024", months: list = None, out_dir: Path = REAL_DATA_DIR) -> Path:
    """
    Download ERA5 single-level reanalysis data for India.
    Requires: pip install cdsapi
    Requires: ~/.cdsapirc file with CDS API credentials
    Register at: https://cds.climate.copernicus.eu/

    Args:
        year: Target year
        months: List of month strings ["01", "02", ...]
        out_dir: Output directory
    Returns:
        Path to downloaded NetCDF file
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError("Install cdsapi: pip install cdsapi. Then configure ~/.cdsapirc")

    if months is None:
        months = [f"{m:02d}" for m in range(1, 13)]

    client = cdsapi.Client()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_india_{year}.nc"

    request = {
        "product_type": ["reanalysis"],
        "format": "netcdf",
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "boundary_layer_height",
            "total_precipitation",
        ],
        "year": [year],
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [38, 68, 5, 98],  # [North, West, South, East]
    }

    logger.info(f"Submitting ERA5 download request for {year}...")
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    logger.info(f"Downloaded ERA5 data to {out_path}")
    return out_path


# ============================================================
# MASTER DOWNLOAD FUNCTION
# ============================================================

def download_all_available(out_dir: Path = REAL_DATA_DIR):
    """
    Download all data that doesn't require API keys or special authentication.
    Currently this is just FIRMS fire data (direct CSV download).
    """
    logger.info("=" * 60)
    logger.info("ISRO AQI PROJECT — REAL DATA DOWNLOAD")
    logger.info("=" * 60)

    results = {}

    # 1. FIRMS Fire Data (NO API KEY NEEDED)
    logger.info("\n[1/4] Downloading NASA FIRMS fire data (no API key needed)...")
    try:
        fire_df = download_firms_fire_data("7d", "VIIRS_SNPP", out_dir)
        results["firms_viirs"] = f"✅ {len(fire_df)} fire detections downloaded"
    except Exception as e:
        results["firms_viirs"] = f"❌ Failed: {e}"

    try:
        fire_df_modis = download_firms_fire_data("7d", "MODIS", out_dir)
        results["firms_modis"] = f"✅ {len(fire_df_modis)} fire detections downloaded"
    except Exception as e:
        results["firms_modis"] = f"❌ Failed: {e}"

    # 2-4: These require API keys / authentication
    logger.info("\n[2/4] Sentinel-5P TROPOMI — Requires GEE auth or DLR download")
    results["sentinel5p"] = "⚠️ Requires GEE authentication (earthengine authenticate)"

    logger.info("\n[3/4] CPCB/OpenAQ — Requires free API key")
    results["cpcb"] = "⚠️ Requires OpenAQ API key (register at explore.openaq.org/register)"

    logger.info("\n[4/4] ERA5 Reanalysis — Requires CDS API key")
    results["era5"] = "⚠️ Requires CDS API key (register at cds.climate.copernicus.eu)"

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for source, status in results.items():
        logger.info(f"  {source}: {status}")

    return results


if __name__ == "__main__":
    download_all_available()
