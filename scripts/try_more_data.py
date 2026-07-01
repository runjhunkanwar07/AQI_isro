"""
Attempt to download Sentinel-5P HCHO and NO2 data directly from DLR.
Also try to fetch CPCB data from OpenAQ without API key.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import re
import pandas as pd
from io import StringIO

REAL_DATA_DIR = Path("data/real")
REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Try DLR Sentinel-5P download (HCHO + NO2)
# ============================================================
print("=" * 60)
print("[1] Attempting Sentinel-5P HCHO download from DLR...")
print("=" * 60)

# Try a recent date
for days_ago in range(1, 10):
    from datetime import datetime, timedelta
    dt = datetime.now() - timedelta(days=days_ago)
    year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    base_url = f"https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/{year}/{month}/{day}/"
    
    print(f"  Checking {base_url} ...")
    try:
        resp = requests.get(base_url, timeout=15)
        if resp.status_code == 200:
            # Look for HCHO folders
            folders = re.findall(r'href="([^"]*)"', resp.text)
            hcho_folders = [f for f in folders if 'HCHO' in f.upper()]
            no2_folders = [f for f in folders if 'NO2' in f.upper()]
            
            print(f"  Found {len(hcho_folders)} HCHO products, {len(no2_folders)} NO2 products")
            if hcho_folders:
                print(f"  HCHO folders: {hcho_folders[:3]}")
            if no2_folders:
                print(f"  NO2 folders: {no2_folders[:3]}")
            
            # Try to download HCHO
            if hcho_folders:
                folder_url = base_url + hcho_folders[0]
                resp2 = requests.get(folder_url, timeout=15)
                nc_files = re.findall(r'href="([^"]*\.nc)"', resp2.text)
                if nc_files:
                    file_url = folder_url + nc_files[0]
                    print(f"\n  Downloading: {nc_files[0]} ...")
                    resp3 = requests.get(file_url, stream=True, timeout=120)
                    if resp3.status_code == 200:
                        out_path = REAL_DATA_DIR / f"S5P_HCHO_{dt.strftime('%Y-%m-%d')}.nc"
                        total = 0
                        with open(out_path, "wb") as f:
                            for chunk in resp3.iter_content(chunk_size=8192):
                                f.write(chunk)
                                total += len(chunk)
                        print(f"  ✅ Downloaded! Size: {total / 1024 / 1024:.1f} MB -> {out_path}")
                        break
                    else:
                        print(f"  ❌ HTTP {resp3.status_code} - may need authentication")
                        # Try without auth headers
            break
        else:
            print(f"  HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================
# 2. Try OpenAQ without API key
# ============================================================
print("\n" + "=" * 60)
print("[2] Attempting CPCB data via OpenAQ (no API key)...")
print("=" * 60)

try:
    # Try the v3 API without a key
    url = "https://api.openaq.org/v3/locations?country_id=105&limit=10"
    resp = requests.get(url, timeout=15)
    print(f"  OpenAQ v3 response: HTTP {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("results", [])
        print(f"  Found {len(results)} stations (showing first 10)")
        for loc in results[:5]:
            coords = loc.get("coordinates", {})
            print(f"    - {loc.get('name', 'N/A')} ({coords.get('latitude', '?')}, {coords.get('longitude', '?')})")
    elif resp.status_code in (401, 403):
        print("  API key required. Trying v2 API...")
        # Try v2 which may be more open
        url2 = "https://api.openaq.org/v2/locations?country=IN&limit=10"
        resp2 = requests.get(url2, timeout=15)
        print(f"  OpenAQ v2 response: HTTP {resp2.status_code}")
        if resp2.status_code == 200:
            data2 = resp2.json()
            results2 = data2.get("results", [])
            print(f"  Found {len(results2)} stations")
            for loc in results2[:5]:
                coords = loc.get("coordinates", {})
                print(f"    - {loc.get('name', 'N/A')} ({coords.get('latitude', '?')}, {coords.get('longitude', '?')})")
            
            # Download more data
            url_full = "https://api.openaq.org/v2/locations?country=IN&limit=1000"
            resp_full = requests.get(url_full, timeout=30)
            if resp_full.status_code == 200:
                all_data = resp_full.json().get("results", [])
                rows = []
                for loc in all_data:
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
                            "count": param.get("count"),
                        })
                df = pd.DataFrame(rows)
                out_path = REAL_DATA_DIR / "cpcb_openaq_stations.csv"
                df.to_csv(out_path, index=False)
                print(f"\n  ✅ Saved {len(df)} parameter records from {df['station_id'].nunique()} stations to {out_path}")
                
                # Show cities with most stations
                if "city" in df.columns:
                    city_counts = df.groupby("city")["station_id"].nunique().sort_values(ascending=False).head(10)
                    print(f"\n  Top cities by station count:")
                    for city, count in city_counts.items():
                        print(f"    - {city}: {count} stations")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
