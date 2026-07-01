import requests
import pandas as pd
from pathlib import Path

REAL_DATA_DIR = Path("data/real")
REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
API_KEY = "fdc589ee3b1f4bc2fc427575c7c1b3e53cc2da56092d080757f1550b096cd803"
headers = {"X-API-Key": API_KEY}

print("Fetching locations from OpenAQ v3...")
# Get India locations (country_id 105 is India)
resp = requests.get("https://api.openaq.org/v3/locations?country_id=105&limit=100", headers=headers)
locations = resp.json().get("results", [])

rows = []
for i, loc in enumerate(locations):
    loc_id = loc["id"]
    loc_name = loc["name"]
    lat = loc["coordinates"]["latitude"]
    lon = loc["coordinates"]["longitude"]
    
    # Map sensor ID to parameter name
    sensor_map = {}
    for sensor in loc.get("sensors", []):
        sensor_map[sensor["id"]] = sensor["parameter"]["name"]
        
    # Get latest measurements for this location
    print(f"[{i+1}/{len(locations)}] Fetching latest for {loc_name}...")
    latest_resp = requests.get(f"https://api.openaq.org/v3/locations/{loc_id}/latest", headers=headers)
    if latest_resp.status_code == 200:
        measurements = latest_resp.json().get("results", [])
        for m in measurements:
            sensor_id = m.get("sensorsId")
            if sensor_id in sensor_map:
                rows.append({
                    "station_id": loc_id,
                    "station_name": loc_name,
                    "lat": lat,
                    "lon": lon,
                    "parameter": sensor_map[sensor_id],
                    "last_value": m.get("value"),
                    "last_updated": m.get("datetime", {}).get("utc")
                })

df = pd.DataFrame(rows)
out_path = REAL_DATA_DIR / "cpcb_openaq_stations.csv"
df.to_csv(out_path, index=False)
print(f"\nSUCCESS: Saved {len(df)} measurements to {out_path}")
