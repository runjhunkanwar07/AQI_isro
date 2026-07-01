# -*- coding: utf-8 -*-
"""
Download Sentinel-5P HCHO + NO2 data from DLR (fixed encoding for Windows).
"""
import sys
import os
import re
import requests
from pathlib import Path
from datetime import datetime, timedelta

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

REAL_DATA_DIR = Path("data/real")
REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DOWNLOADING SENTINEL-5P HCHO FROM DLR")
print("=" * 60)

downloaded_files = []

for days_ago in range(1, 10):
    dt = datetime.now() - timedelta(days=days_ago)
    year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    date_str = dt.strftime("%Y-%m-%d")
    base_url = f"https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/{year}/{month}/{day}/"

    print(f"\nChecking {date_str} ...")
    try:
        resp = requests.get(base_url, timeout=15)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} - skipping")
            continue

        folders = re.findall(r'href="([^"]*)"', resp.text)
        hcho_folders = [f for f in folders if "HCHO" in f.upper()]

        if not hcho_folders:
            print("  No HCHO products found")
            continue

        print(f"  Found HCHO product: {hcho_folders[0]}")
        folder_url = base_url + hcho_folders[0]
        resp2 = requests.get(folder_url, timeout=15)
        nc_files = re.findall(r'href="([^"]*\.nc)"', resp2.text)

        if not nc_files:
            print("  No NetCDF files in folder")
            continue

        file_url = folder_url + nc_files[0]
        out_path = REAL_DATA_DIR / f"S5P_HCHO_{date_str}.nc"

        print(f"  Downloading {nc_files[0]} ...")
        resp3 = requests.get(file_url, stream=True, timeout=300)

        if resp3.status_code == 200:
            total = 0
            with open(out_path, "wb") as f:
                for chunk in resp3.iter_content(chunk_size=65536):
                    f.write(chunk)
                    total += len(chunk)
            size_mb = total / 1024 / 1024
            print(f"  SUCCESS! Downloaded {size_mb:.1f} MB -> {out_path}")
            downloaded_files.append(str(out_path))
            break  # Got one file, that's enough for now
        else:
            print(f"  HTTP {resp3.status_code} - download failed")

    except Exception as e:
        print(f"  Error: {str(e)[:100]}")

# Also try NO2
print("\n" + "=" * 60)
print("DOWNLOADING SENTINEL-5P NO2 FROM DLR")
print("=" * 60)

for days_ago in range(1, 10):
    dt = datetime.now() - timedelta(days=days_ago)
    year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    date_str = dt.strftime("%Y-%m-%d")
    base_url = f"https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/{year}/{month}/{day}/"

    try:
        resp = requests.get(base_url, timeout=15)
        if resp.status_code != 200:
            continue
        folders = re.findall(r'href="([^"]*)"', resp.text)
        no2_folders = [f for f in folders if "NO2" in f.upper()]

        if not no2_folders:
            continue

        print(f"\n  Found NO2 product for {date_str}: {no2_folders[0]}")
        folder_url = base_url + no2_folders[0]
        resp2 = requests.get(folder_url, timeout=15)
        nc_files = re.findall(r'href="([^"]*\.nc)"', resp2.text)

        if not nc_files:
            continue

        file_url = folder_url + nc_files[0]
        out_path = REAL_DATA_DIR / f"S5P_NO2_{date_str}.nc"

        print(f"  Downloading {nc_files[0]} ...")
        resp3 = requests.get(file_url, stream=True, timeout=300)

        if resp3.status_code == 200:
            total = 0
            with open(out_path, "wb") as f:
                for chunk in resp3.iter_content(chunk_size=65536):
                    f.write(chunk)
                    total += len(chunk)
            size_mb = total / 1024 / 1024
            print(f"  SUCCESS! Downloaded {size_mb:.1f} MB -> {out_path}")
            downloaded_files.append(str(out_path))
            break
        else:
            print(f"  HTTP {resp3.status_code}")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
if downloaded_files:
    for f in downloaded_files:
        print(f"  [OK] {f}")
else:
    print("  No files downloaded")
print("=" * 60)
