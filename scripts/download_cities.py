# -*- coding: utf-8 -*-
"""
scripts/download_cities.py
──────────────────────────
Downloads the GeoNames India dataset and produces data/india_cities.csv
with every Indian city/town/village that has a recorded population.

GeoNames is the authoritative open-source geographic database used by
Google Maps, OpenStreetMap, and most government GIS platforms.

Usage:
    python scripts/download_cities.py

Output:
    data/india_cities.csv   (name, state, lat, lon, population)
"""

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
OUT_CSV   = DATA_DIR / "india_cities.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── GeoNames India dump ────────────────────────────────────────────────────────
GEONAMES_URL = "https://download.geonames.org/export/dump/IN.zip"
ADMIN1_URL   = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"

# GeoNames feature classes we want (populated places)
WANTED_CLASSES  = {"P"}                          # P = populated place
WANTED_CODES    = {                              # only keep city/town/village types
    "PPL",   # populated place
    "PPLA",  # admin capital (state capital)
    "PPLA2", # second-order admin capital (district HQ)
    "PPLA3", # third-order admin capital
    "PPLA4", # fourth-order admin capital
    "PPLC",  # capital of country
    "PPLG",  # seat of government
    "PPLS",  # populated places
    "PPLX",  # populated suburb / section of populated place
    "PPLF",  # farm village
    "PPLQ",  # abandoned populated place
    "PPLR",  # religious populated place
    "PPLT",  # populated territory (historical)
    "STLMT", # israeli settlement
    "PPLW",  # destroyed populated place
    "PPLH",  # historical populated place
}

COLS = [
    "geonameid", "name", "asciiname", "alternatenames",
    "lat", "lon", "feature_class", "feature_code",
    "country_code", "cc2", "admin1_code", "admin2_code",
    "admin3_code", "admin4_code", "population",
    "elevation", "dem", "timezone", "modification_date"
]

def download(url: str, desc: str) -> bytes:
    print(f"  Downloading {desc} ...", end=" ", flush=True)
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    data = b""
    for chunk in r.iter_content(65536):
        data += chunk
    print(f"done ({len(data)/1024/1024:.1f} MB)")
    return data

def main():
    print("\n=== GeoNames India City Database Builder ===\n")

    # 1. Download admin1 (state) codes ─────────────────────────────────────────
    admin1_raw = download(ADMIN1_URL, "Admin1 state codes").decode("utf-8")
    admin1 = {}
    for line in admin1_raw.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].startswith("IN."):
            code  = parts[0].split(".")[1]
            admin1[code] = parts[1].strip()
    print(f"  → {len(admin1)} Indian states/UTs mapped")

    # 2. Download & extract IN.zip ─────────────────────────────────────────────
    zip_bytes = download(GEONAMES_URL, "GeoNames India (IN.zip)")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open("IN.txt") as f:
            raw = f.read().decode("utf-8")
    print("  Parsing GeoNames entries ...")

    # 3. Parse ─────────────────────────────────────────────────────────────────
    rows = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != len(COLS):
            continue
        rec = dict(zip(COLS, parts))
        if rec["feature_class"] not in WANTED_CLASSES:
            continue
        try:
            lat = float(rec["lat"])
            lon = float(rec["lon"])
            pop = int(rec["population"]) if rec["population"] else 0
        except ValueError:
            continue
        # Keep India-only bounding box
        if not (6.5 <= lat <= 37.5 and 68.0 <= lon <= 97.5):
            continue
        state = admin1.get(rec["admin1_code"], "Unknown")
        rows.append({
            "name":       rec["asciiname"] or rec["name"],
            "state":      state,
            "lat":        round(lat, 5),
            "lon":        round(lon, 5),
            "population": pop,
        })

    df = pd.DataFrame(rows)
    print(f"  → {len(df):,} total populated places found")

    # 4. Deduplicate & sort ────────────────────────────────────────────────────
    df = (df
          .sort_values("population", ascending=False)
          .drop_duplicates(subset=["lat", "lon"], keep="first")
          .drop_duplicates(subset=["name", "state"], keep="first")
          .reset_index(drop=True))

    print(f"  → {len(df):,} unique cities/towns after deduplication")

    # 5. Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n  Saved → {OUT_CSV}")
    print(f"  Cities with population > 0   : {(df.population > 0).sum():,}")
    print(f"  Cities with population > 100k: {(df.population > 100_000).sum():,}")
    print(f"  Cities with population > 10k : {(df.population > 10_000).sum():,}")
    print(f"  Total rows                   : {len(df):,}\n")
    print("Done. Re-run app.py to load the new database.")

if __name__ == "__main__":
    main()
