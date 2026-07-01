# -*- coding: utf-8 -*-
"""Download CPCB (via OpenAQ) and ERA5 data using user-provided API keys."""
from pathlib import Path
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi_isro.ingest import download_cpcb_openaq, download_era5, REAL_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

print("=" * 60)
print("DOWNLOADING CPCB DATA VIA OPENAQ")
print("=" * 60)
api_key = "fdc589ee3b1f4bc2fc427575c7c1b3e53cc2da56092d080757f1550b096cd803"
try:
    cpcb_df = download_cpcb_openaq(api_key=api_key, limit=1000, out_dir=REAL_DATA_DIR)
    if not cpcb_df.empty:
        print(f"\nSUCCESS: Downloaded {len(cpcb_df)} ground station records!")
        print("Sample of parameters collected:")
        print(cpcb_df['parameter'].value_counts())
except Exception as e:
    print(f"Failed CPCB: {e}")

print("\n" + "=" * 60)
print("DOWNLOADING ERA5 REANALYSIS DATA (RECENT)")
print("=" * 60)
# Download just the current month to be quick
import datetime
now = datetime.datetime.now()
current_year = str(now.year)
current_month = f"{now.month:02d}"

try:
    era5_path = download_era5(year=current_year, months=[current_month], out_dir=REAL_DATA_DIR)
    print(f"\nSUCCESS: Downloaded ERA5 meteorological data to {era5_path}")
except Exception as e:
    print(f"Failed ERA5: {e}")

print("\nDONE")
