"""
Download Real Data for ISRO AQI Project.
Run this script to fetch actual satellite and fire data.

Usage:
    python scripts/download_real_data.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi_isro.ingest import download_all_available, REAL_DATA_DIR

if __name__ == "__main__":
    print("=" * 60)
    print("ISRO AQI PROJECT — DOWNLOADING REAL DATA")
    print("=" * 60)
    download_all_available(REAL_DATA_DIR)
