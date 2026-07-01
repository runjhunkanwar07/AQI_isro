from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi_isro.demo import DEFAULT_DATA_DIR, generate_demo_data


if __name__ == "__main__":
    generate_demo_data(DEFAULT_DATA_DIR)
    print(f"Demo data written to {DEFAULT_DATA_DIR}")
