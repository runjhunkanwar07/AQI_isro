from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi_isro.demo import DEFAULT_DATA_DIR
from src.aqi_isro.pipeline import load_demo_data, train_and_save_model


if __name__ == "__main__":
    spatial_df, station_df = load_demo_data(DEFAULT_DATA_DIR)
    artifact_path = train_and_save_model(spatial_df, station_df, DEFAULT_DATA_DIR)
    print(f"Model artifact saved to {artifact_path}")
