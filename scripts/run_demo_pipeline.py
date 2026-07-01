from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi_isro.demo import DEFAULT_DATA_DIR, generate_demo_data
from src.aqi_isro.pipeline import load_artifact, load_demo_data, make_aqi_predictions, prepare_hcho_hotspots, train_and_save_model


if __name__ == "__main__":
    generate_demo_data(DEFAULT_DATA_DIR)
    spatial_df, station_df = load_demo_data(DEFAULT_DATA_DIR)
    artifact_path = train_and_save_model(spatial_df, station_df, DEFAULT_DATA_DIR)
    artifact = load_artifact(artifact_path)
    latest_day = spatial_df["date"].max()
    daily_df = spatial_df.loc[spatial_df["date"] == latest_day].copy()
    pred_df = make_aqi_predictions(daily_df, artifact)
    hotspot_df, summary = prepare_hcho_hotspots(daily_df)

    print(f"Demo data rows: {len(spatial_df):,}")
    print(f"Station rows: {len(station_df):,}")
    print(f"Model artifact: {artifact_path}")
    print(f"AQI prediction range: {pred_df['aqi_pred'].min():.1f} to {pred_df['aqi_pred'].max():.1f}")
    print(f"HCHO hotspot pixels: {summary['hotspot_pixels']}")
    print(f"Top likely source region: {summary['top_source_region']}")
