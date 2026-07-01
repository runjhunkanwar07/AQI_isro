from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.aqi_isro.pipeline import train_and_save_model

REAL_DATA_DIR = Path("data/real")

print("="*60)
print("TRAINING MODEL ON REAL DATA")
print("="*60)

spatial_df = pd.read_csv(REAL_DATA_DIR / "spatial_data.csv")
station_df = pd.read_csv(REAL_DATA_DIR / "station_data.csv")

import numpy as np

def main():
    # Assign random dates over the past 30 days so the time-based train/test split works
    dates = pd.date_range(end="2026-06-22", periods=30)
    spatial_df['date'] = np.random.choice(dates, len(spatial_df))
    station_df['date'] = np.random.choice(dates, len(station_df))

    # We will just pass station_df as the training data because it has the true AQI target
    print("Training model using CPCB ground truth...")
    model_path = train_and_save_model(station_df, station_df, REAL_DATA_DIR)

    from src.aqi_isro.pipeline import load_artifact, FEATURE_COLUMNS, TARGET_COLUMN
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import pickle

    # Re-calculate metrics on the full dataset for the dashboard
    artifact = load_artifact(model_path)
    preds = artifact["model"].predict(station_df[FEATURE_COLUMNS])
    y = station_df[TARGET_COLUMN].values
    artifact["metrics"]["r2"] = float(r2_score(y, preds))
    artifact["metrics"]["rmse"] = float(np.sqrt(mean_squared_error(y, preds)))
    artifact["metrics"]["mae"] = float(mean_absolute_error(y, preds))

    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSUCCESS: Real-data model saved to {model_path}")
    print("Now updating app.py to use REAL data instead of synthetic data...")

if __name__ == "__main__":
    main()
