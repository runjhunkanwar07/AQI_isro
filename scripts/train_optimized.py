import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS = ["aod", "no2", "so2", "co", "o3", "hcho", "temp", "humidity", "wind_u", "wind_v", "fire_count", "pm_proxy_index", "fire_wind_interaction", "urban_density_proxy"]
TARGET_COLUMN = "aqi"

def main():
    print("Loading datasets...")
    spatial_df = pd.read_csv("data/real/spatial_data.csv")
    station_df = pd.read_csv("data/real/station_data.csv")

    dates = pd.date_range(end="2026-06-22", periods=30)
    spatial_df['date'] = np.random.choice(dates, len(spatial_df))
    station_df['date'] = np.random.choice(dates, len(station_df))

    print("Training the RandomForestRegressor with optimized parameters...")
    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=1, max_depth=None, random_state=42, n_jobs=1)
    
    # We use station_df for training because it has the AQI target
    model.fit(station_df[FEATURE_COLUMNS], station_df[TARGET_COLUMN])

    print("Calculating metrics...")
    preds = model.predict(station_df[FEATURE_COLUMNS])
    y = station_df[TARGET_COLUMN].values
    
    metrics = {
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds))
    }

    print(f"Metrics: {metrics}")

    artifact = {
        "model": model,
        "metrics": metrics,
        "model_type": "Random Forest",
        "features": FEATURE_COLUMNS
    }

    print("Saving artifact...")
    with open("data/real/aqi_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print("Done!")

if __name__ == "__main__":
    main()
