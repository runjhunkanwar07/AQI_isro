from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import pickle

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:  # Keep the core demo runnable even before dependencies are installed.
    RandomForestRegressor = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None


FEATURE_COLUMNS = ["aod", "no2", "so2", "co", "o3", "hcho", "temp", "humidity", "wind_u", "wind_v", "fire_count", "pm_proxy_index", "fire_wind_interaction", "urban_density_proxy"]
TARGET_COLUMN = "aqi"

SOURCE_REGIONS = {
    "Indo-Gangetic Plain": (27.5, 79.0),
    "Punjab-Haryana crop residue belt": (30.0, 76.0),
    "Central India fire belt": (22.5, 82.0),
    "Eastern India forest-fire zone": (23.5, 86.0),
    "Western India urban-industrial belt": (22.8, 72.6),
}


def load_demo_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    spatial_df = pd.read_csv(data_dir / "demo_spatial_data.csv", parse_dates=["date"])
    station_df = pd.read_csv(data_dir / "station_data.csv", parse_dates=["date"])
    return spatial_df, station_df


def _train_test_split_time(df: pd.DataFrame, date_col: str = "date", frac: float = 0.8):
    dates = sorted(df[date_col].unique())
    cutoff = dates[int(len(dates) * frac) - 1]
    train = df.loc[df[date_col] <= cutoff].copy()
    test = df.loc[df[date_col] > cutoff].copy()
    return train, test


def train_model(spatial_df: pd.DataFrame) -> dict:
    train_df, test_df = _train_test_split_time(spatial_df)
    if RandomForestRegressor is not None:
        return _train_random_forest(train_df, test_df)
    return _train_ridge(train_df, test_df)


def _train_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_depth=None,
        random_state=42,
        n_jobs=1
    )
    
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    preds = model.predict(test_df[FEATURE_COLUMNS])
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    feature_importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    return {
        "model_type": "Random Forest",
        "model": model,
        "metrics": metrics,
        "feature_importance": feature_importance.to_dict(),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
    }


def _train_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    scaler_mean = x_train.mean(axis=0)
    scaler_std = x_train.std(axis=0)
    scaler_std[scaler_std < 1e-6] = 1.0

    x_train_scaled = (x_train - scaler_mean) / scaler_std
    x_test_scaled = (x_test - scaler_mean) / scaler_std

    alpha = 2.5
    y_mean = y_train.mean()
    xtx = x_train_scaled.T @ x_train_scaled
    ridge = xtx + alpha * np.eye(xtx.shape[0])
    beta = np.linalg.solve(ridge, x_train_scaled.T @ (y_train - y_mean))
    intercept = float(y_mean)
    preds = x_test_scaled @ beta + intercept
    metrics = {
        "rmse": float(np.sqrt(np.mean((y_test - preds) ** 2))),
        "mae": float(np.mean(np.abs(y_test - preds))),
        "r2": float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2)),
    }
    feature_importance = pd.Series(np.abs(beta), index=FEATURE_COLUMNS).sort_values(ascending=False)
    artifact = {
        "model_type": "Ridge",
        "coef": beta,
        "intercept": intercept,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "metrics": metrics,
        "feature_importance": feature_importance.to_dict(),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
    }
    return artifact


def train_and_save_model(spatial_df: pd.DataFrame, station_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = train_model(spatial_df)
    path = out_dir / "aqi_model.pkl"
    with path.open("wb") as f:
        pickle.dump(artifact, f)
    return path


def load_artifact(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def make_aqi_predictions(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    pred_df = df.copy()
    pred_df["aqi_pred"] = _predict_from_artifact(pred_df[FEATURE_COLUMNS], artifact)
    pred_df["aqi_pred"] = pred_df["aqi_pred"].clip(0.0, 500.0)
    pred_df["uncertainty"] = _estimate_uncertainty(pred_df[FEATURE_COLUMNS], artifact)
    pred_df["aqi_class"] = pred_df["aqi_pred"].apply(_aqi_class)
    return pred_df


def _predict_from_artifact(features: pd.DataFrame, artifact: dict) -> np.ndarray:
    if artifact.get("model_type") == "Random Forest":
        return artifact["model"].predict(features)
    x = features.to_numpy(dtype=float)
    x_scaled = (x - artifact["scaler_mean"]) / artifact["scaler_std"]
    return x_scaled @ artifact["coef"] + artifact["intercept"]


def _estimate_uncertainty(features: pd.DataFrame, artifact: dict) -> pd.Series:
    if artifact.get("model_type") == "Random Forest":
        x = features.to_numpy(dtype=float)
        tree_preds = np.vstack([tree.predict(x) for tree in artifact["model"].estimators_])
        return pd.Series(tree_preds.std(axis=0), index=features.index)
    x = features.to_numpy(dtype=float)
    x_scaled = (x - artifact["scaler_mean"]) / artifact["scaler_std"]
    leverage = np.sqrt(np.sum(x_scaled**2, axis=1))
    base = np.std(x_scaled @ artifact["coef"]) * 0.12
    uncertainty = base * (1.0 + leverage / (np.percentile(leverage, 90) + 1e-6))
    return pd.Series(uncertainty, index=features.index)


def _aqi_class(value: float) -> str:
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def prepare_demo_metrics(station_df: pd.DataFrame, artifact: dict) -> Dict[str, float]:
    preds = _predict_from_artifact(station_df[FEATURE_COLUMNS], artifact)
    y = station_df[TARGET_COLUMN].to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((y - preds) ** 2))),
        "mae": float(np.mean(np.abs(y - preds))),
        "r2": float(1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def prepare_hcho_hotspots(daily_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    df = daily_df.copy()
    baseline = df["hcho"].median()
    spread = float(np.mean(np.abs(df["hcho"] - baseline)))
    spread = spread if spread and spread > 1e-6 else 0.1
    df["hcho_anomaly"] = df["hcho"] - baseline
    df["anomaly_z"] = df["hcho_anomaly"] / spread
    df["wind_speed"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)
    df["hotspot_score"] = (df["anomaly_z"].clip(lower=0) * 0.6) + (df["fire_count"] / (df["fire_count"].max() + 1e-6)) * 0.25 + (df["wind_speed"] / (df["wind_speed"].max() + 1e-6)) * 0.15
    df["is_hotspot"] = df["hotspot_score"] >= df["hotspot_score"].quantile(0.92)
    df["likely_source_region"] = df.apply(_nearest_source_region, axis=1)
    top_region = (
        df.loc[df["is_hotspot"]]
        .groupby("likely_source_region")["hotspot_score"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    summary = {
        "hotspot_pixels": int(df["is_hotspot"].sum()),
        "mean_anomaly": float(df["hcho_anomaly"].mean()),
        "fire_link_score": float(np.corrcoef(df["hcho"], df["fire_count"])[0, 1]) if df["fire_count"].nunique() > 1 else 0.0,
        "top_source_region": top_region[0] if len(top_region) else "No hotspot",
    }
    return df, summary


def _nearest_source_region(row: pd.Series) -> str:
    lat = float(row["lat"])
    lon = float(row["lon"])
    distances = {
        name: np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
        for name, (center_lat, center_lon) in SOURCE_REGIONS.items()
    }
    return min(distances, key=distances.get)
