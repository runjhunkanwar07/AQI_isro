from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aqi_isro.demo import DEFAULT_DATA_DIR
from src.aqi_isro.pipeline import load_artifact, load_demo_data, make_aqi_predictions, prepare_hcho_hotspots


if __name__ == "__main__":
    output_dir = Path("outputs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_df, _ = load_demo_data(DEFAULT_DATA_DIR)
    artifact = load_artifact(DEFAULT_DATA_DIR / "aqi_model.pkl")
    latest_day = spatial_df["date"].max()
    daily_df = spatial_df.loc[spatial_df["date"] == latest_day].copy()

    aqi_df = make_aqi_predictions(daily_df, artifact)
    hotspot_df, summary = prepare_hcho_hotspots(daily_df)

    aqi_out = output_dir / "latest_aqi_predictions.csv"
    hcho_out = output_dir / "latest_hcho_hotspots.csv"
    summary_out = output_dir / "latest_hcho_summary.csv"

    aqi_df.to_csv(aqi_out, index=False)
    hotspot_df.to_csv(hcho_out, index=False)
    summary_out.write_text(
        "\n".join(f"{key},{value}" for key, value in summary.items()),
        encoding="utf-8",
    )

    print(f"AQI predictions exported to {aqi_out}")
    print(f"HCHO hotspots exported to {hcho_out}")
    print(f"HCHO summary exported to {summary_out}")
