import pandas as pd

print("=" * 60)
print("PROOF: REAL NASA FIRE DATA DOWNLOADED")
print("=" * 60)

df = pd.read_csv("data/real/firms_fire_VIIRS_SNPP_7d.csv")
print(f"\nTotal VIIRS fire detections in India: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['acq_date'].min()} to {df['acq_date'].max()}")
print(f"Lat range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
print(f"Lon range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
print(f"\nFire Radiative Power (FRP) statistics:")
print(df["frp"].describe())

print(f"\n{'='*60}")
print("TOP 15 FIRE HOTSPOTS BY FIRE RADIATIVE POWER (FRP)")
print("=" * 60)
top = df.nlargest(15, "frp")[["latitude", "longitude", "acq_date", "frp", "confidence", "daynight"]]
print(top.to_string(index=False))

df2 = pd.read_csv("data/real/firms_fire_MODIS_7d.csv")
print(f"\n{'='*60}")
print(f"MODIS fire detections in India: {len(df2)}")
print(f"Date range: {df2['acq_date'].min()} to {df2['acq_date'].max()}")
print("=" * 60)
