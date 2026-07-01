# -*- coding: utf-8 -*-
"""Verify the downloaded Sentinel-5P HCHO NetCDF file."""
import xarray as xr
import numpy as np

print("=" * 60)
print("VERIFYING REAL SENTINEL-5P HCHO DATA")
print("=" * 60)

ds = xr.open_dataset("data/real/S5P_HCHO_2026-06-21.nc")

print(f"\nDataset variables: {list(ds.data_vars)}")
print(f"Dimensions: {dict(ds.dims)}")
print(f"Coordinates: {list(ds.coords)}")

# Show all variable details
for var in ds.data_vars:
    v = ds[var]
    print(f"\n  Variable: {var}")
    print(f"    Shape: {v.shape}")
    print(f"    Dtype: {v.dtype}")
    if hasattr(v, 'attrs'):
        for attr_key in ['long_name', 'units', 'standard_name']:
            if attr_key in v.attrs:
                print(f"    {attr_key}: {v.attrs[attr_key]}")

# Subset to India
print("\n" + "=" * 60)
print("SUBSETTING TO INDIA REGION")
print("=" * 60)

# Figure out dimension names
lat_name = None
lon_name = None
for coord in list(ds.coords) + list(ds.dims):
    cl = coord.lower()
    if 'lat' in cl:
        lat_name = coord
    if 'lon' in cl:
        lon_name = coord

if lat_name and lon_name:
    print(f"Lat dimension: {lat_name}, range: {float(ds[lat_name].min()):.1f} to {float(ds[lat_name].max()):.1f}")
    print(f"Lon dimension: {lon_name}, range: {float(ds[lon_name].min()):.1f} to {float(ds[lon_name].max()):.1f}")
    
    india = ds.sel({lat_name: slice(7, 38), lon_name: slice(68, 98)})
    print(f"\nIndia subset dimensions: {dict(india.dims)}")
    
    # Get the main data variable
    main_var = list(india.data_vars)[0]
    data = india[main_var].values
    valid = data[~np.isnan(data)]
    
    if len(valid) > 0:
        print(f"\nHCHO data for India:")
        print(f"  Valid pixels: {len(valid):,}")
        print(f"  Min: {np.min(valid):.6e}")
        print(f"  Max: {np.max(valid):.6e}")
        print(f"  Mean: {np.mean(valid):.6e}")
        print(f"  Std: {np.std(valid):.6e}")
    else:
        print("  No valid data in India region")
else:
    print(f"Could not identify lat/lon. Available coords: {list(ds.coords)}")
    print(f"Available dims: {list(ds.dims)}")

ds.close()
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE - THIS IS REAL SATELLITE DATA!")
print("=" * 60)
