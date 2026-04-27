"""
Extract daily climate timeseries for each olive farm from AgERA5.

For each farm in olive_groves_catalonia.gpkg:
  1. Read its matched grid cell indices (grid_lat_idx, grid_lon_idx)
  2. Extract daily timeseries for all climate variables (2015-2024)
  3. Apply elevation-based temperature correction (-5.5°C/km lapse rate)
  4. Compute VPD from corrected tmax + ea
  5. Output: agera5_daily_olive_farms.csv (482 farms × 3653 days = ~1.8M rows)

Output columns:
  farm_id, date, comarca, area_ha,
  et0, precip,
  tmax_raw, tmax_cor, tmin_raw, tmin_cor,
  ea, vpd_raw, vpd_cor
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR   = Path(__file__).parent.parent / "data"
AGERA5_DIR = DATA_DIR / "agera5_catalonia"
FARMS_PATH = DATA_DIR / "olive_groves_catalonia.gpkg"

YEARS = list(range(2015, 2025))

# Maps filename stem key -> output column name
VARIABLE_MAP = {
    "reference_evapotranspiration": "et0",      # mm/day, no conversion
    "precipitation_flux":           "precip",   # mm/day, no conversion
    "vapour_pressure":              "ea",       # hPa → kPa
    "2m_temperature_max":           "tmax",     # K → °C
    "2m_temperature_min":           "tmin",     # K → °C
}


def _normalise_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for dim in list(ds.dims) + list(ds.coords):
        if dim.lower() == "latitude"  and "lat" not in ds.dims: rename[dim] = "lat"
        if dim.lower() == "longitude" and "lon" not in ds.dims: rename[dim] = "lon"
    return ds.rename(rename) if rename else ds


def _get_data_var(ds: xr.Dataset) -> str:
    coord_names = set(ds.coords)
    spatial = [
        v for v in ds.data_vars
        if v not in coord_names
        and any(d.lower() in ("lat", "latitude")  for d in ds[v].dims)
        and any(d.lower() in ("lon", "longitude") for d in ds[v].dims)
    ]
    if len(spatial) == 1:
        return spatial[0]
    if len(spatial) > 1:
        return max(spatial, key=lambda v: len(ds[v].dims))
    raise ValueError(f"Cannot identify spatial data variable. data_vars={list(ds.data_vars)}")


def _compute_vpd(tmax_c: np.ndarray, ea_kpa: np.ndarray) -> np.ndarray:
    """FAO-56 Tetens formula: VPD = es(Tmax) - ea"""
    es = 0.6108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))
    return np.maximum(es - ea_kpa, 0.0)


def extract_year(year: int, farms_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Extract daily climate for all farms for one year"""

    # Load all variables for this year
    data = {}
    for nc in sorted(AGERA5_DIR.glob(f"*_{year}.nc")):
        m = re.match(r"^(.+)_\d{4}$", nc.stem)
        if not m:
            continue
        key = m.group(1)
        if key not in VARIABLE_MAP:
            continue
        ds = _normalise_dims(xr.open_dataset(nc, engine="netcdf4"))
        data[key] = ds

    if not data:
        print(f"  No files for {year} — skipping")
        return pd.DataFrame()

    # Get time dimension from first dataset
    first_ds = next(iter(data.values()))
    times = pd.to_datetime(first_ds["time"].values)
    n_days = len(times)

    records = []

    for idx, row in farms_gdf.iterrows():
        farm_id = row['OBJECTID']  # or use idx if no ID column
        comarca = row['comarca']
        area_ha = row['Area_Ha']
        li = int(row['grid_lat_idx'])
        lj = int(row['grid_lon_idx'])
        temp_correction = row['temp_correction']  # Already computed: -5.5°C/km × elev_delta

        farm_data = {
            'farm_id': np.full(n_days, farm_id, dtype=int),
            'date': times,
            'comarca': comarca,
            'area_ha': area_ha,
        }

        # Extract each variable at this farm's grid cell
        for key, col_name in VARIABLE_MAP.items():
            if key not in data:
                farm_data[col_name] = np.full(n_days, np.nan)
                continue

            ds = data[key]
            var_name = _get_data_var(ds)
            vals = ds[var_name].values[:, li, lj].astype(float)  # (time,) array

            # Unit conversions
            if key == "vapour_pressure":
                vals = vals / 10.0  # hPa → kPa
            elif "temperature" in key:
                vals = vals - 273.15  # K → °C

            farm_data[col_name] = vals

        # Apply elevation correction to temperatures
        if 'tmax' in farm_data:
            farm_data['tmax_raw'] = farm_data['tmax'].copy()
            farm_data['tmax_cor'] = farm_data['tmax'] + temp_correction
        if 'tmin' in farm_data:
            farm_data['tmin_raw'] = farm_data['tmin'].copy()
            farm_data['tmin_cor'] = farm_data['tmin'] + temp_correction

        # Compute VPD (raw and corrected)
        if 'tmax' in farm_data and 'ea' in farm_data:
            farm_data['vpd_raw'] = _compute_vpd(farm_data['tmax_raw'], farm_data['ea'])
            farm_data['vpd_cor'] = _compute_vpd(farm_data['tmax_cor'], farm_data['ea'])
        else:
            farm_data['vpd_raw'] = np.full(n_days, np.nan)
            farm_data['vpd_cor'] = np.full(n_days, np.nan)

        # Drop uncorrected temp columns (keep only raw/cor)
        if 'tmax' in farm_data:
            del farm_data['tmax']
        if 'tmin' in farm_data:
            del farm_data['tmin']

        df_farm = pd.DataFrame(farm_data)
        records.append(df_farm)

    # Close datasets
    for ds in data.values():
        ds.close()

    return pd.concat(records, ignore_index=True)


def main():
    if not FARMS_PATH.exists():
        raise FileNotFoundError(f"Missing {FARMS_PATH}")

    print(f"Loading {FARMS_PATH}...")
    farms_gdf = gpd.read_file(FARMS_PATH)
    print(f"  {len(farms_gdf)} farms loaded")

    # Check required columns
    required = ['OBJECTID', 'comarca', 'Area_Ha', 'grid_lat_idx', 'grid_lon_idx', 'temp_correction']
    missing = [c for c in required if c not in farms_gdf.columns]
    if missing:
        raise ValueError(f"Missing columns in GeoPackage: {missing}")

    all_data = []
    for year in YEARS:
        print(f"Extracting {year}...")
        df_year = extract_year(year, farms_gdf)
        if not df_year.empty:
            all_data.append(df_year)

    if not all_data:
        raise RuntimeError("No data extracted")

    df = pd.concat(all_data, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['farm_id', 'date']).reset_index(drop=True)

    out_path = DATA_DIR / "agera5_daily_olive_farms.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} rows to {out_path}")
    print(f"Farms: {df['farm_id'].nunique()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"\nSample data:\n{df.head(10)}")


if __name__ == "__main__":
    main()
