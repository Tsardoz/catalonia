"""
extract_climate.py  —  Step 3b
Extract and spatially aggregate AgERA5 climate variables per comarca.

For each comarca, all AgERA5 grid cells (~0.1°) whose centres fall within the
comarca polygon are identified via a spatial join. Daily min, mean, and max are
then computed across those cells, giving a spatially representative summary
rather than a single centroid value.

Fallback: if a comarca polygon contains no grid cell centres (very small
comarques), the nearest cell to the centroid is used instead.

Unit conversions:
  - Temperature: K → °C  (subtract 273.15)
  - Vapour pressure: hPa → kPa  (divide by 10)
  - ET0: mm/day  (no conversion)
  - Precipitation: mm/day  (no conversion)

VPD is computed from ea (actual vapour pressure) and tmax using FAO-56 Tetens:
  es(Tmax) = 0.6108 * exp(17.27 * Tmax_C / (Tmax_C + 237.3))  [kPa]
  vpd = max(es - ea, 0)

Output CSV columns (one row per comarca-day):
  date, comarca,
  et0_mean, et0_min, et0_max,
  ea_mean,  ea_min,  ea_max,
  tmax_mean, tmax_min, tmax_max,
  tmin_mean, tmin_min, tmin_max,
  vpd_mean,  vpd_min,  vpd_max,
  precip_mean, precip_min, precip_max  (if precipitation_flux downloaded)
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

DATA_DIR   = Path(__file__).parent.parent / "data"
AGERA5_DIR = DATA_DIR / "agera5_catalonia"
POLY_PATH  = DATA_DIR / "comarca_polygons.gpkg"

YEARS = list(range(2015, 2025))

# Maps filename stem key -> (output base name, unit conversion or None)
VARIABLE_MAP = {
    "reference_evapotranspiration": ("et0",    None),
    "vapour_pressure":              ("ea",     lambda x: x / 10.0),
    "2m_temperature_max":           ("tmax",   lambda x: x - 273.15),
    "2m_temperature_min":           ("tmin",   lambda x: x - 273.15),
    "precipitation_flux":           ("precip", None),
}


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


def _normalise_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for dim in list(ds.dims) + list(ds.coords):
        if dim.lower() == "latitude"  and "lat" not in ds.dims: rename[dim] = "lat"
        if dim.lower() == "longitude" and "lon" not in ds.dims: rename[dim] = "lon"
    return ds.rename(rename) if rename else ds


def _compute_vpd(tmax_c: np.ndarray, ea_kpa: np.ndarray) -> np.ndarray:
    es = 0.6108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))
    return np.maximum(es - ea_kpa, 0.0)


def build_comarca_mask(
    gdf: gpd.GeoDataFrame, lats: np.ndarray, lons: np.ndarray
) -> dict:
    """
    Spatial join: for each comarca, return indices of AgERA5 grid cells
    whose centres fall within its polygon.
    Falls back to nearest cell for very small comarques.
    """
    lat_idx, lon_idx = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing="ij")
    lat_val, lon_val = np.meshgrid(lats, lons, indexing="ij")

    grid_gdf = gpd.GeoDataFrame(
        {
            "lat_idx": lat_idx.ravel().astype(int),
            "lon_idx": lon_idx.ravel().astype(int),
            "geometry": [Point(lo, la) for la, lo in zip(lat_val.ravel(), lon_val.ravel())],
        },
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(grid_gdf, gdf[["comarca", "geometry"]], how="left", predicate="within")

    mask = {}
    for comarca, grp in joined.dropna(subset=["comarca"]).groupby("comarca"):
        mask[comarca] = (grp["lat_idx"].values, grp["lon_idx"].values)

    # Fallback for comarques with no cell centres inside polygon
    for _, row in gdf.iterrows():
        comarca = row["comarca"]
        if comarca not in mask:
            cx = row["geometry"].centroid
            li = int(np.argmin(np.abs(lats - cx.y)))
            lj = int(np.argmin(np.abs(lons - cx.x)))
            mask[comarca] = (np.array([li]), np.array([lj]))
            print(f"  Fallback to nearest cell for '{comarca}'")

    return mask


def extract_year(year: int, mask: dict) -> pd.DataFrame:
    avail = {}
    for nc in sorted(AGERA5_DIR.glob(f"*_{year}.nc")):
        m = re.match(r"^(.+)_\d{4}$", nc.stem)
        if not m: continue
        key = m.group(1)
        if key not in VARIABLE_MAP: continue
        ds = _normalise_dims(xr.open_dataset(nc, engine="netcdf4"))
        avail[key] = (ds, VARIABLE_MAP[key])

    if not avail:
        print(f"  No files for {year} — skipping")
        return pd.DataFrame()

    missing = [k for k in VARIABLE_MAP if k not in avail]
    if missing:
        print(f"  {year}: missing {missing} — will be NaN")

    first_ds = next(iter(avail.values()))[0]
    times = pd.to_datetime(first_ds["time"].values)
    T = len(times)

    # Build in WARP-specified column order: et0, ea, tmax, tmin, vpd, precip
    OUTPUT_ORDER = ["et0", "ea", "tmax", "tmin", "precip"]

    records = []
    for comarca, (li, lj) in mask.items():
        row = {"date": times, "comarca": comarca}
        arrays = {}

        for key in VARIABLE_MAP:
            if key not in avail:
                continue
            ds, (base_name, transform) = avail[key]
            var = _get_data_var(ds)
            vals = ds[var].values[:, li, lj].astype(float)  # (T, C)
            if transform is not None:
                vals = transform(vals)
            arrays[base_name] = vals
            row[f"{base_name}_mean"] = vals.mean(axis=1)
            row[f"{base_name}_min"]  = vals.min(axis=1)
            row[f"{base_name}_max"]  = vals.max(axis=1)

        # VPD per cell then aggregate
        if "ea" in arrays and "tmax" in arrays:
            vpd = _compute_vpd(arrays["tmax"], arrays["ea"])
            row["vpd_mean"] = vpd.mean(axis=1)
            row["vpd_min"]  = vpd.min(axis=1)
            row["vpd_max"]  = vpd.max(axis=1)
        else:
            row["vpd_mean"] = row["vpd_min"] = row["vpd_max"] = np.full(T, np.nan)

        # Enforce WARP column order: date, comarca, et0, ea, tmax, tmin, vpd, precip
        ordered_cols = ["date", "comarca"]
        for base in OUTPUT_ORDER:
            for stat in ("mean", "min", "max"):
                col = f"{base}_{stat}"
                if col in row:
                    ordered_cols.append(col)
        for stat in ("mean", "min", "max"):
            ordered_cols.append(f"vpd_{stat}")
        df_row = pd.DataFrame(row)
        records.append(df_row[ordered_cols])

    for ds, _ in avail.values():
        ds.close()

    return pd.concat(records, ignore_index=True)


def extract_all(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    sample_nc = next(iter(sorted(AGERA5_DIR.glob("*.nc"))), None)
    if sample_nc is None:
        raise FileNotFoundError(f"No .nc files in {AGERA5_DIR}")
    ds0 = _normalise_dims(xr.open_dataset(sample_nc, engine="netcdf4"))
    lats, lons = ds0["lat"].values, ds0["lon"].values
    ds0.close()

    print(f"Building comarca mask ({len(gdf)} comarques, {len(lats)}×{len(lons)} grid)...")
    mask = build_comarca_mask(gdf, lats, lons)
    n_cells = [len(li) for li, _ in mask.values()]
    print(f"  Cells per comarca: min={min(n_cells)}, mean={np.mean(n_cells):.1f}, max={max(n_cells)}")

    frames = []
    for year in YEARS:
        print(f"  Extracting {year} ...")
        df = extract_year(year, mask)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No data extracted — check {AGERA5_DIR}")

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["comarca", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    if not POLY_PATH.exists():
        raise FileNotFoundError(f"Missing {POLY_PATH} — run get_centroids.py first")

    gdf = gpd.read_file(POLY_PATH)
    print(f"Loaded {len(gdf)} comarca polygons")

    df = extract_all(gdf)

    out = DATA_DIR / "agera5_daily_catalonia.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows to {out}")
    print(f"Date range:  {df['date'].min()} → {df['date'].max()}")
    print(f"Comarques:   {df['comarca'].nunique()}")
    print(f"Columns:     {list(df.columns)}")
    print(f"NaN counts:\n{df.isna().sum().to_string()}")
