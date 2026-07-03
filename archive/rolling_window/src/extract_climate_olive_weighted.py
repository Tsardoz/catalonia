"""
extract_climate_olive_weighted.py
Olive-area-weighted version of extract_climate.py.

For each comarca, AgERA5 grid cells are weighted by the DUN olive area
they contain (year 2024, parcels >= 0.5 ha, S+R). Cells with zero olive
area get zero weight; comarcas with no olive parcels are skipped.

Per-day comarca aggregates:
  *_mean = area-weighted mean across olive-containing cells
  *_min  = unweighted min across olive-containing cells
  *_max  = unweighted max across olive-containing cells

Output: data/agera5_daily_catalonia_oliveweighted.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from shapely import wkt
from shapely.errors import GEOSException

DATA_DIR = Path(__file__).parent.parent / "data"
AGERA5_DIR = DATA_DIR / "agera5_catalonia"
DUN_CSV = DATA_DIR / "dun" / "olives.csv"
OUT_CSV = DATA_DIR / "agera5_daily_catalonia_oliveweighted.csv"

YEARS = list(range(2015, 2025))
DUN_YEAR = "2024"
MIN_PARCEL_HA = 0.5

USE = ["the_geom",
       "Campanya (CAMPANYA)",
       "Nom comarca (NOM_COM)",
       "Superf\u00edcie neta en hect\u00e0rees (SUP_NETA_H)",
       "Sistema d\u2019explotaci\u00f3 (SIST_EXPL)"]
RENAME = dict(zip(USE, ["wkt", "year", "comarca", "ha", "sist"]))

VARIABLE_MAP = {
    "reference_evapotranspiration": ("et0",    None),
    "vapour_pressure":              ("ea",     lambda x: x / 10.0),
    "2m_temperature_max":           ("tmax",   lambda x: x - 273.15),
    "2m_temperature_min":           ("tmin",   lambda x: x - 273.15),
    "precipitation_flux":           ("precip", None),
}


def _normalise_dims(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for d in list(ds.dims) + list(ds.coords):
        if d.lower() == "latitude" and "lat" not in ds.dims: rename[d] = "lat"
        if d.lower() == "longitude" and "lon" not in ds.dims: rename[d] = "lon"
    return ds.rename(rename) if rename else ds


def _get_data_var(ds: xr.Dataset) -> str:
    coords = set(ds.coords)
    spatial = [v for v in ds.data_vars if v not in coords
               and any(d.lower() in ("lat", "latitude") for d in ds[v].dims)
               and any(d.lower() in ("lon", "longitude") for d in ds[v].dims)]
    return max(spatial, key=lambda v: len(ds[v].dims))


def _vpd(tmax_c: np.ndarray, ea_kpa: np.ndarray) -> np.ndarray:
    es = 0.6108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))
    return np.maximum(es - ea_kpa, 0.0)


def load_grid_lats_lons() -> tuple[np.ndarray, np.ndarray]:
    nc = sorted(AGERA5_DIR.glob("*.nc"))[0]
    ds = _normalise_dims(xr.open_dataset(nc))
    lats, lons = ds["lat"].values, ds["lon"].values
    ds.close()
    return lats, lons


def build_olive_weights(lats: np.ndarray, lons: np.ndarray) -> dict:
    """Stream DUN; for each (comarca, cell) sum olive area; return
    {comarca: (lat_idx_arr, lon_idx_arr, weight_arr)}."""
    rows = []
    for chunk in pd.read_csv(DUN_CSV, usecols=USE, dtype=str, chunksize=100_000):
        chunk = chunk.rename(columns=RENAME)
        chunk = chunk[(chunk["year"] == DUN_YEAR) & chunk["sist"].isin(["S", "R"])]
        chunk["ha"] = pd.to_numeric(chunk["ha"].str.replace(",", ".", regex=False), errors="coerce")
        chunk = chunk[(chunk["ha"] >= MIN_PARCEL_HA)].dropna(subset=["wkt", "comarca"])
        if chunk.empty: continue
        cx, cy, keep = [], [], []
        for w in chunk["wkt"].values:
            try:
                c = wkt.loads(w).centroid
                cx.append(c.x); cy.append(c.y); keep.append(True)
            except (GEOSException, ValueError, TypeError):
                keep.append(False)
        chunk = chunk.loc[keep].copy()
        chunk["lat_idx"] = np.abs(np.array(cy)[:, None] - lats[None, :]).argmin(axis=1)
        chunk["lon_idx"] = np.abs(np.array(cx)[:, None] - lons[None, :]).argmin(axis=1)
        rows.append(chunk[["comarca", "ha", "lat_idx", "lon_idx"]])
    parc = pd.concat(rows, ignore_index=True)
    agg = parc.groupby(["comarca", "lat_idx", "lon_idx"], as_index=False)["ha"].sum()
    out = {}
    for comarca, g in agg.groupby("comarca"):
        out[comarca] = (g["lat_idx"].values.astype(int),
                        g["lon_idx"].values.astype(int),
                        g["ha"].values.astype(float))
    return out


def extract_year(year: int, weights: dict) -> pd.DataFrame:
    avail = {}
    for nc in sorted(AGERA5_DIR.glob(f"*_{year}.nc")):
        m = re.match(r"^(.+)_\d{4}$", nc.stem)
        if not m: continue
        if m.group(1) in VARIABLE_MAP:
            avail[m.group(1)] = (_normalise_dims(xr.open_dataset(nc, engine="netcdf4")),
                                 VARIABLE_MAP[m.group(1)])
    if not avail: return pd.DataFrame()
    times = pd.to_datetime(next(iter(avail.values()))[0]["time"].values)
    T = len(times)
    OUT_ORDER = ["et0", "ea", "tmax", "tmin", "precip"]
    records = []
    for comarca, (li, lj, w) in weights.items():
        row = {"date": times, "comarca": comarca}
        arrays = {}
        for key, (ds, (base, transform)) in avail.items():
            v = ds[_get_data_var(ds)].values[:, li, lj].astype(float)
            if transform is not None: v = transform(v)
            arrays[base] = v
            row[f"{base}_mean"] = np.average(v, axis=1, weights=w)
            row[f"{base}_min"] = v.min(axis=1)
            row[f"{base}_max"] = v.max(axis=1)
        if "ea" in arrays and "tmax" in arrays:
            vpd = _vpd(arrays["tmax"], arrays["ea"])
            row["vpd_mean"] = np.average(vpd, axis=1, weights=w)
            row["vpd_min"] = vpd.min(axis=1); row["vpd_max"] = vpd.max(axis=1)
        else:
            row["vpd_mean"] = row["vpd_min"] = row["vpd_max"] = np.full(T, np.nan)
        cols = ["date", "comarca"]
        for b in OUT_ORDER:
            for s in ("mean", "min", "max"):
                if f"{b}_{s}" in row: cols.append(f"{b}_{s}")
        for s in ("mean", "min", "max"): cols.append(f"vpd_{s}")
        records.append(pd.DataFrame({k: row[k] for k in cols}))
    for ds, _ in avail.values(): ds.close()
    return pd.concat(records, ignore_index=True)


if __name__ == "__main__":
    print("Loading AgERA5 grid + DUN olive parcels ...")
    lats, lons = load_grid_lats_lons()
    weights = build_olive_weights(lats, lons)
    n_cells = [len(w) for _, _, w in weights.values()]
    print(f"  {len(weights)} comarques with olives; cells/comarca: "
          f"min={min(n_cells)}, mean={np.mean(n_cells):.1f}, max={max(n_cells)}")
    frames = []
    for y in YEARS:
        print(f"  {y} ...")
        fr = extract_year(y, weights)
        if not fr.empty: frames.append(fr)
    df = pd.concat(frames, ignore_index=True).sort_values(["comarca", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df):,} rows, {df['comarca'].nunique()} comarques)")
