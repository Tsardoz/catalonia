"""
elevation_correction.py
Per-comarca temperature lapse-rate correction for olive analysis.

For each comarca:
  z_olive   = DUN-area-weighted mean parcel elevation (DEM-sampled centroids)
  z_climate = mean elevation of AgERA5 grid cells assigned to the comarca
              (same spatial join as extract_climate.py)
  dT        = lapse_rate * (z_olive - z_climate)   [default -5.5 C / km]

Outputs:
  data/comarca_olive_elevation_correction.csv  (one row per comarca)
  --apply: data/agera5_daily_catalonia_elevcor.csv (daily climate +
           tmax_*_elevcor, tmin_*_elevcor, vpd_*_elevcor columns)
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from shapely import wkt
from shapely.errors import GEOSException

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
DUN_CSV = DATA / "dun" / "olives.csv"
POLY = DATA / "comarca_polygons.gpkg"
AGERA5_DIR = DATA / "agera5_catalonia"
DEM_TIF = DATA / "dem_catalonia.tif"
DAILY_CSV = DATA / "agera5_daily_catalonia.csv"
DAILY_CSV_OW = DATA / "agera5_daily_catalonia_oliveweighted.csv"
OUT_CORR = DATA / "comarca_olive_elevation_correction.csv"
OUT_CORR_OW = DATA / "comarca_olive_elevation_correction_oliveweighted.csv"
OUT_DAILY = DATA / "agera5_daily_catalonia_elevcor.csv"
OUT_DAILY_OW = DATA / "agera5_daily_catalonia_oliveweighted_elevcor.csv"

USE = ["the_geom",
       "Campanya (CAMPANYA)",
       "Nom comarca (NOM_COM)",
       "Superf\u00edcie neta en hect\u00e0rees (SUP_NETA_H)",
       "Sistema d\u2019explotaci\u00f3 (SIST_EXPL)"]
RENAME = dict(zip(USE, ["wkt", "year", "comarca", "ha", "sist"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lapse-rate", type=float, default=-5.5,
                   help="degC per km (default -5.5; negative means cooler higher)")
    p.add_argument("--year", default="2024", help="DUN campaign year for olive footprint")
    p.add_argument("--min-parcel-ha", type=float, default=0.5)
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--apply", action="store_true",
                   help="also write daily climate file with elev-corrected columns")
    p.add_argument("--plot", action="store_true",
                   help="write per-comarca diagnostic bar chart")
    p.add_argument("--plot-top", type=int, default=20,
                   help="show top-N comarques by olive area (default 20)")
    p.add_argument("--olive-weighted", action="store_true",
                   help="compute z_climate as olive-area-weighted mean of cell DEM "
                        "elevations (matches extract_climate_olive_weighted.py); "
                        "with --apply also reads/writes the olive-weighted daily file")
    return p.parse_args()


def plot_correction(corr: pd.DataFrame, top: int) -> None:
    df = corr.dropna(subset=["dT_C"]).copy()
    df = df[df["area_ha"] > 0].sort_values("area_ha", ascending=False).head(top)
    df = df.iloc[::-1]  # largest at top of horizontal plot
    colors = ["#c0392b" if v >= 0 else "#2980b9" for v in df["dT_C"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, 0.32 * len(df))),
                                   sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    ax1.barh(df["comarca"], df["dT_C"], color=colors, edgecolor="black", linewidth=0.4)
    ax1.axvline(0, color="black", linewidth=0.6)
    for y, (v, dz) in enumerate(zip(df["dT_C"], df["dz_m"])):
        ax1.text(v + (0.02 if v >= 0 else -0.02), y,
                 f"{v:+.2f}°C ({dz:+.0f} m)",
                 va="center", ha="left" if v >= 0 else "right", fontsize=8)
    xmax = max(0.05, df["dT_C"].abs().max() * 1.35)
    ax1.set_xlim(-xmax, xmax)
    ax1.set_xlabel("Temperature offset dT (°C)  =  -5.5/1000 × (z_olive − z_climate)")
    ax1.set_title(f"Elevation correction per comarca (top {len(df)} by olive area)")
    ax2.barh(df["comarca"], df["area_ha"], color="#7f8c8d", edgecolor="black", linewidth=0.4)
    for y, a in enumerate(df["area_ha"]):
        ax2.text(a, y, f"  {a:,.0f}", va="center", fontsize=8)
    ax2.set_xlabel("DUN olive area (ha, 2024, ≥0.5 ha)")
    ax2.set_xlim(0, df["area_ha"].max() * 1.18)
    fig.tight_layout()
    out = FIG / "elevation_correction_per_comarca.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def load_grid_lats_lons() -> tuple[np.ndarray, np.ndarray]:
    nc = sorted(AGERA5_DIR.glob("*.nc"))[0]
    ds = xr.open_dataset(nc)
    rename = {c: ("lat" if c.lower() == "latitude" else "lon")
              for c in ds.coords if c.lower() in ("latitude", "longitude")}
    if rename: ds = ds.rename(rename)
    lats, lons = ds["lat"].values, ds["lon"].values
    ds.close()
    return lats, lons


def cells_per_comarca(gdf: gpd.GeoDataFrame, lats: np.ndarray, lons: np.ndarray) -> dict:
    li, lj = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing="ij")
    yv, xv = np.meshgrid(lats, lons, indexing="ij")
    grid = gpd.GeoDataFrame(
        {"lat_idx": li.ravel(), "lon_idx": lj.ravel(),
         "geometry": gpd.points_from_xy(xv.ravel(), yv.ravel())}, crs="EPSG:4326")
    joined = gpd.sjoin(grid, gdf[["comarca", "geometry"]], how="left", predicate="within")
    out = {c: (g["lat_idx"].values, g["lon_idx"].values)
           for c, g in joined.dropna(subset=["comarca"]).groupby("comarca")}
    for _, row in gdf.iterrows():
        c = row["comarca"]
        if c not in out:
            cx = row["geometry"].centroid
            out[c] = (np.array([int(np.argmin(np.abs(lats - cx.y)))]),
                      np.array([int(np.argmin(np.abs(lons - cx.x)))]))
    return out


def stream_olive_parcels(year: str, min_ha: float, chunksize: int) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(DUN_CSV, usecols=USE, dtype=str, chunksize=chunksize):
        chunk = chunk.rename(columns=RENAME)
        chunk = chunk[(chunk["year"] == year) & chunk["sist"].isin(["S", "R"])]
        chunk["ha"] = pd.to_numeric(chunk["ha"].str.replace(",", ".", regex=False), errors="coerce")
        chunk = chunk[(chunk["ha"] >= min_ha)].dropna(subset=["wkt", "comarca"])
        if chunk.empty: continue
        cx, cy, keep = [], [], []
        for w in chunk["wkt"].values:
            try:
                c = wkt.loads(w).centroid
                cx.append(c.x); cy.append(c.y); keep.append(True)
            except (GEOSException, ValueError, TypeError):
                keep.append(False)
        chunk = chunk.loc[keep].copy()
        chunk["lon"], chunk["lat"] = cx, cy
        parts.append(chunk[["comarca", "ha", "lon", "lat"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sample_dem(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    with rasterio.open(DEM_TIF) as src:
        vals = np.fromiter((v[0] for v in src.sample(zip(lons, lats))), dtype=float, count=len(lons))
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
    return vals


def main() -> None:
    args = parse_args()
    print("Loading grid + comarca polygons + DEM ...")
    lats, lons = load_grid_lats_lons()
    gdf = gpd.read_file(POLY)
    cells = cells_per_comarca(gdf, lats, lons)

    rows = []
    olive = stream_olive_parcels(args.year, args.min_parcel_ha, args.chunksize)
    print(f"  {len(olive):,} olive parcels (year={args.year}, >= {args.min_parcel_ha} ha)")
    olive["z"] = sample_dem(olive["lon"].values, olive["lat"].values)

    if args.olive_weighted:
        olive["lat_idx"] = np.abs(olive["lat"].values[:, None] - lats[None, :]).argmin(axis=1)
        olive["lon_idx"] = np.abs(olive["lon"].values[:, None] - lons[None, :]).argmin(axis=1)
        cell_area = (olive.dropna(subset=["z"])
                          .groupby(["comarca", "lat_idx", "lon_idx"], as_index=False)["ha"].sum())

    for comarca in sorted(gdf["comarca"].unique()):
        if args.olive_weighted:
            ca = cell_area.loc[cell_area["comarca"] == comarca]
            if ca.empty:
                z_clim, n_cells = np.nan, 0
            else:
                z_cells = sample_dem(lons[ca["lon_idx"].values], lats[ca["lat_idx"].values])
                w = ca["ha"].values
                m = ~np.isnan(z_cells)
                z_clim = float(np.average(z_cells[m], weights=w[m])) if m.any() else np.nan
                n_cells = int(m.sum())
        else:
            li, lj = cells[comarca]
            z_clim = float(np.nanmean(sample_dem(lons[lj], lats[li])))
            n_cells = len(li)
        sub = olive.loc[olive["comarca"] == comarca].dropna(subset=["z"])
        if sub.empty or sub["ha"].sum() == 0:
            z_ol, n, area = np.nan, 0, 0.0
        else:
            z_ol = float(np.average(sub["z"].values, weights=sub["ha"].values))
            n, area = len(sub), float(sub["ha"].sum())
        dz = z_ol - z_clim if not (np.isnan(z_ol) or np.isnan(z_clim)) else np.nan
        dT = (args.lapse_rate / 1000.0) * dz if not np.isnan(dz) else np.nan
        rows.append({"comarca": comarca, "n_parcels": n, "area_ha": area,
                     "z_olive_m": z_ol, "z_climate_m": z_clim,
                     "dz_m": dz, "dT_C": dT, "n_grid_cells": n_cells})

    corr = pd.DataFrame(rows)
    out_corr = OUT_CORR_OW if args.olive_weighted else OUT_CORR
    corr.to_csv(out_corr, index=False)
    print(f"\nWrote {out_corr}")
    print(corr.dropna(subset=["dT_C"]).sort_values("dT_C").to_string(index=False))

    if args.plot:
        plot_correction(corr, args.plot_top)

    if args.apply:
        in_csv = DAILY_CSV_OW if args.olive_weighted else DAILY_CSV
        out_csv = OUT_DAILY_OW if args.olive_weighted else OUT_DAILY
        print(f"\nApplying correction to {in_csv} ...")
        d = pd.read_csv(in_csv, parse_dates=["date"])
        d = d.merge(corr[["comarca", "dT_C"]], on="comarca", how="left")
        for stat in ("mean", "min", "max"):
            for v in ("tmax", "tmin"):
                col = f"{v}_{stat}"
                if col in d.columns:
                    d[f"{col}_elevcor"] = d[col] + d["dT_C"]
            tcol = f"tmax_{stat}_elevcor"
            ea = d.get(f"ea_{stat}")
            if tcol in d.columns and ea is not None:
                es = 0.6108 * np.exp(17.27 * d[tcol] / (d[tcol] + 237.3))
                d[f"vpd_{stat}_elevcor"] = np.maximum(es - ea, 0.0)
        d = d.drop(columns=["dT_C"])
        d.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}  ({len(d):,} rows, {len(d.columns)} cols)")


if __name__ == "__main__":
    main()
