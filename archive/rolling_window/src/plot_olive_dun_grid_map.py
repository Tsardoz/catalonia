"""
plot_olive_dun_grid_map.py
Aggregate DUN olive parcels (one campaign year) to AgERA5 0.1 deg grid cells
and plot one dot per occupied cell: size = total olive area in cell,
colour = rainfed area fraction (S / (S + R)).

Usage:
  python src/plot_olive_dun_grid_map.py --year 2024 --min-parcel-ha 0.5
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import xarray as xr
from shapely import wkt
from shapely.errors import GEOSException

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
DUN_CSV = DATA / "dun" / "olives.csv"
POLY = DATA / "comarca_polygons.gpkg"
AGERA5_DIR = DATA / "agera5_catalonia"
WHITELIST = DATA / "dun" / "comarca_olive_whitelist.csv"
NE_DIR = DATA / "ne"
NE_RIVERS = [NE_DIR / "ne_10m_rivers_lake_centerlines.zip",
             NE_DIR / "ne_10m_rivers_europe.zip"]
NE_LAKES = [NE_DIR / "ne_10m_lakes.zip",
            NE_DIR / "ne_10m_lakes_europe.zip"]
OSM_RIVERS = DATA / "osm" / "catalonia_rivers.gpkg"
OSM_WATER = DATA / "osm" / "catalonia_water.gpkg"
OSM_MIN_LAKE_KM2 = 0.5
DEM_TIF = DATA / "dem_catalonia.tif"

USE = ["the_geom",
       "Campanya (CAMPANYA)",
       "Nom comarca (NOM_COM)",
       "Superf\u00edcie neta en hect\u00e0rees (SUP_NETA_H)",
       "Sistema d\u2019explotaci\u00f3 (SIST_EXPL)"]
RENAME = {USE[0]: "wkt", USE[1]: "year", USE[2]: "comarca",
          USE[3]: "ha", USE[4]: "sist"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", default="2024")
    p.add_argument("--min-parcel-ha", type=float, default=0.5)
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--elevation", action="store_true",
                   help="overlay DEM as a low-alpha raster background")
    p.add_argument("--elevation-alpha", type=float, default=0.35)
    p.add_argument("--elevation-cmap", default="terrain",
                   help="matplotlib colormap (e.g. terrain, gist_earth, gray)")
    p.add_argument("--elevation-max-px", type=int, default=1200,
                   help="downsample DEM so the longer side is <= this many pixels")
    return p.parse_args()


def load_dem(bbox: tuple[float, float, float, float], max_px: int
             ) -> tuple[np.ndarray, tuple[float, float, float, float]] | tuple[None, None]:
    if not DEM_TIF.exists():
        print(f"  DEM not found at {DEM_TIF}; skipping elevation overlay")
        return None, None
    xmin, ymin, xmax, ymax = bbox
    with rasterio.open(DEM_TIF) as src:
        win = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, src.transform)
        h, w = max(1, int(win.height)), max(1, int(win.width))
        scale = max(1, int(np.ceil(max(h, w) / max_px)))
        out_h, out_w = h // scale, w // scale
        arr = src.read(1, window=win, out_shape=(out_h, out_w),
                       resampling=Resampling.average)
        t = src.window_transform(win)
        left = t.c
        top = t.f
        right = left + t.a * w
        bottom = top + t.e * h
        nd = src.nodata
    if nd is not None:
        arr = np.where(arr == nd, np.nan, arr)
    return arr, (left, right, bottom, top)


def load_hydro(bbox: tuple[float, float, float, float]) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    xmin, ymin, xmax, ymax = bbox
    def _load_zips(paths):
        parts = []
        for p in paths:
            if not p.exists(): continue
            g = gpd.read_file("zip://" + str(p)).to_crs("EPSG:4326")
            parts.append(g.cx[xmin:xmax, ymin:ymax])
        if not parts: return None
        out = pd.concat(parts, ignore_index=True)
        return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    if OSM_RIVERS.exists():
        rivers = gpd.read_file(OSM_RIVERS).to_crs("EPSG:4326")
        rivers = rivers[rivers["waterway"] == "river"].cx[xmin:xmax, ymin:ymax]
    else:
        rivers = _load_zips(NE_RIVERS)

    if OSM_WATER.exists():
        lakes = gpd.read_file(OSM_WATER).to_crs("EPSG:4326")
        if "area_km2" not in lakes.columns:
            lakes["area_km2"] = lakes.to_crs("EPSG:25831").area / 1e6
        lakes = lakes[lakes["area_km2"] >= OSM_MIN_LAKE_KM2].cx[xmin:xmax, ymin:ymax]
    else:
        lakes = _load_zips(NE_LAKES)

    return rivers, lakes


def load_grid() -> tuple[np.ndarray, np.ndarray]:
    nc = sorted(AGERA5_DIR.glob("*.nc"))[0]
    ds = xr.open_dataset(nc)
    rename = {}
    for c in list(ds.coords):
        if c.lower() == "latitude":  rename[c] = "lat"
        if c.lower() == "longitude": rename[c] = "lon"
    if rename: ds = ds.rename(rename)
    lats = ds["lat"].values
    lons = ds["lon"].values
    ds.close()
    return lats, lons


def stream_centroids(year: str, min_parcel_ha: float, chunksize: int) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(DUN_CSV, usecols=USE, dtype=str, chunksize=chunksize):
        chunk = chunk.rename(columns=RENAME)
        chunk = chunk[chunk["year"] == year]
        chunk["ha"] = pd.to_numeric(chunk["ha"].str.replace(",", ".", regex=False), errors="coerce")
        chunk = chunk[(chunk["ha"] >= min_parcel_ha) & chunk["sist"].isin(["S", "R"])]
        chunk = chunk.dropna(subset=["wkt"])
        if chunk.empty:
            continue
        cx, cy, keep = [], [], []
        for w in chunk["wkt"].values:
            try:
                c = wkt.loads(w).centroid
                cx.append(c.x); cy.append(c.y); keep.append(True)
            except (GEOSException, ValueError, TypeError):
                keep.append(False)
        chunk = chunk.loc[keep].copy()
        chunk["lon"], chunk["lat"] = cx, cy
        parts.append(chunk[["comarca", "ha", "sist", "lon", "lat"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def aggregate_to_grid(parcels: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    li = np.abs(parcels["lat"].values[:, None] - lats[None, :]).argmin(axis=1)
    lj = np.abs(parcels["lon"].values[:, None] - lons[None, :]).argmin(axis=1)
    parcels = parcels.assign(lat_idx=li, lon_idx=lj,
                             cell_lat=lats[li], cell_lon=lons[lj])
    long = (parcels.groupby(["lat_idx", "lon_idx", "cell_lat", "cell_lon", "sist"], as_index=False)
                   .agg(area_ha=("ha", "sum"), n_parcels=("ha", "size")))
    wide = long.pivot_table(index=["lat_idx", "lon_idx", "cell_lat", "cell_lon"],
                            columns="sist", values=["area_ha", "n_parcels"], fill_value=0).reset_index()
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
    for col in ["area_ha_S", "area_ha_R", "n_parcels_S", "n_parcels_R"]:
        if col not in wide.columns: wide[col] = 0
    wide["total_ha"] = wide["area_ha_S"] + wide["area_ha_R"]
    wide["rainfed_frac"] = wide["area_ha_S"] / wide["total_ha"].where(wide["total_ha"] > 0)
    return wide.sort_values("total_ha", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    print("Loading AgERA5 grid ...")
    lats, lons = load_grid()
    print(f"  {len(lats)}x{len(lons)} cells, dlat={lats[1]-lats[0]:.3f}, dlon={lons[1]-lons[0]:.3f}")

    print(f"Streaming DUN parcels (year={args.year}, >= {args.min_parcel_ha} ha) ...")
    parcels = stream_centroids(args.year, args.min_parcel_ha, args.chunksize)
    print(f"  {len(parcels):,} parcels parsed")

    cells = aggregate_to_grid(parcels, lats, lons)
    print(f"  {len(cells)} occupied grid cells, total area {cells['total_ha'].sum():,.0f} ha")

    csv_out = DATA / "dun" / f"olive_dun_grid_cells_{args.year}.csv"
    cells.to_csv(csv_out, index=False)
    print(f"  Wrote {csv_out}")

    comarcas = gpd.read_file(POLY)
    wl = pd.read_csv(WHITELIST)["comarca"].str.strip().unique()
    comarcas["wl"] = comarcas["comarca"].isin(wl)
    xmin, ymin, xmax, ymax = comarcas.total_bounds
    pad = 0.15
    in_x = (lons >= xmin - pad) & (lons <= xmax + pad)
    in_y = (lats >= ymin - pad) & (lats <= ymax + pad)
    gx, gy = np.meshgrid(lons[in_x], lats[in_y])

    rivers, lakes = load_hydro((xmin - pad, ymin - pad, xmax + pad, ymax + pad))

    fig, ax = plt.subplots(figsize=(12, 9))
    if args.elevation:
        dem, dem_extent = load_dem((xmin - pad, ymin - pad, xmax + pad, ymax + pad),
                                   args.elevation_max_px)
        if dem is not None:
            ax.imshow(dem, extent=dem_extent, origin="upper",
                      cmap=args.elevation_cmap, alpha=args.elevation_alpha,
                      interpolation="bilinear", zorder=0.5)
            print(f"  DEM overlay: {dem.shape}, range "
                  f"{np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} m, "
                  f"alpha={args.elevation_alpha}, cmap={args.elevation_cmap}")
    fill = "none" if args.elevation else "white"
    comarcas.plot(ax=ax, facecolor=fill, edgecolor="#999", linewidth=0.5, zorder=1.2)
    wl_fill = "none" if args.elevation else "#eef9f1"
    comarcas[comarcas["wl"]].plot(ax=ax, facecolor=wl_fill, edgecolor="#27ae60",
                                   linewidth=0.9, zorder=1.5)
    if rivers is not None and not rivers.empty:
        rivers.plot(ax=ax, color="#7fb0e0", linewidth=0.5, alpha=0.45, zorder=1.7)
    if lakes is not None and not lakes.empty:
        lakes.plot(ax=ax, color="#cfe2f3", edgecolor="#7fb0e0", linewidth=0.3,
                   alpha=0.55, zorder=1.8)
    ax.scatter(gx, gy, s=5, color="#9ec5e8", alpha=0.5, marker=".", zorder=2,
               label="AgERA5 grid cell")

    max_ha = float(cells["total_ha"].max())
    sizes = 10 + 240 * (cells["total_ha"] / max_ha)
    sc = ax.scatter(cells["cell_lon"], cells["cell_lat"],
                    c=cells["rainfed_frac"], cmap="RdYlGn", vmin=0, vmax=1,
                    s=sizes, edgecolor="black", linewidth=0.3, alpha=0.92, zorder=4)
    plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.01,
                 label="Rainfed fraction in cell (S / (S + R) area)")

    for ha_ref in [100, 500, 2000]:
        s = 10 + 240 * (ha_ref / max_ha)
        ax.scatter([], [], s=s, color="#888", edgecolor="black", linewidth=0.3,
                   label=f"{ha_ref} ha")
    ax.legend(title="DUN olive area / cell", loc="lower right", fontsize=8,
              labelspacing=1.3, framealpha=0.92)
    ax.set_xlim(xmin - pad, xmax + pad); ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"DUN olive parcels ({args.year}, >= {args.min_parcel_ha} ha) "
                 f"aggregated to AgERA5 grid cells\n"
                 f"Green-outlined comarques are the >= 80% rainfed whitelist",
                 fontsize=10)
    fig.tight_layout()
    suffix = "_elev" if args.elevation else ""
    out = FIG / f"olive_dun_grid_map_{args.year}{suffix}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
