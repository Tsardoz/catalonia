#!/usr/bin/env python3
"""
fetch_dun_crops.py

Extract parcel-level DUN declarations per crop from the Catalan open-data
portal (analisi.transparenciacatalunya.cat), dataset "Parcel·les i cultius
de les explotacions (DUN)" (id uwe8-jqcu) -- the same source that
data/dun/olives.csv was originally extracted from (filtered there to
nprod = OLIVERA). This script generalises that extraction to other crops
and additionally performs the AgERA5 grid-cell spatial join that was
previously done ad hoc for olive in
archive/rolling_window/src/extract_climate_olive_weighted.py.

For each registered crop:
  1. Paginate the SODA API (https://dev.socrata.com/docs/paging.html)
     filtered to one or more --nprod crop names. Perennial crops (olive,
     vine, nuts, stone fruit) default to campaign 2024 only, matching the
     existing olive convention -- DUN area/cultivar mix is treated as a
     static single-year snapshot, which is reasonable since perennial
     area changes <2%/year. Annual crops (corn, oats, wheat, soybean)
     default to ALL available campaigns (2023-2025), since their sown
     area genuinely rotates year to year and a single-year snapshot could
     be an unrepresentative fluke -- fetching all years lets you check
     that the area/spatial pattern is stable before trusting it. Writes
     data/dun/{crop}.csv (all fetched years combined, with a campanya
     column) with columns:
       the_geom, campanya, comarca, sup_neta_h, sist_expl, nvar, nprod,
       id_exp, id_rec
  2. Computes each parcel polygon's centroid (shapely), assigns it to the
     nearest AgERA5 grid cell (grid read from data/agera5_catalonia/*.nc,
     same nearest-neighbour approach as extract_climate_olive_weighted.py),
     and aggregates net declared area (ha) per grid cell x irrigation
     regime, SEPARATELY PER YEAR. Writes one
     data/dun/{crop}_dun_grid_cells_{year}.csv per campaign year present
     (or the unsuffixed data/dun/{crop}_dun_grid_cells.csv when only one
     year was fetched, for backward compatibility with the perennial-crop
     files already generated). Schema matches the existing
     data/dun/olive_dun_grid_cells_2024.csv:
       lat_idx, lon_idx, cell_lat, cell_lon, area_ha_R, area_ha_S,
       n_parcels_R, n_parcels_S, total_ha, rainfed_frac

CLI:
  python src/fetch_dun_crops.py --crop vinya
  python src/fetch_dun_crops.py --crop blat_de_moro --years 2023,2024,2025
  python src/fetch_dun_crops.py --all
"""
import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely import wkt
from shapely.errors import GEOSException

DATA_DIR = Path(__file__).parent.parent / "data"
DUN_DIR = DATA_DIR / "dun"
AGERA5_DIR = DATA_DIR / "agera5_catalonia"

DATASET_ID = "uwe8-jqcu"
BASE_URL = f"https://analisi.transparenciacatalunya.cat/resource/{DATASET_ID}.csv"
ALL_CAMPAIGNS = ["2023", "2024", "2025"]  # full range available in this dataset
PAGE_SIZE = 20_000
REQUEST_TIMEOUT = 300

SELECT_COLS = [
    "the_geom", "campanya", "nom_com", "sup_neta_h", "sist_expl",
    "nvar", "nprod", "id_exp", "id_rec",
]

# crop slug -> list of nprod values to match (some crops have multiple
# DUN product names, e.g. peach vs flat peach are kept separate here to
# match the distinct crop_en entries already used elsewhere in this project)
CROPS: dict[str, list[str]] = {
    "vinya": ["VINYA"],
    "ametller": ["AMETLLER"],
    "avellaner": ["AVELLANER"],
    "cirerer": ["CIRERER"],
    "presseguer": ["PRESSEGUER"],
    "pressec_pla": ["PRÉSSEC PLA"],
    "blat_de_moro": ["BLAT DE MORO"],
    "civada": ["CIVADA"],
    "blat_tou": ["BLAT TOU"],
    "blat_dur": ["BLAT DUR"],
    "soia": ["SOIA"],
    "coto": ["COTÓ"],
}

# Annual crops rotate their sown area year to year, so default to fetching
# every available campaign instead of a single-year snapshot (unlike the
# perennial crops above, which default to campaign 2024 only).
ANNUAL_CROPS: set[str] = {"blat_de_moro", "civada", "blat_tou", "blat_dur", "soia"}


def default_years_for(slug: str) -> list[str]:
    return list(ALL_CAMPAIGNS) if slug in ANNUAL_CROPS else ["2024"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--crop", choices=sorted(CROPS), help="Single crop slug to process")
    g.add_argument("--all", action="store_true", help="Process all registered crops")
    p.add_argument(
        "--years", default=None,
        help=(
            "Comma-separated DUN campaign years to fetch (e.g. 2023,2024,2025). "
            "Default: all available years (2023-2025) for annual crops "
            "(blat_de_moro, civada, blat_tou, blat_dur, soia); 2024 only for "
            "perennial crops. Applies to every crop processed in this run."
        ),
    )
    return p.parse_args()


def soql_in_clause(field: str, values: list[str]) -> str:
    quoted = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"{field} in ({quoted})"


def fetch_crop_parcels(nprod_values: list[str], years: list[str]) -> pd.DataFrame:
    """Paginate the SODA API for all parcels matching nprod_values in any
    of the given campaign years. Returns the raw (string-typed) parcel
    table (with a campanya column so years can be split back out later)."""
    where = soql_in_clause("nprod", nprod_values) + " AND " + soql_in_clause("campanya", years)
    frames = []
    offset = 0
    while True:
        params = {
            "$select": ",".join(SELECT_COLS),
            "$where": where,
            "$order": ":id",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
        resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        page = pd.read_csv(StringIO(resp.text), dtype=str)
        if page.empty:
            break
        frames.append(page)
        print(f"    offset={offset:>7} +{len(page):>6} rows", flush=True)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=SELECT_COLS)
    return pd.concat(frames, ignore_index=True)


def load_grid_lats_lons() -> tuple[np.ndarray, np.ndarray]:
    nc = sorted(AGERA5_DIR.glob("*.nc"))[0]
    ds = xr.open_dataset(nc)
    rename = {}
    for d in list(ds.dims) + list(ds.coords):
        if d.lower() == "latitude" and "lat" not in ds.dims:
            rename[d] = "lat"
        if d.lower() == "longitude" and "lon" not in ds.dims:
            rename[d] = "lon"
    if rename:
        ds = ds.rename(rename)
    lats, lons = ds["lat"].values, ds["lon"].values
    ds.close()
    return lats, lons


def assign_grid_cells(df: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """Compute each parcel's centroid from its WKT geometry and assign the
    nearest AgERA5 grid cell index. Rows with unparseable geometry are
    dropped."""
    cx, cy, keep = [], [], []
    for w in df["the_geom"].values:
        try:
            c = wkt.loads(w).centroid
            cx.append(c.x)
            cy.append(c.y)
            keep.append(True)
        except (GEOSException, ValueError, TypeError, AttributeError):
            keep.append(False)
    out = df.loc[keep].copy()
    cx_arr = np.array(cx)
    cy_arr = np.array(cy)
    out["lat_idx"] = np.abs(cy_arr[:, None] - lats[None, :]).argmin(axis=1)
    out["lon_idx"] = np.abs(cx_arr[:, None] - lons[None, :]).argmin(axis=1)
    out["cell_lat"] = lats[out["lat_idx"].values]
    out["cell_lon"] = lons[out["lon_idx"].values]
    return out


def build_grid_cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate net declared area (ha) per grid cell x irrigation regime,
    matching the schema of the existing
    data/dun/olive_dun_grid_cells_2024.csv."""
    df = df.copy()
    df["ha"] = pd.to_numeric(df["sup_neta_h"], errors="coerce")
    df = df.dropna(subset=["ha", "sist_expl"])
    df = df[df["sist_expl"].isin(["S", "R"])]

    agg = (
        df.groupby(["lat_idx", "lon_idx", "cell_lat", "cell_lon", "sist_expl"])
        .agg(ha=("ha", "sum"), n=("ha", "size"))
        .unstack("sist_expl", fill_value=0.0)
    )
    agg.columns = [f"{stat}_{regime}" for stat, regime in agg.columns]
    agg = agg.rename(columns={
        "ha_R": "area_ha_R", "ha_S": "area_ha_S",
        "n_R": "n_parcels_R", "n_S": "n_parcels_S",
    })
    for c in ["area_ha_R", "area_ha_S", "n_parcels_R", "n_parcels_S"]:
        if c not in agg.columns:
            agg[c] = 0.0
    agg = agg.reset_index()
    agg["total_ha"] = agg["area_ha_R"] + agg["area_ha_S"]
    agg["rainfed_frac"] = agg["area_ha_S"] / agg["total_ha"].where(agg["total_ha"] > 0)
    cols = ["lat_idx", "lon_idx", "cell_lat", "cell_lon", "area_ha_R", "area_ha_S",
            "n_parcels_R", "n_parcels_S", "total_ha", "rainfed_frac"]
    return agg[cols].sort_values(["lat_idx", "lon_idx"]).reset_index(drop=True)


def process_crop(slug: str, nprod_values: list[str], years: list[str],
                  lats: np.ndarray, lons: np.ndarray) -> None:
    print(f"\n=== {slug} ({', '.join(nprod_values)}) -- years {years} ===", flush=True)
    print("  Fetching parcels ...", flush=True)
    df = fetch_crop_parcels(nprod_values, years)
    print(f"  Fetched {len(df):,} parcel rows", flush=True)
    if df.empty:
        print(f"  WARNING: no rows for {slug}; skipping", flush=True)
        return

    parcels_out = df.rename(columns={"nom_com": "comarca"})
    parcels_path = DUN_DIR / f"{slug}.csv"
    parcels_out.to_csv(parcels_path, index=False)
    print(f"  Saved: {parcels_path} ({len(parcels_out):,} rows)", flush=True)
    if "campanya" in parcels_out.columns:
        print(
            "  Rows per year: "
            + ", ".join(f"{y}={n}" for y, n in parcels_out["campanya"].value_counts().sort_index().items()),
            flush=True,
        )

    gridded = assign_grid_cells(df, lats, lons)
    n_dropped = len(df) - len(gridded)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with unparseable geometry", flush=True)

    years_present = sorted(gridded["campanya"].dropna().unique())
    for year in years_present:
        year_df = gridded[gridded["campanya"] == year]
        grid_table = build_grid_cell_table(year_df)
        suffix = f"_{year}" if len(years_present) > 1 else ""
        grid_path = DUN_DIR / f"{slug}_dun_grid_cells{suffix}.csv"
        grid_table.to_csv(grid_path, index=False)
        print(
            f"  Saved: {grid_path} ({len(grid_table)} grid cells, "
            f"total_ha={grid_table['total_ha'].sum():,.0f})",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    DUN_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading AgERA5 grid ...", flush=True)
    lats, lons = load_grid_lats_lons()
    print(f"  Grid: {len(lats)} lats x {len(lons)} lons", flush=True)

    crops = {args.crop: CROPS[args.crop]} if args.crop else CROPS
    override_years = [y.strip() for y in args.years.split(",")] if args.years else None
    for slug, nprod_values in crops.items():
        years = override_years if override_years is not None else default_years_for(slug)
        process_crop(slug, nprod_values, years, lats, lons)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
