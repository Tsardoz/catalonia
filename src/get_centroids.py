"""
get_centroids.py  —  Step 2
Fetch comarca boundary polygons from the ICC WFS and compute centroids.

Output CSV columns: comarca, lat, lon

Note: comarca names come from the ICC. They should match the yield data names
(both use official Catalan names), but verify with check_name_match() if
the join in join_dataset.py produces unexpected NaNs.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"

# ICC WFS endpoint — administrative divisions of Catalonia
WFS_BASE = "https://geoserveis.icgc.cat/servei/catalunya/divisions-administratives/wfs"

# Full namespaced layer names as reported by GetCapabilities (WFS 2.0.0).
# The service does not support GeoJSON output; it returns GML 3.2 only.
CANDIDATE_LAYERS = [
    "divisions_administratives_wfs:divisions_administratives_comarques_5000",
    "divisions_administratives_wfs:divisions_administratives_comarques_50000",
]

EXPECTED_COUNT = 42


def _build_url(layer: str) -> str:
    return (
        f"{WFS_BASE}?SERVICE=WFS&REQUEST=GetFeature&VERSION=2.0.0"
        f"&TYPENAMES={layer}&SRSNAME=EPSG%3A4326"
    )


def _find_name_column(gdf: gpd.GeoDataFrame) -> str:
    """Return the column most likely to hold the comarca name."""
    candidates = [
        "nomcomar", "nom_comar", "nomcomarc", "nom_comarc",
        "nom", "name", "comarca", "NOMCOMAR", "NOM",
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    # Fallback: first non-geometry string column
    for c in gdf.columns:
        if c == "geometry":
            continue
        if gdf[c].dtype == object:
            return c
    raise ValueError(
        f"Cannot identify comarca name column. Available columns: {list(gdf.columns)}"
    )


def fetch_comarques() -> gpd.GeoDataFrame:
    """Download comarca polygons from ICC WFS (GML 3.2) and return as GeoDataFrame."""
    last_err = None
    for layer in CANDIDATE_LAYERS:
        url = _build_url(layer)
        try:
            print(f"Trying layer '{layer}' ...")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            # Detect server-side exception (service returns 200 even for errors)
            if b"ExceptionReport" in resp.content[:200]:
                raise RuntimeError("WFS returned ExceptionReport")
            # Write to a temp file with .gml extension so GDAL can detect format
            with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            gdf = gpd.read_file(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            print(f"  OK — {len(gdf)} features, columns: {list(gdf.columns)}")
            return gdf
        except Exception as e:
            print(f"  Failed: {e}")
            last_err = e
    raise RuntimeError(
        f"Could not fetch any WFS layer from {WFS_BASE}. Last error: {last_err}\n"
        "Check available layers at:\n"
        f"  {WFS_BASE}?SERVICE=WFS&REQUEST=GetCapabilities"
    )


# Name normalisations: WFS name → yield data name
NAME_MAP = {
    "Val d'Aran": "Aran",
}


def get_comarca_centroids() -> pd.DataFrame:
    gdf = fetch_comarques()

    if len(gdf) != EXPECTED_COUNT:
        print(f"Note: WFS returned {len(gdf)} comarques (expected {EXPECTED_COUNT})")

    # Ensure geographic CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Compute centroids in projected CRS (UTM 31N) to avoid geographic-CRS warning,
    # then extract lat/lon from the reprojected points.
    gdf_proj = gdf.to_crs("EPSG:25831")
    centroids_proj = gdf_proj.geometry.centroid
    centroids_geo = centroids_proj.to_crs("EPSG:4326")

    name_col = _find_name_column(gdf)
    print(f"Using '{name_col}' as comarca name column")

    names = gdf[name_col].str.strip().map(lambda n: NAME_MAP.get(n, n))
    df = pd.DataFrame({
        "comarca": names.values,
        "lat":     centroids_geo.y.values,
        "lon":     centroids_geo.x.values,
    })
    df = df.sort_values("comarca").reset_index(drop=True)
    return df


def check_name_match(centroids: pd.DataFrame, yield_csv: Path) -> None:
    """Print any comarca names that don't match between centroids and yield data."""
    yield_df = pd.read_csv(yield_csv)
    yield_names = set(yield_df["comarca"].unique())
    centroid_names = set(centroids["comarca"].unique())
    only_yield = yield_names - centroid_names
    only_centroids = centroid_names - yield_names
    if only_yield:
        print(f"\nIn yield but NOT in centroids ({len(only_yield)}):")
        for n in sorted(only_yield):
            print(f"  '{n}'")
    if only_centroids:
        print(f"\nIn centroids but NOT in yield ({len(only_centroids)}):")
        for n in sorted(only_centroids):
            print(f"  '{n}'")
    if not only_yield and not only_centroids:
        print("\nAll comarca names match between yield data and centroids.")


def save_polygons(out_path: Path) -> None:
    """
    Fetch comarca polygons and save as GeoPackage for use in spatial aggregation.
    Used by extract_climate.py to build the comarca→grid-cell mask.
    """
    gdf = fetch_comarques()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    name_col = _find_name_column(gdf)
    gdf = gdf.rename(columns={name_col: "comarca"})
    gdf["comarca"] = gdf["comarca"].str.strip().map(lambda n: NAME_MAP.get(n, n))
    gdf = gdf[["comarca", "geometry"]]
    gdf.to_file(out_path, driver="GPKG")
    print(f"Saved {len(gdf)} comarca polygons to {out_path}")


if __name__ == "__main__":
    df = get_comarca_centroids()
    out = DATA_DIR / "comarca_centroids.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} centroids to {out}")
    print(df.to_string(index=False))

    poly_out = DATA_DIR / "comarca_polygons.gpkg"
    save_polygons(poly_out)

    yield_csv = DATA_DIR / "catalan_woody_yield_raw.csv"
    if yield_csv.exists():
        check_name_match(df, yield_csv)
    else:
        print("\n(Run parse_yield.py first to check name matching)")
