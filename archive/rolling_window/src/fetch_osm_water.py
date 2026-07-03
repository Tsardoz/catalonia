"""
fetch_osm_water.py
Query the OSM Overpass API for rivers/canals (waterway) and water bodies
(natural=water, including reservoirs) over the Catalonia bbox, parse the
inlined geometry and cache as GeoPackages under data/osm/.

Usage:
  python src/fetch_osm_water.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "osm"
RIVERS_GPKG = OUT_DIR / "catalonia_rivers.gpkg"
WATER_GPKG = OUT_DIR / "catalonia_water.gpkg"

OVERPASS = "https://overpass-api.de/api/interpreter"
BBOX = (40.4, 0.0, 42.9, 3.4)  # south, west, north, east
BBOX_STR = ",".join(f"{v}" for v in BBOX)
HEADERS = {"User-Agent": "catalonia-research/0.1 (academic; contact: phd)"}

QUERY = f"""
[out:json][timeout:180];
(
  way["waterway"~"^(river|canal)$"]({BBOX_STR});
  way["natural"="water"]({BBOX_STR});
  relation["natural"="water"]({BBOX_STR});
);
out geom;
"""


def fetch() -> dict:
    print(f"POST {OVERPASS} (bbox={BBOX_STR}) ...")
    r = requests.post(OVERPASS, data={"data": QUERY}, timeout=300, headers=HEADERS)
    if r.status_code != 200:
        print(f"  status {r.status_code}; first 400 chars of body:\n{r.text[:400]}")
    r.raise_for_status()
    js = r.json()
    print(f"  {len(js.get('elements', []))} elements")
    return js


def way_coords(el: dict) -> list[tuple[float, float]]:
    return [(p["lon"], p["lat"]) for p in el.get("geometry", [])]


def parse_rivers(elements: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        wt = tags.get("waterway")
        if wt not in ("river", "canal"):
            continue
        coords = way_coords(el)
        if len(coords) < 2:
            continue
        rows.append({"id": el["id"], "waterway": wt,
                     "name": tags.get("name"), "geometry": LineString(coords)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def parse_water(elements: list[dict]) -> gpd.GeoDataFrame:
    polys: list[tuple[Polygon, dict]] = []
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        if tags.get("natural") != "water":
            continue
        coords = way_coords(el)
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        try:
            p = Polygon(coords)
            if p.is_valid and not p.is_empty:
                polys.append((p, {"id": el["id"], "name": tags.get("name"),
                                  "water": tags.get("water"), "src": "way"}))
        except Exception:
            continue
    for el in elements:
        if el.get("type") != "relation":
            continue
        tags = el.get("tags", {})
        if tags.get("natural") != "water":
            continue
        outer_lines = []
        for m in el.get("members", []):
            if m.get("role") != "outer" or m.get("type") != "way":
                continue
            coords = [(p["lon"], p["lat"]) for p in m.get("geometry", [])]
            if len(coords) >= 2:
                outer_lines.append(LineString(coords))
        if not outer_lines:
            continue
        merged = unary_union(outer_lines)
        for poly in polygonize(merged):
            if poly.is_valid and not poly.is_empty:
                polys.append((poly, {"id": el["id"], "name": tags.get("name"),
                                     "water": tags.get("water"), "src": "rel"}))
    rows = [{**props, "geometry": geom} for geom, props in polys]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    js = fetch()
    elements = js.get("elements", [])

    rivers = parse_rivers(elements)
    print(f"Rivers/canals: {len(rivers)} linestrings")
    rivers.to_file(RIVERS_GPKG, driver="GPKG")
    print(f"  wrote {RIVERS_GPKG}")

    water = parse_water(elements)
    print(f"Water bodies: {len(water)} polygons "
          f"({(water['src']=='way').sum()} ways, {(water['src']=='rel').sum()} relations)")
    water["area_km2"] = water.to_crs("EPSG:25831").area / 1e6
    print("  largest 10 by area:")
    print(water.sort_values("area_km2", ascending=False)
                .loc[:, ["name", "water", "area_km2"]].head(10).to_string(index=False))
    water.to_file(WATER_GPKG, driver="GPKG")
    print(f"  wrote {WATER_GPKG}")


if __name__ == "__main__":
    main()
