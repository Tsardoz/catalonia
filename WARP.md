# Catalan Woody Crop Climate Dataset Pipeline

## Project Purpose

Assemble a comarca-level rainfed woody crop yield + climate dataset for Catalonia (2015–2024).
This is an intermediate fine-tune dataset between CropClimateX (US annual crops) and Tatura
(Australian perennial orchards) in the Tatura Transfer Strategy.

`PLAN.md` is the current working analysis document for the olive distributed-lag regression.
This file (`WARP.md`) documents the broader data pipeline and repository structure.

---

## Environment

- Python 3.12, venv at `.venv`
- Always activate: `source .venv/bin/activate`
- All code goes in `src/`
- All data goes in `data/` — raw source files are never modified

---

## Project Structure

```
catalonia/
  data/
    Produccions_comarcals_YYYY.xlsx   # raw Gencat yield files, 2015–2024
    crops_woody_en.xlsx               # pre-filtered woody crops, all years, English names
    comarca_centroids.csv             # computed from ICC WFS
    comarca_polygons.gpkg             # comarca polygon boundaries for spatial aggregation
    agera5_catalonia/                 # downloaded AgERA5 .nc files (variable_year.nc)
    olive_groves_catalonia.gpkg       # CORINE 2018 olive grove polygons with climate matching
    dem_catalonia.tif                 # Copernicus GLO-30 DEM mosaic for Catalonia
    srtm/                            # raw SRTM/Copernicus DEM tiles
    sigpac_datasets_dades_obertes.xlsx  # SIGPAC parcel registry open data (reference)
  src/
    parse_yield.py                    # Step 1 — yield parsing and filtering
    get_centroids.py                  # Step 2 — comarca centroid computation
    download_agera5.py                # Step 3 — AgERA5 download via CDS API
    extract_climate.py                # Step 3b — spatial aggregation from .nc per comarca polygon
    aggregate_seasonal.py             # Step 4 — seasonal weather aggregation
    join_dataset.py                   # Step 5 — join yield + climate
    plot_scatter.py                   # Step 6 — scatter plots
    elasticnet_olive_lag.py           # downstream olive distributed-lag analysis
  figures/
    olive_agera5_matching_topo.png    # map: olive farms → AgERA5 grid cells with topography
  PLAN.md
  WARP.md
```

---

## Data Sources

### Yield
- Gencat "Produccions comarcals" xlsx files, one per year, 2015–2024
- Sheet: `LLENYOSOS` (woody/perennial crops — NOT `RESTA HERBACIS` or `HORTA I FLOR`)
- **Always use openpyxl directly — never pandas `read_excel` on these files.**
  Pandas mishandles the merged cells in column A (comarca) and B (grup), causing comarca
  names to be lost or misassigned. openpyxl returns `None` for merged cell continuations,
  which we forward-fill manually. This is the correct approach.

### Weather
- **AgERA5** via Copernicus CDS API (`sis-agrometeorological-indicators`)
- AgERA5 is not raw ERA5 — it is a bias-corrected, gridded agrometeorological product at
  ~9 km resolution. ET0 is **pre-computed** in the dataset.
- Do **not** implement Hargreaves-Samani. ET0 is from AgERA5 directly.
- **VPD is computed in `extract_climate.py`** from the downloaded `vapour_pressure` (ea, hPa)
  and `2m_temperature` max (K), using the FAO-56 Tetens formula:
  `es(Tmax) = 0.6108 · exp(17.27 · Tmax_C / (Tmax_C + 237.3))` → `VPD = max(es - ea, 0)` (kPa).
  The CDS API rejects `vapour_pressure_deficit_at_daily_maximum_temperature` requests.
  `version="2_0"` is required for ET0, precipitation, and temperature; `vapour_pressure`
  requires `version="1_0"` (with v2.0 fallback for 2023–2024). See AgERA5 Variables.

### Comarca boundaries
- Derived programmatically from the ICC (Institut Cartogràfic i Geològic de Catalunya)
  public WFS or equivalent open data endpoint. Do not hardcode centroids.
- 42 comarques total in Catalonia; 41 appear in the rainfed yield data.
  Alta Ribagorça (high mountain) and Lluçanès (new small comarca) have no rainfed
  records for any target crop across 2015–2024.

### Olive sample coverage (comarca × year)
- **36 comarcas** report olive yield at least once across 2015–2024 (out of 41 in the dataset).
- **354 total olive rows** (comarca × year observations).
- Panel is slightly unbalanced in early years: 34–35 comarcas in 2015–2018, rising to a
  stable **36 comarcas from 2019–2024**.

| Year | Comarcas |
|------|----------|
| 2015 | 34 |
| 2016 | 35 |
| 2017 | 35 |
| 2018 | 34 |
| 2019 | 36 |
| 2020 | 36 |
| 2021 | 36 |
| 2022 | 36 |
| 2023 | 36 |
| 2024 | 36 |

### Parcel registry (reference only)
- `data/sigpac_datasets_dades_obertes.xlsx` — SIGPAC open-data export of agricultural
  parcel declarations in Catalonia. Not used directly in the pipeline but useful for
  cross-checking rainfed crop area and parcel distribution per comarca.

---

## AgERA5 Variables

Download these variables, all years 2015–2024, Catalonia bounding box
`[North=42.51, West=-0.09, South=40.31, East=3.19]`:

```python
# No statistic required. version="2_0" required.
SINGLE_VARIABLES = [
    "reference_evapotranspiration",  # FAO-56 Penman-Monteith ET0 (mm/day)
    "precipitation_flux",            # precipitation (mm/day)
]

# Requires statistic="24_hour_mean".
# version="1_0" used; auto-falls back to "2_0" for years not covered by v1.0
# (vapour_pressure v1.0 cuts off mid-2023; v2.0 used for 2023 full year and 2024)
MEAN_VARIABLES = [
    "vapour_pressure",  # actual vapour pressure ea (hPa)
]

# Requires statistic + version="2_0".
TEMP_STATISTICS = {
    "24_hour_maximum": "max",  # daily max temp → tmax
    "24_hour_minimum": "min",  # daily min temp → tmin
}
```

CDS API notes (determined empirically):
- `vapour_pressure_deficit_*` variables rejected by API despite being in docs
- `version="2_0"` required for `reference_evapotranspiration`, `precipitation_flux`, `2m_temperature`
- `vapour_pressure` uses `version="1_0"` + `statistic="24_hour_mean"`; v1.0 cuts off mid-2023
- `vapour_pressure` 2023 and 2024 require `version="2_0"` (handled automatically by fallback)
- Temperature stat names: `24_hour_maximum` / `24_hour_minimum` (not `Max-Day-Time`)

Download structure (50 files total: 5 variables × 10 years):
- `agera5_catalonia/reference_evapotranspiration_{year}.nc`
- `agera5_catalonia/precipitation_flux_{year}.nc`
- `agera5_catalonia/vapour_pressure_{year}.nc`
- `agera5_catalonia/2m_temperature_max_{year}.nc`
- `agera5_catalonia/2m_temperature_min_{year}.nc`

---

## Pipeline Steps

### Step 1 — Parse and Filter Yield (`src/parse_yield.py`)

- Read all 10 `Produccions_comarcals_YYYY.xlsx` files using **openpyxl**
- Sheet: `LLENYOSOS`
- Forward-fill `comarca` (col 0) and `grup` (col 1) on `None` values
- Stack into single long-format DataFrame
- Columns: `year`, `comarca`, `grup`, `crop_catalan`, `seca_ha`, `seca_kg_ha`
- Filter: keep only rows where `seca_ha > 0` AND `seca_kg_ha > 0`
- Compute: `yield_tha = seca_kg_ha / 1000` (kg/ha → t/ha; area-normalised)
- **Do not use `PRODUCCIO_TOTAL_T`** — there is no rainfed-specific total column in the source

#### Target crops

| crop_catalan | crop_en | crop_group |
|---|---|---|
| Pomera | Apple | pome_fruit |
| Perera | Pear | pome_fruit |
| Presseguer | Peach | stone_fruit |
| Pruner | Plum | stone_fruit |
| Albercoquer | Apricot | stone_fruit |
| Nectariner | Nectarine | stone_fruit |
| Prèssec plà | Flat peach | stone_fruit |
| Platerina | Flat nectarine | stone_fruit |
| Cirerer i guinder | Cherry and sour cherry | stone_fruit |
| Ametller | Almond | nut |
| Avellaner | Hazelnut | nut |
| Noguera | Walnut | nut |
| Vinya de raïm per a vi | Wine grape | vine |
| Vinya de raïm per a panses | Raisin grape | vine |
| Vinya de raïm per a  panses | Raisin grape | vine |
| Olivera per a oliva d'oli | Oil olive | olive |
| Magraner | Pomegranate | other |
| Caqui | Persimmon | other |
| Nashi | Asian pear | other |

Note: `Vinya de raïm per a panses` appears with double-space in some years — handle both.

Output: `data/catalan_woody_yield_raw.csv`

---

### Step 2 — Comarca Centroids (`src/get_centroids.py`)

- Fetch comarca boundary polygons from ICC public endpoint (WFS or GeoJSON)
- Compute centroid of each polygon
- Write `data/comarca_centroids.csv` with columns: `comarca`, `lat`, `lon`
- Also write `data/comarca_polygons.gpkg` for downstream spatial aggregation
- 42 comarques expected
- Outputs: `data/comarca_centroids.csv`, `data/comarca_polygons.gpkg`

---

### Step 3 — Download AgERA5 (`src/download_agera5.py`)

- CDS API client (`cdsapi`)
- Skip files that already exist (resume-safe)
- One `.nc` per variable per year
- See AgERA5 Variables section above

---

### Step 3b — Extract Climate per Comarca (`src/extract_climate.py`)

- Requires `data/comarca_polygons.gpkg` (from `get_centroids.py`)
- For each comarca, identifies all AgERA5 grid cells (~0.1°, ~11km) whose centres fall
  within the comarca polygon via a `geopandas` spatial join.
  Fallback to nearest cell for very small comarques.
- Grid: 22 lats × 32 lons = 704 cells over Catalonia bounding box.
  328 cells assigned to comarques; 376 are sea or outside borders.
  Cell counts range from 1 (Barcelonès, Garraf, Aran — fallback) to 19 (Noguera).
  3 fallback comarques: Aran (narrow Pyrenean valley), Barcelonès (tiny urban), Garraf (small coastal).
- Extracts the full (time, n_cells) array per variable per comarca, then computes
  spatial **mean, min, max** across those cells for each day.
- VPD is computed per cell first (FAO-56 Tetens from tmax + ea), then aggregated.
- Output columns: `date`, `comarca`,
  `et0_mean/min/max`, `ea_mean/min/max`,
  `tmax_mean/min/max`, `tmin_mean/min/max`,
  `vpd_mean/min/max`,
  `precip_mean/min/max` (if `precipitation_flux` downloaded)
- Output: `data/agera5_daily_catalonia.csv`

---

### Step 4 — Seasonal Aggregation (`src/aggregate_seasonal.py`)  <!-- was mislabelled Step 5 -->

For each `comarca × crop × year`, compute weather summaries over phenological periods.

#### Phenological windows (approximate, calendar-based)

| crop_group | period | months |
|---|---|---|
| stone_fruit (cherry) | pre_flower | Jan–Feb |
| stone_fruit (cherry) | flower_set | Mar–Apr |
| stone_fruit (cherry) | fruit_dev | May–Jun |
| stone_fruit (other) | pre_flower | Feb–Mar |
| stone_fruit (other) | fruit_set | Apr–May |
| stone_fruit (other) | fruit_fill | Jun–Aug |
| pome_fruit | pre_flower | Mar–Apr |
| pome_fruit | cell_div | May–Jun |
| pome_fruit | cell_exp | Jul–Sep |
| apricot | flower | Feb–Mar |
| apricot | fruit_set | Apr–May |
| apricot | harvest | Jun–Jul |
| almond | flower | Feb–Mar |
| almond | kernel_fill | Apr–Jun |
| almond | maturation | Jul–Aug |
| hazelnut | flower | Feb–Mar |
| hazelnut | fill | Apr–Jul |
| hazelnut | maturation | Aug–Sep |
| walnut | flower | Apr–May |
| walnut | fill | Jun–Aug |
| walnut | maturation | Sep–Oct |
| vine | bud_break | Mar–Apr |
| vine | flower_veraison | May–Jul |
| vine | veraison_harvest | Aug–Sep |
| olive | flower | May–Jun |
| olive | fruit_set | Jul–Aug |
| olive | maturation | Sep–Nov |
| other | season | Apr–Sep |

#### Aggregates per period
- `mean_vpd` — mean daily VPD at Tmax (kPa)
- `max_vpd` — mean of top-10% daily VPD values (stress peaks)
- `n_days_vpd_gt2` — count of days VPD > 2 kPa
- `cum_et0` — cumulative ET0 (mm)
- `mean_tmax` — mean daily maximum temperature (°C)

Output: `data/agera5_seasonal_catalonia.csv`

---

### Step 5 — Join Yield and Climate (`src/join_dataset.py`)  <!-- was mislabelled Step 6 -->

- Join `catalan_woody_yield_raw.csv` × `agera5_seasonal_catalonia.csv` on `comarca` + `year`
- Final columns: `comarca`, `year`, `crop_catalan`, `crop_group`, `pheno_key`, `seca_ha`, `yield_tha`,
  then all `{period}_{aggregate}` columns
- Columns `crop_en`, `seca_kg_ha`, `grup` are dropped from the yield data at this stage
- Output: `data/catalan_woody_yield_climate.csv`

---

### Step 6 — Scatter Plots (`src/plot_scatter.py`)  <!-- was mislabelled Step 7 -->

#### Cherry period analysis (4-panel)
- Rows: VPD, ET0
- Columns: period (pre-flower, flower-set, fruit-dev, full-season mean)
- x-axis: climate variable (kPa or mm), y-axis: yield_tha
- Points = comarca-year observations, colour = year
- OLS fit line + R²

#### Cross-crop comparison (2-panel)
- Panel A: Catalan cherry fruit_dev VPD vs yield_tha
- Panel B: CropClimateX soybean pod-fill VPD vs yield (matched methodology)

Formatting:
- Physical axis labels (kPa, t/ha, mm)
- No log transform unless distributions require it
- Output: `figures/cherry_period_scatter.png`, `figures/cross_crop_vpd_yield.png`

---

## Olive Farm Spatial Analysis

### Olive grove locations
- Source: CORINE Land Cover 2018, class 223 (Olive groves), downloaded from EEA ArcGIS REST API
- 482 polygons after clipping to Catalonia boundary, ~1,237 km² total
- Stored in `data/olive_groves_catalonia.gpkg`
- CORINE area (~123,700 ha) is ~18% larger than DARP productive area (~105k ha in 2018),
  consistent with CORINE including non-productive olive land cover

### Yield area consistency (DARP spreadsheets 2015–2024)
- Total olive area stable 2015–2022 (~103–108k ha, year-on-year changes ≤2.2%)
- Sharp ~20% drop in 2023 (to ~82k ha) persists in 2024 — likely methodological change
- Irrigated share rising steadily: 16.7% (2015) → 27.4% (2024)
- Oil olives dominate (>99% of area); table olives negligible

### Elevation tagging
- Copernicus GLO-30 DEM downloaded from AWS (10 tiles merged into `data/dem_catalonia.tif`)
- Zonal stats per polygon: mean, min, max, std elevation
- Farm elevation range: 7–777 m, median 264 m, most below 500 m
- Within-polygon relief typically ~21 m std (polygons are fairly flat)

### AgERA5 grid cell matching
- Each olive polygon centroid matched to nearest AgERA5 0.1° grid cell
- Each polygon also assigned to its comarca via spatial join
- Comparison of nearest-cell vs comarca-average Tmax (2024):
  - 80% of farms differ by <1°C from comarca average
  - 20% differ by >1°C, mainly in topographically varied comarcas
    (Pallars Jussà +1.6°C, Anoia +1.0°C, Noguera +0.8°C)
  - Flat comarcas (Ribera d'Ebre, Terra Alta) show <0.1°C difference

### Elevation-based temperature correction
- Grid cell mean DEM elevation computed per AgERA5 cell (zonal stats over 0.1° boxes)
- Elevation delta = farm elevation − grid cell elevation
  - Mean: −19 m, std: 91 m, range: −298 to +254 m
  - 27% of farms have |delta| > 100 m; 5% have |delta| > 200 m
- Lapse rate correction: T_corrected = T_agera5 + (−5.5°C/km) × (elev_delta / 1000)
  - Correction range: −1.40 to +1.64°C
  - VP correction negligible (~3% max); not applied
- Raw and corrected Tmax/Tmin extracted for all years 2015–2024

### GeoPackage columns (`olive_groves_catalonia.gpkg`)
- Location: `centroid_lon`, `centroid_lat`, `comarca`
- Elevation: `elev_mean`, `elev_min`, `elev_max`, `elev_std`
- Grid matching: `grid_lat`, `grid_lon`, `grid_lat_idx`, `grid_lon_idx`, `grid_elev`
- Corrections: `elev_delta`, `temp_correction`
- Climate: `tmax_raw_YYYY`, `tmax_cor_YYYY`, `tmin_raw_YYYY`, `tmin_cor_YYYY` (2015–2024)
- Summaries: `tmax_raw_mean`, `tmax_cor_mean`, `tmin_raw_mean`, `tmin_cor_mean`

---

## Known Limitations

- ERA5/AgERA5 comarca centroids do not represent rainfed parcel conditions — rainfed tree
  crops cluster on south-facing slopes in interior comarques, not valley floors where
  centroids often fall. Consistent with CropClimateX methodology (county centroid weather).
- Yield data is a comarca aggregate — single representative weather point is always an
  approximation.
- Hargreaves-Samani is not used. ET0 is FAO-56 Penman-Monteith from AgERA5, which is more
  accurate but still a gridded estimate, not a station measurement.
- VPD is computed using FAO-56 Tetens: es(Tmax) - ea, where ea is the 24h mean actual
  vapour pressure from AgERA5 `vapour_pressure`. Column names: `vpd_mean`, `vpd_min`, `vpd_max`.
- Spatial aggregation (mean/min/max) is over comarca polygon cells, not a single centroid.
  3 small comarques (Aran, Barcelonès, Garraf) fall back to nearest cell.
- Olive farm elevation correction uses a fixed lapse rate (−5.5°C/km). In practice lapse
  rates vary seasonally (lower in winter, higher in summer) and by weather regime.
  Dutra et al. (2020) found −4.5°C/km for ERA5 in western US; Mediterranean values may differ.
- CORINE 2018 olive polygons are a single-year snapshot. Olive area is effectively static
  (perennial crop, <2% year-on-year change 2015–2022) so this is acceptable.

---

## Output Files

| File | Description |
|---|---|
| `data/catalan_woody_yield_raw.csv` | Parsed, filtered, rainfed-only yield data |
| `data/comarca_centroids.csv` | Comarca name, lat, lon |
| `data/comarca_polygons.gpkg` | Comarca polygon boundaries (GeoPackage) |
| `data/agera5_daily_catalonia.csv` | Daily AgERA5 weather per comarca (spatial mean/min/max) |
| `data/agera5_seasonal_catalonia.csv` | Seasonal aggregates per comarca × crop × year |
| `data/catalan_woody_yield_climate.csv` | Final joined dataset |
| `data/olive_groves_catalonia.gpkg` | Olive farm polygons with elevation, grid matching, corrected temperatures |
| `data/dem_catalonia.tif` | Merged Copernicus GLO-30 DEM for Catalonia |
| `figures/cherry_period_scatter.png` | 4-panel cherry period scatter |
| `figures/cross_crop_vpd_yield.png` | Cross-crop VPD-yield comparison |
| `figures/olive_agera5_matching_topo.png` | Olive farms matched to AgERA5 grid cells with topographic contours |
