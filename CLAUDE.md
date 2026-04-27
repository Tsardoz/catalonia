# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a PhD research project that assembles comarca-level rainfed woody crop yield and climate data for Catalonia (2015-2024). It investigates how climate variables (VPD, precipitation, temperature, ET0) affect perennial crop yields across phenological stages. This dataset serves as an intermediate fine-tuning dataset between CropClimateX (US annual crops) and Tatura (Australian perennial orchards).

Repository: https://github.com/Tsardoz/catalonia.git

## Environment Setup

```bash
# Always activate the virtual environment first
source .venv/bin/activate
```

Python 3.12 with ~79 scientific packages (pandas, numpy, xarray, geopandas, scikit-learn, statsmodels, streamlit).

## Running the Pipeline

### Standard Pipeline (Centroid-Based, All Crops)

Execute these steps sequentially:

```bash
# Step 1: Parse rainfed woody crop yield from Excel files
python src/parse_yield.py

# Step 2: Fetch comarca boundaries and compute centroids
python src/get_centroids.py

# Step 3: Download AgERA5 climate data (requires CDS API credentials)
python src/download_agera5.py

# Step 3b: Extract daily climate per comarca with spatial aggregation
python src/extract_climate.py

# Step 4: Aggregate into phenological windows
python src/aggregate_seasonal.py

# Step 5: Join yield + climate into final dataset
python src/join_dataset.py
```

### Area-Weighted Pipeline (Olive-Specific)

For olive yield analysis, the unweighted comarca mean is replaced by a single-step
area-weighted aggregator that uses DUN 2024 olive area per AgERA5 cell as weights:

```bash
# Olive-area-weighted daily climate (36 olive comarques, 131,508 rows)
python src/extract_climate_olive_weighted.py

# Optional residual lapse-rate correction on top of the weighted file
python src/elevation_correction.py --olive-weighted --apply --plot
```

**For mixed-cultivar olive panel analysis (rainfed-80 robustness check):** use
`data/agera5_daily_catalonia_oliveweighted.csv` with `comarca_olive_whitelist.csv`.

**For headline cultivar-restricted Arbequina analysis:** use the **unweighted**
`data/agera5_daily_catalonia.csv` with `data/dun/comarca_arbequina_whitelist.csv`. The
olive-area-weighted file weights cells by total DUN olive area (rainfed + irrigated), which
biases the climate signal toward irrigated valley cells in Garrigues and Segrià.

The previous hybrid pipeline (`*_hybrid_olive*` files, `extract_climate_olive_farms_daily.py`,
`run_olive_weighted_pipeline.py`, `create_hybrid_olive_dataset.py`, etc.) has been retired
and is preserved under `archive/src/hybrid/` and `archive/src/olive_weighted/`. The current
pipeline is single-source area-weighted, not a two-source hybrid.

## Analysis Tools

```bash
# Sliding-window OLS screen — headline Arbequina spec (see PLAN.md)
python src/sliding_window_regression.py --crop olive --var tmax_mean \
    --agg mean --window 21 --scan-start 06-01 --scan-end 09-15 \
    --comarca-whitelist data/dun/comarca_arbequina_whitelist.csv \
    --tag arbequina    # uses unweighted agera5_daily_catalonia.csv by default

# Mixed-cultivar rainfed-80 robustness check (older canonical spec)
python src/sliding_window_regression.py --crop olive --var tmax_mean \
    --threshold 32 --agg count --window 14 --scan-start 06-01 --scan-end 09-15 \
    --comarca-whitelist data/dun/comarca_olive_whitelist.csv \
    --climate-csv data/agera5_daily_catalonia_oliveweighted.csv --tag oliveweighted

# First-difference variant (no FE; eliminates biennial bearing and slow drift)
python src/sliding_window_regression_fd.py --crop olive --var tmax_mean \
    --agg mean --window 21 --scan-start 06-01 --scan-end 09-15 \
    --comarca-whitelist data/dun/comarca_arbequina_whitelist.csv --tag arbequina

# Olive distributed-lag impulse response regression
python src/elasticnet_olive_lag.py

# DUN rainfed-fraction whitelist (used to build comarca_olive_whitelist.csv)
python src/dun_rainfed_fraction.py

# Visual check: olive DUN parcels vs AgERA5 grid cells per comarca
python src/plot_olive_dun_grid_map.py
```

Exploratory tooling (LPF features, Streamlit explorer, per-scan screens, hybrid
regressions, scatter plots) lives under `archive/src/exploratory/`,
`archive/src/scans_screens/`, and `archive/src/hybrid/` and is no longer part of
the active pipeline.

## Architecture

### Data Flow

```
Excel yield files + AgERA5 NetCDF + ICC boundaries
  ↓
parse_yield.py → catalan_woody_yield_raw.csv
  ↓
get_centroids.py → comarca_centroids.csv, comarca_polygons.gpkg
  ↓
download_agera5.py → agera5_catalonia/*.nc (50 files)
  ↓
extract_climate.py → agera5_daily_catalonia.csv (~157K rows, GITIGNORED)
  ↓
aggregate_seasonal.py → agera5_seasonal_catalonia.csv
  ↓
join_dataset.py → catalan_woody_yield_climate.csv (final dataset)
  ↓
Analysis scripts → figures/*.png
```

### Key Components

- **Spatial aggregation**: Each comarca gets multiple AgERA5 grid cells (~0.1° resolution) with spatial mean/min/max, not just centroid lookup. 328/704 grid cells assigned to 42 comarques. 3 small comarques (Aran, Barcelonès, Garraf) fall back to nearest cell.

- **Phenological windows**: Defined once in `aggregate_seasonal.py` (`PHENO_WINDOWS` constant). Crop-specific windows (e.g., olive: flower [May-Jun], fruit_set [Jul-Aug], maturation [Sep-Nov]).

- **Olive area weighting**: `extract_climate_olive_weighted.py` aggregates AgERA5 cells per comarca using DUN 2024 olive area per cell as weights (`data/dun/olive_dun_grid_cells_2024.csv`). This replaces the unweighted comarca mean for olive analysis.

- **Sliding-window screening**: `sliding_window_regression.py` is the current primary screening tool. It runs one-predictor OLS with comarca fixed effects across a scan of climate-window endpoints to identify vulnerable timing windows. Reports both total R² and within-R² (the share of variance explained by climate after stripping out comarca FEs). `sliding_window_regression_fd.py` runs the same scan in first differences (Δlog-yield on Δclimate) as a no-FE robustness check.

- **Cultivar restriction**: For the headline Arbequina analysis, the panel is restricted to 10 comarques where Arbequina is ≥ 90 % of rainfed olive area and rainfed Arbequina ≥ 500 ha (DUN 2024 cultivar shares; whitelist at `data/dun/comarca_arbequina_whitelist.csv`). This restriction tripled the within-R² of the canonical Tmax timing regression despite a 43 % drop in n, by removing cross-cultivar heterogeneity.

- **Fixed effects**: Comarca dummy variables absorb baseline yield differences in regression models (yield varies ~5× across comarcas for structural reasons).

## Critical Implementation Details

### Excel Parsing

**ALWAYS use openpyxl directly, NEVER pandas.read_excel()** on Gencat yield files. Pandas mishandles merged cells in columns A-B, causing comarca names to be lost. Example:

```python
import openpyxl
wb = openpyxl.load_workbook(path)
ws = wb['LLENYOSOS']
# Forward-fill None values in comarca/grup columns manually
```

### VPD Computation

VPD is **not available** from the CDS API despite being in the docs. Compute it manually in `extract_climate.py` using FAO-56 Tetens formula:

```python
# es(Tmax) in kPa
es_tmax = 0.6108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))
# VPD = max(es - ea, 0)
vpd = np.maximum(es_tmax - ea_kpa, 0)
```

where `ea` is actual vapour pressure from AgERA5 (converted from hPa to kPa) and `tmax_c` is max temperature in Celsius.

### AgERA5 Variables

Download these 5 variables for 2015-2024 (50 files total):

- `reference_evapotranspiration` (ET0, FAO-56 Penman-Monteith) - version="2_0", no statistic
- `precipitation_flux` - version="2_0", no statistic
- `vapour_pressure` (ea) - version="1_0" with statistic="24_hour_mean"; auto-fallback to version="2_0" for 2023-2024
- `2m_temperature` with statistic="24_hour_maximum" - version="2_0"
- `2m_temperature` with statistic="24_hour_minimum" - version="2_0"

**Do not implement Hargreaves-Samani**: ET0 is pre-computed in AgERA5.

### Agronomic Year

Olive yield in year Y is driven by climate from Dec 1 (Y-1) through Nov 30 (Y). This is the standard agronomic calendar for Mediterranean perennials (flowering May-Jun, harvest Oct-Nov).

### Target Crops

19 rainfed woody crops across 6 groups:
- pome_fruit: apple, pear
- stone_fruit: peach, plum, apricot, nectarine, flat peach, flat nectarine, cherry
- nut: almond, hazelnut, walnut
- vine: wine grape, raisin grape
- olive: oil olive
- other: pomegranate, persimmon, asian pear

Filter criterion: `seca_ha > 0 AND seca_kg_ha > 0` (rainfed area and yield both positive).

## Key Data Files

| File | Size | Status |
|------|------|--------|
| catalan_woody_yield_raw.csv | 121KB | Tracked |
| comarca_centroids.csv | 2KB | Tracked |
| comarca_polygons.gpkg | 11MB | Tracked |
| agera5_daily_catalonia.csv | 52MB (~157K rows) | **GITIGNORED** (regenerate with extract_climate.py) |
| agera5_daily_catalonia_elevcor.csv | 73MB | **GITIGNORED** (regenerate with elevation_correction.py) |
| agera5_daily_catalonia_oliveweighted.csv | 44MB (131,508 rows, 36 olive comarques) | **GITIGNORED** (regenerate with extract_climate_olive_weighted.py) |
| agera5_daily_catalonia_oliveweighted_elevcor.csv | 64MB | **GITIGNORED** (regenerate with elevation_correction.py --olive-weighted) |
| agera5_seasonal_catalonia.csv | 1.9MB | Tracked |
| catalan_woody_yield_climate.csv | 809KB | Tracked (final joined dataset) |
| olive_groves_catalonia.gpkg | 12MB | Tracked (CORINE 2018 + DEM + corrected temps) |
| dun/olive_dun_grid_cells_2024.csv | 21KB | Tracked (DUN olive area per AgERA5 cell, used as weights) |
| dun/comarca_olive_whitelist.csv | <1KB | Tracked (18 comarques where olive is ≥80% rainfed-area) |
| dun/comarca_arbequina_whitelist.csv | <1KB | Tracked (10 comarques where Arbequina is ≥90% of rainfed olive area AND ≥500 ha; headline cultivar subset) |
| dun/comarca_olive_rainfed_fraction.csv | 3KB | Tracked (DUN rainfed-fraction per comarca) |

## Documentation

- **WARP.md**: Complete pipeline documentation (data sources, variable definitions, spatial methodology, output schemas). **Single source of truth** for pipeline operations.
- **PLAN.md**: Current olive analysis documentation (area-weighted daily climate, sliding-window timing results, current interpretation, next steps). Updated as analysis progresses.

Refer to these files for detailed specifications. They document not just what the code does but **why design decisions were made** (e.g., why 4-week bins, why openpyxl, why comarca fixed effects).

## Known Limitations

- **Centroid approximation**: Rainfed crops cluster on south-facing slopes, not valley floors where centroids often fall. Spatial aggregation (mean across multiple grid cells per comarca) partially mitigates this.
- **Fixed lapse rate**: Olive farm temperature correction uses constant -5.5°C/km; real lapse rates vary seasonally.
- **No true forecast validation**: ElasticNet 5-fold CV is used only as a rough comparative guide. OLS R² is in-sample and should not be interpreted as out-of-sample predictive skill.
- **Sample size**: 320 olive rows (36 comarques × 9 years) limits interaction detection. VPD × rainfall interactions likely require soil moisture modeling or larger datasets.

## Key Findings

From the current `PLAN.md` timing analysis:

1. **Cultivar restriction is the largest single source of signal-to-noise improvement.** Restricting the olive panel to 10 Arbequina-dominant comarques (Arbequina ≥ 90 % of rainfed olive area, ≥ 500 ha rainfed) raised within-R² from ~0.09 (mixed-cultivar rainfed-80, n=174) to ~0.32 (Arbequina-only, n=100) on the headline summer Tmax regression, despite a 43 % drop in n.

2. **Timing converges on mid-to-late August for Arbequina.** The 14-, 21-, and 28-day mean-Tmax windows all peak with window-end between 17 and 24 August. The 21-day window ending 17 August (covering ~28 July → 17 August) is the recommended primary specification: t = −6.51, p = 4.3e-9, within-R² = 0.32.

3. **Continuous `mean Tmax` outperforms threshold counts on the cultivar-restricted set.** At the same 14-day window, `mean Tmax` gives within-R² = 0.23 versus 0.12 for `Tmax ≥ 32 °C count`. The threshold operator was discarding usable temperature information once cross-cultivar heterogeneity was removed.

4. **`mean VPD` agrees on the same vulnerable period but is weaker than `mean Tmax`.** Temperature itself, not VPD, is the dominant one-predictor signal on the Arbequina panel.

5. **Full-year `P - ET0` is more useful as background context than as a clean summer timing driver.** It shows a plausible positive winter signal (late Dec / early Jan), but mixed-sign summer windows.
