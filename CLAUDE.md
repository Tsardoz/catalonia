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

For olive yield analysis using climate weighted by actual farm locations:

```bash
# Run complete pipeline (4 steps, ~3 minutes)
python src/run_olive_weighted_pipeline.py

# Or run steps individually:
python src/extract_climate_olive_farms_daily.py      # Farm-level daily climate
python src/aggregate_farms_to_comarca.py             # Area-weighted comarca averages
python src/aggregate_seasonal_olive_weighted.py      # Phenological windows
python src/join_dataset_olive_weighted.py            # Join with yield

# Compare with centroid approach
python src/compare_weighted_vs_centroid.py

# Create hybrid dataset (RECOMMENDED for final analysis)
python src/create_hybrid_olive_dataset.py
```

**Key differences:**
- Centroid: 36 comarques, 320 observations, all crops, uniform weighting
- Area-weighted: 23 comarques, 207 observations, olive only, farm area weighted, elevation-corrected
- **Hybrid (RECOMMENDED)**: 36 comarques, 320 observations, area-weighted where available + centroid fallback
- Correlation between methods: r = 0.93 for VPD

**For olive yield analysis, use:** `data/catalan_olive_yield_climate_hybrid.csv` (320 obs, best of both approaches)

**For olive daily timing analysis, use:** `data/agera5_daily_hybrid_olive.csv` (131,508 rows, 36 olive comarques)

See [HYBRID_DATASET.md](HYBRID_DATASET.md) and [OLIVE_WEIGHTED_PIPELINE.md](OLIVE_WEIGHTED_PIPELINE.md) for details

## Analysis Tools

```bash
# LPF (low-pass filter) feature extraction with top-5 scatter plots
python src/lpf_features.py

# Olive distributed-lag impulse response regression
python src/elasticnet_olive_lag.py
python src/elasticnet_olive_lag_hybrid.py

# Olive timing / threshold screening on hybrid daily climate
python src/olive_summer_tmax_threshold_screen.py
python src/olive_tmax_timing_scan.py
python src/olive_vpd_timing_scan.py
python src/olive_cwb_timing_scan.py
python src/olive_tmax_window_compare.py

# Publication-quality 4-panel plots (CLI)
python src/plot_features.py --crop apricot --var vpd_mean --feat lpf_peak --tau 30
python src/plot_features.py --list-crops
python src/plot_features.py --list-vars

# Interactive Streamlit dashboard
streamlit run src/explorer.py
```

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

- **Phenological windows**: Defined once in `aggregate_seasonal.py` and imported by all analysis scripts. Crop-specific windows (e.g., olive: flower [May-Jun], fruit_set [Jul-Aug], maturation [Sep-Nov]).

- **Causal filtering**: `lpf_features.py` applies IIR low-pass filters per comarca to capture temporal dynamics (τ = 3, 7, 14, 30 days). Extracts peak value, peak DOY, and peak width.

- **Fixed effects**: Comarca dummy variables absorb baseline yield differences in regression models (yield varies ~5× across comarcas for structural reasons).

- **Shared modules**: `features.py` provides filtering functions imported by `explorer.py` and `plot_features.py`. `aggregate_seasonal.py` exports `PHENO_WINDOWS` constant.

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
| agera5_daily_catalonia.csv | ~157K rows | **GITIGNORED** (regenerate with extract_climate.py) |
| agera5_seasonal_catalonia.csv | 1.9MB | Tracked |
| catalan_woody_yield_climate.csv | 809KB | Tracked (final joined dataset) |
| olive_groves_catalonia.gpkg | 12MB | Tracked (CORINE 2018 + DEM + corrected temps) |
| lpf_r2_table.csv | 76KB | Tracked |

## Documentation

- **WARP.md**: Complete pipeline documentation (data sources, variable definitions, spatial methodology, output schemas). **Single source of truth** for pipeline operations.
- **PLAN.md**: Current olive analysis documentation (hybrid dataset status, exploratory timing scans, current interpretation, next steps). Updated as analysis progresses.

Refer to these files for detailed specifications. They document not just what the code does but **why design decisions were made** (e.g., why 4-week bins, why openpyxl, why comarca fixed effects).

## Known Limitations

- **Centroid approximation**: Rainfed crops cluster on south-facing slopes, not valley floors where centroids often fall. Spatial aggregation (mean across multiple grid cells per comarca) partially mitigates this.
- **Fixed lapse rate**: Olive farm temperature correction uses constant -5.5°C/km; real lapse rates vary seasonally.
- **No true forecast validation**: ElasticNet 5-fold CV is used only as a rough comparative guide. OLS R² is in-sample and should not be interpreted as out-of-sample predictive skill.
- **Sample size**: 320 olive rows (36 comarques × 9 years) limits interaction detection. VPD × rainfall interactions likely require soil moisture modeling or larger datasets.

## Key Findings

From the current `PLAN.md` timing analysis:

1. **Timing matters more than whole-year bins for olive.** The clearest negative heat/dry-air signal appears in **late July to late August**, not uniformly across the agronomic year.

2. **`Tmax` hot-day counts are currently the cleanest olive screening variable.** Days with `Tmax >= 32°C` gave the strongest and most interpretable one-predictor timing result on the hybrid daily climate.

3. **`VPD >= 3.0 kPa` tells a similar timing story, but not a cleaner one.** VPD threshold-day scans broadly match the Tmax timing peak, while longest consecutive-run metrics are weaker than total counts.

4. **Full-year `P - ET0` is more useful as background context than as a clean summer timing driver.** It shows a plausible positive winter signal (late Dec / early Jan), but mixed-sign summer windows.
