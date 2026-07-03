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

## Olive Water Sensitivity Pipeline

The current olive analysis uses a hierarchical Bayesian scalar-on-function regression
(see `olive_water_sensitivity_plan.md` for the full design). The literature-derived
GDD axis has been retired; the current pipeline uses fixed calendar windows anchored
to observed Arbequina phenology in Garrido et al. 2021. Execute steps sequentially:

```bash
# Step 1: Load cohort-filtered yield + daily climate, compute lag yield
python src/01_load_data.py --cohort both

# Step 2: Prepare the fixed calendar DOY phenology axis
python src/02_prepare_doy_axis.py

# Step 3: Build WD (P − ET0) and VPD predictor matrices on the calendar DOY axis
python src/03_compute_water_deficit.py

# Step 4: Write calendar-window phenological anchors (S5–S8)
python src/04_align_phenology.py

# Step 5: Construct B-spline basis and pre-compute reduced functional predictors on the DOY grid
python src/05_build_basis.py             # --n-basis 4/5/6/8 for sensitivity

# Step 6: Bambi hierarchical sanity check (stage-window aggregated predictors)
python src/06_fit_bambi_sanity.py --cohort both --predictor wd
```

Steps 7–10 (`07_fit_numpyro_main.py` through `10_sensitivity_analysis.py`) are
implemented and have been run on the calendar DOY axis. Step 10 is partial
(Arbequina CWB-state sweep complete; VPD and all-olive sweeps pending). The
retired GDD-axis artifacts have been deleted from `results/`.

### Supporting tools

```bash
# DUN rainfed-fraction whitelist (used to build comarca_olive_whitelist.csv)
python src/dun_rainfed_fraction.py

# DUN cultivar-share whitelist for a given crop / regime / cultivar
python src/dun_cultivar_fraction.py --crop olive --regime S --cultivar ARBEQUINA \
    --campaign 2024 --threshold 0.90 --min-cult-ha 500 --input data/dun/olives.csv
```

Prior sliding-window OLS and elastic-net analyses have been retired to
`archive/rolling_window/` and `archive/src/`. They suffered from window-overlap
artifacts and are superseded by the Bayesian scalar-on-function approach.

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

- **Scalar-on-function regression**: The current olive analysis (`src/01_load_data.py` through `src/10_sensitivity_analysis.py`) estimates phenological-stage-specific water sensitivity via a hierarchical Bayesian model with B-spline basis on a fixed calendar day-of-year axis (Arbequina windows from Garrido et al. 2021; the older GDD axis is retired). See `olive_water_sensitivity_plan.md` for the full design.

- **Cultivar restriction**: For the headline Arbequina analysis, the panel is restricted to 10 comarques where Arbequina is ≥ 90 % of rainfed olive area and rainfed Arbequina ≥ 500 ha (DUN 2024 cultivar shares; whitelist at `data/dun/comarca_arbequina_whitelist.csv`). This restriction tripled the within-R² of the canonical Tmax timing regression despite a 43 % drop in n, by removing cross-cultivar heterogeneity.

- **Within-estimator centering**: Step 1 subtracts per-(cohort, comarca) means from `yield_tha` and `lag_yield`, so the production model does not include a comarca random effect.

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

Olive yield in year Y is driven by climate accumulated during the calendar year Y.
The current functional olive pipeline uses fixed calendar day-of-year windows from
January 1 through harvest truncation at DOY 327 (~23 November). The broader
seasonal pipeline (`aggregate_seasonal.py`) may reference a Dec–Nov agronomic
window for precipitation, but the scalar-on-function olive pipeline uses the
calendar year. A PEP725-derived GDD axis is future work; the older
literature-derived GDD axis has been retired.

### Target Crops

19 rainfed woody crops across 6 groups:
- pome_fruit: apple, pear
- stone_fruit: peach, plum, apricot, nectarine, flat peach, flat nectarine, cherry
- nut: almond, hazelnut, walnut
- vine: wine grape, raisin grape
- olive: oil olive
- other: pomegranate, persimmon, asian pear

Filter criterion: row is kept if **either** rainfed (`seca_ha > 0 AND seca_kg_ha > 0`)
or irrigated (`regadiu_ha > 0 AND regadiu_kg_ha > 0`) is positive. The dropped regime's
`yield_*_tha` is `NaN`. Output columns: `seca_ha`, `seca_kg_ha`, `yield_tha` (rainfed)
plus `regadiu_ha`, `regadiu_kg_ha`, `yield_irrig_tha` (irrigated). Existing rainfed
analyses that filter on `yield_tha > 0` are unaffected.

## Key Data Files

| File | Size | Status |
|------|------|--------|
| catalan_woody_yield_raw.csv | 230KB | Tracked (rainfed + irrigated columns; see parse_yield.py) |
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
| dun/comarca_arbequina_whitelist.csv | <1KB | Tracked (10 comarques where Arbequina is ≥90% of rainfed olive area AND ≥500 ha; headline rainfed cultivar subset) |
| dun/comarca_irrigated_arbequina_whitelist.csv | <1KB | Tracked (8 comarques where Arbequina is ≥90% of *irrigated* olive area AND ≥200 ha; irrigated sign-test subset) |
| dun/comarca_olive_rainfed_fraction.csv | 3KB | Tracked (DUN rainfed-fraction per comarca) |

## Documentation

- **WARP.md**: Complete pipeline documentation (data sources, variable definitions, spatial methodology, output schemas). **Single source of truth** for pipeline operations.
- **olive_water_sensitivity_plan.md**: Current olive analysis design (hierarchical Bayesian scalar-on-function regression, calendar DOY phenology windows, within-estimator centering, LOYO/sensitivity checks). **Primary source** for olive methodology.

Refer to these files for detailed specifications. They document not just what the code does but **why design decisions were made** (e.g., why openpyxl, why fixed calendar windows, why not rolling-window OLS). If older prose elsewhere conflicts with the olive plan, `olive_water_sensitivity_plan.md` wins for the olive pipeline.

## Known Limitations

- **Centroid approximation**: Rainfed crops cluster on south-facing slopes, not valley floors where centroids often fall. Spatial aggregation (mean across multiple grid cells per comarca) partially mitigates this.
- **Fixed lapse rate**: Olive farm temperature correction uses constant -5.5°C/km; real lapse rates vary seasonally.
- **Sample size**: Arbequina panel is ~90 comarca-years (10 comarques × 9 effective years after lag). With n ≈ 9 independent years, the analysis is exploratory / hypothesis-generating; no inferential framework can make it confirmatory.
- **Do not report Bayesian R² as a headline metric** — use posterior predictive checks and LOYO stability instead.

## Current Analysis Status

The olive analysis has transitioned from sliding-window OLS (archived) to a hierarchical
Bayesian scalar-on-function regression. The prior sliding-window approach suffered from
window-overlap artifacts and physiologically implausible sign-stable negative precipitation
coefficients. See `olive_water_sensitivity_plan.md` for the full rationale.

The current olive methodology uses fixed calendar windows (Garrido et al. 2021) on a
DOY axis; the earlier literature-derived GDD axis was retired after sensitivity work
showed the headline contrast could flip under plausible base-temperature choices.

Steps 1–10 have been implemented and run on the current calendar DOY axis; the
retired GDD-axis artifacts have been deleted. Step 10 is partial: the Arbequina
CWB-state sensitivity sweep (τ × k × year-RE, ±10-day window shift, irrigated
negative control) is complete; the VPD predictor sweep and the all-olive Step 10
remain. Both cohorts (Arbequina n = 90; all-olive ≥ 500 ha n = 162) converged
cleanly in Steps 7–9.

**Design decisions carried forward from prior work:**
- Two-cohort comparison: Arbequina (headline) + All olive (breadth check)
- Within-estimator centering of `yield_tha` and `lag_yield`, so no comarca RE in the production model
- Lag yield retained as a covariate, with year RE as a sensitivity comparison
- Irrigated panel as a negative control
- Headline posterior contrast defined as `Δ(S7 − S6)` on calendar windows; the
  most robust result is the `Δ(S7 − S_pre)` contrast (P(diff) 0.95–0.99, LOYO-stable),
  while `Δ(S7 − S6)` is directional but LOYO-sensitive
