# Area-Weighted Olive Climate Pipeline

## Overview

This pipeline improves on the standard comarca-level climate aggregation by weighting climate variables according to the **actual spatial distribution of rainfed olive farms**.

Instead of assuming climate is uniform across comarca polygon boundaries, we:
1. Extract climate at each olive farm's location (482 farms)
2. Apply elevation corrections to temperature and VPD
3. Compute **area-weighted comarca averages**: `weighted_mean = sum(value × area) / sum(area)`

This gives comarca climate that reflects where the crops actually are, not where comarca boundaries happen to fall.

## Why This Matters

**Problem with centroid approach:**
- Comarca centroids often fall in valleys or urban centers
- Rainfed olive farms cluster on south-facing slopes at higher elevations
- Temperature can vary ±2°C within a comarca due to elevation (lapse rate: -5.5°C/km)
- VPD varies accordingly

**Area-weighted approach:**
- Each farm gets climate from its matched AgERA5 grid cell
- Temperatures are elevation-corrected using DEM data
- Comarca average weighted by farm area reflects true exposure

## Data Flow

```
olive_groves_catalonia.gpkg (482 farms with grid cell matching + elevation)
  ↓
extract_climate_olive_farms_daily.py
  → agera5_daily_olive_farms.csv (~1.8M rows: 482 farms × 3653 days)
    [farm_id, date, comarca, area_ha, et0, precip, tmax_cor, tmin_cor, vpd_cor, ...]
  ↓
aggregate_farms_to_comarca.py
  → agera5_daily_comarca_olive_weighted.csv (~84K rows: 23 comarques × 3653 days)
    [comarca, date, total_area_ha, n_farms, et0, precip, tmax_cor, tmin_cor, vpd_cor, ...]
  ↓
aggregate_seasonal_olive_weighted.py
  → agera5_seasonal_comarca_olive_weighted.csv (207 rows: 23 comarques × 9 years)
    [comarca, year, flower_mean_vpd, flower_cum_precip, fruit_set_mean_vpd, ...]
  ↓
join_dataset_olive_weighted.py
  → catalan_olive_yield_climate_weighted.csv (final dataset)
    [comarca, year, yield_tha, seca_ha, flower_mean_vpd, fruit_set_max_vpd, ...]
```

## Running the Pipeline

### Option 1: Run all steps at once

```bash
source .venv/bin/activate
python src/run_olive_weighted_pipeline.py
```

### Option 2: Run steps individually

```bash
source .venv/bin/activate

# Step 1: Extract farm-level daily climate (~2-3 minutes)
python src/extract_climate_olive_farms_daily.py

# Step 2: Area-weighted comarca aggregation (~30 seconds)
python src/aggregate_farms_to_comarca.py

# Step 3: Bin into phenological windows (~5 seconds)
python src/aggregate_seasonal_olive_weighted.py

# Step 4: Join with yield data (~1 second)
python src/join_dataset_olive_weighted.py
```

## Output Files

| File | Size | Tracked in Git? | Description |
|------|------|-----------------|-------------|
| `agera5_daily_olive_farms.csv` | ~150 MB | No (gitignored) | Daily climate per farm with elevation correction |
| `agera5_daily_comarca_olive_weighted.csv` | ~15 MB | Yes | Daily comarca averages weighted by farm area |
| `agera5_seasonal_comarca_olive_weighted.csv` | ~50 KB | Yes | Seasonal aggregates per comarca-year |
| `catalan_olive_yield_climate_weighted.csv` | ~100 KB | Yes | Final dataset: yield + weighted climate |

## Key Differences from Standard Pipeline

| Aspect | Standard (`extract_climate.py`) | Area-Weighted (this pipeline) |
|--------|--------------------------------|-------------------------------|
| **Spatial extent** | All grid cells within comarca polygon | Only cells containing olive farms |
| **Weighting** | Equal weight to all cells | Weighted by olive farm area |
| **Temperature** | Raw AgERA5 | Elevation-corrected (-5.5°C/km) |
| **VPD** | Computed from raw tmax | Computed from corrected tmax |
| **Coverage** | 42 comarques (all crops) | 23 comarques (olive farms captured by CORINE) |
| **Use case** | Multi-crop analysis | Olive-specific analysis |

## Expected Results

### Sample size
- **23 comarques** with usable CORINE olive-farm coverage
- **9 years** (2016-2024, agronomic years; 2015 dropped due to missing Dec 2014 data)
- **207 comarca-year observations** for weighted-only yield analysis

For full-sample olive analysis, the current recommendation is to use the **hybrid dataset**:

- `data/catalan_olive_yield_climate_hybrid.csv`
- 36 comarques, 320 observations
- area-weighted where available + centroid fallback elsewhere

### Comarca farm area distribution
- Total olive farm area per comarca: 10-15,000 ha (large interior comarques) down to 100-500 ha (small coastal comarques)
- Number of farms per comarca: 1-30 (median ~13)

### Climate corrections
- Temperature correction range: -1.4 to +1.6°C (depends on farm elevation vs grid cell elevation)
- VPD changes accordingly (VPD increases with temperature via Tetens formula)
- Precipitation and ET0: no correction applied (not elevation-dependent at this scale)

## Comparison with Centroid Approach

To assess the impact of area weighting, compare:

```python
import pandas as pd

# Centroid-based (from standard pipeline)
df_centroid = pd.read_csv("data/catalan_woody_yield_climate.csv")
df_centroid = df_centroid[df_centroid['pheno_key'] == 'olive']

# Area-weighted (from this pipeline)
df_weighted = pd.read_csv("data/catalan_olive_yield_climate_weighted.csv")

# Join on comarca + year
merged = df_centroid.merge(
    df_weighted[['comarca', 'year', 'flower_mean_vpd', 'fruit_set_mean_vpd']],
    on=['comarca', 'year'],
    suffixes=('_centroid', '_weighted')
)

# Compare
print("Flowering VPD difference (weighted - centroid):")
print((merged['flower_mean_vpd_weighted'] - merged['flower_mean_vpd_centroid']).describe())
```

Expected: small systematic differences (0.1-0.3 kPa VPD) in topographically varied comarques, near-zero in flat comarques.

## Integration with Existing Analysis

### Adapt `elasticnet_olive_lag.py`

The existing distributed-lag regression script reads `agera5_daily_catalonia.csv`. To use area-weighted data:

```python
# OLD (line ~50)
daily = pd.read_csv("data/agera5_daily_catalonia.csv", parse_dates=["date"])

# NEW
daily = pd.read_csv("data/agera5_daily_comarca_olive_weighted.csv", parse_dates=["date"])
# Use vpd_cor instead of vpd_mean
```

Or create a new script: `elasticnet_olive_lag_weighted.py`

In practice, the repo now uses:

- `elasticnet_olive_lag_hybrid.py` for hybrid daily climate
- timing scans on `agera5_daily_hybrid_olive.csv` for current exploratory olive results

### Adapt `explorer.py` (Streamlit dashboard)

Add a toggle to switch between centroid-based and area-weighted data sources.

## Prerequisites

1. `olive_groves_catalonia.gpkg` must exist with columns:
   - `OBJECTID`, `comarca`, `Area_Ha`
   - `grid_lat_idx`, `grid_lon_idx` (from AgERA5 grid matching)
   - `temp_correction` (from elevation-based lapse rate)

2. AgERA5 NetCDF files must be downloaded:
   - `data/agera5_catalonia/*.nc` (50 files: 5 variables × 10 years)

3. Yield data:
   - `data/catalan_woody_yield_raw.csv`

## Notes

- **Agronomic year**: Dec 1 (Y-1) → Nov 30 (Y), yield attributed to year Y
- **Elevation correction**: Already computed in `olive_groves_catalonia.gpkg` (column: `temp_correction`)
- **VPD formula**: FAO-56 Tetens, computed per-farm then area-averaged
- **Phenological windows**: Same as standard pipeline (from `aggregate_seasonal.py`)
  - Flower: May-Jun
  - Fruit set: Jul-Aug
  - Maturation: Sep-Nov
