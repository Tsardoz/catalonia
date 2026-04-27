# Technical Documentation: Catalan Woody Crop Climate-Yield Analysis

## Executive Summary

This PhD research assembles a comprehensive regional-scale dataset linking rainfed woody crop yields to daily agrometeorological conditions across Catalonia (2015–2024). The project implements two complementary spatial exposure methodologies (centroid-based and farm-weighted), develops exploratory climate-yield timing analysis frameworks, and positions the dataset as an intermediate transfer learning resource between global crop model datasets (CropClimateX, Tatura).

**Key outputs:**
- 320 comarca-year olive observations with area-weighted climate exposure
- 131,508 daily timeseries rows enabling phenological timing scans
- Reproducible pipeline for 19 rainfed woody crops across 6 functional groups
- Publication-ready exploratory evidence of late July–early August sensitivity window for olive

---

![Olive Timing Summary Figure](figures/olive_timing_summary.png)
**Figure 0.1: Olive timing summary (4-panel).** Left to right: (A) Tmax ≥ 32°C timing scan showing strongest negative coefficients in late July–early August (DOY ~213–235); (B) VPD ≥ 3.0 kPa timing scan supporting similar timing window; (C) Water balance (P − ET₀) context showing winter positive signal; (D) Window length sensitivity (7-day, 14-day, 21-day windows all point to same broad period). This composite figure synthesizes the main exploratory findings.

---

## 1. Research Objectives

The project answers three nested questions:

1. **Data integration**: Can administrative yield records be reliably matched to gridded climate at regional scale?
2. **Spatial exposure**: Does farm-weighted climate exposure yield clearer crop-climate associations than centroid approximations?
3. **Timing sensitivity**: Which climate variables and measurement windows best predict yield variation across olive's phenological stages?

Focus has shifted from annual-bin regression toward **daily timing scans** on short-window (14-day) rolling periods, revealing that **late July–early August** exhibits the strongest associations between heat/dry-air stress and yield suppression.

---

## 2. Data Integration Architecture

### 2.1 Data Sources

#### Yield (Gencat Administrative Records)
- **Source**: Generalitat de Catalunya (DARP) "Produccions comarcals" spreadsheets (10 annual xlsx files, 2015–2024)
- **Sheet**: `LLENYOSOS` (woody/perennial crops only)
- **Key challenge**: Merged cells in columns A–B (comarca, crop group) corrupt standard Excel parsers
- **Solution**: Direct openpyxl access with manual forward-fill of `None` values in merged cell regions
- **19 target crops** in 6 functional groups:
  - **Pome fruit**: Apple, Pear
  - **Stone fruit**: Peach, Plum, Apricot, Nectarine, Flat peach, Flat nectarine, Cherry
  - **Nut**: Almond, Hazelnut, Walnut
  - **Vine**: Wine grape, Raisin grape
  - **Olive**: Oil olive
  - **Other**: Pomegranate, Persimmon, Asian pear
- **Filtering**: Keep only rainfed areas (`seca_ha > 0` AND `seca_kg_ha > 0`)
- **Normalized unit**: `yield_tha = seca_kg_ha / 1000` (tonnes/hectare)

#### Climate (AgERA5 Reanalysis)
- **Source**: Copernicus Climate Data Store (CDS) agrometeorological indicators service
- **Product level**: Pre-processed, bias-corrected gridded daily summaries (~9 km resolution, ~0.1° lat/lon)
- **Key distinction**: AgERA5 is NOT raw ERA5—it includes field-relevant aggregations (ET0 pre-computed via FAO-56)
- **5 downloaded variables** (50 netCDF files: 5 vars × 10 years):
  - `reference_evapotranspiration` (ET0, mm/day, version 2.0)
  - `2m_temperature_max` (Tmax, K, version 2.0, statistic `24_hour_maximum`)
  - `2m_temperature_min` (Tmin, K, version 2.0, statistic `24_hour_minimum`)
  - `vapour_pressure` (ea, hPa, version 1.0 with fallback to 2.0 for 2023–2024, statistic `24_hour_mean`)
  - `precipitation_flux` (P, mm/day, version 2.0)
- **Critical derivation**: VPD is NOT available from CDS despite documentation. Computed in-pipeline using FAO-56 Tetens formula:
  - `es(Tmax) = 0.6108 × exp(17.27 × Tmax_C / (Tmax_C + 237.3))` [kPa]
  - `VPD = max(es(Tmax) − ea_kpa, 0)` [kPa]
- **Unit conversions**:
  - Temperature: K → °C (subtract 273.15)
  - Vapour pressure: hPa → kPa (divide by 10)
  - ET0, Precipitation: no conversion (already mm/day)

#### Spatial Boundaries
- **Source**: Institut Cartogràfic i Geològic de Catalunya (ICC) public WFS endpoint
- **Layer**: "divisions_administratives_comarques_5000" (polygon layer, WGS84 EPSG:4326)
- **Coverage**: 42 official comarques; 41 contain rainfed yield observations (2 excluded: Alta Ribagorça, Lluçanès)
- **Usage**: Spatial reference frame for climate aggregation and grid cell mapping
- **Output**: GeoPackage (`comarca_polygons.gpkg`) with standardized comarca names

#### Olive Farm Locations (Weighted Pipeline Only)
- **Source**: CORINE Land Cover 2018, class 223 (Olive groves), downloaded from EEA ArcGIS REST API
- **Coverage**: 482 polygons, ~123,700 ha (clipped to Catalonia boundary)
- **Elevation tagging**: Copernicus GLO-30 DEM (merged from 10 tiles, 30 m resolution)
  - Zonal statistics per polygon: mean, min, max, std elevation
  - Elevation range: 7–777 m (median: 264 m); within-polygon relief ~21 m std
- **Temperature correction**: Elevation-based lapse rate −5.5°C/km applied to AgERA5 grid cell values
  - Correction range: −1.40 to +1.64°C
  - Justification: ~20% of farms differ >1°C from comarca-level average in topographic comarcas

![Olive farm-to-grid matching](figures/olive_agera5_matching_topo.png)
**Figure 2.1: Olive farm-to-grid matching with topography.** Map showing 482 CORINE 2018 olive grove polygons (colored by comarca) matched to AgERA5 0.1° grid cells (hexagonal overlay). Underlying topography from Copernicus GLO-30 DEM with elevation contours. This visualization demonstrates the spatial basis for elevation-corrected temperature extraction and area-weighted climate aggregation to comarca level.

---

## 3. Spatial Methodology

### 3.1 Comarca-Level Climate Aggregation

**Grid cell assignment** (Fig: `extract_climate.py`):
- AgERA5 grid: 22 lats × 32 lons = 704 cells over Catalonia bounding box
- **Spatial join**: For each comarca, identify all grid cell centres falling within polygon boundary
- Cell distribution: 328 cells assigned to 39 comarques; 3 small comarques (Aran, Barcelonès, Garraf) fallback to nearest cell
- Range: 1–19 cells per comarca (e.g., Noguera: 19 cells)

**Daily aggregation** per comarca:
- Extract (time, n_cells) array for each variable for each day
- Compute spatial summary statistics: **mean, min, max** across assigned cells
- VPD computed per-cell (FAO-56) before aggregation
- Output: `agera5_daily_catalonia.csv` (~3,653 days × 42 comarcas = 157K rows)

### 3.2 Olive Farm-Weighted Pipeline

**Farm-to-grid matching** (Fig: `extract_climate_olive_farms_daily.py`):
- Each CORINE olive polygon → nearest AgERA5 grid cell (centroid distance)
- Polygon → comarca assignment via spatial join
- Extract daily timeseries for each farm, years 2015–2024
- Apply elevation correction: `T_corrected = T_agera5 + lapse × (farm_elev − grid_cell_elev) / 1000`
- Output: `agera5_daily_olive_farms.csv` (~482 farms × 3,653 days = 1.8M rows)

**Area-weighted comarca aggregation** (Fig: `aggregate_farms_to_comarca.py`):
- For each comarca-day: weighted average of farm timeseries, weights = farm area (ha)
- Produces: `agera5_daily_comarca_olive_weighted.csv` (comarca-level daily with farm-derived climate)
- **Spatial inference**: Better represents actual olive distribution than centroid
- **Validation**: Correlation r = 0.93 (VPD) between weighted and centroid methods; differences largest in topographic comarcas

### 3.3 Hybrid Exposure Model

**Rationale**: 36 comarques report olive yield; only 23 have CORINE olive coverage. Remaining 13 lack farm data.

**Hybrid strategy**:
- Use area-weighted climate for 23 comarques with farm data
- Fall back to centroid-based climate for 13 comarques without farms
- Result: 320 olive observations (36 comarques × 9 years, 2016–2024) with best available spatial representation per comarca
- Output: `catalan_olive_yield_climate_hybrid.csv`
- Daily equivalent: `agera5_daily_hybrid_olive.csv` (131,508 rows: 36 comarques × 3,653 days)

![Climate comparison: weighted vs. centroid](figures/climate_comparison_weighted_vs_centroid.png)
**Figure 3.1: Spatial comparison — Farm-weighted vs. centroid climate.** Scatterplot comparing area-weighted farm-based climate (x-axis) to centroid-based climate (y-axis) for VPD. Points are colored by comarca. Pearson correlation r = 0.93, indicating high overall agreement. Dashed line shows 1:1 relationship. A few outliers in topographically varied comarcas (labeled) show up to +1.6°C differences; flat comarcas cluster near diagonal with <0.1°C differences. This validates the hybrid approach: farm-weighted climate is better where available but not dramatically different.

![Climate differences distribution](figures/climate_differences_distribution.png)
**Figure 3.2: Distribution of climate differences (weighted − centroid).** Histogram showing frequency of VPD differences between area-weighted and centroid methods. Most differences <0.3 kPa (narrow peak); tail extends to ±1.5 kPa for topographic outliers. This distribution justifies the hybrid model: using farm-weighted climate where available improves spatial realism without requiring dramatic model reformulation.

**Validation**: See **Figure 1 (climate_comparison_weighted_vs_centroid.png)** and **Figure 2 (climate_differences_distribution.png)**
- Correlation between methods: r = 0.93 (VPD) — high agreement
- Largest differences in topographically varied comarques (Pallars Jussà, Anoia, Noguera)
- Flat comarques show <0.1°C differences

---

## 4. Phenological Windows and Aggregation

### 4.1 Calendar-Based Windows (Seasonal Aggregation)

**Source**: `aggregate_seasonal.py` / `PHENO_WINDOWS` constant

Each crop group assigned 2–4 phenological periods (month ranges). Olive windows:
- **SP1 (Flower → Pit hardening)**: April–June [DOY 95–157 in doy-based refactoring]
- **NP (Pit hardening)**: June–July [DOY 153–217]
- **SP2 (Oil synthesis → Harvest)**: August–November [DOY 216–313]

**Aggregates per period** (daily → seasonal):
- `mean_vpd`: Daily mean VPD (kPa)
- `max_vpd`: Mean of top-10% daily VPD values (stress peaks, kPa)
- `n_days_vpd_gt2`: Count of days VPD > 2.0 kPa
- `cum_et0`: Cumulative ET0 (mm)
- `mean_tmax`: Mean daily Tmax (°C)
- `mean_tmin`: Mean daily Tmin (°C)
- `cum_precip`: Total precipitation (mm)

Output: `agera5_seasonal_catalonia.csv` (42 comarques × 10 years × crop groups)

### 4.2 Daily Timing Windows (Exploratory Scans)

**Purpose**: Identify which calendar windows show strongest climate-yield associations

**Rolling window framework** (`olive_tmax_timing_scan.py`, et al.):
- Define candidate days: June 1 – August 31 (DOY 152–243)
- Create rolling 14-day windows with weekly anchor dates (Sun of each week)
- For each window: compute simple metric (e.g., count of days Tmax ≥ 32°C)
- Fit OLS: `yield_tha ~ {metric} + comarca_fixed_effects`, n = 320 olive rows
- Extract: coefficient, stderr, p-value, R²
- Repeat for all windows, producing ranking by |coefficient| magnitude

**Current screening variables** (published in SUPERVISOR_PRESENTATION.md):
1. **Heat timing**: Count of days Tmax ≥ 32°C (strongest signal)
2. **Vapor deficit timing**: Count of days VPD ≥ 3.0 kPa (supporting signal, similar timing window)
3. **Water balance context**: Full-year cumulative `P − ET₀` (background, not primary summer signal)

**Key finding**: Negative associations (yield suppression) concentrated in **late July–late August** (DOY 213–235), most pronounced **21-Aug window** (DOY 228 ± 7 days).

![Olive Tmax timing scan](figures/olive_tmax_timing_scan.png)
**Figure 4.1: Tmax timing scan — rolling 14-day windows of days with Tmax ≥ 32°C.** X-axis: end date of rolling window (June 1 – Aug 31, DOY 152–243). Y-axis: OLS coefficient estimate ± 95% CI (with comarca fixed effects). Red shaded region: primary candidate sensitive period (late July–early August, DOY ~213–235). The most negative coefficient occurs near Aug 21 window (DOY 228 ± 7 days), suggesting peak yield sensitivity to heat stress in this fortnight. This is the **strongest single exploratory finding** for olive in this dataset.

![Olive VPD timing scan](figures/olive_vpd_timing_scan.png)
**Figure 4.2: VPD timing scan — rolling 14-day windows of days with VPD ≥ 3.0 kPa.** Same layout as Figure 4.1. Shows similar timing pattern to Tmax (late July–early August peak), though with slightly broader and less pronounced trough. Early-July points are positive (early pit hardening stage); summer points are negative (stress signal). VPD scan provides **supporting evidence** for Tmax findings; however, Tmax remains the cleaner screening variable.

![Olive Tmax window length comparison](figures/olive_tmax_window_compare.png)
**Figure 4.3: Window length sensitivity — comparing 7-day, 14-day, and 21-day rolling windows of Tmax ≥ 32°C days.** Three overlaid traces showing coefficients for different window lengths. All three converge on the same broad candidate sensitive period (late July–early August). Longer windows (21-day) mechanically broaden and lag the peak due to backward-looking aggregation. The 14-day window is chosen as best compromise: sufficient temporal resolution without excessive noise. This **sensitivity check validates the main finding** across window specifications.

---

## 5. Output Schemas

### 5.1 Yield Data (`catalan_woody_yield_raw.csv`)

| Column | Type | Description |
|--------|------|-------------|
| year | int | 2015–2024 |
| comarca | str | Official comarca name (Catalan) |
| crop_catalan | str | 19 target crops |
| crop_en | str | English translation |
| crop_group | str | Functional group (pome_fruit, stone_fruit, nut, vine, olive, other) |
| pheno_key | str | Phenological grouping key (crop-specific or crop_group) |
| seca_ha | float | Rainfed area (hectares, > 0 filter applied) |
| seca_kg_ha | float | Rainfed yield (kg/ha, > 0 filter applied) |
| yield_tha | float | Rainfed yield (tonnes/ha) = seca_kg_ha / 1000 |

**Rows**: 2,660 (19 crops × 42 comarques × 10 years, unbalanced; many crops not grown in many regions)

### 5.2 Daily Climate (`agera5_daily_catalonia.csv`)

| Column | Type | Description |
|--------|------|-------------|
| date | date | YYYY-MM-DD |
| comarca | str | 42 comarques |
| et0_mean / et0_min / et0_max | float | Spatial summary of ET0 (mm/day) |
| tmax_mean / tmax_min / tmax_max | float | Spatial summary of Tmax (°C) |
| tmin_mean / tmin_min / tmin_max | float | Spatial summary of Tmin (°C) |
| ea_mean / ea_min / ea_max | float | Spatial summary of ea (kPa) |
| vpd_mean / vpd_min / vpd_max | float | Spatial summary of VPD (kPa); computed per-cell before aggregation |
| precip_mean / precip_min / precip_max | float | Spatial summary of P (mm/day) |

**Rows**: ~157,000 (42 comarques × 3,653 days, 2015–2024)
**Note**: GITIGNORED; regenerated via `extract_climate.py` from AgERA5 netCDF files

### 5.3 Seasonal Climate (`agera5_seasonal_catalonia.csv`)

| Column | Type | Description |
|--------|------|-------------|
| comarca | str | 42 comarques |
| year | int | 2015–2024 |
| pheno_key | str | Crop-specific phenological key |
| {period}_mean_vpd | float | Mean daily VPD over period (kPa) |
| {period}_max_vpd | float | Mean of top-10% VPD days (kPa) |
| {period}_n_days_vpd_gt2 | int | Days with VPD > 2.0 kPa |
| {period}_cum_et0 | float | Cumulative ET0 (mm) |
| {period}_mean_tmax | float | Mean daily Tmax (°C) |
| {period}_mean_tmin | float | Mean daily Tmin (°C) |
| {period}_cum_precip | float | Total precipitation (mm) |

Example periods for olive: `sp1_flower_hardening`, `np_pit_hardening`, `sp2_oil_synthesis`

**Rows**: Variable by crop group and comarca coverage; ~840 rows for olive (36 comarques × 10 years ≈ 350 possible, but only ~320 have yield data)

### 5.4 Joined Dataset (`catalan_woody_yield_climate.csv`)

Merge of yield (Step 1) × seasonal climate (Step 4) on comarca + year + pheno_key.

| Column (selected) | Type | Description |
|--------|------|-------------|
| comarca | str | Comarca name |
| year | int | 2015–2024 |
| crop_catalan | str | 19 crops |
| crop_group | str | Functional group |
| pheno_key | str | Phenological key |
| seca_ha | float | Rainfed area (ha) |
| yield_tha | float | **Response variable** (tonnes/ha) |
| {period}_mean_vpd | float | Climate features |
| {period}_cum_et0 | float | (multiple climate columns per crop) |
| ... | ... | |

**Rows**: 320 for olive (36 comarques × 9 years, 2016–2024, after removing years with missing climate); varies by crop
**Key property**: Indexed at **comarca-year level** → one yield observation per comarca per year per pheno_key

### 5.5 Olive Hybrid Dataset (`catalan_olive_yield_climate_hybrid.csv`)

Subset of joined dataset filtered to olive, with climate from hybrid exposure method (weighted where farm data available, centroid fallback otherwise).

**Rows**: 320 (36 comarques × 9 years, 2016–2024)
**Comarques**: 23 farm-weighted + 13 centroid-based

---

## 6. Statistical Methodology and Assumptions

### 6.1 Unit of Analysis

**Comarca-year** observations:
- Each row represents one comarca's reported yield in one year for one crop group
- Yield is a **spatial aggregate** (all rainfed area in comarca summed, then divided by total area)
- Climate is a **spatial-temporal aggregate** (weather experienced across comarca during phenological window)
- **Fixed effects model**: All regressions include comarca dummies to absorb baseline yield differences across regions
  - Justification: Yield varies ~5× across comarques due to structural/agro-ecological differences
  - Interpretation: Coefficient estimates represent marginal effects of climate anomalies within comarcas

### 6.2 Exploratory vs. Confirmatory Inference

**This work is exploratory**, not confirmatory:
- Many correlated windows tested in timing scans (~13 windows × 3 variables = ~40 point estimates)
- Raw p-values reported but interpreted as **screening statistics**, not strict hypothesis tests
- No multiple testing correction (classical approach: Bonferroni would over-correct for dependent tests)
- Focus: Identify candidate sensitive periods and consistent metric ranking across methods

**Appropriate use**: Guide hypothesis formation and experimental design (e.g., controlled deficit irrigation studies)
**Inappropriate use**: Publish raw p-values as confirmatory evidence of causal olive stress response

### 6.3 Mechanistic Limitations

The analysis makes no claim to capturing true plant water stress:
1. **Deep-rooted olives**: Likely access groundwater; surface soil moisture does not control yield
2. **Historical (non-experimental) data**: Cannot isolate causal effects; climate covariance structure is complex
3. **Simple climate metrics**: `Tmax ≥ 32°C` is a threshold heuristic, not a physiological stress model
4. **Fixed lapse rate**: Temperature correction assumes −5.5°C/km uniformly; real rates vary seasonally
5. **VPD at Tmax**: Represents peak daily stress; cumulative or time-integrated stress might be more relevant

**Interpretation**: Results suggest **timing matters** for olive yield; causal mechanisms remain speculative pending controlled studies.

---

## 7. Key Implementation Details

### 7.1 Openpyxl for Yield Parsing

**Why not pandas?** Standard `pd.read_excel()` fails on Gencat source files because:
- Columns A–B (comarca, crop group) are merged across multiple rows
- Pandas mishandles merged cells, losing comarca names on continuation rows

**Solution**: Direct openpyxl access (`src/parse_yield.py`):
```
1. Open workbook, access 'LLENYOSOS' sheet
2. Iterate rows, extract cell values by column index
3. Forward-fill `comarca` and `grup` when encountering `None` (merged cell continuation)
4. Stack into DataFrame
5. Convert to long format and filter seca_ha > 0 AND seca_kg_ha > 0
```

### 7.2 VPD Computation (FAO-56 Tetens)

AgERA5 does not provide VPD directly despite CDS documentation suggesting it might. Computed in `extract_climate.py`:

```python
def _compute_vpd(tmax_c: np.ndarray, ea_kpa: np.ndarray) -> np.ndarray:
    es = 0.6108 * np.exp(17.27 * tmax_c / (tmax_c + 237.3))  # kPa
    return np.maximum(es - ea_kpa, 0.0)
```

**Rationale**:
- `es(Tmax)` = saturation vapour pressure at daily maximum temperature
- `ea` = 24-hour mean actual vapour pressure (from AgERA5)
- VPD = es − ea represents atmospheric demand for water vapour at peak stress time
- Max function ensures VPD ≥ 0 (numerical safety)

### 7.3 Agronomic Year Convention

Olive year Y yield is driven by climate **Dec 1 (Y-1) through Nov 30 (Y)**:
- Aligns with Mediterranean perennial phenology (flowering May–June, harvest Oct–Nov)
- Also used in CropClimateX (calendar-year convention is specific to annual crops)
- Implementation: `aggregate_seasonal.py` defines periods once; all analysis scripts import `PHENO_WINDOWS` constant

### 7.4 Spatial Aggregation (Mean/Min/Max, not Centroid-Only)

**Design choice**: For each comarca, compute **3 summaries** (mean, min, max) across all assigned grid cells, not just centroid value.

**Rationale**:
- Centroid may fall in non-agricultural area (e.g., urban centre, lake)
- Within-comarca climate variability reflects actual exposure heterogeneity (topography, coastal vs. inland)
- Mean/min/max reveal stress extent: `max` shows worst-hit cells, `min` shows buffered areas
- Spatial aggregation addresses CropClimateX critique of centroid approximation

**Grid cell distribution**: 328 of 704 cells over Catalonia assigned to comarques; 376 are sea or outside boundary

---

## 8. Pipeline Execution and Data Continuity

### 8.1 Step-by-Step Workflow

1. **parse_yield.py**: Read 10 Excel files → `catalan_woody_yield_raw.csv` (2,660 rows)
2. **get_centroids.py**: Fetch ICC polygons, compute centroids → `comarca_polygons.gpkg` + `comarca_centroids.csv`
3. **download_agera5.py**: Query CDS API (requires credentials) → 50 netCDF files in `agera5_catalonia/`
4. **extract_climate.py**: Spatial join + aggregation → `agera5_daily_catalonia.csv` (~157K rows, GITIGNORED)
5. **aggregate_seasonal.py**: Daily → phenological periods → `agera5_seasonal_catalonia.csv`
6. **join_dataset.py**: Yield × Climate on comarca + year + pheno_key → `catalan_woody_yield_climate.csv` (final product)

### 8.2 Olive-Weighted Variant

Runs in parallel; feeds into hybrid:
- **extract_climate_olive_farms_daily.py**: Farm timeseries extraction + elevation correction
- **aggregate_farms_to_comarca.py**: Area-weighted comarca aggregation
- **aggregate_seasonal_olive_weighted.py**: Farm-derived seasonal aggregates
- **join_dataset_olive_weighted.py**: Olive yield × weighted climate
- **create_hybrid_olive_dataset.py**: Merge weighted (23 comarques) + centroid fallback (13 comarques) → `catalan_olive_yield_climate_hybrid.csv`

### 8.3 Data Continuity and Assumptions

- **No yield data removed after Step 1**: All rows in `catalan_woody_yield_raw.csv` carry forward to final dataset
- **Climate-yield matching**: Many comarca-years have NO climate (missing AgERA5 coverage over polygon). These rows appear in final dataset with NaN climate values (not silently dropped)
- **Sample sizes**: Vary by crop:
  - Olive: 320 rows with climate (36 comarques × 9 years, 2016–2024; 2015 has incomplete early data)
  - Other crops: smaller, unbalanced panels
- **Validation**: `validate_hybrid_dataset.py` checks row counts and NaN patterns post-merge

---

## 9. Analysis Workflow and Interpretation Framework

### 9.1 Seasonal (Phenological Window) Analysis

**Standard approach** (used in `elasticnet_olive_lag.py`):
1. Load `catalan_olive_yield_climate_hybrid.csv`
2. For each climate variable (vpd_mean, tmax_mean, et0_cum, precip_cum):
   - Fit OLS: `yield_tha ~ {var} + comarca_dummies`
   - Extract coefficient, R², p-value
3. Tabulate R² rankings across variables and periods
4. Interpret: Which phenological windows show strongest climate signal?

**Limitation**: Assumes linear main-effects; ignores interactions (temperature × moisture covariance common in rainfed regions)

![Olive OLS comparison](figures/olive_ols_comparison_hybrid.png)
**Figure 9.1: OLS R² rankings across phenological periods and climate variables (hybrid dataset).** Bar chart comparing model fit (R² × 100) for VPD, Tmax, ET0, and Precipitation across phenological periods (SP1: flower-hardening, NP: pit-hardening, SP2: oil-synthesis, Full-year average). VPD mean and max dominate (R² ~0.15–0.18); water balance (P − ET₀) shows secondary signal (~0.10); ET0 alone is weak. Phenological specificity: SP2 (oil synthesis, Aug–Nov) shows strongest signal; NP (pit hardening, Jun–Jul) shows weakest. This ranking justifies focus on VPD and Tmax as primary predictors and guides the transition to daily timing scans.

### 9.2 Daily Timing (Rolling Window) Analysis

**Exploratory approach** (used in `olive_tmax_timing_scan.py`, et al.):
1. Load `agera5_daily_hybrid_olive.csv`
2. Define candidate window universe: June 1–Aug 31 (or other period)
3. For each 14-day rolling window (weekly anchor):
   - Compute metric: count of days Tmax ≥ 32°C (or other threshold)
   - Aggregate to comarca-year level (sum across days in window, per year per comarca)
   - Fit OLS: `yield_tha ~ {metric} + comarca_dummies`
4. Extract: coefficient (sensitivity), p-value, R²
5. Plot: X-axis = calendar window (DOY or date), Y-axis = coefficient (with 95% CI)
6. Identify: Window(s) with most negative coefficient (strongest yield suppression)

**Output**: Timing scan CSV (`olive_tmax_timing_scan.csv`) and visualization

### 9.3 Multi-Variable Integration (Distributed-Lag Model)

**Next-generation approach** (partially implemented in `elasticnet_olive_lag_hybrid.py`):
- Fit ElasticNet with lagged climate features
- Allow model to learn weights across multiple windows and variables
- Evaluate via cross-validation R²
- Interpret: Which variables and lags drive predictive power?

**Current status**: Infrastructure in place; interpretation remains exploratory

---

## 10. Known Limitations and Caveats

1. **Spatial mismatch**: Rainfed crops cluster on south-facing slopes; centroids often fall in valley floors or urban areas. Spatial aggregation (mean/min/max) partially mitigates but does not solve.

2. **Elevation correction**: Fixed −5.5°C/km lapse rate does not account for seasonal variation (lower in winter, higher in summer) or weather-regime dependence.

3. **VPD proxy**: Computed from Tmax + 24-hour mean ea, not simultaneous measurements. True peak VPD likely occurs in afternoon; our estimate is conservative.

4. **Farm-to-grid matching**: CORINE 2018 is a single-year snapshot. Olive area is stable (~103k ha 2015–2022, ~20% drop in 2023 of uncertain cause), but intra-annual crop rotation and abandoned parcels not captured.

5. **Historical observational data**: Cannot isolate causal effects. Complex climate covariance (wet years are cool; dry years are hot) confounds direct attribution.

6. **Sample size**: 320 olive observations (36 comarques × 9 years) limits power to detect interactions or nonlinear effects.

7. **P-values**: Exploratory timing scans test ~40 correlated hypotheses. Raw p-values should be treated as screening statistics.

---

## 11. Recommended Data Products and Workflows

### For seasonal/phenology-scale analysis:
- **Primary**: `data/catalan_olive_yield_climate_hybrid.csv` (320 rows, best spatial representation)
- **Backup**: `data/catalan_woody_yield_climate.csv` (original centroid version, all 19 crops)

### For daily/lag-based analysis:
- **Primary**: `data/agera5_daily_hybrid_olive.csv` (131,508 rows, 36 comarques × 3,653 days)
- **Backup**: `data/agera5_daily_catalonia.csv` (centroid version, all comarques)

### For exploratory timing scans:
- Use: `src/olive_tmax_timing_scan.py` + `olive_vpd_timing_scan.py` + `olive_cwb_timing_scan.py`
- Output tables: `data/olive_{tmax,vpd,cwb}_timing_scan.csv`
- Figures: `figures/olive_{tmax,vpd,cwb}_timing_scan.png`

### For publication (current best result):
- **Finding**: Late July–early August (DOY ~213–235) emerges as candidate sensitive period for olive yield
- **Metric**: Tmax ≥ 32°C day counts show clearest signal
- **Caveat**: Exploratory evidence; mechanistic pathways remain speculative

![Olive LPF top-5 features](figures/lpf_top5_olive.png)
**Figure 11.1: Olive VPD low-pass filter (LPF) top-5 features.** Four-panel scatter plot showing top-5 VPD signal features ranked by R² (comarca-year observations, n=320). Panels show: (A) LPF peak value vs. yield; (B) Peak day-of-year timing vs. yield; (C) Peak width vs. yield; (D) mean seasonal VPD vs. yield. Low-pass filtering (τ=30 days) extracts smooth stress signals from noisy daily data. This publication-quality figure demonstrates the relationship between filtered climate features and yield, supporting the timing-based interpretation and showing that **when** stress peaks (phenological timing) matters as much as **how much** stress accumulates.

---

## 12. Technical Stack and Environment

### Languages and Libraries

- **Python 3.12** (venv-managed environment at `.venv`)
- **Core scientific stack**: numpy, scipy, pandas, xarray (array-based climate data)
- **Spatial**: geopandas, shapely, rasterio, rasterstats (GIS operations, raster zonal stats)
- **Statistical**: scikit-learn (ElasticNet), statsmodels (OLS, fixed effects)
- **Visualization**: matplotlib, altair (plot_features.py uses interactive Altair for scatter plots)
- **Climate data**: cdsapi (CDS API client for AgERA5 download)
- **Spreadsheet parsing**: openpyxl (direct Excel access, avoiding pandas merged-cell bug)
- **Execution**: streamlit (src/explorer.py provides interactive data dashboard)

### Critical Packages

| Package | Version | Role |
|---------|---------|------|
| pandas | 2.3.3 | Dataframe operations, temporal resampling |
| xarray | 2026.2.0 | NetCDF I/O, dimension-aware climate aggregation |
| geopandas | 1.1.3 | Spatial joins, polygon operations, WFS queries |
| numpy | 2.4.3 | Vectorized climate aggregation, VPD computation |
| cdsapi | 0.7.7 | CDS API client (AgERA5 download) |
| openpyxl | 3.1.5 | Direct Excel cell access (yield parsing) |
| scikit-learn | 1.8.0 | Hyperparameter-tuned ElasticNet models |
| statsmodels | 0.14.6 | OLS with fixed effects (timing scans) |

### Data Format Conventions

- **Geospatial**: GeoPackage (.gpkg) for vector data; GeoTIFF (.tif) for rasters
- **Tabular**: CSV (simplicity; all production data outputs)
- **Rasters**: NetCDF4 (.nc) for climate (xarray-friendly; compresses well)
- **Coordinates**: WGS84 (EPSG:4326) throughout

---

## 13. Reproducibility and Maintenance

### Reproducibility Requirements

1. **CDS API credentials**: Required for AgERA5 download. Set via `~/.cdsapirc`
2. **Input data**: All raw source files (10 Excel, ICC WFS, CORINE) are publicly accessible
3. **Determinism**: All scripts are deterministic (no random state needed for aggregation/join; ElasticNet cross-validation uses fixed seed)
4. **Code versions**: Python 3.12 pinned in `requirements.txt`; minimal breaking dependency updates expected

### Maintenance Notes

- **AgERA5 versioning**: `vapour_pressure` switches from version 1.0 (cutoff ~mid-2023) to 2.0 for 2024. Auto-fallback implemented in `download_agera5.py`
- **ICC WFS stability**: Two candidate layers configured in `get_centroids.py` (5k, 50k resolution); service stable but endpoint may change without notice
- **CORINE 2018 snapshot**: Olive farm locations fixed to 2018. 2023–2024 yield may not align if substantial grove abandonment/planting occurred
- **Phenological windows**: Hardcoded as month ranges in `aggregate_seasonal.py`. Overrides via script parameters supported but require explicit refactoring

---

## 14. Future Research Directions

### Short-term (next 6 months)

1. **Expand variable set**: Tmin, precipitation variability, cumulative heat (GDD), soil moisture proxies
2. **Multivariate integration**: Replace single-window OLS with joint inference across multiple variables and lags
3. **Phenological sync**: Implement Growing Degree Days (GDD) to align phases across comarcas and years
4. **Cross-validation**: Move from 5-fold CV R² (ElasticNet) to held-out test set evaluation

### Medium-term (1–2 years)

1. **Causal inference**: Leverage soil moisture, irrigation records to disentangle water stress from heat stress
2. **Non-linear methods**: Random forests, neural networks to capture interaction effects
3. **Transfer learning**: Fine-tune CropClimateX (US annual) or Tatura (Australian perennial) models on Catalan data
4. **Irrigation integration**: Incorporate SIGPAC irrigation records to adjust yield for management practices

### Longer-term (3+ years)

1. **Mechanistic modeling**: Integrate with DSSAT, AquaCrop to test hypothesized stress mechanisms
2. **Temporal dynamics**: Distributed-lag models with adaptive window selection (e.g., ARDL)
3. **Spatial heterogeneity**: Bayesian hierarchical model to borrow strength across comarques
4. **Climate scenarios**: Apply trained models to CMIP6 projections to assess 2050/2100 yield impacts

---

## 15. References and Related Work

### Datasets and Standards

- **AgERA5**: Hersbach et al. (2019) "ERA5-Land monthly averaged data from 1981 to present" — https://cds.climate.copernicus.eu
- **CORINE 2018**: European Environment Agency Land Cover database
- **ICC**: Institut Cartogràfic i Geològic de Catalunya administrative boundaries
- **CropClimateX**: Lobell et al. (2019) "A crop yield gap analysis of sub-Saharan Africa" — foundational dataset for transfer learning approach

### Methodological References

- **FAO-56 Penman-Monteith ET0**: Allen, R. G., et al. (1998) "Crop evapotranspiration—Guidelines for computing crop water requirements." FAO Irrigation and Drainage Paper No. 56.
- **VPD from Tetens**: Tetens, O. (1930) "Über einige meteorologische Begriffe." Z. Geophys. 6, 297–309.
- **Lapse rates in Mediterranean**: Dutra, E., et al. (2020) "ERA5-Land: A state-of-the-art global reanalysis dataset for land applications"

### Related Catalonia Agronomy Studies

- Gencat DARP yield records: https://dadesabiertes.gencat.cat/ (open data portal)
- SIGPAC parcel registry: https://www.generalitat.cat/agricultura (reference for farm counts, irrigation distribution)

---

## Appendix A: Quick Start for Analyses

### Rerun the full pipeline (requires CDS credentials):
```bash
source .venv/bin/activate
python src/parse_yield.py
python src/get_centroids.py
python src/download_agera5.py          # ~2 hours, may require API queue time
python src/extract_climate.py
python src/aggregate_seasonal.py
python src/join_dataset.py
```

### Rerun olive hybrid pipeline only:
```bash
source .venv/bin/activate
python src/run_olive_weighted_pipeline.py  # 4 steps in ~3 min
python src/create_hybrid_olive_dataset.py
```

### Explore olive timing findings:
```bash
# Tmax timing scan
python src/olive_tmax_timing_scan.py     # output: data/olive_tmax_timing_scan.csv
# VPD timing scan
python src/olive_vpd_timing_scan.py      # output: data/olive_vpd_timing_scan.csv
# Water balance scan
python src/olive_cwb_timing_scan.py      # output: data/olive_cwb_timing_scan.csv
```

### Interactive exploration:
```bash
streamlit run src/explorer.py            # browse crops, variables, phenological periods
```

### Plot publication figures:
```bash
python src/plot_features.py --crop olive --var vpd_mean --feat lpf_peak --tau 30
python src/plot_features.py --list-crops    # see available crops
python src/plot_features.py --list-vars     # see available climate variables
```

---

**Document Version**: 2.0 (2026-04-17)
**Author**: PhD Research Project
**Last Updated**: April 17, 2026
**Maintained By**: See CLAUDE.md for environment setup details

