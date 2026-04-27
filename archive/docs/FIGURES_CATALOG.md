# Figure Catalog: Generated Visualizations

**Total figures generated:** 78 PNG files in `/home/tsardoz/phd/catalonia/figures/`

**Total disk space:** ~19 MB

---

## 1. Olive Timing Analysis Figures (MAIN RESULTS)

### Key Timing Scan Figures
- **`olive_tmax_timing_scan.png`** — 14-day rolling windows of days Tmax ≥ 32°C vs. yield
  - Shows strongest negative signal in late July–early August (DOY ~213–235)
  - **Most important exploratory result**

- **`olive_vpd_timing_scan.png`** — 14-day rolling windows of VPD ≥ 3.0 kPa days vs. yield
  - Similar timing pattern to Tmax, supporting evidence
  
- **`olive_cwb_timing_scan.png`** — Full-year P − ET₀ water balance rolling windows
  - Shows positive winter signal; mixed summer; more contextual than primary driver
  
- **`olive_cwb_deficit_timing_scan.png`** — Water stress deficit periods

### Window Length Sensitivity
- **`olive_tmax_window_compare.png`** — Compares 7-day, 14-day, 21-day windows
  - All point to same broad candidate sensitive period (late July–early August)
  - 14-day chosen as best presentation compromise

### Summary Figure
- **`olive_timing_summary.png`** — 4-panel overview combining Tmax, VPD, and CWB timing results
  - Publication-ready composite figure

---

## 2. Olive Screening/Exploratory Figures

### Summer Threshold Screens
- **`olive_summer_tmax_threshold_screen.png`** — Single-window late-July/early-August Tmax ≥ 32°C screening
- **`olive_summer_vpd_threshold_screen.png`** — Single-window VPD ≥ 3.0 kPa screening
- **`olive_summer_vpd_run_screen.png`** — Consecutive-run metrics for VPD
- **`olive_summer_cwb_deficit_screen.png`** — Water deficit screening

### Pit Hardening Period Analysis
- **`olive_pithardening_vpd_red_scan.png`** — VPD "red zone" (≥ 3.5 kPa) during pit hardening (NP)

---

## 3. Olive OLS Period Analysis Figures

### Seasonal Window Comparisons
- **`olive_tmax_yield_periods.png`** — 4-panel OLS: Tmax across SP1/NP/SP2/full-year
- **`olive_tmin_yield_periods.png`** — 4-panel OLS: Tmin across phenological periods
- **`olive_vpd_yield_periods.png`** — 4-panel OLS: VPD across phenological periods
- **`olive_maxvpd_yield_periods.png`** — 4-panel OLS: Max VPD (top-10% stress peaks)
- **`olive_ndays_vpd_yield_periods.png`** — 4-panel OLS: Count of VPD > 2.0 kPa days
- **`olive_precip_yield_periods.png`** — 4-panel OLS: Cumulative precipitation
- **`olive_et0_yield_periods.png`** — 4-panel OLS: Cumulative ET0

### Comparative/Overlay Figures
- **`olive_ols_comparison.png`** — Side-by-side bar chart of R² across all climate variables
- **`olive_ols_comparison_hybrid.png`** — Same comparison using hybrid dataset
- **`olive_ols_vpd_with_cwb_overlay.png`** — VPD OLS with water balance context overlay
- **`olive_ols_vpd_with_cwb_overlay_hybrid.png`** — Hybrid version
- **`olive_ols_et0_with_cwb_overlay.png`** — ET0 OLS with water balance overlay
- **`olive_ols_precip_with_cwb_overlay.png`** — Precipitation OLS with water balance overlay
- **`olive_ols_vpd_with_et0_overlay.png`** — VPD OLS with ET0 overlay
- **`olive_ols_vpd_with_precip_overlay.png`** — VPD OLS with precipitation overlay
- **`olive_ols_vpd_with_precip_overlay_hybrid.png`** — Hybrid version

---

## 4. Olive Distributed-Lag (ElasticNet) Figures

### Impulse Response Functions
- **`olive_irf_vpd.png`** — Impulse response: VPD lags
- **`olive_irf_vpd_hybrid.png`** — Hybrid dataset version
- **`olive_irf_vpd_max.png`** — IRF for max VPD
- **`olive_irf_vpd_max_hybrid.png`** — Hybrid version
- **`olive_irf_vpd_vs_et0.png`** — Comparative IRF: VPD vs. ET0
- **`olive_irf_vpd_vs_et0_hybrid.png`** — Hybrid version
- **`olive_irf_vpd_dry.png`** — IRF for "dry" VPD definition
- **`olive_irf_precip.png`** — Impulse response: Precipitation lags
- **`olive_irf_precip_hybrid.png`** — Hybrid version
- **`olive_irf_nvpd.png`** — Impulse response: Normalized VPD
- **`olive_irf_nvpd_hybrid.png`** — Hybrid version

---

## 5. Cross-Crop Phenological Analysis (LPF Features)

### Top-5 VPD-Yield Scatter Figures (Low-Pass Filter Peak Features)
- **`lpf_top5_olive.png`** — Top 5 VPD signal features for olive
- **`lpf_top5_almond.png`** — Almond
- **`lpf_top5_apricot.png`** — Apricot
- **`lpf_top5_cherry.png`** — Cherry
- **`lpf_top5_hazelnut.png`** — Hazelnut
- **`lpf_top5_walnut.png`** — Walnut
- **`lpf_top5_pome_fruit.png`** — Pome fruit (apple, pear)
- **`lpf_top5_stone_fruit.png`** — Stone fruit aggregated
- **`lpf_top5_vine.png`** — Vine (wine, raisin grape)
- **`lpf_top5_other.png`** — Other crops (pomegranate, persimmon, etc.)

### Detailed LPF Timing Analysis
- **`apricot_vpd_mean_lpf30_by_season.png`** — Apricot VPD with LPF τ=30 by season
- **`apricot_vpd_mean_lpf30_peak_doy.png`** — Peak day-of-year timing for apricot
- **`apricot_vpd_lpf30_three_features.png`** — Peak value, peak DOY, peak width
- **`apricot_vpd_lpf30_windows_plus_doy.png`** — Seasonal windows + DOY overlay
- **`apricot_raw_vpd_windows_plus_doy.png`** — Raw VPD (unfiltered) windows

### Crop-Specific Filter Studies
- **`apple_vpd_max_lpf_doy_tau14.png`** — Apple: max VPD with LPF τ=14
- **`apricot_tmax_mean_lpf_peak_tau14.png`** — Apricot: Tmax with τ=14
- **`apricot_vpd_max_lpf_doy_tau7.png`** — Apricot: VPD max with τ=7
- **`apricot_vpd_mean_lpf_peak_tau30.png`** — Apricot: VPD mean with τ=30

### Individual Crop Pheno Analysis
- **`cherry_flower_set_vpd_yield.png`** — Cherry: flower-set VPD vs. yield
- **`cherry_fruit_dev_vpd_yield.png`** — Cherry: fruit-dev VPD vs. yield
- **`almond_vpd_yield_periods.png`** — Almond period comparison
- **`almond_precip_yield_periods.png`** — Almond precipitation periods
- **`hazelnut_vpd_max_lpf_doy_tau7.png`** — Hazelnut max VPD timing (τ=7)
- **`hazelnut_vpd_max_lpf_width_tau7.png`** — Hazelnut VPD peak width
- **`best_feature_hazelnut_vpd_peak_doy.png`** — Best hazelnut feature
- **`peach_vpd_max_lpf_doy_tau7.png`** — Peach VPD timing
- **`plum_vpd_max_lpf_doy_tau7.png`** — Plum VPD timing
- **`stonefruit_maxvpd_yield_periods.png`** — Stone fruit max VPD periods
- **`stonefruit_precip_yield_periods.png`** — Stone fruit precipitation periods
- **`vine_vpd_yield_periods.png`** — Vine VPD periods

---

## 6. Spatial & Methodology Figures

### Spatial Comparison (Centroid vs. Area-Weighted)
- **`climate_comparison_weighted_vs_centroid.png`** — Map: farm-weighted climate vs. centroid
  - Shows ~0.93 correlation in VPD; largest differences in topographic comarques
  
- **`climate_differences_distribution.png`** — Histogram of climate differences by variable

### Farm-to-Grid Matching
- **`olive_agera5_matching_topo.png`** — Map showing 482 olive farm polygons matched to AgERA5 grid cells
  - Includes topographic contours from DEM
  - Shows elevation correction zones

---

## 7. Timeseries and Diagnostic Figures

### Temperature Diagnostics
- **`tmax_by_month_year.png`** — Heatmap of mean Tmax by month and year (2015–2024)
- **`tmax_heatmap_month_year.png`** — Alternative heatmap visualization
- **`tmax_timeseries.png`** — Daily timeseries of Tmax across Catalonia
- **`vpd_max_timeseries.png`** — Daily timeseries of max VPD

---

## Figure Generation Scripts

**Primary script**: `src/plot_features.py`
- CLI interface: `python src/plot_features.py --crop {crop} --var {var} --feat {feat} --tau {tau}`
- Supports: almond, apricot, cherry, hazelnut, walnut, olive, pome_fruit, stone_fruit, vine, other
- Variables: vpd_mean, vpd_max, precip_mean, et0_mean, tmax_mean
- Features: lpf_peak, lpf_peak_doy, lpf_peak_width, raw_window_mean

**Secondary scripts**:
- `src/olive_tmax_timing_scan.py` → `olive_tmax_timing_scan.png`
- `src/olive_vpd_timing_scan.py` → `olive_vpd_timing_scan.png`
- `src/olive_cwb_timing_scan.py` → `olive_cwb_timing_scan.png`
- `src/olive_tmax_window_compare.py` → `olive_tmax_window_compare.png`
- `src/elasticnet_olive_lag.py` (and `_hybrid.py`) → IRF figures
- `src/plot_olive_timing_summary.py` → `olive_timing_summary.png`
- `src/compare_weighted_vs_centroid.py` → climate comparison figures

