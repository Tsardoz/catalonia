# Olive Climate–Yield Analysis — Current State and Next Steps

## Purpose

Assemble the current status of the olive analysis in Catalonia: what data products are now available, which workflow is recommended, what findings are established, and which questions remain open.

This file now supersedes the older interpretation of the project as only a single `elasticnet_olive_lag.py` experiment on comarca-average climate.

## Headline result (current best olive finding)

- The clearest current olive timing result is a **late July to late August** vulnerable period on the **hybrid daily climate**.
- The strongest simple screening variable is **14-day count of days with `Tmax >= 32°C`**.
- Matching **`VPD >= 3.0 kPa`** timing scans point to the **same late-summer window**, but do not tell a cleaner story than `Tmax`.
- Full-year **`P - ET₀`** scans are useful mainly as **winter/background context**, showing a plausible positive late-December / early-January recharge signal but mixed summer signs.
- These are still **exploratory comarca-year screening results**, not a mechanistic olive water-stress model.

## Current recommended datasets

### Primary seasonal analysis table

- `data/catalan_olive_yield_climate_hybrid.csv`
- **320 rows**, **29 columns**, **36 comarques**, **2016–2024**
- Uses **area-weighted farm-based climate** where available and **centroid fallback** elsewhere
- This is the **recommended dataset for olive regression analysis**

### Supporting daily climate tables

- `data/agera5_daily_catalonia.csv`
  - Original comarca-average daily climate from AgERA5
  - Used by the older annual distributed-lag analysis
- `data/agera5_daily_comarca_olive_weighted.csv`
  - Daily comarca climate weighted by olive farm area
  - Derived from farm-matched AgERA5 cells and elevation-corrected temperature/VPD
- `data/agera5_daily_hybrid_olive.csv`
  - Corrected hybrid daily climate used by the hybrid lag and timing scripts
  - **131,508 rows**, **36 olive comarques**

## What changed in the project

### 1. From comarca-average climate to olive-farm-weighted climate

The original workflow averaged climate across all AgERA5 grid cells inside each comarca polygon.

The project now also supports an olive-specific pipeline that:

1. matches climate to actual olive farm locations
2. applies elevation correction to temperature
3. recomputes VPD using corrected temperature
4. aggregates back to comarca using **farm area weighting**

This is a substantial methodological upgrade for olive analysis because climate is now closer to where olive farms actually occur, instead of where comarca polygons happen to lie.

### 2. Introduction of a hybrid olive dataset

The weighted pipeline covers only the comarques with usable olive farm coverage. To recover full sample size, the project now combines:

- **23 comarques / 207 rows** from the area-weighted farm-based pipeline
- **13 comarques / 113 rows** from the centroid-based pipeline

This yields the hybrid table with **320 total observations**.

### 3. Analysis has moved beyond a single annual lag model

The older annual distributed-lag analysis still exists and remains useful as a baseline, but it is no longer the whole project.

Current work now spans three layers:

- baseline annual lag analysis on comarca-average climate
- improved spatial exposure via weighted/hybrid climate products
- exploratory short-timescale stress analysis focused on pit hardening and red-zone VPD days

## Implemented pipelines and scripts

### Baseline centroid pipeline

- `src/extract_climate.py`
- `src/aggregate_seasonal.py`
- `src/join_dataset.py`
- `src/elasticnet_olive_lag.py`

### Olive farm-weighted pipeline

- `src/extract_climate_olive_farms_daily.py`
- `src/aggregate_farms_to_comarca.py`
- `src/aggregate_seasonal_olive_weighted.py`
- `src/join_dataset_olive_weighted.py`
- `src/run_olive_weighted_pipeline.py`

### Hybrid analysis layer

- `src/compare_weighted_vs_centroid.py`
- `src/create_hybrid_daily_climate.py`
- `src/create_hybrid_olive_dataset.py`
- `src/validate_hybrid_dataset.py`
- `src/elasticnet_olive_lag_hybrid.py`

### Exploratory timing and screening scripts

- `src/olive_summer_red_vpd.py`
- `src/olive_summer_vpd_threshold_screen.py`
- `src/olive_summer_vpd_run_screen.py`
- `src/olive_summer_tmax_threshold_screen.py`
- `src/olive_tmax_timing_scan.py`
- `src/olive_vpd_timing_scan.py`
- `src/olive_tmax_window_compare.py`
- `src/olive_cwb_timing_scan.py`

These scripts are exploratory, but they now provide the clearest current picture of **when** olive yield is most sensitive to hot and dry-air conditions.

## Current validated status

### Hybrid dataset validation

`src/validate_hybrid_dataset.py` runs successfully on `data/catalan_olive_yield_climate_hybrid.csv` and reports:

- **320 rows** loaded
- **no missing values**
- **no duplicate comarca-year rows**
- **207 area-weighted rows**
- **113 centroid rows**
- physically reasonable ranges for yield, VPD, temperature, and precipitation

Minor source-data caveat:

- `Garrotxa` has 8 years of data
- `Pallars Sobirà` has 6 years of data

but the hybrid dataset is otherwise considered ready for analysis.

### Weighted vs centroid comparison

For the overlapping olive comarques, the weighted and centroid products are similar but not identical.

Documented comparison result:

- flowering mean VPD correlation ≈ **0.93** between methods

Interpretation:

- the weighted pipeline changes exposures in a real but not catastrophic way
- topographically varied comarques are the most likely to benefit from the weighted approach
- the hybrid dataset is therefore a pragmatic compromise between better exposure and full sample size

## Established findings from the older annual lag analysis

The older `src/elasticnet_olive_lag.py` results should now be treated as a **baseline exploratory result**, not the final biological interpretation.

Still-useful takeaways from that stage:

1. **Aggregation matters.** Peak-like summaries (for example max VPD) recovered more signal than broad means.
2. **Precipitation performed strongly** at 4-week resolution in ElasticNet.
3. **Whole-year 4-week bins are probably too coarse for VPD stress timing.**
4. **Single-variable annual lag models are informative but hard to interpret biologically** because they spread short stress episodes over long bins.

## Current interpretation of the olive problem

The central question is no longer just “which variable wins in a 13-bin annual lag model?”

The project now appears to be converging on a more specific question:

> Which climate summaries best capture damaging stress to rainfed olive yield once weather is matched more closely to where olive farms actually are?

This shifts attention toward:

- farm-weighted or hybrid climate exposure
- stage-specific analysis rather than whole-year analysis
- short-timescale stress summaries rather than monthly-like averages
- simple screening variables that do not overclaim plant water status

## Current exploratory timing results

### 1. Summer `Tmax` is the cleanest screening variable so far

Using `data/agera5_daily_hybrid_olive.csv`, one-predictor OLS models with comarca fixed effects show that summer hot-day counts are more informative than broad annual-bin summaries.

For the simple summer window `06-20` to `08-15`:

- days with `Tmax >= 30°C`, `32°C`, and `34°C` are all negatively associated with yield
- `Tmax >= 32°C` gives the clearest screening result
- `Tmax >= 36°C` appears too rare to be useful

### 2. The main susceptible period is late July to late August

Timing scans using **14-day rolling counts** and weekly anchor dates show that the strongest negative windows for both:

- `Tmax >= 32°C`
- `VPD >= 3.0 kPa`

cluster in **late July through late August**, with the strongest individual windows ending around **Aug 21**.

This is the clearest current answer to the question of **when** olives appear most sensitive.

### 3. VPD timing broadly agrees with Tmax, but Tmax is simpler

`VPD >= 3.0 kPa` timing scans produce a pattern very similar to the `Tmax >= 32°C` scans:

- positive windows in early July
- strongest negative windows in late July to late August

However, `Tmax` is simpler to interpret and currently gives the cleaner screening story for olive.

### 4. Counts work better than longest consecutive runs

For summer VPD stress screening:

- **total number** of high-VPD days above threshold was informative
- **longest consecutive run** of high-VPD days was weaker and not significant

So cumulative exposure currently looks more useful than a single uninterrupted spell metric.

### 5. Window-length comparison does not change the basic heat-timing story

For `Tmax >= 32°C`, the `7-day`, `14-day`, and `21-day` timing scans all point to the same broad vulnerable period:

- **late July to late August**

Longer windows do mechanically broaden and lag the apparent peak because they are backward-looking rolling sums, so the cleanest compromise is to use the **14-day window** as the default presentation.

### 6. Full-year `P - ET₀` is more useful for background context than for summer timing

The agronomic-year `P - ET₀` scan (`Dec 1 → Nov 30`) is informative, but less clean than `Tmax`:

- it shows a plausible **positive winter signal** in late December / early January
- it still shows mixed-sign summer windows

Current interpretation:

- `P - ET₀` may be useful as a **background recharge / water-balance context** variable
- `Tmax` is the better current variable for identifying the main summer susceptible period

### 7. Why this is still exploratory

These are still simple comarca-year screening models. They should be treated as:

- exploratory associations
- useful for identifying candidate phases and metrics
- not a mechanistic olive water-stress model

## Current recommended workflow

### For seasonal/phenology-scale modelling

Use:

- `data/catalan_olive_yield_climate_hybrid.csv`

This is the best current balance of:

- sample size
- spatial realism
- reproducibility

### For daily or lag-based experimentation

Prefer:

- `data/agera5_daily_hybrid_olive.csv`

over the original comarca-average daily table when the goal is olive-specific inference.

### For current olive timing work

Use:

- `src/olive_tmax_timing_scan.py`
- `data/olive_tmax_timing_scan.csv`
- `figures/olive_tmax_timing_scan.png`

as the main exploratory result, with:

- `src/olive_vpd_timing_scan.py` as a supporting comparison
- `src/olive_cwb_timing_scan.py` as background-context comparison

## Open questions / next steps

1. **Should the main olive analysis move fully to the hybrid daily climate?**
   - likely yes for olive-specific lag work

2. **Are whole-year 4-week bins still the right temporal representation for olive stress timing?**
   - probably not; the timing scans are more informative

3. **Should the primary olive result now be framed around a late-July to late-August heat window?**
   - likely yes, with explicit exploratory caveats

4. **Should stress be represented by threshold-day metrics rather than means?**
   - yes, this now looks more plausible for olive timing work

5. **How should `P - ET₀` be used?**
   - likely as winter/background context rather than the main summer timing variable

6. **Can weather be matched even more directly to olive farms/grid cells for final inference?**
   - this is still the most promising spatial refinement

## Practical recommendation right now

If a regression analysis must be run today, the safest current recommendation is:

1. use `catalan_olive_yield_climate_hybrid.csv` for seasonal models
2. treat the older annual lag results as baseline/exploratory
3. use the weighted/hybrid climate products as the main spatial data layer going forward
4. use the `Tmax >= 32°C` **14-day timing scan** as the clearest current exploratory olive result
5. use `VPD >= 3.0 kPa` timing scans as supporting evidence, not the main headline
6. keep `P - ET₀` framed as background context rather than a clean summer stress metric

## Key commands

```bash
source .venv/bin/activate

# Run weighted olive pipeline
python src/run_olive_weighted_pipeline.py

# Build hybrid datasets
python src/create_hybrid_daily_climate.py
python src/create_hybrid_olive_dataset.py

# Validate hybrid seasonal dataset
python src/validate_hybrid_dataset.py

# Exploratory olive timing scans
python src/olive_tmax_timing_scan.py
python src/olive_vpd_timing_scan.py
python src/olive_cwb_timing_scan.py
python src/olive_tmax_window_compare.py
```
