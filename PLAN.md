# Olive Climate–Yield Analysis — Current State

`WARP.md` is the broader pipeline reference. This file is the working analysis document
for the rainfed olive timing analysis: which datasets are now in use, which findings
are established, which questions remain open.

The previous hybrid pipeline (`*_hybrid_olive*` files, `extract_climate_olive_farms_daily.py`,
`run_olive_weighted_pipeline.py`, etc.) is archived under `archive/`. The current pipeline is
a single-step area-weighted aggregator built on the DUN parcel registry, not a two-source
hybrid.

## Headline result

- The clearest current olive timing result is a **late-July to late-August** vulnerable
  period for rainfed yield.
- The strongest simple screening variable is the **14-day count of days with `Tmax ≥ 32°C`**,
  with the peak window ending around **24 August** (DOY 236).
- `VPD ≥ 3.0 kPa` timing scans agree with the same late-summer window but do not give a
  cleaner story than `Tmax`.
- Full-year `P − ET₀` is more useful as winter-recharge background context (positive late-
  December / early-January signal) than as a clean summer timing variable.
- These remain **exploratory comarca-year screening results**, not a mechanistic olive
  water-stress model.

## Current datasets

### Daily climate

| File | Rows | Comarques | Aggregation |
|---|---:|---:|---|
| `data/agera5_daily_catalonia.csv` | 157,079 | 42 | Unweighted spatial mean across all comarca cells |
| `data/agera5_daily_catalonia_elevcor.csv` | 157,079 | 42 | + uniform lapse-rate `*_elevcor` cols |
| **`data/agera5_daily_catalonia_oliveweighted.csv`** | **131,508** | **36** | **Mean weighted by DUN olive area per cell (recommended for olive)** |
| `data/agera5_daily_catalonia_oliveweighted_elevcor.csv` | 131,508 | 36 | + residual lapse `*_elevcor` cols |

### Yield + supporting tables

- `data/catalan_woody_yield_raw.csv` — DARP/Gencat yield, all woody crops, all years (rainfed only).
- `data/dun/comarca_olive_whitelist.csv` — 18 comarques where olive is ≥80 % rainfed-area (DUN 2024, parcels ≥ 0.5 ha).
- `data/dun/olive_dun_grid_cells_2024.csv` — olive area per (comarca, AgERA5 cell), used as area weights.
- `data/comarca_olive_elevation_correction_oliveweighted.csv` — residual `dT_C` after area-weighting.

## Pipeline summary (current, end-to-end)

```bash
source .venv/bin/activate
python src/parse_yield.py                          # yield panel
python src/get_centroids.py                        # comarca polygons
python src/download_agera5.py                      # 50 .nc files (CDS API)
python src/extract_climate.py                      # baseline daily climate (unweighted)
python src/extract_climate_olive_weighted.py       # olive-area-weighted daily climate
python src/elevation_correction.py --olive-weighted --apply --plot   # residual lapse
python src/sliding_window_regression.py --crop olive --var tmax_mean \
    --threshold 32 --agg count --window 14 --scan-start 06-01 --scan-end 09-15 \
    --comarca-whitelist data/dun/comarca_olive_whitelist.csv \
    --climate-csv data/agera5_daily_catalonia_oliveweighted.csv --tag oliveweighted
```

## Sliding-window OLS comparison (Tmax≥32 °C count, 14-day window, rainfed-80 whitelist, n=174)

| pipeline | top window | β | t | p | R² |
|---|---|---:|---:|---|---:|
| equal-cell (baseline) | 08-24 | −0.0573 | −4.28 | 3.3e-5 | 0.427 |
| equal-cell + lapse | 08-24 | −0.0456 | −4.06 | 7.9e-5 | 0.421 |
| **olive-area-weighted** | **08-24** | **−0.0447** | **−3.85** | **1.7e-4** | **0.415** |
| olive-area-weighted + lapse | 08-24 | −0.0436 | −3.87 | 1.6e-4 | 0.416 |

- Same peak window, same sign, same shape across all four.
- β shrinks ~22 % after weighting (yield loss per extra Tmax≥32 day in late August: 5.7 % → 4.5 %).
- Adding the residual lapse on top of the weighted file moves β by 0.001 — negligible.
- Area-weighting absorbs ~95 % of the lapse-rate signal because the climate is already biased
  toward cells where olives sit. Residual `dT_C` shrinks 5–10× in the main olive-belt
  comarques (Garrigues +1.27 → −0.10 °C, Segrià +1.16 → −0.07 °C, Baix Ebre +0.83 → +0.16 °C).

The regression is robust to the choice because (a) the rainfed-80 whitelist already excludes
the most heterogeneous comarques (Baix Llobregat, Garraf, Maresme), (b) comarca fixed effects
absorb uniform level shifts, and (c) the threshold-count operator smooths sub-degree changes.
Continuous variables, full-comarca panels, extreme percentiles, and absolute-level metrics
(CWB, VPD threshold scans) would shift more.

## Salvaged analytical findings (still valid from the archived analysis)

1. **Counts > runs.** Total number of high-VPD or high-Tmax days above threshold is more
   informative than the longest consecutive run.
2. **Window length.** 7-day, 14-day, and 21-day Tmax≥32 °C scans all point to the same
   late-July to late-August window. 14-day is the cleanest default.
3. **Threshold choice.** `Tmax ≥ 30/32/34 °C` are all negatively associated; `≥ 32 °C` is
   the clearest. `≥ 36 °C` is too rare in this panel.
4. **Early-July positive windows** in the 14-day scans are a backward-looking artefact —
   a window ending early July covers late June, which overlaps the onset of pit hardening
   (consistent with olive RDI practice, not contradictory).
5. **Statistical caveat.** Each OLS uses one climate-window predictor + comarca fixed
   effects. Raw p-values exist, but because the scan tests many correlated windows they
   should be treated as exploratory screening statistics, not confirmatory significance tests.

## Open questions / next steps

1. **Rerun VPD and CWB threshold scans on the area-weighted file.** Continuous and absolute-
   level metrics are expected to shift more than the Tmax-count regression did. The scripts
   in `archive/src/` referenced in the original plan need to be either revived or
   reimplemented as parameter sets for `sliding_window_regression.py`.
2. **Almond.** Once olive is finalised, repeat the area-weighted DUN pipeline for almond
   (pending almond DUN file confirmation from user).
3. **Pit-hardening framing.** The early-July positive / late-August negative pattern
   appears consistent with the established olive water-stress literature; revisit framing
   when writing.
4. **Spatial refinement.** The current weighting uses DUN 2024 only. Sensitivity to using
   a different DUN year (or a multi-year average) is unexplored.

## Known limitations specific to the olive analysis

- DUN olive footprint is a single year (2024). Olive area is essentially static (perennial,
  <2 % year-on-year change), so this is acceptable.
- Lapse rate is fixed at −5.5 °C/km. Real lapse rates vary seasonally (lower in winter,
  higher in summer) and by weather regime. Mediterranean values may differ from the
  −4.5 °C/km Dutra et al. (2020) reported for ERA5 in western US.
- Sample size is 320 olive comarca-years (or 174 in the rainfed-80 subset) — enough for
  one-predictor screening, marginal for interaction detection.
- No true out-of-sample validation. R² values are in-sample.
