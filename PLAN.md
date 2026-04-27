# Olive Climate–Yield Analysis — Current State

`WARP.md` is the broader pipeline reference. This file is the working analysis document
for the rainfed olive timing analysis: which datasets are now in use, which findings
are established, which questions remain open.

The previous hybrid pipeline (`*_hybrid_olive*` files, `extract_climate_olive_farms_daily.py`,
`run_olive_weighted_pipeline.py`, etc.) is archived under `archive/`. The current pipeline is
a single-step area-weighted aggregator built on the DUN parcel registry, not a two-source
hybrid.

## Headline result

- The clearest current olive timing result is a **two-channel summer signal** on
  rainfed Arbequina yield in western and southern interior Catalonia: late-July to
  mid-August **heat** and June–July **rain**, both negatively associated with yield,
  partially correlated within comarca, and jointly more informative than either alone.
- On the headline 10-comarca Arbequina panel (Arbequina ≥ 90 % of rainfed olive area,
  ≥ 500 ha rainfed; DUN 2024) the bivariate `log_yield ~ FE + meanTmax + sumPrecip`
  gives within-R² = **0.42** (n=100): summer Tmax β = −0.13 t/ha/°C (t = −2.3), summer
  precip β = −0.010 t/ha/mm (t = −3.9). Univariate within-R² is 0.32 for Tmax alone and
  **0.39 for precip alone** — summer rain is in fact the *stronger* single predictor on
  this panel.
- **Heat channel.** 21-day mean `Tmax` ending **17 August** (DOY 229): t = −6.51,
  p = 4.3e-9, within-R² = 0.32, β = −0.29 t/ha/°C. Sharpens monotonically with window
  length from 7 → 21 d and plateaus by 28 d (~3–4 week stress integration window).
  Continuous `mean Tmax` outperforms the previous canonical `Tmax ≥ 32 °C count` operator
  on the cultivar-restricted set (within-R² 0.32 vs 0.12). `mean VPD` agrees on the same
  vulnerable period but is weaker (within-R² 0.11) — temperature itself, not VPD, is the
  dominant heat signal here.
- **Rain channel.** 30-day precipitation sum ending **19 July** (DOY 200): t = −7.53,
  p = 4.0e-11, within-R² = 0.39, β = −0.0146 t/ha/mm. Same calendar peak across 30/60/90-day
  windows. Signal scales with rain *intensity*, not occurrence: `count ≥ 5 mm` reproduces
  most of the effect (within-R² 0.36) while `count ≥ 1 mm` is weaker (0.24), suggesting
  convective-storm / hail / disease pathways rather than pure cloud-cover (PAR/Tmax-suppression).
- **Winter recharge null.** Adding Dec–Mar `P` or `P − ET₀` to the headline Tmax model on
  this Arbequina panel does not improve fit (within-R² 0.336 → 0.336) and the winter
  coefficient is indistinguishable from zero (t ≈ −0.2 to +0.05). The earlier full-panel
  positive winter signal does not survive cultivar restriction.
- These remain **exploratory comarca-year screening results**, not a mechanistic olive
  water-stress model.

## Current datasets

### Daily climate

| File | Rows | Comarques | Aggregation |
|---|---:|---:|---|
| **`data/agera5_daily_catalonia.csv`** | **157,079** | **42** | **Unweighted spatial mean (used for cultivar-restricted Arbequina analysis to avoid irrigated-cell weighting bias)** |
| `data/agera5_daily_catalonia_elevcor.csv` | 157,079 | 42 | + uniform lapse-rate `*_elevcor` cols |
| `data/agera5_daily_catalonia_oliveweighted.csv` | 131,508 | 36 | Mean weighted by DUN olive area per cell (S+R); used for full-panel rainfed-80 robustness |
| `data/agera5_daily_catalonia_oliveweighted_elevcor.csv` | 131,508 | 36 | + residual lapse `*_elevcor` cols |

The olive-area-weighted file weights AgERA5 cells by total DUN olive area (rainfed +
irrigated) per cell. For Arbequina-dominant comarques like Garrigues and Segrià this
weighting biases the climate signal toward irrigated valley cells, so the headline
Arbequina analysis uses the unweighted file. A future rebuild of the weighted file
using only rainfed parcels is an open task.

### Yield + supporting tables

- `data/catalan_woody_yield_raw.csv` — DARP/Gencat yield, all woody crops, all years (rainfed only).
- **`data/dun/comarca_arbequina_whitelist.csv`** — **10 comarques where Arbequina is ≥ 90 % of rainfed olive area AND rainfed Arbequina ≥ 500 ha (DUN 2024, parcels ≥ 0.5 ha). Used for the headline analysis.**
- `data/dun/comarca_olive_whitelist.csv` — 18 comarques where olive is ≥ 80 % rainfed-area (DUN 2024, parcels ≥ 0.5 ha). Used for the older mixed-cultivar robustness check.
- `data/dun/olive_dun_grid_cells_2024.csv` — olive area per (comarca, AgERA5 cell), used as area weights for the olive-weighted climate file.
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
    --agg mean --window 21 --scan-start 06-01 --scan-end 09-15 \
    --comarca-whitelist data/dun/comarca_arbequina_whitelist.csv \
    --tag arbequina    # uses unweighted agera5_daily_catalonia.csv by default
```

## Sliding-window OLS — headline (Arbequina-restricted, mean Tmax, n=100, 10 comarques)

| window length | best window-end | β (t/ha per °C) | t | p | within-R² |
|---|---|---:|---:|---|---:|
| 7-day  | 09-14 | −0.162 | −4.26 | 5.1e-5 | 0.169 |
| 14-day | 08-17 | −0.240 | −5.19 | 1.4e-6 | 0.232 |
| **21-day** | **08-17** | **−0.290** | **−6.51** | **4.3e-9** | **0.323** |
| 28-day | 08-24 | −0.269 | −6.61 | 2.8e-9 | 0.329 |

- Window peak converges on **mid-to-late August** (08-17 to 08-24) for 14d/21d/28d scans.
- Signal sharpens monotonically with window length and plateaus by 21–28 days, indicating a
  ~3–4 week stress integration period rather than a 2-week pulse.
- 21-day window ending 17 August (covers DOY 209–229, ~28 July → 17 August) is the recommended
  primary specification: same within-R² as 28d to within 0.7 pp, slightly tighter physiological
  window matching pit-hardening, more room for a non-overlapping winter window in any future
  bivariate model.
- Effect size at 21d/08-17: β = −0.29 t/ha per °C of mean Tmax. At a panel-mean rainfed
  Arbequina yield of ~2 t/ha that is roughly **−14 % yield per °C of August mean Tmax**.

## Sliding-window OLS — operator and variable comparison at 14-day window (Arbequina, n=100)

| variable | operator | best window-end | β | t | p | within-R² |
|---|---|---|---:|---:|---|---:|
| `tmax_mean` | `count ≥ 32 °C` | 08-17 | −0.086 | −3.48 | 7.9e-4 | 0.120 |
| **`tmax_mean`** | **`mean`**     | **08-17** | **−0.240** | **−5.19** | **1.4e-6** | **0.232** |
| `vpd_mean`  | `mean`         | 07-27 | −0.861 | −3.38 | 1.1e-3 | 0.114 |

- Continuous `mean Tmax` roughly doubles the within-R² of the threshold-count operator on
  the same 14-day window, confirming that on the cultivar-restricted set the count operator
  was discarding usable temperature information.
- `mean VPD` agrees on the same vulnerable period but is weaker than `mean Tmax`. Temperature
  itself, not VPD, is the dominant one-predictor signal here.

## Summer-rain channel and bivariate model (Arbequina, n=100)

Sliding-window scan over `precip_mean` (`agg=sum`) on the Arbequina panel reveals a
strong negative June–July rain signal that survives all reasonable window-length and
threshold choices:

| spec | best end | β | t | p | within-R² |
|---|---|---:|---:|---|---:|
| precip sum, 30d | **07-19** | −0.0146 | **−7.53** | 4.0e-11 | **0.389** |
| precip sum, 60d | 07-19 (sec.) | −0.0079 | −6.42 | 6.4e-9 | 0.317 |
| precip sum, 90d | 07-22 | −0.0059 | −5.86 | 7.8e-8 | 0.278 |
| count days ≥ 1 mm, 30d | 07-19 | −0.125 | −5.29 | 8.5e-7 | 0.240 |
| **count days ≥ 5 mm, 30d** | **07-19** | **−0.224** | **−7.09** | **3.1e-10** | **0.361** |
| count days ≥ 10 mm, 30d | 07-19 | −0.240 | −4.02 | 1.2e-4 | 0.154 |

- All thresholds and window lengths peak on the **same calendar window-end (19 July)**.
  The signal is a real calendar feature, not a scan artefact.
- **Intensity scales the signal, not occurrence.** `count ≥ 5 mm` (within-R² 0.36) is
  almost as strong as `precip sum` (0.39); `count ≥ 1 mm` (0.24) is much weaker. Rule-out:
  if the mechanism were pure cloud-cover (PAR / Tmax suppression), the ≥1 mm count would
  be the strongest predictor, since cloudy days reduce PAR independent of intensity.
  Result is more consistent with **convective-storm damage** (hail, wind, mechanical
  fruit drop) and/or **disease pressure** (humidity-driven sporulation thresholds).

### Bivariate `Tmax + precip` (Arbequina, n=100)

Heat window: 21d mean Tmax ending 08-17. Rain window: 30d precip sum ending 07-19.
Within-comarca correlation between the two predictors is **r = 0.70** (positive: in
this panel, rainier July ⇒ hotter August, consistent with convective-summer regime
rather than synoptic frontal cooling).

| spec | within-R² | summer Tmax β / t | summer precip β / t |
|---|---:|---|---|
| Univariate Tmax (21d, 08-17) | 0.323 | −0.290 / **−6.51** | — |
| Univariate precip (30d, 07-19) | **0.389** | — | **−0.0146 / −7.53** |
| **Bivariate Tmax + precip** | **0.423** | −0.132 / −2.27 | −0.0104 / −3.91 |

- Both channels retain independent statistical significance, but Tmax's coefficient is
  more than halved and its t-statistic drops from −6.5 to −2.3. Precip's t drops from
  −7.5 to −3.9. The univariate Tmax effect was **partially borrowing strength from the
  precip signal** through their within-comarca correlation; precip is now the more
  robust single-predictor channel on this panel.
- Bivariate within-R² = 0.42 vs 0.32 (Tmax-only) or 0.39 (precip-only): both predictors
  carry independent variance, supporting a "both heat and rain hurt" reading rather
  than either being a pure confounder of the other.

### Winter-recharge null result (Arbequina, n=90)

Adding a fixed Dec–Mar window of `precip` or `P − ET₀` to the headline Tmax model on
the Arbequina panel does not improve fit:

| spec | within-R² | summer Tmax β / t | winter β / t |
|---|---:|---|---|
| Univariate summer Tmax (matched n=90) | 0.336 | −0.369 / −6.32 | — |
| Univariate winter precip (Dec–Mar) | 0.004 | — | −4e-4 / −0.55 |
| Univariate winter P−ET₀ (Dec–Mar) | 0.001 | — | −2e-4 / −0.26 |
| Bivariate Tmax + winter precip | 0.336 | −0.369 / −6.25 | −1e-4 / −0.20 |
| Bivariate Tmax + winter P−ET₀ | 0.336 | −0.370 / −6.27 | +3e-5 / +0.05 |

- Winter Dec–Mar precipitation is essentially **orthogonal** to the headline summer Tmax
  predictor (Pearson r = −0.10) and adds nothing to within-comarca yield variance on the
  Arbequina panel. The earlier full-panel positive winter signal does not survive
  cultivar restriction.
- Winter `P − ET₀` ≈ winter `P` shifted by a constant (correlation 0.97) because winter
  ET₀ is small (~200 mm) and very stable (sd 18 mm) while winter precip is highly variable
  (sd 65 mm). The two carry the same information in this season.
- Implication: the long-standing recharge hypothesis is not supported by an additive
  fixed-window test on this panel. Spring rain, concurrent August rain, and cumulative
  running balance remain untested.

## Sliding-window OLS — robustness check (rainfed-80 whitelist, mixed cultivars, Tmax≥32 count, 14-day window, n=174)

| pipeline | top window | β | t | p | within-R² | total R² |
|---|---|---:|---:|---|---:|---:|
| equal-cell (baseline) | 08-24 | −0.057 | −4.28 | 3.3e-5 | — | 0.427 |
| equal-cell + lapse | 08-24 | −0.046 | −4.06 | 7.9e-5 | — | 0.421 |
| olive-area-weighted | 08-24 | −0.045 | −3.85 | 1.7e-4 | **0.087** | 0.415 |
| olive-area-weighted + lapse | 08-24 | −0.044 | −3.87 | 1.6e-4 | — | 0.416 |

- Same peak window (late August), same sign, same shape across all four pipelines — but
  within-R² is only ~0.09 once comarca fixed effects are stripped out.
- Cultivar restriction (Arbequina) raises within-R² from 0.09 → 0.32 at the comparable
  spec, despite a 43 % drop in n. Comarca-level cultivar/management heterogeneity was the
  dominant noise source in the mixed-cultivar panel.

### Summer-rain channel on the rainfed-80 panel (mixed cultivars, n=174)

| spec | best end | β | t | p | within-R² |
|---|---|---:|---:|---|---:|
| precip sum, 30d (oliveweighted) | **07-19** | −0.0048 | −3.96 | 1.1e-4 | 0.092 |
| precip sum, 60d (oliveweighted) | 08-16 | −0.0039 | −4.00 | 9.8e-5 | 0.094 |
| count days ≥ 5 mm, 30d (oliveweighted) | 07-19 | −0.052 | −2.70 | 7.7e-3 | 0.045 |

- **Same calendar peak (19 July)** as the Arbequina panel for both `precip sum 30d` and
  `count ≥ 5 mm 30d`, confirming the timing is pan-Catalan rather than Arbequina-specific.
- **Effect roughly 4× weaker** than on the Arbequina panel (within-R² 0.09 vs 0.39 for
  precip sum; 0.045 vs 0.36 for count ≥ 5 mm). Same dilution pattern as the Tmax channel:
  cross-cultivar heterogeneity in the rainfed-80 panel obscures both the heat and rain
  signals identically. Direction and timing reproduce; magnitude is recoverable only on
  the cultivar-restricted set.

## Salvaged analytical findings (still valid from the archived analysis)

1. **Counts > runs.** Total number of high-VPD or high-Tmax days above threshold is more
   informative than the longest consecutive run.
2. **Window length.** 7-day, 14-day, and 21-day Tmax scans all point to the same
   late-July to late-August window. On the Arbequina panel 21-day is the cleanest default;
   on the mixed-cultivar panel 14-day was.
3. **Threshold choice.** `Tmax ≥ 30/32/34 °C` are all negatively associated; `≥ 32 °C` is
   the clearest. `≥ 36 °C` is too rare in this panel. `mean Tmax` outperforms all three
   on the cultivar-restricted set.
4. **Early-July positive windows** in the 14-day scans are a backward-looking artefact —
   a window ending early July covers late June, which overlaps the onset of pit hardening
   (consistent with olive RDI practice, not contradictory).
5. **Statistical caveat.** Each OLS uses one climate-window predictor + comarca fixed
   effects. Raw p-values exist, but because the scan tests many correlated windows they
   should be treated as exploratory screening statistics, not confirmatory significance tests.

## Open questions / next steps

1. **Irrigated-olive sign test (next priority).** Refit the headline bivariate
   `Tmax + precip` on *irrigated* Arbequina yield in Segrià / Urgell / Pla d'Urgell
   (Canal d'Urgell zone) using `regadiu_kg_ha` from the DARP yield panel. If the negative
   summer-precip coefficient survives under irrigation it confirms the rain channel is
   storm/disease/PAR-mediated rather than compounded drought; if it vanishes the rainfed
   precip signal is partly water-supply-confounded. Requires (a) a `regadiu`-based
   cultivar-share whitelist from `olives.csv`, (b) `parse_yield.py` already exposes
   `regadiu_ha` and `regadiu_kg_ha`, (c) decision on which climate file to weight by
   (probably unweighted comarca mean for the sign test, irrigated-area-weighted for a
   precise effect size).
2. **Solar radiation channel.** Download AgERA5 `solar_radiation_flux` for 2015–2024,
   add `srad_mean` to `extract_climate.py`, and refit `Tmax + precip + srad` on the
   Arbequina panel. If `srad` absorbs the precip signal the mechanism is PAR loss;
   otherwise the residual precip effect is hail / disease / mechanical damage.
   Recommended only after the irrigated test, which already separates the water-supply
   channel from the storm/disease/PAR channels.
3. **Spring and concurrent moisture windows.** The winter Dec–Mar fixed window was null;
   spring (Apr–Jun) rain, August concurrent rain, and a cumulative running `P − ET₀`
   from 1 October Y-1 remain untested as potential moderators of the summer Tmax channel.
4. **2-D scan of the precip window.** The 19-July peak is robust across 30/60/90-day
   lengths and 1/5/10 mm thresholds, but a joint heatmap of (window-end × window-length)
   would show whether the peak is a sharp ridge or a broad plateau.
5. **Other cultivars.** Repeat the cultivar-restricted bivariate scan for Morruda /
   Empeltre on the southern Tarragona comarques (Baix Ebre, Montsià, Terra Alta) as a
   cross-cultivar comparison. The 4× signal dilution on the rainfed-80 panel may unwind
   when each cultivar is analysed separately.
6. **Rebuild olive-weighted climate using rainfed-only weights.** Modify
   `extract_climate_olive_weighted.py` to filter `sist == "S"` before computing weights,
   then rerun both Arbequina (Tmax) and rainfed-80 (Tmax + precip) scans. Cleanest
   possible specification.
7. **Make the cultivar whitelist a CLI step.** Add `--cultivar` to `dun_rainfed_fraction.py`
   so `comarca_arbequina_whitelist.csv` is regenerable from the pipeline rather than ad-hoc.
8. **Almond.** Once olive is finalised, repeat the area-weighted DUN pipeline for almond
   (pending almond DUN file confirmation from user).

## Known limitations specific to the olive analysis

- DUN olive footprint is a single year (2024). Olive area / cultivar mix is essentially
  static (perennial, <2 % year-on-year change), so this is acceptable.
- Lapse rate is fixed at −5.5 °C/km. Real lapse rates vary seasonally (lower in winter,
  higher in summer) and by weather regime. Mediterranean values may differ from the
  −4.5 °C/km Dutra et al. (2020) reported for ERA5 in western US.
- Sample size is 100 olive comarca-years on the headline Arbequina subset (320 across the
  full olive panel, 174 in the rainfed-80 subset). Adequate for one-predictor screening,
  marginal for interaction detection.
- All 10 Arbequina-dominant comarques cluster in western and southern interior Catalonia
  (Lleida + Tarragona interior). Spatial autocorrelation in climate is high; OLS standard
  errors may be slightly under-estimated. A comarca-clustered or bootstrapped SE would be
  more conservative.
- No true out-of-sample validation. R² values are in-sample.
