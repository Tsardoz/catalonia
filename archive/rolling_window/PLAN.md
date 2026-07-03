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
  rainfed Arbequina yield in western and southern interior Catalonia: mid-August
  **night-time heat (Tmin)** and June–July **rain**, both negatively associated with
  yield, partially correlated within comarca, and jointly the strongest specification
  found.
- On the headline 10-comarca Arbequina panel (Arbequina ≥ 90 % of rainfed olive area,
  ≥ 500 ha rainfed; DUN 2024) the bivariate `log_yield ~ FE + meanTmin + sumPrecip`
  gives within-R² = **0.47** (n=100): summer Tmin β = −0.275 t/ha/°C (t = −3.64),
  summer precip β = −0.0076 t/ha/mm (t = −2.87). Univariate within-R² is **0.42 for Tmin**,
  0.39 for precip, 0.32 for Tmax.
- **Tmin absorbs Tmax.** When the same model carries both temperature predictors, Tmax
  collapses (Tmin t = −4.19 vs Tmax t = −1.58 in the bivariate; Tmin t = −2.84 vs Tmax
  t = −0.68 in the trivariate Tmin + Tmax + precip). Tmax was a noisier proxy for the
  same underlying heat-stress signal that Tmin carries directly. Adding Tmax to a
  Tmin + precip model raises within-R² by only 0.003 (0.469 → 0.472).
- **Heat channel (Tmin).** 21-day mean `Tmin` ending **17 August** (DOY 229): t = −8.02,
  p = 4.0e-12, within-R² = 0.42, β = −0.43 t/ha/°C. Same calendar peak as the Tmax scan,
  same monotonic sharpening with window length, but with stronger signal at every length
  (7d/14d/21d/28d Tmin within-R²: 0.21 / 0.30 / 0.42 / 0.35 vs Tmax 0.17 / 0.23 / 0.32 / 0.33).
  Mechanism is consistent with night-respiration drain on photosynthate, loss of the
  overnight stomatal-recovery window, and oil-quality penalty under tropical-night-style
  minima — all well-documented in the olive-yield literature.
- **Heat channel (Tmax, demoted).** 21-day mean `Tmax` ending **17 August**: t = −6.51,
  within-R² = 0.32, β = −0.29 t/ha/°C. Continuous `mean Tmax` outperforms the previous
  canonical `Tmax ≥ 32 °C count` operator on the cultivar-restricted set (0.32 vs 0.12).
  Retained as a familiar reference variable but not the primary heat predictor.
  `mean VPD` (within-R² 0.11 at the same window) is weaker than both temperature
  predictors — VPD adds nothing on this panel.
- **Rain channel.** 30-day precipitation sum ending **19 July** (DOY 200): t = −7.53,
  p = 4.0e-11, within-R² = 0.39, β = −0.0146 t/ha/mm. Same calendar peak across 30/60/90-day
  windows. Signal scales with rain *intensity*, not occurrence: `count ≥ 5 mm` reproduces
  most of the effect (within-R² 0.36) while `count ≥ 1 mm` is weaker (0.24), suggesting
  convective-storm / hail / disease pathways rather than pure cloud-cover (PAR/Tmax-suppression).
- **Winter recharge null.** Adding Dec–Mar `P` or `P − ET₀` to the headline temperature
  model on this Arbequina panel does not improve fit (within-R² 0.336 → 0.336) and the
  winter coefficient is indistinguishable from zero (t ≈ −0.2 to +0.05). The earlier
  full-panel positive winter signal does not survive cultivar restriction.
- **Irrigated-olive sign test (passes for both channels).** On an 8-comarca irrigated
  Arbequina panel (≥ 90 % cultivar share, ≥ 200 ha rainfed; n=80) the univariate
  Tmin coefficient is **β = −0.519 t/ha/°C (t = −3.58)** and the univariate precip
  coefficient is **β = −0.0183 t/ha/mm (t = −3.83)** — same sign and *larger* magnitude
  than the rainfed-control 8-comarca panel (Tmin β = −0.366; precip β = −0.0109).
  Within-R² roughly halves under irrigation (0.40 → 0.19 for the bivariate), consistent
  with irrigation absorbing the water-supply half of both stresses but leaving the
  non-water mechanisms (night-respiration / oil-quality for Tmin; storm / disease /
  PAR for precip) intact. See *Irrigated-olive sign test* below.
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

- `data/catalan_woody_yield_raw.csv` — DARP/Gencat yield, all woody crops, all years.
  Schema as of Apr 2026: `seca_ha`, `seca_kg_ha`, `yield_tha` (rainfed) plus
  `regadiu_ha`, `regadiu_kg_ha`, `yield_irrig_tha` (irrigated). A row is kept if
  *either* regime has positive area and yield; the dropped regime's `yield_*_tha`
  is `NaN`. Existing rainfed analyses that filter on `yield_tha > 0` are unaffected.
- **`data/dun/comarca_arbequina_whitelist.csv`** — **10 comarques where Arbequina is ≥ 90 % of rainfed olive area AND rainfed Arbequina ≥ 500 ha (DUN 2024, parcels ≥ 0.5 ha). Used for the headline rainfed analysis.**
- **`data/dun/comarca_irrigated_arbequina_whitelist.csv`** — **8 comarques where Arbequina is ≥ 90 % of *irrigated* olive area AND irrigated Arbequina ≥ 200 ha (DUN 2024). Used for the irrigated sign test.**
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

## Sliding-window OLS — Tmin scan (Arbequina, mean Tmin, n=100)

| window length | best window-end | β (t/ha per °C) | t | p | within-R² |
|---|---|---:|---:|---|---:|
| 7-day  | 09-07 | −0.154 | −4.86 | 5.0e-6 | 0.210 |
| 14-day | 08-17 | −0.341 | −6.19 | 1.8e-8 | 0.301 |
| **21-day** | **08-17** | **−0.433** | **−8.02** | **4.0e-12** | **0.419** |
| 28-day | 08-31 | −0.571 | −6.80 | 1.2e-9 | 0.342 |

- **Same calendar peak as the Tmax scan** (mid-to-late August window-end), same monotonic
  sharpening with window length, but **Tmin gives a stronger signal at every window length**
  (within-R² 0.21 / 0.30 / 0.42 / 0.34 vs Tmax 0.17 / 0.23 / 0.32 / 0.33).
- The 21-day window ending 17 August is the global maximum on the Arbequina panel for any
  single climate predictor tested to date.
- A wide-scan (03-01 → 11-15, anchor-step 7 d) confirms the August peak is the global
  maximum: no comparable spring or autumn Tmin signal.
- Effect size at 21d/08-17: β = −0.43 t/ha per °C of mean Tmin. At a panel-mean rainfed
  Arbequina yield of ~2 t/ha that is roughly **−21 % yield per °C of August mean Tmin**.

## Temperature decomposition: Tmin vs Tmax (Arbequina, 21d / 08-17, n=100)

When Tmin and Tmax compete in the same FE-OLS, **Tmax collapses to non-significance**:

| spec | within-R² | summer Tmin β / t | summer Tmax β / t | summer precip β / t |
|---|---:|---|---|---|
| Tmin alone | **0.419** | **−0.433 / −8.02** | — | — |
| Tmax alone | 0.323 | — | −0.290 / −6.51 | — |
| Tmin + Tmax | 0.435 | **−0.338 / −4.19** | −0.097 / −1.58 (ns) | — |
| Tmin + precip | **0.469** | **−0.275 / −3.64** | — | **−0.0076 / −2.87** |
| Tmax + precip | 0.423 | — | −0.132 / −2.27 | −0.0104 / −3.91 |
| Tmin + Tmax + precip | 0.472 | **−0.246 / −2.84** | −0.044 / −0.68 (ns) | **−0.0069 / −2.45** |

- **Within-comarca correlations** at the headline windows: Tmin ↔ Tmax = 0.75, Tmin ↔ precip
  = 0.73, Tmax ↔ precip = 0.70. All three predictors carry overlapping information; each
  also carries some independent variance.
- **Tmin absorbs Tmax, not the other way around.** In every multivariate spec containing
  both temperature predictors, Tmin survives at p ≪ 0.01 while Tmax falls to p > 0.1.
  The previous "heat channel" in this dataset is best read as a *night-warmth* signal that
  Tmax was tagging only as a noisy proxy.
- **Adding Tmax to a Tmin + precip model raises within-R² by 0.003** (0.469 → 0.472). Tmax
  is statistically and practically redundant once Tmin is in the model.
- **The new clean two-channel headline is `Tmin + precip`** (within-R² = 0.47, both
  channels significant), not `Tmax + precip` (0.42).

### Why Tmin may carry more signal than Tmax here

1. **Night-active oil biosynthesis.** Mesocarp lipid synthesis peaks during dark hours;
   high Tmin sustains respiration that drains the photosynthate pool feeding it.
2. **Loss of stomatal-recovery window.** Tropical-night-style minima eliminate the
   overnight rehydration / cavitation-recovery period — the same mechanism implicated
   in the Andalusian olive-yield decline literature for the past decade.
3. **Oil-quality penalty.** High Tmin during the same window shifts fatty-acid composition
   (less oleic, more linoleic), a quality-side cost even when yield is preserved.
4. **Spatial decorrelation from elevation.** Tmin varies more sharply than Tmax with
   elevation, slope position and cold-air drainage, so within-comarca cell-to-cell
   heterogeneity in Tmin is larger and the comarca mean has *more signal-relative-to-noise*
   for Tmin in this panel. Part of the win may be measurement, not biology.

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

### Bivariate `Tmax + precip` (Arbequina, n=100, superseded by `Tmin + precip` above)

Heat window: 21d mean Tmax ending 08-17. Rain window: 30d precip sum ending 07-19.
Within-comarca correlation between the two predictors is **r = 0.70** (positive: in
this panel, rainier July ⇒ hotter August, consistent with convective-summer regime
rather than synoptic frontal cooling).

| spec | within-R² | summer Tmax β / t | summer precip β / t |
|---|---:|---|---|
| Univariate Tmax (21d, 08-17) | 0.323 | −0.290 / **−6.51** | — |
| Univariate precip (30d, 07-19) | **0.389** | — | **−0.0146 / −7.53** |
| **Bivariate Tmax + precip** | **0.423** | −0.132 / −2.27 | −0.0104 / −3.91 |

- Retained as a reference specification; replaced as headline by `Tmin + precip`
  (within-R² 0.469) once Tmin was added to the candidate set. Both channels here
  retain independent significance, but the Tmax coefficient drops by more than half
  vs the univariate, and the entire Tmax signal is in turn absorbed by Tmin once
  the trivariate is run (see decomposition table above).
- Within-comarca correlation pattern (Tmax ↔ precip = 0.70) is the canonical
  convective-summer regime signature for inland Catalonia: hot Augusts go *with*
  rainier Julys here, not against them.

## Irrigated-olive sign test (Arbequina, headline windows, n=80)

The two-channel `Tmin + precip` headline was refit on an **irrigated** Arbequina
panel built from `regadiu_kg_ha` in the regenerated yield file. Whitelist criterion:
Arbequina ≥ 90 % of *irrigated* olive area AND ≥ 200 ha irrigated Arbequina (DUN 2024).
This produces 8 comarques (Garrigues, Segrià, Baix Camp, Priorat, Urgell, Noguera,
Tarragonès, Alt Camp) covering the Canal d'Urgell zone plus the Camp de Tarragona /
Priorat irrigated belts. Same headline windows as the rainfed model (Tmin 21d / 08-17,
precip 30d / 07-19). The 200 ha threshold (relaxed from the rainfed 500 ha) is needed
because irrigated Arbequina area is smaller than rainfed in most comarques outside
Garrigues / Segrià / Baix Camp.

| panel | n | spec | Tmin β / t | precip β / t | within-R² |
|---|--:|---|---|---|--:|
| Rainfed Arbequina (10-comarca headline) | 100 | Tmin alone | **−0.433 / −8.02** | — | **0.419** |
| Rainfed Arbequina (10-comarca headline) | 100 | precip alone | — | **−0.0146 / −7.53** | **0.389** |
| Rainfed Arbequina (10-comarca headline) | 100 | Tmin + precip | −0.275 / −3.64 | −0.0076 / −2.87 | **0.469** |
| Rainfed control, *same 8* as irrigated | 80 | Tmin alone | −0.366 / −6.60 | — | 0.381 |
| Rainfed control, *same 8* as irrigated | 80 | precip alone | — | −0.0109 / −5.51 | 0.299 |
| Rainfed control, *same 8* as irrigated | 80 | Tmin + precip | −0.277 / −3.43 | −0.0041 / −1.50 (ns) | 0.400 |
| **Irrigated Arbequina (8-comarca)** | **80** | **Tmin alone** | **−0.519 / −3.58** | — | 0.153 |
| **Irrigated Arbequina (8-comarca)** | **80** | **precip alone** | — | **−0.0183 / −3.83** | 0.171 |
| **Irrigated Arbequina (8-comarca)** | **80** | **Tmin + precip** | −0.252 / −1.20 (ns) | −0.0122 / −1.73 (p=.089) | 0.188 |

- **Both univariate channels keep the predicted negative sign under irrigation.** The
  precip coefficient is *larger* in absolute terms on the irrigated panel than on the
  rainfed control (β = −0.0183 vs −0.0109); the Tmin coefficient is also larger
  (β = −0.519 vs −0.366). The sign test passes for both: neither signal collapses to
  zero or flips sign once water-supply confounding is removed.
- **Within-R² roughly halves** (0.40 → 0.19 for the bivariate). Approximately half of
  the rainfed within-comarca variance was indeed water-supply-mediated and is removed
  by irrigation; the remaining half is the non-water mechanisms.
- **Bivariate split is unstable at n=80 with high collinearity.** Within-comarca
  Tmin ↔ precip correlation is r = 0.73 on this panel and only ~19 % of within
  variance remains to share between the two predictors. Both lose individual
  significance in the bivariate (Tmin t = −1.20, precip t = −1.73). The univariate
  sign tests are the cleaner result; the bivariate decomposition should not be
  over-interpreted at this n.
- **Tmax stays demoted under irrigation.** Bivariate Tmin + Tmax on the irrigated
  panel: Tmin t = −2.61, Tmax t = +0.23 (ns). The Tmin > Tmax dominance is preserved
  across both regimes.
- **Mechanism implication.** If the rainfed precip signal had been primarily a hidden
  drought-compounding proxy, irrigation should have eliminated it. It did not. The
  surviving (and amplified) precip coefficient is consistent with the
  storm/disease/PAR mechanism class. Symmetrically, if the rainfed Tmin signal had
  been primarily transpirational night-cooling loss, irrigation should have dampened
  it. It did not. The surviving Tmin coefficient is consistent with the
  night-respiration / oil-quality mechanism class.
- **What this does *not* say.** The panel measures yield, not applied irrigation
  volume. It does not address whether irrigated comarques could maintain yields
  with less summer water — only that the *residual* yield variance, conditional on
  whatever irrigation was actually applied, is dominated by mechanisms irrigation
  cannot offset.

### Winter-recharge null result (Arbequina, n=90)

Adding a fixed Dec–Mar window of `precip` or `P − ET₀` to a summer-Tmax model on
the Arbequina panel does not improve fit (winter null is expected to carry over to
a Tmin model — the winter signal is weaker than the within-comarca SE either way,
but a Tmin + winter retest is an open task):

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

1. ~~**Irrigated-olive sign test.**~~ **Done (Apr 2026).** See *Irrigated-olive sign
   test* section above. Both univariate channels survive irrigation with the same
   negative sign and slightly larger magnitude, supporting the non-water mechanism
   class for both Tmin and precip. Bivariate split unstable at n=80 with r=0.73
   collinearity.
2. ~~**Phenological-stage (GDD) alignment of the headline windows.**~~ **Done
   (Apr 2026).** GDD₁₀ anchoring with per-comarca-year *local* cumgdd-vs-date
   mapping was implemented in `src/compute_gdd.py` and as `--anchor gdd` in
   `src/sliding_window_regression.py`. **Result: GDD anchoring weakens the
   headline within-R² on both panels for both channels** (Arbequina rainfed
   Tmin 21d: 0.42 → 0.23; precip 30d: 0.39 → 0.25; irrigated 8-comarca:
   0.15 → 0.14 and 0.17 → 0.13). See *GDD-sync test result* below. The
   precip null was predicted by the calendar-driven storm/disease/PAR
   mechanism class. The Tmin null is more informative: within the warm
   western/southern Arbequina comarques cross-(comarca, year) variation in
   heat-accumulation *timing* is itself yield-relevant, so forcing the same
   physiological-stage comparison destroys signal rather than tightening it.
3. **Tmin replication on the rainfed-80 panel.** The Tmin > Tmax finding has only been
   demonstrated on the Arbequina panel. A scan of `tmin_mean` 21d on the rainfed-80
   mixed-cultivar panel would confirm whether night-warmth dominance is Arbequina-specific
   or pan-Catalan. Cheap (one CLI invocation).
4. ~~**Solar radiation channel.**~~ **Done (Apr 2026).** AgERA5
   `solar_radiation_flux` 2015–2024 added to the daily climate file; the trivariate
   `Tmin + precip + srad` refit is reported in the *Solar radiation (PAR) channel
   result* section below. PAR-loss is refuted; srad enters as a third independent
   negative channel rather than absorbing the precip signal.
5. **Spring and concurrent moisture windows.** The winter Dec–Mar fixed window was null
   against Tmax. Cumulative running `P − ET₀` from 1 October Y−1 was tested in Apr 2026
   and is also null on both panels (see *Cumulative agronomic-year P − ET₀ result* below).
   Spring (Apr–Jun) rain and August concurrent rain remain untested. Re-run with Tmin as
   the heat predictor.
6. **2-D scan of the precip window.** The 19-July peak is robust across 30/60/90-day
   lengths and 1/5/10 mm thresholds, but a joint heatmap of (window-end × window-length)
   would show whether the peak is a sharp ridge or a broad plateau.
7. **Other cultivars.** Repeat the cultivar-restricted Tmin + precip bivariate for
   Morruda / Empeltre on the southern Tarragona comarques (Baix Ebre, Montsià, Terra Alta)
   as a cross-cultivar comparison. The 4× signal dilution on the rainfed-80 panel may
   unwind when each cultivar is analysed separately.
8. **Rebuild olive-weighted climate using rainfed-only weights.** Modify
   `extract_climate_olive_weighted.py` to filter `sist == "S"` before computing weights,
   then rerun both Arbequina (Tmin) and rainfed-80 (Tmin + precip) scans. Cleanest
   possible specification.
9. ~~**Make the cultivar whitelist a CLI step.**~~ **Done (Apr 2026).** Subsumed by
   `src/dun_cultivar_fraction.py`, which takes `--crop / --regime / --cultivar /
   --campaign / --threshold / --min-cult-ha` and writes `comarca_{regime}_{cultivar}_*.csv`
   pairs. Both `comarca_arbequina_whitelist.csv` (rainfed) and
   `comarca_irrigated_arbequina_whitelist.csv` are now reproducible from the CLI.
10. **Almond.** Once olive is finalised, repeat the area-weighted DUN pipeline for
    almond (pending almond DUN file confirmation from user).
11. **Comarca-clustered / HC1 standard errors.** Current SEs are OLS, likely
    under-estimated under spatial autocorrelation. Wrap the FE-OLS in
    `statsmodels` cluster-robust SEs or a simple block bootstrap on comarca and
    re-report the headline t-statistics for both rainfed and irrigated panels.

## GDD-sync test result (Apr 2026)

**Hypothesis (going in).** The 0.47 within-R² of the rainfed Arbequina headline
(and the 0.19 under irrigation) is partly capped by *phenological misalignment*:
the calendar window 28 Jul → 17 Aug catches mid-pit-hardening / oil-accumulation
in cooler upland Priorat or Tarragona interior years and early-fruit-set in
warmer Lleida-valley years. Aligning each comarca-year to the same accumulated
heat should reduce within-comarca residual variance and tighten the headline
t-stats.

**Implementation.** Cumulative growing-degree-days were computed per
(comarca, day) from 1 January with base temperature 10 °C (single-base method,
standard for olive; cf. De Melo-Abreu et al. 2004; Aguilera & Valenzuela 2012):

    GDD(d) = max( 0.5 (Tmax + Tmin) − 10, 0 )
    cumGDD(d) = Σ GDD(d') for d' = Jan 1 … d

`src/compute_gdd.py` writes `data/agera5_gdd_catalonia.csv` (157,079 rows,
43 comarques, 2015–2024); end-of-year cumGDD ranges across comarques from ~270
in cold Pyrenean Aran to ~2900 in interior Lleida/Tarragona, year-median
~2100. `src/sliding_window_regression.py --anchor gdd` then scans over cumGDD
endpoints; **for each (comarca, year) row separately** it looks up the first
date on which that row's local cumGDD crosses the scan endpoint and aggregates
the climate window ending on that local date. The mapping is **per-comarca-year
local** — there is no single panel-level cumGDD→date table — so each row is
genuinely anchored to its own physiological stage rather than to a panel
average.

Reference olive phenological anchors (literature ranges, used here only for
interpretation, not as fixed scan points):

  - **A1: end of flowering / fruit set onset** ≈ 600–800 GDD₁₀ from Jan 1
  - **A2: pit hardening onset** ≈ 1400–1700 GDD₁₀
  - **A3: oil accumulation onset** ≈ 2200–2500 GDD₁₀
  - **A4: full ripeness / harvest readiness** ≈ 3000–3500 GDD₁₀

**Result: GDD anchoring weakens the signal on every panel × channel tested.**

| panel | predictor | calendar within-R² (anchor) | GDD within-R² (best end / panel-median end date) |
|---|---|---:|---:|
| Rainfed Arbequina (n=100, 10 comarques) | Tmin 21d | **0.419** (08-17) | 0.232 (cumGDD 1500 / 08-07) |
| Rainfed Arbequina (n=100, 10 comarques) | precip 30d | **0.389** (07-19) | 0.254 (cumGDD 1000 / 07-10) |
| Irrigated Arbequina (n=80, 8 comarques) | Tmin 21d | **0.153** (08-17) | 0.137 (cumGDD 1700 / 08-21) |
| Irrigated Arbequina (n=80, 8 comarques) | precip 30d | **0.171** (07-19) | 0.129 (cumGDD 1000 / 07-22) |

GDD scan grid: 800 → 2400 in steps of 50 (Tmin) and 600 → 2200 in steps of 50
(precip). Scan range covers anchors A1 → A3. The "panel-median end date" column
is the modal MM-DD of the per-comarca-year window-end dates that the
local-cumGDD lookup produces at the best-fitting cumGDD endpoint, given here
only as a calendar-context aid; the regression itself uses the per-row local
end date, not the modal date.

**Interpretation.**

1. **Precip null was predicted.** If the rainfed precip damage is storm /
   disease / PAR-driven (mechanism class also supported by the irrigated sign
   test), it should follow the *calendar* — convective season, hail climatology,
   fly population dynamics — not the phenology. The drop from 0.39 → 0.25 on
   the rainfed panel and 0.17 → 0.13 on the irrigated panel is consistent with
   that mechanism class. GDD anchoring is the wrong reference frame for this
   channel.
2. **Tmin null is more informative.** The Arbequina panel sits inside a
   relatively narrow climatic envelope (warm western/southern interior
   Catalonia; end-of-year cumGDD on these 10 comarques ranges only ~1900–2700,
   not the full 270–2900 spread of the AgERA5 grid). Within that envelope,
   cross-(comarca, year) variation in heat-accumulation *timing* is itself
   yield-relevant: a year that hits cumGDD 1500 by 7 Aug behaves differently
   from a year that hits it 17 Aug, even if both are at the "same"
   physiological stage. Forcing a same-stage comparison destroys that
   timing-of-heat-arrival information without buying a compensating reduction
   in residual variance, so within-R² falls.
3. **The calendar anchoring already captures most of the available
   physiological alignment** for this cultivar-restricted, climate-narrow
   panel. The mid-August calendar window happens to catch a similar
   physiological window across these 10 comarques because they are similar
   climatically; the small remaining heat-accumulation timing differences
   carry their own signal that calendar mode preserves but GDD mode normalises
   away.
4. **What this does *not* rule out.** A GDD anchor might still help on a
   broader, more climatically heterogeneous panel (e.g. the rainfed-80 mixed
   cultivar set, which spans cold-upland and warm-interior comarques). It is
   also possible that olive-specific phenological clocks — chilling-fulfilment
   plus heat-forcing (UNIFORC / chill-portions), rather than vegetative GDD₁₀
   from 1 January — are needed to capture the right physiological frame.
   Both remain open extensions if the broader panel rehabilitates the GDD
   approach.

**Headline windows therefore stay calendar-anchored.** No revision to the
`Tmin 21d ending 08-17` / `precip 30d ending 07-19` headline specification.

## Cumulative agronomic-year P − ET₀ result (Apr 2026)

**Hypothesis (going in).** The fixed Dec–Mar `P − ET₀` window was null against
Tmax (winter recharge null), but a fixed-window test cannot capture *chronic*
balance: a dry winter followed by an average summer would look identical to
an average winter followed by an average summer in any 30-day summer window.
Cumulative running `P − ET₀` from 1 October Y−1 (the standard agronomic-year
start for Mediterranean perennials) is the sharpest possible single-number
test of whether the rainfed precip channel is in fact a hidden moisture-supply
proxy: if the precip damage were chronically water-mediated, this predictor
should outperform any localised calendar window.

**Implementation.** `src/compute_water_balance.py` writes
`data/agera5_water_balance_catalonia.csv` (157,079 rows; daily and cumulative
`P − ET₀`, reset on 1 October so each comarca-year accumulates over 1 Oct Y−1
→ 30 Sep Y). End-of-agro-year (30 Sep) totals across comarques span ~−1000 mm
(dry interior Lleida) to ~+1000 mm (wet Pyrenees), with median ~−400 mm.
Scanned with `sliding_window_regression.py --var cum_p_minus_et0 --agg mean
--window 1` (window = 1 day reduces the rolling mean to the cumulative balance
on the endpoint date itself), 04-01 → 09-15, weekly endpoints.

**Result: null on both panels.**

| panel | best endpoint | β (t/ha per mm) | t | p | within-R² | (cf. calendar precip 30d, 07-19) |
|---|---|---:|---:|---:|---:|---:|
| Rainfed Arbequina (n=100) | 07-08 | −0.00048 | −1.22 | 0.22 | **0.017** | 0.389 |
| Irrigated Arbequina (n=80) | 08-26 | −0.00165 | −1.86 | 0.067 | **0.047** | 0.171 |

Across the full scan grid every endpoint stays at within-R² < 0.02 on the
rainfed panel and < 0.05 on the irrigated panel; signs are uniformly negative
(more deficit → lower yield, directionally correct) but t-statistics never
exceed 2 in magnitude.

**Interpretation.** This is the second independent failed test of the
chronic-moisture-supply hypothesis (the first was the Dec–Mar fixed window,
within-R² 0.001–0.004). Cumulative `P − ET₀` from the agronomic-year start
explains roughly **4 % as much within-comarca variance as the 30-day
calendar precip window** on the rainfed panel. If the rainfed precip channel
were a hidden chronic-water-balance proxy, the ranking would be reversed —
cum P − ET₀ would dominate and the calendar window would be a noisy summary
of it. The opposite is true.

Combined with the irrigated sign test (which showed the precip coefficient
*survives and amplifies* under irrigation), this triangulates the rainfed
precip channel away from any moisture-supply mechanism class and toward
the calendar-localised mechanism classes: convective-storm damage (hail /
wind / mechanical fruit drop), humidity-driven disease pressure (anthracnose,
peacock spot, olive fruit fly), and PAR loss from convective-cloud cover.
The forthcoming AgERA5 `solar_radiation_flux` test (next-steps item 4) will
discriminate the PAR-loss subset from the storm/disease subset.

**Unchanged headline.** No revision to `Tmin 21d ending 08-17 + precip 30d
ending 07-19`. Univariate cum P − ET₀ adds no fit and was therefore not
carried to a bivariate `Tmin + cum_pet` confirmation; with t = −1.22 in the
univariate it would be subsumed by either headline channel.

## Solar radiation (PAR) channel result (Apr 2026)

**Hypothesis (going in).** The rainfed precip channel could in principle be
a PAR-loss signal in disguise: convective cloud cover during 30 days ending
19 July reduces incoming shortwave, which reduces photosynthate accumulation
during fruit set / early oil deposition, which reduces yield. Under this
mechanism, daily-integrated solar radiation `srad` would enter the regression
**positively** (sunnier = more PAR = more carbon = more yield), and adding
`srad` to `Tmin + precip` would absorb a substantial share of the negative
precip coefficient (because `srad` is the upstream variable and precip is its
downstream proxy). Refuting this requires a sign test on `srad` and a
controlled-coefficient test on `precip` after `srad` is included.

**Implementation.** AgERA5 `solar_radiation_flux` (J m⁻² day⁻¹) added to
`SINGLE_VARIABLES` in `src/download_agera5.py` (version `2_0`, no statistic);
10 annual NetCDFs downloaded (2015–2024). Spatial aggregation in
`src/extract_climate.py` extended with `solar_radiation_flux → srad` (divide
by 1e6 for MJ m⁻² day⁻¹). Daily comarca CSV regenerated; magnitudes are
correct for ~41 °N inland Mediterranean (Garrigues June peak ~26 MJ m⁻² d⁻¹,
December low ~7 MJ m⁻² d⁻¹). Univariate sliding-window scans (`srad_mean`,
30-day mean, 06-01 → 09-15) on both Arbequina panels:

| panel | best endpoint | β (t/ha per MJ m⁻² d⁻¹) | t | p | within-R² |
|---|---|---:|---:|---:|---:|
| Rainfed Arbequina (n=100), w=30d | 08-10 | **−0.216** | −4.11 | 8.7×10⁻⁵ | **0.160** |
| Rainfed Arbequina (n=100), w=21d | 08-03 | −0.214 | −4.22 | 5.9×10⁻⁵ | 0.167 |
| Irrigated Arbequina (n=80), w=30d | 08-03 | −0.505 | −2.86 | 5.6×10⁻³ | 0.103 |

**The univariate srad coefficient is NEGATIVE on both panels.** That single
fact refutes PAR-loss as the dominant mechanism for the rainfed precip
channel. Under PAR-loss, more sunshine in early August would have to mean
*more* yield. It does not — sunnier early-August fortnights predict *lower*
yield. Srad is acting as a heat / radiation-load indicator, not as a PAR
resource. The irrigated sign test confirms the sign and roughly doubles the
magnitude (β = −0.505 vs −0.216), the same sign-keeping / magnitude-amplifying
pattern observed on Tmin and precip.

**Trivariate confirmation** (`src/trivariate_arbequina_par.py`,
rainfed Arbequina, n=100):

```
                                    within-R²    Tmin t    precip t   srad t
Univariate Tmin (21d / 08-17)         0.419     −8.02       —         —
Univariate precip (30d / 07-19)       0.389       —       −7.53       —
Univariate srad (30d / 08-10)         0.160       —         —       −4.11
Bivariate Tmin + precip               0.469     −3.64     −2.87       —
Bivariate Tmin + srad                 0.463     −7.05       —       −2.67
Bivariate precip + srad               0.416       —       −6.21     −1.99
Trivariate Tmin + precip + srad       0.493     −3.64     −2.26     −2.02
```

All three predictors keep significance (|t| > 2) in the trivariate.
Adding srad to the headline `Tmin + precip` model lifts within-R² from
0.469 to 0.493 (+0.024, ~5 % relative gain). Within-comarca correlations
in the cultivar-restricted panel: Tmin × precip = 0.73, precip × srad =
0.40, Tmin × srad = 0.31; collinearity is non-trivial but does not
collapse any channel.

**Critical controlled-coefficient test.** The precip coefficient under
`Tmin + precip` is β = −0.0076; under the trivariate `Tmin + precip +
srad` it is β = −0.0061. Adding srad shrinks the precip coefficient by
about 20 %, but precip remains highly significant (t = −2.26, p = 0.026)
and keeps its negative sign. **Cloudiness, as captured directly by the
inverse of srad, accounts for at most ~20 % of the precip damage; the
other ~80 % of the precip channel is intrinsic to rain hitting the
orchard.** This is exactly the residual structure expected if the
mechanism is convective-storm damage (hail / mechanical fruit drop) and
humidity-driven disease pressure (anthracnose, peacock spot, olive fruit
fly), which require water on the trees rather than absent sunshine.

**Mechanism assignment after Apr 2026.**

* **Tmin (21d / 08-17)** — night-respiration / oil-deposition channel.
  Untouched by srad; loses only ~0.04 t/ha/°C of magnitude in the
  trivariate (β = −0.275 → −0.271).
* **precip (30d / 07-19)** — calendar-localised storm-damage and
  disease-pressure channel. ~80 % of its coefficient survives controlling
  for cloudiness; the remaining ~20 % is consistent with a small PAR-loss
  contribution as a secondary mechanism.
* **srad (30d / 08-10)** — independent heat / radiation-load channel. The
  best window (ends 08-10) sits between the precip window (ends 07-19)
  and the Tmin window (ends 08-17), suggesting srad picks up the same
  early-August stress complex from a third angle (insolation-driven leaf
  temperature, surface heating, evaporative demand) that neither Tmin nor
  precip alone fully captures.

**Headline expansion.** The rainfed Arbequina spec is now best read as a
**three-channel** model (`Tmin + precip + srad`, within-R² 0.493) rather
than a two-channel one (within-R² 0.469). All three channels are
calendar-anchored, all three are non-water-supply, and all three survive
the irrigated sign test in expected directions. The two-channel headline
remains valid as a parsimonious specification.

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
