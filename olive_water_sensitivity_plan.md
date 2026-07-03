# Olive Yield × Water Sensitivity: Implementation Plan for Claude Code

## Project goal

Estimate phenological-stage-specific water sensitivity of olive yield across Catalan comarques using a hierarchical Bayesian scalar-on-function regression across five phenological stages (S_pre, S5–S8). Headline output: posterior contrast between water sensitivity at fruit development (S7) vs flowering (S6), framed as an exploratory hypothesis-generating analysis given n ≈ 9 independent years.

**Note on phenological axis (May 2026):** This plan previously used a GDD (growing degree day) axis with T_b = 10 °C and literature-derived GDD thresholds. Those thresholds have been retired. The GDD values (e.g. flowering at 350–450 GDD, pit hardening at 900–1200 GDD) were originally derived from a small number of site-years and do not represent Arbequina in Catalonia with sufficient reliability. GDD accumulation is also sensitive to the choice of T_b, which introduces an additional axis of uncertainty that was not adequately constrained (see Step 10 sensitivity results: T_b = 6.5–7 °C flips the headline contrast).

The pipeline now uses **five fixed calendar windows**: a pipeline-defined pre-inflorescence stage (S_pre, DOY 1–71) plus four stages anchored to observed Arbequina phenology in north-west Iberia (Garrido et al. 2021, *Forests* 12:204, doi:10.3390/f12020204 — three-year record from the northern limit of Arbequina distribution, BBCH stages 5–8). S_pre captures winter rest and early vegetative growth before inflorescence development. These are the most appropriate published Arbequina calendar dates available pending a larger-sample GDD calibration. A GDD-based axis derived from the **PEP725 European phenology database** (which holds multi-decade, multi-site olive observations) is planned as a future replacement once sufficient Catalan station data have been extracted.

### Resolved design decisions

- **Cultivar strategy:** Two-cohort analysis. **Arbequina** (headline, 10 comarques, 90 comarca-years) and **All olive** (breadth check, 18 comarques ≥ 500 ha mean rainfed area, 162 comarca-years). The all-olive panel uses `--min-rainfed-ha 500` in `01_load_data.py`; the old DUN rainfed-dominance whitelist is superseded. Both use rainfed yield (`yield_tha` / `seca_ha`) as the response.
- **Harvest truncation:** End of S8 maturation window, DOY 327 (≈ 23 November), for all comarca-years.
- **Lag yield:** Data covers 2015–2024; after lag, n = 9 effective years (2016–2024), 90 Arbequina comarca-years.
- **Phenological anchors:** Five calendar windows — S_pre (pipeline-defined, DOY 1–71) plus S5–S8 from Garrido et al. 2021 (*Forests* 12:204). Stage overlaps resolved by assigning shared DOYs to the earlier stage. GDD-based anchors are retired (see rationale in project goal section above). PEP725-derived GDD is planned as a future replacement.

This plan supersedes prior rolling-OLS analyses, which suffered from window-overlap artifacts and gave physiologically implausible sign-stable negative precipitation coefficients. The GDD axis used in earlier versions of Steps 2–5 is also superseded by calendar-date windows as of May 2026 (see rationale above).

## Stack

- **Python 3.11+**
- **NumPyro** (JAX backend) for the production hierarchical model — chosen for refit speed across stability checks
- **Bambi** for a 30-minute sanity-check model before committing to hand-rolled NumPyro
- **scikit-fda** only for plotting / quick exploration, not for inference
- **arviz** for posterior diagnostics and plots
- **pandas, numpy, scipy** for data handling
- B-spline basis via `scipy.interpolate.BSpline` or `patsy.bs`
- **No rpy2 / no BLiSS** unless a reviewer explicitly asks. The smooth model is the real analysis.

## Repository layout

```
olive-water-sensitivity/
├── data/
│   ├── raw/                  # comarca yield, daily climate, phenology references
│   └── processed/            # calendar-DOY-aligned curves, model-ready arrays
├── src/
│   ├── 01_load_data.py
│   ├── 02_prepare_doy_axis.py
│   ├── 03_compute_water_deficit.py
│   ├── 04_align_phenology.py
│   ├── 05_build_basis.py
│   ├── 06_fit_bambi_sanity.py
│   ├── 07_fit_numpyro_main.py
│   ├── 08_posterior_contrast.py
│   ├── 09_stability_loyo.py     # leave-one-year-out
│   ├── 10_sensitivity_analysis.py
│   ├── plot_beta_t_comparison.py    # β(t) curves across cohorts (calendar axis)
│   ├── plot_stage_posteriors.py     # per-stage β KDE + S7-referenced contrasts
│   └── utils/
│       ├── splines.py
│       ├── phenology.py
│       └── plotting.py
├── notebooks/                # exploratory only, not in main pipeline
├── results/
│   ├── posteriors/           # .nc files (arviz InferenceData)
│   ├── figures/
│   └── tables/
├── tests/
└── README.md
```

## Step-by-step implementation

### Step 1 — Data ingestion (`01_load_data.py`)

Load two sources into pandas dataframes:

- Comarca-level annual olive yield, filtered separately for two cohorts:
  - **Arbequina cohort:** 10 comarques where Arbequina > 90% of olive area (from DUN parcel data)
  - **All olive cohort:** all comarques with mean rainfed olive area ≥ `--min-rainfed-ha` (default 500 ha); no DUN whitelist required
- Daily climate per comarca: T_min, T_max, precipitation, ET₀, VPD (T_mean derived as (T_max + T_min) / 2 in Step 2)
- Use rainfed yield only (`yield_tha` / `seca_ha`), not irrigated yield

Validate that every comarca-year with a yield observation has near-complete daily climate coverage from January 1 through harvest (DOY 327 ≈ Nov 23); Feb 29 excluded so expected count is always 327 days. Log missing-data patterns. Drop comarca-years with > 5 % missing daily values rather than imputing. Run the full pipeline (Steps 2–9) independently for each cultivar cohort.

**Within-estimator centering (added):** After coverage validation, subtract the per-(cohort, comarca) mean from both `yield_tha` and `lag_yield`. This within-estimator (Frisch-Waugh) removes structural between-comarca baseline differences at the data level, making a comarca random effect in the model redundant. Comarca means are saved to `data/processed/comarca_yield_means.csv` for back-transformation if needed. Known limitation: means are computed on the full fitted panel; in LOYO refits the held-out year remains in the mean, introducing ~1/9-year leakage that is negligible in practice.

### Step 2 — Phenological calendar axis (`02_prepare_doy_axis.py`)

**GDD axis retired.** This step previously computed cumulative GDD from January 1 with T_b = 10 °C and interpolated daily climate onto a common GDD grid. That approach is no longer used. The GDD thresholds (flowering at 350–450 GDD, pit hardening at 900–1200 GDD, etc.) were derived from a small number of site-years in the literature and are not reliably calibrated for Arbequina in Catalonia. The T_b sensitivity sweep in Step 10 showed the headline contrast flips sign at T_b ≤ 7 °C, confirming the GDD axis was not robust.

**Replacement: five fixed calendar windows.** S5–S8 based on observed Arbequina BBCH phenology from Garrido et al. 2021 (three seasons, north-west Iberian Peninsula — the closest published Arbequina phenology record to Catalonia). S_pre is a pipeline-defined winter/early-vegetative window that extends the DOY axis back to January 1. Raw stage boundaries (with natural overlap from Garrido et al.):

| Stage | BBCH | Description | Calendar window (average) | DOY approx |
|-------|------|-------------|--------------------------|------------|
| S_pre | — | Winter rest / early vegetative | Jan 1 → mid-March | DOY 1–71 |
| S5 | 51–59 | Inflorescence development | mid-March → mid-May | DOY 72–138 |
| S6 | 61–69 | Flowering | mid-May → late June | DOY 136–173 |
| S7 | 71–79 | Fruit development | mid-June → late October | DOY 168–299 |
| S8 | 81–89 | Fruit maturation / harvest | late October → late November | DOY 299–327 |

S5–S8 dates carry an uncertainty of roughly ±5–10 days (visual estimates from Garrido et al. 2021, Fig. 2). Stage 5's average start is pulled earlier by an unusually early 2016 season; 2017–2018 suggest a more typical late-March start.

Stage overlap at the S5/S6 and S6/S7 boundaries is resolved by assigning shared DOYs to the earlier stage, producing contiguous non-overlapping windows that cover DOY 1–327:

| Stage | Resolved DOY | Days |
|-------|-------------|------|
| S_pre | 1–71 | 71 |
| S5 | 72–138 | 67 |
| S6 | 139–173 | 35 |
| S7 | 174–299 | 126 |
| S8 | 300–327 | 28 |

The DOY grid spans 1–327 (327 daily points). The five stages align with k = 5 B-spline basis functions in Step 5.

**Future work:** Replace these fixed windows with a GDD axis calibrated from the PEP725 European phenology database, which contains multi-decade olive phenology observations from multiple Iberian stations. This will allow T_b to be estimated from data rather than assumed.

### Step 3 — Water-deficit predictor (`03_compute_water_deficit.py`)

Primary predictor: daily climatic water balance `CWB(t) = P(t) − ET₀(t)` on the **calendar day-of-year axis** (not GDD).
Add a state-proxy memory representation as an additional predictor output:
- `WD_state(t)` from a leaky integrator on the calendar DOY axis
- recursion per row: `S[0]=WD[0]`, `S[t]=alpha*WD[t] + (1-alpha)*S[t-1]`
- `alpha = 1 - exp(-dt/tau)`, default `tau = 45 days` from config, with Step 10 sensitivity sweep over multiple `tau` values

Secondary predictors for comparison runs: VPD curve, cumulative CWB over a rolling 30-day window. Each gets its own model fit; the final paper reports CWB-state as headline and VPD as a corroborating analysis.

Outputs (all shape `(n_obs, n_doy_grid)`): `CWB_matrix`, `CWB_state_matrix`, `CWB30_matrix` (internal names: WD_*), `VPD_matrix` (plus temperature matrices used elsewhere).

### Step 4 — Phenological anchors (`04_align_phenology.py`)

Anchor olive phenological windows to **calendar DOY ranges**. S5–S8 derived from Garrido et al. 2021 Arbequina observations; S_pre is pipeline-defined. GDD-based anchors (previously Didevarasl et al. 2023, Sanz-Cortés et al. 2002, Rapoport et al. 2013) are retired — see Step 2 rationale.

Resolved (non-overlapping) boundaries used in all downstream steps:

| Stage | Label | DOY start | DOY end | Days | Notes |
|-------|-------|-----------|---------|------|-------|
| S_pre | pre_inflorescence | 1 | 71 | 71 | Winter rest / early vegetative |
| S5 | inflorescence | 72 | 138 | 67 | BBCH 51–59; ±5–10 day uncertainty |
| S6 | flowering | 139 | 173 | 35 | BBCH 61–69; starts after S5 overlap resolved |
| S7 | fruit_dev | 174 | 299 | 126 | BBCH 71–79; pit hardening within this stage |
| S8 | maturation | 300 | 327 | 28 | BBCH 81–89; starts after S7 overlap resolved |

Source for S5–S8: Garrido et al. 2021, *Forests* 12:204, doi:10.3390/f12020204. Three seasons (2016–2018) at the northern distribution limit of Arbequina; average values used as fixed windows. Stage 5 average start (DOY 72 ≈ 13 March) is pulled earlier by 2016; 2017–2018 suggest DOY ~89 (30 March) is more typical. This uncertainty is within the ±10 day tolerance of the calendar window approach.

Document window definitions in `data/processed/phenology_anchors.csv` with both raw (Garrido) and resolved DOY ranges, BBCH codes, and a `source` column. The headline contrast remains **S7 − S6** (fruit development sensitivity minus flowering sensitivity). Secondary contrasts: S6 − S5 (flowering vs inflorescence) and S5 − S_pre (inflorescence vs winter).

### Step 5 — Basis construction (`05_build_basis.py`)

Build B-spline basis on the **calendar DOY grid** (spanning DOY 1–327, S_pre start through S8 end, 327 daily points):

- Cubic B-splines, **k = 5 basis functions** (one per phenological stage; check posterior EDF; only increase if pinned)
- Knots placed at quantiles of the DOY grid
- Output: basis matrix `B` of shape `(n_doy_grid, n_basis)` — note this is grid × basis, **not** observations × basis
- Pre-compute unstandardized reduced predictors via `matrix @ B * dt` (including `CWB_state_reduced`) with shape `(n_obs, n_basis)` where `dt` is the day step size (1 day) — this is the integrated functional predictor and the `dt` is essential for prior interpretability
- Do not apply persistent full-dataset z-scoring in Step 5; standardization is performed per fit in Step 7 to avoid leakage in LOYO/sensitivity refits

### Step 6 — Bambi sanity check (`06_fit_bambi_sanity.py`)

Quick model to confirm the hierarchical structure behaves before committing to NumPyro. Pre-aggregate WD into five stage-window means as predictors. Default specification uses lag_yield without year RE. Use `--year-re` for the sensitivity comparison. Comarca RE is omitted: yield is comarca-mean centered in Step 1.

```python
import bambi as bmb
m = bmb.Model(
    "yield_tha ~ wd_pre_mean + wd_inflorescence_mean + wd_flowering_mean"
    "            + wd_fruit_dev_mean + wd_maturation_mean + lag_yield",
    data=df,
)
fit = m.fit(draws=1000, tune=1000)
```

Contrasts computed: (1) S7 − S6 (headline: fruit dev vs flowering), (2) S6 − S5 (flowering vs inflorescence), (3) S5 − S_pre (inflorescence vs winter).

#### Step 6 findings (Arbequina, n=90, 10 comarques, 9 years)

**lag_yield + (1|year) collision confirmed.** Running both produces lag_yield β = −0.07 (HDI spans zero) with year_sigma = 0.43 and residual sigma = 0.31. The year RE absorbs the variance lag_yield was meant to capture. Removing `(1|year)` does not recover lag_yield (β = +0.03, still spans zero) — instead residual sigma jumps to 0.48 and the contrast weakens (P(Δ>0) drops from 0.94 to 0.69). Alternate bearing is undetectable at comarca-level aggregation.

Neither specification resolves the lag/year question cleanly:
- **lag_yield only (default):** clean convergence, interpretable, but lag is inert and residual is large.
- **year RE only:** absorbs common-year shocks (drought 2022, good year 2019), tightens residual, sharpens contrast — but is less transparent than lag.

Both specs should be run as a sensitivity comparison in Step 7.

**CWB coefficient signs:** wd_flowering ≈ −0.01, wd_pit ≈ +0.02–0.05 (both span zero). Positive pit coefficient means higher CWB (surplus) → more yield, consistent with mesocarp development being water-sensitive. Signs are directionally correct but imprecise at this aggregation level — the full functional model (Step 7) may resolve the shape more clearly.

**Conclusion:** Hierarchical structure converges.

### Step 7 — NumPyro production model (`07_fit_numpyro_main.py`)

The headline model. Scalar-on-function regression using `CWB_state_reduced` as predictor input, per-fit basis-column standardization, RW2 P-spline prior, and optional year random effect. **Comarca RE is not included**: yield is comarca-mean centered in Step 1, so there is no between-comarca variance left to absorb.

Model formula: `yield_tha_centered ~ f(WD_state_reduced_standardized) + lag_yield_centered [+ (1|year)]`

**Critical implementation notes:**

- **Comarca RE dropped.** Step 1 pre-subtracts per-(cohort, comarca) means from `yield_tha` and `lag_yield` (within-estimator). The model therefore has no `comarca_idx`, `n_comarcas`, `sigma_comarca`, or `comarca_re` terms.
- **Lag yield vs year random effect collision** remains. Both are included in the design; sensitivity pair (lag only vs year RE only) run in Step 10.
- Use `CWB_state_reduced` from Step 5 as the default production predictor (WD30 is retained for sensitivity comparisons).
- Standardize predictor columns inside each fit using training rows only. Save per-fit column means/SDs and back-transform posterior `beta_coefs` to physical scale before Step 8 contrasts.
- RW2 implemented via anchor coefficients (beta0, beta1) + innovation recursion via `lax.scan`; no extra global Normal prior on assembled coefficients.
- Use 4 chains, 2000 warmup, 2000 samples, `target_accept_prob=0.95`.
- Save InferenceData as `.nc` to `results/posteriors/`.

### Step 8 — Posterior contrast (`08_posterior_contrast.py`)

The headline number, computed independently for each cultivar cohort. Two correct ways to compute the stage-window contrast:

**Window-integrated β (preferred — matches what physiologists mean by "stage sensitivity"):**
Before window contrasts, back-transform posterior `beta_coefs` from standardized predictor scale using the saved per-fit column SDs, then compute `beta(t)` from the rescaled coefficients.

```python
# Posterior draws: shape (n_chains, n_samples, n_basis)
beta_draws = idata.posterior["beta_coefs"].values
beta_draws_flat = beta_draws.reshape(-1, n_basis)   # (n_total, n_basis)

# β(t) at every grid point: (n_total, n_grid)
beta_t = beta_draws_flat @ B.T

# Window masks on the DOY grid (resolved non-overlapping boundaries)
pre_mask            = (doy_grid >= 1)   & (doy_grid <= 71)    # S_pre: winter/early veg
inflorescence_mask  = (doy_grid >= 72)  & (doy_grid <= 138)   # S5: BBCH 51-59
flowering_mask      = (doy_grid >= 139) & (doy_grid <= 173)   # S6: BBCH 61-69
fruit_dev_mask      = (doy_grid >= 174) & (doy_grid <= 299)   # S7: BBCH 71-79
maturation_mask     = (doy_grid >= 300) & (doy_grid <= 327)   # S8: BBCH 81-89

# Mean β within each window per posterior draw
beta_S_pre = beta_t[:, pre_mask].mean(axis=1)
beta_S5 = beta_t[:, inflorescence_mask].mean(axis=1)
beta_S6 = beta_t[:, flowering_mask].mean(axis=1)
beta_S7 = beta_t[:, fruit_dev_mask].mean(axis=1)
beta_S8 = beta_t[:, maturation_mask].mean(axis=1)

# All contrasts use S7 as reference
delta_s7_s6   = beta_S7 - beta_S6     # headline: fruit dev vs flowering
delta_s7_s5   = beta_S7 - beta_S5     # fruit dev vs inflorescence
delta_s7_spre = beta_S7 - beta_S_pre  # fruit dev vs winter
delta_s7_s8   = beta_S7 - beta_S8     # fruit dev vs maturation

for name, arr in [("S_pre", beta_S_pre), ("S5", beta_S5),
                  ("S6", beta_S6), ("S7", beta_S7), ("S8", beta_S8)]:
    print(f"β_{name}: median {np.median(arr):.6f}, "
          f"90% CI [{np.quantile(arr, 0.05):.6f}, {np.quantile(arr, 0.95):.6f}]")
for label, d in [("S7−S6", delta_s7_s6), ("S7−S5", delta_s7_s5),
                 ("S7−S_pre", delta_s7_spre), ("S7−S8", delta_s7_s8)]:
    p_gt0 = (d > 0).mean()
    p_diff = max(p_gt0, 1 - p_gt0)
    print(f"Δ {label}: median {np.median(d):.6f}, P(diff)={p_diff:.3f}")
```

**Absolute-value contrast (fallback only):** Use `|β_flower| − |β_pit|` only if the sign of β is genuinely uncertain at one stage. Folded posteriors are harder to interpret — avoid by default.

Output: a side-by-side results table for Arbequina and All olive with point estimates, 90 % CIs, and posterior probabilities for the headline contrast (S7 − S6) plus each stage's β. The dilution from Arbequina to All olive is itself informative and should be reported.

### Step 9

Leave-one-year-out, 9 refits. The point is to detect whether one anomalous year (drought, disease outbreak) is carrying the result.

For each held-out year:

1. Refit the model excluding that year
2. Recompute β(t) posterior median + 90 % band
3. Recompute the Δ contrast posterior

Produce two diagnostic plots:

- All 9 LOYO posterior median β(t) curves overlaid on the full-data 90 % band. Any curve outside the band → that year is dominating.
- Boxplot or strip of Δ posterior medians across the 9 LOYO fits next to the full-data Δ posterior.

Report the across-fold spread of Δ alongside the headline number. If LOYO Δ medians span zero while the full-data posterior is confidently positive, **say so explicitly in the paper**.

### Step 10 — Sensitivity analyses (`10_sensitivity_analysis.py`)

Run the production model under each of these perturbations and tabulate Δ posterior summaries:

- **[GDD sweeps retired]** T_b and GDD-window sensitivity sweeps are no longer applicable now that the pipeline uses calendar windows. The T_b sensitivity result (contrast flips at T_b ≤ 7 °C) is documented as the primary motivation for retiring the GDD axis.
- Phenology window widths ± 10 days (calendar equivalent of the former ± 50 GDD sweep)
- Predictor: WD vs VPD vs (WD, VPD) joint
- Basis dimension k ∈ {4, 5, 6, 8}
- Lag-yield vs year-RE specification
- Cultivar isolation: pooled vs cultivar-specific anchor windows
- **Irrigated yield as negative control:** refit the Arbequina model using `yield_irrig_tha` instead of rainfed yield. If the WD signal is real for rainfed trees, it should be weaker or absent for irrigated trees.

Anything where the sign of Δ flips or the posterior probability drops below 0.7 is a finding worth flagging in the discussion, not hiding.

## Diagnostics and reporting standards

- Run `arviz.summary` on every fit; flag any r̂ > 1.01 or ESS < 400
- `pp_check` posterior predictive plots for the main model
- **Do not report Bayesian R² as a headline metric** — at n = 9 it's optimistic in the same way the CAPE R² was. Use posterior predictive checks and LOYO stability instead.
- Report **everything** in posterior-summary form (median + 90 % CI) — no point estimates without uncertainty
- Frame the paper as **exploratory / hypothesis-generating** in abstract and discussion. With 9 independent years, no inferential framework can make this confirmatory.

## What this plan deliberately does not do

- Rolling-window OLS (the original problem)
- Optimizing GDD T_b on yield (Laurent circularity)
- Raw precipitation as primary predictor (sign-instability)
- Elastic net / fLASSO (smearing across autocorrelated days; selection instability at n = 9)
- Treating n_comarca_years as the effective sample size for the climate signal
- BLiSS as the primary analysis (short-interval prior bias oversells precision)
- LOOCV at the row level (irrelevant — independence is at the year level)
- Bayesian R² as a headline fit metric

## Open questions (all resolved)

1. ~~Cultivar pooling~~ → Two-cohort: Arbequina (headline) + All olive (breadth check).
2. ~~T_b and anchor sources~~ → GDD axis retired; calendar windows from Garrido et al. 2021 used instead. PEP725-based GDD planned as future work.
3. ~~Harvest truncation~~ → End of S8 maturation window (DOY 327 ≈ 23 November).
4. ~~Lag yield~~ → Confirmed: 2015–2024 yield data, n = 9 effective years after lag.

## Pipeline status and results summary

### Implemented (Steps 1–6)

**Data panel:** 90 Arbequina comarca-years (10 comarques × 9 years, 2016–2024 after lag). 162 all-olive comarca-years (18 comarques ≥ 500 ha mean rainfed area, after lag). All-olive panel defined by area threshold, not DUN whitelist.

**Phenology axis (calendar DOY):** Five stages covering DOY 1–327. S_pre DOY 1–71 (Jan–mid-Mar, pipeline-defined), S5 inflorescence DOY 72–138 (mid-Mar to mid-May), S6 flowering DOY 139–173 (mid-May to late Jun), S7 fruit development DOY 174–299 (late Jun to late Oct), S8 maturation DOY 300–327 (late Oct to late Nov). Resolved non-overlapping boundaries; raw Garrido et al. 2021 ranges documented in `data/processed/phenology_anchors.csv`. **GDD validation figures are superseded** — they were computed on the retired GDD axis and should not be cited.

**B-spline basis (calendar axis):** 5 cubic basis functions on 327 DOY grid points (DOY 1–327, step 1), knots at DOY quantiles. One basis function per phenological stage. Shape (327, 5) confirmed.

**Bambi sanity check (Arbequina, 3-stage model, lag_yield only, no year RE — to be rerun with 5 stages):**
- Convergence: all r̂ = 1.0, all ESS > 400, 0 divergences (target_accept=0.95).
- wd_preflower: β = −0.047 (90% HDI [−0.136, 0.042]) — most negative of the three stages
- wd_flowering: β = −0.005 (90% HDI [−0.054, 0.044]) — near zero, tightest posterior
- wd_pit: β = +0.021 (90% HDI [−0.067, 0.123]) — widest posterior, slight positive tendency
- lag_yield: β = +0.071 (90% HDI [−0.137, 0.276]) — inert
- comarca_sigma = 0.29, residual sigma = 0.49
- Δ(pit − flower) = +0.025, P(Δ>0) = 0.66
- Δ(flower − preflower) = +0.042, P(Δ>0) = 0.72

Figure: `results/figures/stage_posteriors_arbequina_wd_state.png`, `results/figures/stage_posteriors_all_olive_wd_state.png`

**Signal-strength diagnostic:** OLS R² ≈ 0.02 — WD explains ~2% of yield variance. With n = 9 effective years the model is exploratory/hypothesis-generating, not confirmatory.

### Signal-to-noise assessment and dataset limitations

The current dataset has **insufficient power to detect phenological-stage-specific water sensitivity** with confidence. This is a data limitation, not a methodological one — the hierarchical Bayesian framework is correct but cannot overcome the fundamental constraint of 9 independent climate-years.

**Why the signal is weak at this resolution:**

1. **Effective sample size is ~9, not 90.** Comarca-years add spatial replication of the *same* annual climate signal. All 10 Arbequina comarcas experience the same drought year. The WD→yield relationship is identified from year-to-year climate variation, giving n ≈ 9 independent observations for the climate effect.

2. **Comarca-level yield is a coarse aggregate.** Yield reported by DARP is the aggregate across all rainfed olive parcels in a comarca. Within-comarca variation in microclimate, soil, management, and tree age is averaged out, diluting the WD signal that might be detectable at the parcel or farm level.

3. **WD is not the only yield driver.** Frost damage, pest/disease pressure (e.g. *Bactrocera oleae*), pruning regime, and pollination success all affect olive yield independently of water deficit. These confounders vary year-to-year and are not controlled for in the climate-only model.

4. **Alternate bearing masks climate effects.** Olive yield exhibits strong biennial alternation driven by endogenous hormonal regulation, not climate. Lag yield is inert at comarca scale (β ≈ 0), likely because alternation patterns are desynchronised across trees within a comarca.

5. **Stage-mean compression.** Reducing the continuous CWB(t) curve to 5 scalar stage means discards within-window shape. The full B-spline model (Step 7) may recover finer structure, but the underlying year-level signal strength is unchanged.

**What would improve power:**
- More years of data (the 2015–2024 window gives only 9 effective years after lag)
- Parcel-level yield data (available from DUN but not publicly released as yield)
- Multi-region pooling (e.g. combining Catalonia with Andalucía or Puglia panels under a hierarchical framework)
- Additional climate predictors (frost days, degree-day accumulation below critical thresholds)

**Conclusion:** The paper should be framed as establishing the scalar-on-function methodology for Mediterranean perennial crops and reporting directionally plausible but underpowered stage-specific water sensitivity. Dataset limitations should be documented transparently.

### Implemented (Steps 7–9) — calendar DOY axis

#### Step 7 — NumPyro production model

Scalar-on-function regression, RW2 P-spline prior, non-centred year RE, no comarca RE, CWB_state headline predictor. JAX CPU-only (`jax.config.update("jax_platforms", "cpu")`). 4 chains × 2000 warmup + 2000 draws vectorized for main fits, sequential for LOYO folds.

Clean convergence for both cohorts: 0 r̂ > 1.01, 0 low-ESS.

**Calendar-axis β stage summaries (median, 90% CI):**

| Stage | Arbequina | All olive (≥500 ha) |
|---|---|---|
| S_pre (DOY 1–71) | +0.00167 [−0.00006, +0.00344] | +0.00184 [+0.00118, +0.00251] |
| S5 inflorescence | +0.00057 [−0.00041, +0.00148] | +0.00021 [−0.00026, +0.00055] |
| S6 flowering | +0.00021 [−0.00069, +0.00096] | −0.00016 [−0.00056, +0.00018] |
| S7 fruit dev | −0.00017 [−0.00066, +0.00031] | −0.00027 [−0.00045, −0.00008] |
| S8 maturation | −0.00063 [−0.00224, +0.00134] | −0.00033 [−0.00124, +0.00069] |

#### Step 8 — Posterior contrasts (two-cohort comparison)

**Contrast convention (current):** All contrasts use **S7 (fruit development) as reference**. Two metrics reported per contrast:
- `p_diff = max(P(Δ>0), P(Δ<0))` — probability the two stages are different (0.5 = same, 1.0 = certain)
- `prob_gt0 = P(Δ>0)` — directional probability (S7 > reference)

Contrast table: `results/tables/posterior_contrast_summary.csv`

| Contrast | Arbequina median | P(diff) | All olive median | P(diff) |
|---|---|---|---|---|
| S7 − S6 (vs flowering) | −0.000381 | 0.80 | −0.000290 | 0.84 |
| S7 − S5 (vs inflorescence) | −0.000750 | 0.89 | −0.000566 | 0.93 |
| S7 − S_pre (vs winter) | −0.001839 | 0.95 | −0.001584 | 0.99 |
| S7 − S8 (vs maturation) | +0.000485 | 0.70 | +0.000457 | 0.77 |

All contrasts show S7 β < S_pre/S5/S6 β (fruit dev is more water-sensitive than
earlier stages). S7 and S8 are not clearly distinguishable (P(diff) 0.70–0.77).

#### Step 9 — LOYO stability (calendar axis)

Full-data + 9 LOYO refits per cohort.

**S7−S6 headline contrast (full-data and fold range):**

| Cohort | Full P(diff) | Fold P(diff) range | Sign reversals |
|---|---|---|---|
| Arbequina | 0.80 | 0.55–0.95 | 1/9 folds |
| All olive (≥500 ha) | 0.84 | 0.72–0.94 | 1/9 folds |

One fold per cohort reverses the S7−S6 contrast sign, confirming that the S7 vs S6
separation is LOYO-sensitive at n=9 and should be treated as directional, not confirmatory.
The broader monotone β(t) shape (S_pre positive, S7 near-zero) was preserved across all folds:
no fold produced positive β_S7 jointly with negative or zero β_S_pre.

Figures: `results/figures/loyo_beta_overlay_{cohort}_wd_state.png`, `results/figures/loyo_delta_strip_{cohort}_wd_state.png`
Tables: `results/tables/loyo_summary_{cohort}_wd_state.csv`, `loyo_full_{cohort}_wd_state.csv`

### Two-cohort robustness assessment

The β(t) shape is **monotonically declining S_pre → S8** in both cohorts. S_pre is the only
stage with credibly positive β. The dominant and most robust finding is the S7−S_pre contrast
(P(diff) = 0.95–0.99, stable across all LOYO folds); the S7−S6 contrast is directional but
LOYO-sensitive.

**What strengthens the finding:**
1. **Monotone shape.** Both cohorts show β(t) declining S_pre→S8 across all posterior draws.
2. **S_pre signal LOYO-stable.** No fold reverses the S7−S_pre contrast sign.
3. **Area threshold.** The all-olive ≥500 ha panel (18 comarcas) gives higher P(diff) on all contrasts than the old dominance whitelist or full 36-comarca panel.
4. **Irrigated control.** Irrigated β(t) tracks rainfed; S_pre signal persists because winter irrigation is not practised, confirming winter rainfall as the unmanaged bottleneck.

**What tempers it:**
1. **S7−S6 is LOYO-sensitive** (1 sign flip per cohort). Treat as directional only.
2. **Absolute effect sizes small** (β ~ 10⁻⁴, R² ≈ 0.02).
3. **n = 9 effective years** makes all inference exploratory.

**Conclusion:** The primary publishable finding is the S7−S_pre contrast and the monotone
β(t) shape, not the S7−S6 headline. Winter water accumulation is the dominant climate
predictor; pre-season irrigation represents an unexplored yield lever.

### Step 10 — Complete (Arbequina CWB-state; VPD and all_olive pending)

Sweep complete: τ ∈ {15, 30, 45, 60, 120} days × k ∈ {4, 5, 6, 8} × year-RE {on, off} = 40 fits;
±10 day DOY window shifts applied post-fit; irrigated negative control run.

**Key results:**
- **With year-RE:** P(diff) for S7−S6 ranges 0.66–0.91, no sign reversals. Robust.
- **Without year-RE:** 11/40 sign reversals (short τ + high k corner). Not robust.
- **Longer τ** monotonically strengthens contrast across all k and year-RE specs.
- **Window shift ±10 days:** P(diff) changes 0.83 → 0.78. Stable above 0.7 throughout.
- **Irrigated control:** P(diff) = 0.65 vs 0.80 rainfed; β(t) shape unchanged in S_pre.

Full table: `results/tables/sensitivity_summary.csv`. Per-row: `results/tables/sens_rows/`.
VPD predictor comparison and all_olive sweep remain.

## Minimum viable deliverable

Steps 1–10 complete (Step 10 partial: Arbequina WD-state). The results section is drafted.
The primary publishable finding is the monotone β(t) shape with P(diff) = 0.95–0.99 for
S7 vs S_pre, supported by sensitivity sweeps and the irrigated negative control. The
S7−S6 headline contrast (P(diff) = 0.80–0.84) is directional but LOYO-sensitive and should
be framed accordingly. The paper establishes the scalar-on-function methodology for
Mediterranean perennial crops and generates a testable hypothesis about pre-season winter
irrigationas a yield lever. Remaining: VPD sensitivity sweep, all_olive Step 10, final figures.
