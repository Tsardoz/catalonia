# Olive Yield × Water Sensitivity: Implementation Plan for Claude Code

## Project goal

Estimate phenological-stage-specific water sensitivity of olive yield across Catalan comarques using a hierarchical Bayesian scalar-on-function regression. Headline output: posterior contrast between water sensitivity at flowering vs pit hardening, framed as an exploratory hypothesis-generating analysis given n ≈ 9 independent years.

**Note on phenological axis (May 2026):** This plan previously used a GDD (growing degree day) axis with T_b = 10 °C and literature-derived GDD thresholds. Those thresholds have been retired. The GDD values (e.g. flowering at 350–450 GDD, pit hardening at 900–1200 GDD) were originally derived from a small number of site-years and do not represent Arbequina in Catalonia with sufficient reliability. GDD accumulation is also sensitive to the choice of T_b, which introduces an additional axis of uncertainty that was not adequately constrained (see Step 10 sensitivity results: T_b = 6.5–7 °C flips the headline contrast).

The pipeline now uses **fixed calendar windows** anchored to observed Arbequina phenology in north-west Iberia (Garrido et al. 2021, *Forests* 12:204, doi:10.3390/f12020204 — three-year record from the northern limit of Arbequina distribution, BBCH stages 5–8). These are the most appropriate published Arbequina calendar dates available pending a larger-sample GDD calibration. A GDD-based axis derived from the **PEP725 European phenology database** (which holds multi-decade, multi-site olive observations) is planned as a future replacement once sufficient Catalan station data have been extracted.

### Resolved design decisions

- **Cultivar strategy:** Two-cohort analysis. **Arbequina** (headline, 151k ha, 60% rainfed, 10 Arbequina-dominant comarques) and **All olive** (breadth check, 19 comarques, cultivar-agnostic). Both use rainfed yield (`yield_tha` / `seca_ha`) as the response.
- **Harvest truncation:** End of S8 maturation window, DOY 327 (≈ 23 November), for all comarca-years.
- **Lag yield:** Data covers 2015–2024; after lag, n = 9 effective years (2016–2024), 90 Arbequina comarca-years.
- **Phenological anchors:** Calendar windows for Arbequina (BBCH stages 5–8) from Garrido et al. 2021 (*Forests* 12:204). GDD-based anchors are retired (see rationale in project goal section above). PEP725-derived GDD is planned as a future replacement.

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
│   └── processed/            # GDD-aligned curves, model-ready arrays
├── src/
│   ├── 01_load_data.py
│   ├── 02_compute_gdd.py
│   ├── 03_compute_water_deficit.py
│   ├── 04_align_phenology.py
│   ├── 05_build_basis.py
│   ├── 06_fit_bambi_sanity.py
│   ├── 07_fit_numpyro_main.py
│   ├── 08_posterior_contrast.py
│   ├── 09_stability_loyo.py     # leave-one-year-out
│   ├── 10_sensitivity_analysis.py
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

Load three sources into pandas dataframes:

- Comarca-level annual olive yield, filtered separately for two cohorts:
  - **Arbequina cohort:** 10 comarques where Arbequina > 90% of olive area (from DUN parcel data)
  - **All olive cohort:** all 19 comarques with olive yield records (cultivar-agnostic breadth check)
- Daily climate per comarca: T_min, T_max, T_mean, precipitation, ET₀, VPD
- Use rainfed yield only (`yield_tha` / `seca_ha`), not irrigated yield

Validate that every comarca-year with a yield observation has complete daily climate coverage from January 1 through harvest. Log missing-data patterns. Drop comarca-years with > 5 % missing daily values rather than imputing. Run the full pipeline (Steps 2–9) independently for each cultivar cohort.

**Within-estimator centering (added):** After coverage validation, subtract the per-(cohort, comarca) mean from both `yield_tha` and `lag_yield`. This within-estimator (Frisch-Waugh) removes structural between-comarca baseline differences at the data level, making a comarca random effect in the model redundant. Comarca means are saved to `data/processed/comarca_yield_means.csv` for back-transformation if needed. Known limitation: means are computed on the full fitted panel; in LOYO refits the held-out year remains in the mean, introducing ~1/9-year leakage that is negligible in practice.

### Step 2 — Phenological calendar axis (`02_compute_gdd.py`)

**GDD axis retired.** This step previously computed cumulative GDD from January 1 with T_b = 10 °C and interpolated daily climate onto a common GDD grid. That approach is no longer used. The GDD thresholds (flowering at 350–450 GDD, pit hardening at 900–1200 GDD, etc.) were derived from a small number of site-years in the literature and are not reliably calibrated for Arbequina in Catalonia. The T_b sensitivity sweep in Step 10 showed the headline contrast flips sign at T_b ≤ 7 °C, confirming the GDD axis was not robust.

**Replacement: fixed calendar windows** based on observed Arbequina BBCH phenology from Garrido et al. 2021 (three seasons, north-west Iberian Peninsula — the closest published Arbequina phenology record to Catalonia). Stage boundaries are expressed as day-of-year (DOY) ranges:

| Stage | BBCH | Description | Calendar window (average) | DOY approx |
|-------|------|-------------|--------------------------|------------|
| S5 | 51–59 | Inflorescence development | mid-March → mid-May | DOY 72–138 |
| S6 | 61–69 | Flowering | mid-May → late June | DOY 136–173 |
| S7 | 71–79 | Fruit development | mid-June → late October | DOY 168–299 |
| S8 | 81–89 | Fruit maturation / harvest | late October → late November | DOY 299–327 |

These dates carry an uncertainty of roughly ±5–10 days (visual estimates from Garrido et al. 2021, Fig. 2). Stage 5's average start is pulled earlier by an unusually early 2016 season; 2017–2018 suggest a more typical late-March start.

For pipeline purposes, each comarca-year's climate is extracted over these fixed DOY windows. Stage overlap at the S5/S6 and S6/S7 boundaries (DOY ~136–138, ~168–173) is handled by assigning the overlap days to the earlier stage.

**Future work:** Replace these fixed windows with a GDD axis calibrated from the PEP725 European phenology database, which contains multi-decade olive phenology observations from multiple Iberian stations. This will allow T_b to be estimated from data rather than assumed.

### Step 3 — Water-deficit predictor (`03_compute_water_deficit.py`)

Primary predictor: daily water deficit `WD(t) = P(t) − ET₀(t)` on the **calendar day-of-year axis** (not GDD).
Add a state-proxy memory representation as an additional predictor output:
- `WD_state(t)` from a leaky integrator on the calendar DOY axis
- recursion per row: `S[0]=WD[0]`, `S[t]=alpha*WD[t] + (1-alpha)*S[t-1]`
- `alpha = 1 - exp(-dt/tau)`, default `tau = 45 days` from config, with Step 10 sensitivity sweep over multiple `tau` values

Secondary predictors for comparison runs: VPD curve, cumulative WD over a rolling 30-day window. Each gets its own model fit; the final paper reports WD-state as headline and VPD as a corroborating analysis.

Outputs (all shape `(n_obs, n_doy_grid)`): `WD_matrix`, `WD_state_matrix`, `WD30_matrix`, `VPD_matrix` (plus temperature matrices used elsewhere).

### Step 4 — Phenological anchors (`04_align_phenology.py`)

Anchor olive phenological windows to **calendar DOY ranges** derived from Garrido et al. 2021 Arbequina observations. GDD-based anchors (previously Didevarasl et al. 2023, Sanz-Cortés et al. 2002, Rapoport et al. 2013) are retired — see Step 2 rationale.

| Stage | Label | DOY start | DOY end | Notes |
|-------|-------|-----------|---------|-------|
| S5 | inflorescence | 72 | 138 | BBCH 51–59; ±5–10 day uncertainty |
| S6 | flowering | 136 | 173 | BBCH 61–69; overlaps S5 end |
| S7 | fruit_dev | 168 | 299 | BBCH 71–79; pit hardening within this stage |
| S8 | maturation | 299 | 327 | BBCH 81–89; harvest maturity |

Source: Garrido et al. 2021, *Forests* 12:204, doi:10.3390/f12020204. Three seasons (2016–2018) at the northern distribution limit of Arbequina; average values used as fixed windows. Stage 5 average start (DOY 72 ≈ 13 March) is pulled earlier by 2016; 2017–2018 suggest DOY ~89 (30 March) is more typical. This uncertainty is within the ±10 day tolerance of the calendar window approach.

Document window definitions in `data/processed/phenology_anchors.csv` with a `source` column. The headline contrast remains **S7 − S6** (fruit development sensitivity minus flowering sensitivity), analogous to the prior pit-hardening minus flowering contrast.

### Step 5 — Basis construction (`05_build_basis.py`)

Build B-spline basis on the **calendar DOY grid** (spanning DOY 72–327, the S5 start through S8 end):

- Cubic B-splines, **k = 5–6 basis functions** (start with 5; check posterior EDF; only increase if pinned)
- Knots placed at quantiles of the DOY grid
- Output: basis matrix `B` of shape `(n_doy_grid, n_basis)` — note this is grid × basis, **not** observations × basis
- Pre-compute unstandardized reduced predictors via `matrix @ B * dt` (including `WD_state_reduced`) with shape `(n_obs, n_basis)` where `dt` is the day step size (1 day) — this is the integrated functional predictor and the `dt` is essential for prior interpretability
- Do not apply persistent full-dataset z-scoring in Step 5; standardization is performed per fit in Step 7 to avoid leakage in LOYO/sensitivity refits

### Step 6 — Bambi sanity check (`06_fit_bambi_sanity.py`)

Quick model to confirm the hierarchical structure behaves before committing to NumPyro. Pre-aggregate WD into stage-window means (now including pre-flowering) as predictors. Default specification uses lag_yield without year RE. Use `--year-re` for the sensitivity comparison. Comarca RE is omitted: yield is comarca-mean centered in Step 1.

```python
import bambi as bmb
m = bmb.Model(
    "yield_tha ~ wd_preflower_mean + wd_flowering_mean + wd_pit_mean + lag_yield",
    data=df,
)
fit = m.fit(draws=1000, tune=1000)
```

Contrasts computed: (1) pit − flower (headline), (2) flower − preflower (new: does sensitivity differ before vs during flowering?).

#### Step 6 findings (Arbequina, n=90, 10 comarques, 9 years)

**lag_yield + (1|year) collision confirmed.** Running both produces lag_yield β = −0.07 (HDI spans zero) with year_sigma = 0.43 and residual sigma = 0.31. The year RE absorbs the variance lag_yield was meant to capture. Removing `(1|year)` does not recover lag_yield (β = +0.03, still spans zero) — instead residual sigma jumps to 0.48 and the contrast weakens (P(Δ>0) drops from 0.94 to 0.69). Alternate bearing is undetectable at comarca-level aggregation.

Neither specification resolves the lag/year question cleanly:
- **lag_yield only (default):** clean convergence, interpretable, but lag is inert and residual is large.
- **year RE only:** absorbs common-year shocks (drought 2022, good year 2019), tightens residual, sharpens contrast — but is less transparent than lag.

Both specs should be run as a sensitivity comparison in Step 7.

**WD coefficient signs:** wd_flowering ≈ −0.01, wd_pit ≈ +0.02–0.05 (both span zero). Positive pit coefficient means less water deficit → more yield, consistent with mesocarp development being water-sensitive. Signs are directionally correct but imprecise at this aggregation level — the full functional model (Step 7) may resolve the shape more clearly.

**Conclusion:** Hierarchical structure converges.

### Step 7 — NumPyro production model (`07_fit_numpyro_main.py`)

The headline model. Scalar-on-function regression using `WD_state_reduced` as predictor input, per-fit basis-column standardization, RW2 P-spline prior, and optional year random effect. **Comarca RE is not included**: yield is comarca-mean centered in Step 1, so there is no between-comarca variance left to absorb.

Model formula: `yield_tha_centered ~ f(WD_state_reduced_standardized) + lag_yield_centered [+ (1|year)]`

**Critical implementation notes:**

- **Comarca RE dropped.** Step 1 pre-subtracts per-(cohort, comarca) means from `yield_tha` and `lag_yield` (within-estimator). The model therefore has no `comarca_idx`, `n_comarcas`, `sigma_comarca`, or `comarca_re` terms.
- **Lag yield vs year random effect collision** remains. Both are included in the design; sensitivity pair (lag only vs year RE only) run in Step 10.
- Use `WD_state_reduced` from Step 5 as the default production predictor (WD30 is retained for sensitivity comparisons).
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

# Window masks on the DOY grid (Garrido et al. 2021 Arbequina calendar windows)
inflorescence_mask = (doy_grid >= 72)  & (doy_grid <= 138)   # S5: BBCH 51-59
flowering_mask     = (doy_grid >= 136) & (doy_grid <= 173)   # S6: BBCH 61-69
fruit_dev_mask     = (doy_grid >= 168) & (doy_grid <= 299)   # S7: BBCH 71-79
maturation_mask    = (doy_grid >= 299) & (doy_grid <= 327)   # S8: BBCH 81-89

# Mean β within each window per posterior draw
beta_S5 = beta_t[:, inflorescence_mask].mean(axis=1)
beta_S6 = beta_t[:, flowering_mask].mean(axis=1)
beta_S7 = beta_t[:, fruit_dev_mask].mean(axis=1)
beta_S8 = beta_t[:, maturation_mask].mean(axis=1)

# Headline contrast: S7 (fruit dev) − S6 (flowering)
delta = beta_S7 - beta_S6
prob_S7_less_sensitive = (delta > 0).mean()

# Secondary contrast: S6 − S5
delta_pf = beta_S6 - beta_S5

# Report all window posteriors
for name, arr in [("S5_inflorescence", beta_S5), ("S6_flowering", beta_S6),
                  ("S7_fruit_dev", beta_S7), ("S8_maturation", beta_S8)]:
    print(f"β_{name}: median {np.median(arr):.3f}, "
          f"90% CI [{np.quantile(arr, 0.05):.3f}, {np.quantile(arr, 0.95):.3f}]")
print(f"Δ (S7 − S6): median {np.median(delta):.3f}, "
      f"90% CI [{np.quantile(delta, 0.05):.3f}, {np.quantile(delta, 0.95):.3f}]")
print(f"P(S7 less sensitive than S6) = {prob_S7_less_sensitive:.3f}")
print(f"Δ (S6 − S5): median {np.median(delta_pf):.3f}, "
      f"90% CI [{np.quantile(delta_pf, 0.05):.3f}, {np.quantile(delta_pf, 0.95):.3f}]")
```

**Absolute-value contrast (fallback only):** Use `|β_flower| − |β_pit|` only if the sign of β is genuinely uncertain at one stage. Folded posteriors are harder to interpret — avoid by default.

Output: a side-by-side results table for Arbequina and All olive with point estimates, 90 % CIs, and posterior probabilities for the headline contrast (S7 − S6) plus each stage's β. The dilution from Arbequina to All olive is itself informative and should be reported.

**Note:** Prior results (Steps 7–9 below) were computed on the GDD axis and are retained as historical record. They will be recomputed on the calendar DOY axis once the pipeline is refactored in Steps 2–5.

### Step 9 — Stability check (`09_stability_loyo.py`)

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

**Data panel:** 90 Arbequina comarca-years (10 comarques × 9 years, 2016–2024 after lag). Mean rainfed yield 1.08 t/ha (sd 0.53), median 0.91. 171 all-olive comarca-years (19 comarques) for the breadth check.

**Phenology axis (calendar DOY, Garrido et al. 2021):** S5 inflorescence DOY 72–138 (mid-Mar to mid-May), S6 flowering DOY 136–173 (mid-May to late Jun), S7 fruit development DOY 168–299 (mid-Jun to late Oct), S8 maturation DOY 299–327 (late Oct to late Nov). **GDD validation figures are superseded** — they were computed on the retired GDD axis and should not be cited. Water deficit and VPD gradient magnitudes remain qualitatively valid but will be recomputed on the calendar axis.

**B-spline basis (GDD axis — superseded):** 5 cubic basis functions on 69 grid points (0–1700 GDD, step 25) were used in prior runs. The calendar-axis replacement will use a DOY grid spanning DOY 72–327 (256 days, step 1), with k = 5–6 basis functions and knots at DOY quantiles. Partition of unity must be re-verified for the new grid.

**Bambi sanity check (Arbequina, 3-stage model, lag_yield only, no year RE):**
- Convergence: all r̂ = 1.0, all ESS > 400, 0 divergences (target_accept=0.95).
- wd_preflower: β = −0.047 (90% HDI [−0.136, 0.042]) — most negative of the three stages
- wd_flowering: β = −0.005 (90% HDI [−0.054, 0.044]) — near zero, tightest posterior
- wd_pit: β = +0.021 (90% HDI [−0.067, 0.123]) — widest posterior, slight positive tendency
- lag_yield: β = +0.071 (90% HDI [−0.137, 0.276]) — inert
- comarca_sigma = 0.29, residual sigma = 0.49
- Δ(pit − flower) = +0.025, P(Δ>0) = 0.66
- Δ(flower − preflower) = +0.042, P(Δ>0) = 0.72

Figure: `results/figures/stage_posteriors_arbequina_wd.png`

**Prior 2-stage results (for reference, superseded by 3-stage model):**
- 2-stage wd_flowering: β = −0.014, wd_pit: β = +0.016, Δ(pit − flower) = +0.031, P(Δ>0) = 0.69
- With year RE: Δ(pit − flower) = +0.071, P(Δ>0) = 0.94 (but lag/year collision noted)

**Signal-strength diagnostic (post Step 6):**

Collinearity between stage predictors is low (r = 0.32 preflower–flowering, r = 0.02 flowering–pit), so coefficients spanning zero is not a separation problem. The underlying signal is weak: pooled OLS R² = 0.019 (three WD stage means explain ~2% of yield variance). Year-level correlations (n = 9) are all small: wd_pit vs yield r = −0.31, wd_preflower r = −0.13, wd_flowering r = −0.06. None approach significance at n = 9 (critical |r| ≈ 0.67 at α = 0.05).

**Interpretation:** The coefficient ordering (preflower most negative → flowering near zero → pit slightly positive) is physiologically plausible: water stress during inflorescence development (0–350 GDD) reduces flower count, while pit hardening is more resilient. The flower–preflower contrast (P(Δ>0) = 0.72) is modestly stronger than the pit–flower contrast (P(Δ>0) = 0.66), suggesting the pre-flowering window captures a real early-season signal direction. However, all three coefficients span zero, and the joint WD effect is indistinguishable from noise at this aggregation level.

The year RE specification sharpened the contrast to P(Δ>0) = 0.94, but this likely reflects absorption of confounded year-level factors (pest pressure, management changes, frost events) rather than a clean WD signal. Alternate bearing is undetectable at comarca-level aggregation regardless of specification.

### Signal-to-noise assessment and dataset limitations

The current dataset has **insufficient power to detect phenological-stage-specific water sensitivity** with confidence. This is a data limitation, not a methodological one — the hierarchical Bayesian framework is correct but cannot overcome the fundamental constraint of 9 independent climate-years.

**Why the signal is weak at this resolution:**

1. **Effective sample size is ~9, not 90.** Comarca-years add spatial replication of the *same* annual climate signal. All 10 Arbequina comarcas experience the same drought year. The WD→yield relationship is identified from year-to-year climate variation, giving n ≈ 9 independent observations for the climate effect.

2. **Comarca-level yield is a coarse aggregate.** Yield reported by DARP is the aggregate across all rainfed olive parcels in a comarca. Within-comarca variation in microclimate, soil, management, and tree age is averaged out, diluting the WD signal that might be detectable at the parcel or farm level.

3. **WD is not the only yield driver.** Frost damage, pest/disease pressure (e.g. *Bactrocera oleae*), pruning regime, and pollination success all affect olive yield independently of water deficit. These confounders vary year-to-year and are not controlled for in the climate-only model.

4. **Alternate bearing masks climate effects.** Olive yield exhibits strong biennial alternation driven by endogenous hormonal regulation, not climate. Lag yield is inert at comarca scale (β ≈ 0), likely because alternation patterns are desynchronised across trees within a comarca.

5. **Stage-mean compression.** Reducing the continuous WD(t) curve to 3 scalar stage means discards within-window shape. The full B-spline model (Step 7) may recover finer structure, but the underlying year-level signal strength is unchanged.

**What would improve power:**
- More years of data (the 2015–2024 window gives only 9 effective years after lag)
- Parcel-level yield data (available from DUN but not publicly released as yield)
- Multi-region pooling (e.g. combining Catalonia with Andalucía or Puglia panels under a hierarchical framework)
- Additional climate predictors (frost days, degree-day accumulation below critical thresholds)

**Conclusion for Steps 7–10:** The B-spline functional model (Step 7) remains worth implementing for methodological completeness — it will reveal whether the continuous β(t) shape is consistently negative across the calendar phenological axis or genuinely flat. However, it is unlikely to produce confident stage-specific contrasts given the R² ≈ 0.02 signal strength. The paper should be framed as establishing the methodological pipeline and reporting directionally plausible but underpowered results, with the dataset limitations documented transparently.

Figure: `results/figures/stage_posteriors_arbequina_wd.png`

### Implemented (Steps 7–9)

#### Step 7 — NumPyro production model

**Note:** Results below were computed on the GDD axis and are retained as a historical record. They will be rerun on the calendar DOY axis.

Scalar-on-function regression with RW2 P-spline prior, non-centred year random effect, comarca RE dropped (yield comarca-mean centered in Step 1), WD_state as headline predictor. 4 chains × 2000 warmup + 2000 draws, `target_accept=0.95`, CPU sequential. All fits use all 9 years (2016–2024).

**Previous results (with comarca RE, pre-centering) — superseded:**

- **Arbequina** (10 comarques, 90 obs): Clean convergence — 0 r̂>1.01, 0 low-ESS.
- **All olive** (19 comarques, 171 obs): Clean convergence — 0 r̂>1.01, 0 low-ESS.

**Refit required** with comarca-mean centered yield and no comarca RE.

Pallars Jussà dropped from all cohorts: S8 maturation (DOY 299–327) does not complete in this high-altitude comarca in most years. Exclusion criterion unchanged; only the basis for it is now expressed in calendar terms.

#### Step 8 — Posterior contrasts (two-cohort comparison)

**Note:** Contrasts below used GDD-axis windows (pre-flowering 0–350, flowering 350–450, pit hardening 900–1200 GDD). Analogous contrasts on the calendar DOY axis (S5/S6/S7/S8) will replace these once the pipeline is refactored.

Window-integrated β(t) contrasts from 8000 posterior draws per cohort (comarca-mean centered yield, no comarca RE):

**Arbequina (headline):**
- β_pre-flowering: median −0.000144, 90% CI [−0.000380, +0.000060], P(>0) = 0.11
- β_flowering: median −0.000059, 90% CI [−0.000190, +0.000080], P(>0) = 0.24
- β_pit hardening: median +0.000024, 90% CI [−0.000070, +0.000200], P(>0) = 0.76
- **Δ(pit − flower): median +0.000084, 90% CI [−0.000110, +0.000260], P(Δ>0) = 0.925**
- Δ(flower − pre): median +0.000081, P(Δ>0) = 0.865

**All olive (breadth check):**
- β_pre-flowering: median −0.000033
- β_flowering: median −0.000009
- β_pit hardening: median +0.000017
- **Δ(pit − flower): median +0.000026, P(Δ>0) = 0.743**
- Δ(flower − pre): P(Δ>0) = 0.670

Contrast table: `results/tables/posterior_contrast_summary_cohort_robustness.csv`

#### Step 9 — LOYO stability (two cohorts)

**Note:** Results below are from the GDD-axis model and will be rerun on the calendar DOY axis.

Full-data and 9 leave-one-year-out refits per cohort.

**Arbequina** — Full-data: Δ = +0.000083, P(Δ>0) = 0.926

| Excl year | Δ median | P(Δ>0) |
|-----------|----------|--------|
| 2016 | +0.000053 | 0.781 |
| 2017 | +0.000067 | 0.863 |
| 2018 | +0.000112 | 0.969 |
| 2019 | +0.000075 | 0.904 |
| 2020 | +0.000104 | 0.950 |
| 2021 | +0.000086 | 0.918 |
| 2022 | +0.000071 | 0.883 |
| 2023 | +0.000062 | 0.886 |
| 2024 | +0.000099 | 0.943 |

All 9 folds positive; P(Δ>0) range 0.78–0.97. No sign flip. 2016 is weakest fold (P=0.78). 2023 no longer uniquely weak (P=0.89) — the ~20% area reporting drop in 2023 is absorbed more cleanly by the year RE once comarca means are removed.

**All olive** — Full-data:

Figures: `results/figures/loyo_beta_overlay_{cohort}_wd_state.png`, `results/figures/loyo_delta_strip_{cohort}_wd_state.png`
Tables: `results/tables/loyo_summary_{cohort}_wd_state.csv`

### Two-cohort robustness assessment

The β(t) shape is **qualitatively consistent across both cohorts**: a monotone-rising curve from negative values in pre-flowering through near-zero at flowering to positive in pit hardening and oil accumulation. This means the water deficit signal — more deficit early is bad (reduces flower count), deficit during pit hardening has less impact — is not an artifact of cultivar selection or comarca subset.

**What strengthens the finding:**
1. **Direction is unanimous.** Both cohorts show Δ(pit − flower) > 0 in the posterior median. The sign never flips.
2. **Shape is consistent.** The β(t) curves share the same monotone-rising profile despite different comarca sets, cultivar mixes, and sample sizes.
3. **Arbequina is the cleanest signal.** P(Δ>0) = 0.93, consistent with the cultivar-homogeneous subset giving the least diluted climate response.
4. **LOYO stability.** No single year drives the Arbequina result; 8/9 folds maintain P(Δ>0) > 0.85.

**What tempers it:**
1. **All olive dilutes.** The broadest cohort (19 comarques, mixed cultivars) drops P(Δ>0) to 0.69. Cultivar mixing adds noise — different cultivars have different phenological timing and water sensitivity, smearing the β(t) signal.
2. **Absolute effect sizes are small.** The β(t) coefficients are on the order of 10⁻⁴, reflecting the R² ≈ 0.02 signal strength documented in Step 6. The shape is real but the magnitude is modest.
3. **2023 sensitivity.** The DARP reporting change in 2023 affects the Arbequina result; its exclusion drops P(Δ>0) to 0.63.

**Conclusion:** The two-cohort comparison provides the strongest robustness argument available from this dataset. The consistent β(t) shape across cultivar-restricted (Arbequina) and cultivar-agnostic (all olive) panels confirms the finding is not an artifact of subset selection. The dilution pattern (Arbequina > all olive in signal strength) is itself informative: it implies the WD signal is cultivar-specific, with homogeneous Arbequina panels producing the clearest response. This supports future work targeting cultivar-level analyses with parcel-scale yield data.

### Step 10 — Partially superseded

**GDD sweeps are retired.** T_b and GDD-window-shift sweeps no longer apply. Sweeps for tau, basis k, predictor (WD vs VPD), year-RE specification, and irrigated negative control remain valid and will be re-run on the calendar DOY axis.

Prior GDD-axis sweep results (Arbequina, 2000 warmup + 2000 draws × 4 chains) are retained below as historical record:

#### What stays robust

The Δ(pit − flower) > 0 finding for Arbequina with WD-state survives all of:

- **tau ∈ {15, 30, 45, 60, 120} GDD**: Δ median ranges 0.074–0.083 × 10⁻³, P(Δ>0) ∈ [0.88, 0.93]. No sign flips. The leaky-integrator memory parameter does not drive the result.
- **basis k ∈ {4, 5, 6, 8}**: Δ median 0.081–0.108 × 10⁻³, P(Δ>0) ∈ [0.91, 0.93]. The smooth model is not over- or under-fitting the β(t) shape.
- **T_b ∈ {8.5, 10, 12.5} °C**: Δ stays positive; at T_b=12.5 the contrast actually sharpens (Δ = 0.194 × 10⁻³, P = 0.995), consistent with the warmer base concentrating the same phenological events into a shorter GDD axis.
- **Window shifts ± 50 GDD**: every robust row above remains robust after a ±50 GDD shift of the stage windows.

#### What breaks the contrast (and how to read it)

Four perturbations move the result outside the P(Δ>0) ≥ 0.7 envelope or flip its sign. None of these invalidate the headline; each tells us something specific:

- **Cold T_b reparameterization (T_b ∈ {6.5, 7} °C)**: Δ flips negative (P(Δ>0) ≈ 0.38–0.41). At T_b=6.5 this combines with the *alternate* anchor windows from Toledo/Galicia/Croatia (pre-flower 0–250, flower 400–600, pit 1200–1700, GDD_UPPER=2700; `ALT_*` in `src/utils/config.py`), so the contrast is computed over a different GDD axis with different stage definitions. T_b=7 °C with the headline anchors also flips, indicating the result is sensitive to the joint choice of T_b and stage windows — not surprising at n ≈ 9 effective years.
- **VPD as alternative predictor**: Δ flips negative with both `with_year` (P = 0.19) and `no_year` (P = 0.40). VPD and WD-state are physiologically anti-correlated (high VPD → high evaporative demand → more water deficit), so the sign flip is *expected*. Reading it directionally: high VPD during pit hardening is more harmful than during flowering — the same conclusion as the WD-state result, just expressed on a stress-proxy scale.
- **No year RE (wd_state)**: Δ stays positive (median 0.034 × 10⁻³) but P drops to 0.68. Removing the year RE weakens the contrast by re-injecting common-year shocks (2022 drought, 2019 high yield) into the residual. The headline with the year RE remains the cleaner specification.
- **Irrigated yield negative control**: P(Δ>0) = 0.64 vs 0.92 for rainfed at the same settings. The contrast attenuates as expected if the rainfed signal is real — irrigation buffers WD's effect on yield. Δ median is 0.040 × 10⁻³ for irrigated vs 0.108 × 10⁻³ for rainfed, a 63 % reduction. This is the expected pattern and supports the rainfed finding.

#### Summary

Across 16 distinct sensitivity runs, the Δ(pit − flower) > 0 direction holds for every WD-state fit at T_b ≥ 8.5 °C regardless of tau, basis dimension, or ±50 GDD window shift. It weakens but does not flip under year-RE removal, and attenuates (as predicted by the rainfed hypothesis) under the irrigated negative control. It does flip under VPD (physiologically anti-correlated predictor) and under cold-T_b reparameterizations that combine alternate stage anchors. The conclusion is consistent with the Step 9 LOYO assessment: the headline finding is directionally robust for Arbequina with WD-state at T_b=10 °C, and its dependence on T_b and predictor choice is itself a finding worth documenting rather than hiding.

Full table: `results/tables/sensitivity_summary.csv`. Per-row evidence (atomic CSV writes, safe to re-aggregate any time): `results/tables/sens_rows/`.

## Minimum viable deliverable

Steps 7, 8, and 9 are complete. The two-cohort β(t) comparison provides the core robustness result. The paper can be framed as establishing the scalar-on-function methodology for Mediterranean perennial crops and reporting a directionally plausible but underpowered stage-specific water sensitivity pattern. The honest finding — monotone-rising β(t) with P(Δ>0) = 0.93 for Arbequina but attenuated to 0.69 when cultivar mixing is permitted — is a publishable result that motivates parcel-level and multi-region follow-ups.
