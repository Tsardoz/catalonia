# Olive Yield × Water Sensitivity: Implementation Plan for Claude Code

## Project goal

Estimate phenological-stage-specific water sensitivity of olive yield across Catalan comarques using a hierarchical Bayesian scalar-on-function regression. Headline output: posterior contrast between water sensitivity at flowering vs pit hardening, framed as an exploratory hypothesis-generating analysis given n ≈ 9 independent years.

### Resolved design decisions

- **Cultivar strategy:** Two-cultivar side-by-side comparison. **Arbequina** (151k ha, 60% rainfed, 10 Arbequina-dominant comarques) and **Morruda** (54k ha, 90% rainfed, concentrated in Baix Ebre / Montsià / Terra Alta). Morruda's high rainfed fraction gives a cleaner WD signal; Arbequina's wider geographic spread tests generalisability. Both use rainfed yield (`yield_tha` / `seca_ha`) as the response.
- **Harvest truncation:** Fixed common upper bound at 1700 GDD for all comarca-years.
- **Lag yield:** Data covers 2015–2024; after lag, n = 9 effective years (2016–2024), 90 Arbequina comarca-years. Morruda comarca-year count TBD after whitelist construction.
- **T_b:** 10 °C default (De Melo-Abreu et al. 2004), sensitivity sweep in Step 10.
- **Phenological anchors:** Didevarasl et al. 2023, Sanz-Cortés et al. 2002, Rapoport et al. 2013 (see `data/processed/phenology_anchors.csv`).

This plan supersedes prior rolling-OLS analyses, which suffered from window-overlap artifacts and gave physiologically implausible sign-stable negative precipitation coefficients.

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

- Comarca-level annual olive yield, filtered separately for two cultivar cohorts:
  - **Arbequina cohort:** 10 comarques where Arbequina > 90% of olive area (from DUN parcel data)
  - **Morruda cohort:** comarques where Morruda > 90% of olive area (to be constructed from DUN)
- Daily climate per comarca: T_min, T_max, T_mean, precipitation, ET₀, VPD
- Use rainfed yield only (`yield_tha` / `seca_ha`), not irrigated yield

Validate that every comarca-year with a yield observation has complete daily climate coverage from January 1 through harvest. Log missing-data patterns. Drop comarca-years with > 5 % missing daily values rather than imputing. Run the full pipeline (Steps 2–9) independently for each cultivar cohort.

### Step 2 — GDD axis (`02_compute_gdd.py`)

Compute cumulative GDD per comarca per year from January 1, using base temperature **T_b = 10 °C** for olive (cite De Melo-Abreu et al. 2004 or the specific Spanish source you choose; do **not** optimize T_b on yield data — that creates the Laurent circularity).

```
GDD(d) = Σ_{i=1}^{d} max(0, T_mean(i) − T_b)
```

Produce a common GDD grid (e.g., 0 to 2000 GDD in 25 GDD steps → 81 grid points). Re-express each daily climate series on this GDD axis via linear interpolation. Store as `(n_comarca_years × n_gdd_grid)` arrays.

Add a **sensitivity-analysis hook**: keep T_b as a config parameter so we can refit with T_b ∈ {7, 8.5, 10, 12.5} °C in step 10.

### Step 3 — Water-deficit predictor (`03_compute_water_deficit.py`)

Primary predictor: daily water deficit `WD(t) = P(t) − ET₀(t)` on the GDD axis.

Secondary predictors for comparison runs: VPD curve, cumulative WD over a rolling 30-GDD window. Each gets its own model fit; the final paper reports WD as primary and VPD as a corroborating analysis.

Output: `WD_matrix` of shape `(n_obs, n_gdd_grid)` for each predictor.

### Step 4 — Phenological anchors (`04_align_phenology.py`)

Anchor olive phenological windows to literature GDD values, **not fit from data**:

- Flowering: 350–450 GDD (cite source)
- Pit hardening: 900–1200 GDD (cite source)
- Oil accumulation: 1200–1700 GDD (optional secondary contrast)

These anchors should vary by cultivar group if the literature supports it. If isolating cultivars, store per-cultivar anchor windows. Document every literature source in `data/processed/phenology_anchors.csv` with a `source` column.

### Step 5 — Basis construction (`05_build_basis.py`)

Build B-spline basis on the common GDD grid:

- Cubic B-splines, **k = 5–6 basis functions** (start with 5; check posterior EDF; only increase if pinned)
- Knots placed at quantiles of the GDD grid
- Output: basis matrix `B` of shape `(n_gdd_grid, n_basis)` — note this is grid × basis, **not** observations × basis
- Pre-compute `WD_reduced = WD_matrix @ B * dt` of shape `(n_obs, n_basis)` where `dt` is the GDD step size — this is the integrated functional predictor and the `dt` is essential for prior interpretability

### Step 6 — Bambi sanity check (`06_fit_bambi_sanity.py`)

Quick model to confirm the hierarchical structure behaves before committing to NumPyro. Pre-aggregate WD into stage-window means as predictors. Default specification uses lag_yield without year RE (see findings below). Use `--year-re` for the sensitivity comparison.

```python
import bambi as bmb
m = bmb.Model(
    "yield ~ wd_flowering_mean + wd_pit_mean + lag_yield + (1|comarca)",
    data=df,
)
fit = m.fit(draws=1000, tune=1000)
```

#### Step 6 findings (Arbequina, n=90, 10 comarques, 9 years)

**lag_yield + (1|year) collision confirmed.** Running both produces lag_yield β = −0.07 (HDI spans zero) with year_sigma = 0.43 and residual sigma = 0.31. The year RE absorbs the variance lag_yield was meant to capture. Removing `(1|year)` does not recover lag_yield (β = +0.03, still spans zero) — instead residual sigma jumps to 0.48 and the contrast weakens (P(Δ>0) drops from 0.94 to 0.69). Alternate bearing is undetectable at comarca-level aggregation.

Neither specification resolves the lag/year question cleanly:
- **lag_yield only (default):** clean convergence, interpretable, but lag is inert and residual is large.
- **year RE only:** absorbs common-year shocks (drought 2022, good year 2019), tightens residual, sharpens contrast — but is less transparent than lag.

Both specs should be run as a sensitivity comparison in Step 7.

**WD coefficient signs:** wd_flowering ≈ −0.01, wd_pit ≈ +0.02–0.05 (both span zero). Positive pit coefficient means less water deficit → more yield, consistent with mesocarp development being water-sensitive. Signs are directionally correct but imprecise at this aggregation level — the full functional model (Step 7) may resolve the shape more clearly.

**Morruda (n=18, 2 comarques):** Converges with `(1|comarca)` only (0 divergences, all r̂ ≤ 1.01) but estimates are too imprecise to interpret (90% CIs span ±0.33). Included for completeness; not a viable corroboration cohort at current sample size.

**Conclusion:** Hierarchical structure converges. Proceed to Step 7. Run both lag-yield and year-RE specifications as a sensitivity pair.

### Step 7 — NumPyro production model (`07_fit_numpyro_main.py`)

The headline model. Hierarchical scalar-on-function regression with P-spline smoothness penalty.

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(WD_reduced, B, dt, lag_yield, year_idx, comarca_idx,
          n_years, n_comarcas, n_basis, n_grid, y=None):
    # Smoothness via second-difference penalty on spline coefficients
    sigma_smooth = numpyro.sample("sigma_smooth", dist.HalfNormal(1.0))
    beta_coefs = numpyro.sample(
        "beta_coefs",
        dist.Normal(jnp.zeros(n_basis), 5.0)
    )
    # Second-difference penalty as a log-prior increment
    diff2 = jnp.diff(beta_coefs, n=2)
    numpyro.factor("smooth_pen", -0.5 * jnp.sum(diff2**2) / sigma_smooth**2)

    # Functional term: WD_reduced already includes integration via @ B * dt
    functional_effect = WD_reduced @ beta_coefs

    # Hierarchical random effects
    sigma_year = numpyro.sample("sigma_year", dist.HalfNormal(1.0))
    sigma_comarca = numpyro.sample("sigma_comarca", dist.HalfNormal(1.0))
    year_re = numpyro.sample("year_re", dist.Normal(0, sigma_year).expand([n_years]))
    comarca_re = numpyro.sample(
        "comarca_re", dist.Normal(0, sigma_comarca).expand([n_comarcas])
    )

    # Fixed effects
    alpha = numpyro.sample("alpha", dist.Normal(0, 10))
    lag_coef = numpyro.sample("lag_coef", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    mu = (alpha + functional_effect + lag_coef * lag_yield
          + year_re[year_idx] + comarca_re[comarca_idx])

    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
```

**Critical implementation notes:**

- **Lag yield vs year random effect collision.** Step 6 showed that lag_yield is inert at comarca scale (β ≈ 0 in both specs). Neither removing year_re nor keeping it recovers a meaningful lag coefficient. **Run both specifications as a sensitivity pair** and report which gives a more interpretable posterior. The year-RE spec tightens the residual (0.31 vs 0.48) and sharpens the contrast, but lag-yield is more transparent.
- `WD_reduced` must already incorporate `dt` (see step 5). The integration is baked in pre-MCMC.
- Use 4 chains, 2000 warmup, 2000 samples, `target_accept_prob=0.95`. JAX should make this fast — aim for < 5 min per fit.
- Save InferenceData as `.nc` to `results/posteriors/main.nc` for downstream analysis.

### Step 8 — Posterior contrast (`08_posterior_contrast.py`)

The headline number, computed independently for each cultivar cohort. Two correct ways to compute the stage-window contrast:

**Window-integrated β (preferred — matches what physiologists mean by "stage sensitivity"):**

```python
# Posterior draws: shape (n_chains, n_samples, n_basis)
beta_draws = idata.posterior["beta_coefs"].values
beta_draws_flat = beta_draws.reshape(-1, n_basis)   # (n_total, n_basis)

# β(t) at every grid point: (n_total, n_grid)
beta_t = beta_draws_flat @ B.T

# Window masks on the GDD grid
flowering_mask = (gdd_grid >= 350) & (gdd_grid <= 450)
pit_mask = (gdd_grid >= 900) & (gdd_grid <= 1200)

# Mean β within each window per posterior draw
beta_flower = beta_t[:, flowering_mask].mean(axis=1)
beta_pit = beta_t[:, pit_mask].mean(axis=1)

# Sign-aware contrast (preferred when both expected negative)
delta = beta_pit - beta_flower      # positive ⇒ pit hardening less negative ⇒ less sensitive
prob_pit_less_sensitive = (delta > 0).mean()

# Also report individual window posteriors so reviewers see the signs
print(f"β_flowering: median {np.median(beta_flower):.3f}, "
      f"90% CI [{np.quantile(beta_flower, 0.05):.3f}, {np.quantile(beta_flower, 0.95):.3f}]")
print(f"β_pit:       median {np.median(beta_pit):.3f}, "
      f"90% CI [{np.quantile(beta_pit, 0.05):.3f}, {np.quantile(beta_pit, 0.95):.3f}]")
print(f"Δ (pit − flower): median {np.median(delta):.3f}, "
      f"90% CI [{np.quantile(delta, 0.05):.3f}, {np.quantile(delta, 0.95):.3f}]")
print(f"P(pit less sensitive than flower) = {prob_pit_less_sensitive:.3f}")
```

**Absolute-value contrast (fallback only):** Use `|β_flower| − |β_pit|` only if the sign of β is genuinely uncertain at one stage. Folded posteriors are harder to interpret — avoid by default.

Output: a side-by-side results table for Arbequina and Morruda with point estimates, 90 % CIs, and posterior probabilities for the headline contrast plus each window's β. If both cultivars show the same sign and comparable magnitude for Δ, that strengthens the finding. If they diverge, the discussion should address possible physiological or data-quality explanations (e.g. Morruda's geographic concentration, different phenological timing).

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

- T_b ∈ {7, 8.5, 10, 12.5} °C
- Phenology window widths ± 50 GDD
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

1. ~~Cultivar pooling~~ → Two-cultivar side-by-side: Arbequina (headline) + Morruda (corroboration).
2. ~~T_b and anchor sources~~ → De Melo-Abreu et al. 2004; Didevarasl et al. 2023; Sanz-Cortés et al. 2002.
3. ~~Harvest truncation~~ → Fixed 1700 GDD upper bound.
4. ~~Lag yield~~ → Confirmed: 2015–2024 yield data, n = 9 effective years after lag.

## Pipeline status and results summary

### Implemented (Steps 1–6)

**Data panel:** 90 Arbequina comarca-years (10 comarques × 9 years, 2016–2024 after lag). Mean rainfed yield 1.08 t/ha (sd 0.53), median 0.91. 18 Morruda comarca-years (2 comarques) included for completeness.

**GDD–phenology validation:** GDD thresholds at T_b=10°C map to physiologically correct calendar dates in Catalonia: flowering onset (350 GDD) → mean May 24 (range May 2–Jun 18), pit hardening start (900 GDD) → Jul 9 (Jun 22–Aug 2), harvest truncation (1700 GDD) → Sep 4 (Aug 8–Nov 1). Water deficit deepens from flowering (WD = −3.6 mm/day) to pit hardening (−5.1); VPD rises from 1.78 to 2.51 kPa. Both gradients match expected Mediterranean summer drought intensification.

**B-spline basis:** 5 cubic basis functions on 69 grid points (0–1700 GDD, step 25). Partition of unity confirmed (row sums = 1.0 exactly). One interior knot at GDD 850.

**Bambi sanity check (Arbequina, lag_yield only, no year RE):**
- Convergence: all r̂ = 1.0, all ESS > 400, 0 divergences.
- wd_flowering: β = −0.014 (90% HDI [−0.060, 0.030])
- wd_pit: β = +0.016 (90% HDI [−0.081, 0.106])
- lag_yield: β = +0.034 (90% HDI [−0.163, 0.232]) — inert
- comarca_sigma = 0.30, residual sigma = 0.48
- Δ(pit − flower) = +0.031, P(Δ>0) = 0.69

**Sensitivity comparison (Arbequina, lag_yield + year RE):**
- year_sigma = 0.43, residual sigma = 0.31 (35% tighter)
- lag_yield collapses to β = −0.071 (HDI spans zero)
- Δ(pit − flower) = +0.071, P(Δ>0) = 0.94 (contrast sharpens)
- Collision confirmed: year RE and lag_yield compete for the same variance

**Interpretation:** Coefficient signs are directionally correct — pit hardening shows a more positive WD coefficient than flowering, consistent with mesocarp development being the more water-sensitive stage. However, both coefficients span zero at this aggregation level. The stage-mean predictors compress the functional shape into two scalars; the full B-spline model (Step 7) should resolve finer structure. Alternate bearing is undetectable at comarca-level aggregation regardless of specification.

### Not yet implemented (Steps 7–10)

- **Step 7:** NumPyro production model with P-spline smoothness penalty and full functional predictor (not stage means). Both lag-yield and year-RE specs to be run.
- **Step 8:** Window-integrated β(t) posterior contrast over flowering vs pit hardening GDD ranges.
- **Step 9:** Leave-one-year-out stability (9 refits) to check if any single year drives the result.
- **Step 10:** Sensitivity sweeps (T_b, window width, k, predictor choice, irrigated negative control).

## Minimum viable deliverable

If steps 7, 8, and 9 run end-to-end on real data with sensible posteriors and a plotted Δ contrast, that’s a defensible exploratory paper. Steps 6 and 10 strengthen it. Steps 1–5 are infrastructure that has to work but isn’t the science.
