# fPLS vs. Bayesian Comparison — Results Summary

Summary of the functional Partial Least Squares (fPLS) vs. hierarchical Bayesian
(RW2 P-spline) scalar-on-function comparison for olive yield, implemented in
`src/11_fit_fpls_comparison.py` (extends `src/07_fit_numpyro_main.py`).

All predictive numbers below are **leave-one-year-out (LOYO) cross-validated**
pooled RMSE/R², not in-sample fit. Both estimators use a deliberately
stripped-down single-predictor spec (no `lag_yield`, no year random effect) so
they can be compared on equal footing — see [Caveats](#caveats-read-first)
before drawing conclusions from any single number.

## Caveats (read first)

- **Smoothers, not deployable models.** Stripping `lag_yield` and the year RE
  removes real predictive signal from both sides. No LOYO RMSE/R² below
  reflects "the performance of the project's model" — only the standalone
  contribution of the tested predictor(s).
- **~9 LOYO folds is a noisy, underpowered estimator.** Differences of a few
  hundredths in R² between rows are within fold-to-fold noise, not a
  meaningful ranking.
- **fPLS is supervised (PLS), Bayesian is unsupervised (RW2 smoothness
  prior on a fixed 5-basis reduction)** — different estimator classes, not
  two settings of one knob.
- **Headline fPLS complexity (k=1) is fixed a priori**, independent of any
  LOYO score, to avoid selection leakage. Diagnostic-only `*_component_selection_*.csv`
  sweeps are explicitly selection-optimistic and never used to pick a
  headline model.
- Every finding here has **not yet included `lag_yield`** (deferred to
  follow-up work) — see [What's next](#whats-next).

## 1. Water balance (`wd_state`, CWB-state, tau=45 days)

The headline comparison: `CWB_state(t)` (leaky-integrator climatic water
balance, τ=45d) as the sole functional predictor.

| Cohort | Model | RMSE | R² |
|---|---|---|---|
| arbequina | fPLS (k=1) | 0.457 | −0.077 |
| arbequina | Bayesian (RW2) | 0.473 | −0.153 |
| all_olive | fPLS (k=1) | 0.435 | −0.099 |
| all_olive | Bayesian (RW2) | 0.436 | −0.106 |

fPLS and the Bayesian model land in the same modestly-negative neighborhood in
both cohorts — consistent with a genuinely weak signal (the project's own
in-sample OLS diagnostic put water balance at ≈2% of yield variance) rather
than an estimator-specific problem. The scale-matching check (fPLS `coef_(t)`
vs. Bayesian `beta_phys(t)`) passed cleanly in both cohorts (ratio 1.09–1.13,
not scale-suspect).

## 2. Storm-year attribution (`--drop-years 2018,2024`, arbequina, wd_state)

Dropped 2018 and 2024 (suspected DANA storm years) and refit both estimators
(n: 90→70 obs, 9→7 years).

**Result: a clean sign-recovery.** The Bayesian median and fPLS k=1 both flip
the autumn (S8) sensitivity sign from **negative** (full data) to **positive**
(storm years removed) — the pre-registered "clean win" case, meaning the
full-data negative autumn reading looks storm-driven rather than a real
seasonal effect.

- Component-attribution diagnostic: the fPLS k=1 component carrying the
  full-data negative S8 mass has 2018 at **rank 1 of 90** on `|z|`-score
  (the single most extreme observation in the dataset).
- Unsupervised fPCA check (yield-independent): inconclusive as a *narrow*
  autumn-localization story — the two dominant modes (82% of variance) are
  broad whole-season patterns, not autumn-localized. 2018 *is* a near-record
  outlier (rank 2/90) but on a broad mode, plausibly because the τ=45-day
  leaky-integrator memory smears a discrete storm event across a wide
  temporal window rather than confining it to S8.
- Caveat: n drops to 7 years, so this is a **qualitative sign-recovery test,
  not a precision estimate**; fPLS k=3 is barely identified at this n and
  should be weighted less than the k=1/Bayesian agreement.

Artifacts: `fpls_vs_bayesian_stage_means_drop2018-2024_arbequina.csv`,
`fpls_vs_bayesian_beta_t_drop2018-2024_arbequina.png`,
`fpls_component_attribution_drop2018-2024_arbequina.csv`,
`fpls_fpca_storm_diagnostic_drop2018-2024_arbequina.csv`.

## 3. Cohort check: arbequina vs. all_olive

A quick in-sample OLS screen (5-basis-reduced predictor vs. yield, no CV)
across 7 candidate predictors found arbequina had *higher* in-sample R² than
all_olive for 5 of 7 predictors — the opposite of what "cultivars aren't that
different, use more data" would predict. This is consistent with arbequina's
cultivar-restricted, more homogeneous cohort being the right headline choice,
not a reason to switch to all_olive. Both cohorts are still run side by side
throughout this comparison for transparency.

## 4. Tmin

Despite by far the highest in-sample R² of any predictor tested (0.46
arbequina, 0.37 all_olive, from the same ad hoc OLS screen — **not
cross-validated**), Tmin's genuine LOYO performance is unremarkable:

| Cohort | Model | RMSE | R² |
|---|---|---|---|
| arbequina | fPLS (k=1) | 0.457 | −0.076 |
| arbequina | Bayesian (RW2) | 0.497 | −0.276 |
| all_olive | fPLS (k=1) | 0.445 | −0.150 |
| all_olive | Bayesian (RW2) | 0.458 | −0.224 |

fPLS k=1's LOYO R² is nearly identical to `wd_state`'s (−0.076 vs. −0.077) —
the large in-sample advantage evaporates completely out-of-sample. This is a
concrete demonstration of why in-sample R² is unreliable at this n. The
Bayesian model does *worse* on Tmin than on `wd_state`, suggesting its RW2
posterior overfits Tmin's in-sample shape more than the water-balance curve's.

**The one robust finding:** β(t) is **negative in the S8 (autumn/maturation)
window for all four estimator/cohort combinations** (Bayesian + fPLS k=1/2/3,
both cohorts) — i.e., cooler autumn nights are consistently associated with
higher yield, a physiologically plausible respiration-loss mechanism. The
S7−S6 contrast, by contrast, disagreed in sign between Bayesian (negative) and
fPLS (positive at every k) in both cohorts — an open, unresolved discrepancy.

## 5. VPD

| Cohort | Model | RMSE | R² |
|---|---|---|---|
| arbequina | fPLS (k=1) | 0.535 | −0.478 |
| arbequina | Bayesian (RW2) | 0.496 | −0.271 |
| all_olive | fPLS (k=1) | 0.499 | −0.449 |
| all_olive | Bayesian (RW2) | 0.452 | −0.191 |

VPD is the weakest predictor tested by LOYO R², and the first case where
**Bayesian beats fPLS k=1** (the reverse of `wd_state` and Tmin) — fPLS's
rank-1 fit may be a poor match for VPD's curve shape specifically.

Despite the weak predictive numbers, VPD produced the **cleanest cross-estimator
stage-level sign agreement** of any predictor tested: S_pre confidently
negative for Bayesian in both cohorts (90% CI excludes zero) and negative at
every fPLS k; S8 confidently positive for Bayesian in all_olive (CI excludes
zero) and positive at every fPLS k in both cohorts.

## 6. Combined predictors: `wd_state` + `vpd` (fPLS only)

skfda's `FPLSRegression` has no multi-block support (confirmed against its
source), so combining predictors uses `sklearn.cross_decomposition.PLSRegression`
on the concatenated 5-basis-reduced curves (10 columns total). No Bayesian
counterpart — the Bayesian model is single-predictor only.

Justification check before combining: pointwise correlation between the two
curves ranges from −0.80 to +0.41 across the season (moderate negative on
average, season-dependent) and cross-block basis correlation averages ~0.45
(max 0.85) — correlated but not collinear, and PLS is specifically robust to
correlated predictors.

| Cohort | Model | RMSE | R² |
|---|---|---|---|
| arbequina | combined (wd_state+vpd, k=1) | 0.465 | −0.116 |
| arbequina | wd_state alone | 0.457 | −0.077 |
| arbequina | vpd alone | 0.535 | −0.478 |
| all_olive | combined (wd_state+vpd, k=1) | 0.433 | −0.089 |
| all_olive | wd_state alone | 0.435 | −0.099 |
| all_olive | vpd alone | 0.499 | −0.449 |

**Combining did not achieve positive R².** For arbequina it was *worse* than
`wd_state` alone (the weaker predictor dragged the combination down); for
all_olive it was a ~1-point improvement over `wd_state` alone — within fold
noise, not a real gain. At the fixed headline k=1, both predictors are forced
through a single shared latent component, so their β(t) contributions are
coupled (near-mirror-image shapes, consistent with their negative
correlation) rather than each fitting independently.

## Cross-cutting takeaways

1. **In-sample R² is actively misleading at this n.** Every case where an
   in-sample number looked strong (Tmin: 0.46) collapsed to the same
   underpowered LOYO range (~−0.05 to −0.5) as everything else.
2. **The predictive ceiling (LOYO R² ≈ 0, mostly negative) has not moved**
   across single predictors, multiple predictors, or model flexibility
   (k=1 through k=5). This points to n≈9 independent climate-years as the
   binding constraint, not predictor choice.
3. **β(t) shape/sign agreement across estimators is the more informative
   signal than RMSE/R² throughout** — e.g. Tmin's S8 sign, VPD's S_pre/S8
   signs, and the water-balance storm-year sign recovery all held up
   consistently even where predictive R² did not.
4. **Storm-year contamination is real and attributable**, but its
   fPCA-based signature is broad-seasonal, not narrowly autumn-localized —
   a nuance worth keeping distinct from the fPLS-based (yield-supervised)
   attribution.

## What's next

- **`lag_yield`** has not yet been added to either estimator in this
  comparison (deliberately excluded throughout for a fair single-predictor
  smoother comparison). It's the one lever not yet tried that plausibly
  could move LOYO R² meaningfully, since alternate-bearing is a known strong
  yield driver independent of climate. Deferred to follow-up work.
- Extending `--combine-predictors` to 3+ predictors (e.g. `wd_state,vpd,tmin`)
  or other pairings has not been tried.
