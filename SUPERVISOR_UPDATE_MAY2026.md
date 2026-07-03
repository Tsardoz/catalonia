# Supervisor Update — Winter Recharge as the Primary Climate Signal, with Notes on Autumn DANA Events

**Date:** May 2026
**Purpose:** Discussion point for next meeting
**Supersedes:** `SUPERVISOR_UPDATE_APR2026.md` (earlier sliding-window method was invalid)

## Background term

**DANA** = *Depresión Aislada en Niveles Altos*, a slow-moving cold-drop weather system over the warm Mediterranean in autumn. Produces intense, locally patchy convective storms. Both 2018 and 2024 had major DANAs in late October.

## The puzzle

The olive yield analysis finds that **autumn (Oct–Nov) water is associated with lower yield**. The irrigation literature finds the opposite — autumn root-zone water *helps* oil accumulation. The question is how to reconcile these.

## What the data shows

The autumn signal rests on two years, both DANA years. Autumn (DOY 300–327, late Oct to late Nov) rainfall across the 10 Arbequina comarques:

| Year | Mean rainfall (mm) | Std across comarques (mm) | Storm days (>10 mm) | Max single day (mm) | Within-comarca spatial spread (mm/day) |
|---|---|---|---|---|---|
| 2015 | 47 | 5 | 1.0 | 38 | 0.7 |
| 2016 | 42 | 4 | 1.9 | 15 | 0.7 |
| 2017 | 11 | 2 | 0.2 | 7 | 0.3 |
| **2018** | **139** | **21** | **5.1** | **25** | **1.9** |
| 2019 | 27 | 2 | 0.4 | 10 | 0.7 |
| 2020 | 39 | 7 | 1.5 | 17 | 0.9 |
| 2021 | 68 | 17 | 2.1 | 21 | 1.1 |
| 2022 | 17 | 3 | 0.0 | 6 | 0.5 |
| 2023 | 13 | 3 | 0.0 | 5 | 0.3 |
| **2024** | **149** | **43** | **5.8** | **34** | **2.6** |

2018 and 2024 are extreme on every axis: 3–5× the normal volume, 3× the normal count of heavy-rain days, and 2–4× the normal within-comarca patchiness. In 2024, individual Arbequina comarques received anywhere from 86 mm to 202 mm in the same autumn window.

Drop either 2018 or 2024 from the panel and the autumn effect disappears. Drop any other year and it persists. The **winter** signal (more water → higher yield) is stable in every case.

## Why the literature and our analysis disagree

The two studies are not really comparing the same thing.

The literature isolates *root-zone water supply* in drip-irrigated trials. Drip delivers water directly to the soil and keeps the canopy dry. Our analysis measures *rainfall on rainfed orchards*, which also wets the canopy and the fruit, raises humidity for days, and (in DANAs) brings wind and hail. Three additional pathways are active in our setting that simply do not exist in a drip experiment:

- **Disease.** Anthracnose, peacock spot, and olive fruit fly all require free water on the fruit. Drip orchards never trigger them; rainfed orchards in a wet autumn often do.
- **Harvest disruption.** Saturated ground keeps harvesters out; fallen fruit ferments on wet soil and is downgraded.
- **Mechanical fruit loss.** Ripe Arbequina detaches at a tenth of the force needed earlier in the season. DANA winds and hail strip fruit directly from the tree.

The two findings are answering different questions, not contradicting each other.

## Why the winter signal is the real result

The Catalan model's robust finding is **winter water surplus → higher yield** (S_pre stage, DOY 1–71, stable across every leave-one-year-out fit). Two independent Argentine field studies provide the mechanistic backing. Both used cv. Manzanilla rather than Arbequina; cultivar differences in drought sensitivity exist, but the shoot-elongation → fructification pathway is well established across cultivars.

### Rousseaux et al. 2008

- Mediterranean winter = high rain + low ETo → irrigation suspended. Argentine arid winter = no rain + higher ETo → irrigation must continue. The two systems are mirror images. Catalan winter rainfall is doing what Argentine growers pay to replicate.
- Remove irrigation for 15 days in sandy Argentine soil → soil water at 20–40 cm near wilting point, leaf water potential down 30 %. The tree is actively drawing on winter soil water. This is not dormancy.
- Yield directionally 20 % lower after winter irrigation suspension (16.5 vs 21.1 kg tree⁻¹). Didn't reach significance — n=4, underpowered — but the direction is consistent and the authors say so explicitly.
- VPD dominated stomatal conductance in winter regardless of irrigation level (r²=0.94). VPD is physiologically relevant outside summer. Supports the Catalan model's VPD signal.

### Correa-Tedesco et al. 2010

- Water deficit reduced yield 27 % at Kc=0.50. Mechanism was fruit *number*, not fruit size — individual fruit weight and diameter unchanged across all five irrigation levels. Water stress deletes fruits, not shrinks them.
- The path runs through shoot elongation: Kc=0.50 produced ~50 % of the shoot growth of Kc=1.0. Fewer new nodes → fewer inflorescence sites → fewer fruits *the following season*. This is a carry-forward structural effect, not a within-season physiological one.
- This directly answers Alessio's question: Catalan winter rainfall recharges the soil that supports spring shoot growth. Dry winter → reduced shoot elongation → fewer fructification sites → lower fruit number at harvest, even if summer stress is unremarkable.
- The carry-forward is observable, not theoretical: yield effects appeared in the season following the deficit treatment, not the same season. This is consistent with retaining `lag_yield` as a covariate in our model.
- Above Kc=0.85 yield plateaued despite continued vegetative growth — canopy became too dense for light penetration. Winter recharge is not "more is always better." There is a functional threshold below which the structural crop is compromised.

## Limits of the current data

- Nine independent years, two of which are DANA outliers. With only two extreme events, it is impossible to separate a "DANA effect" from "extreme rainfall" or "extreme storm-day count" — they describe the same two observations.
- The AgERA5 weather grid is 11 km; convective DANA rain is patchy on smaller scales. We have no field-level rainfall.
- Yield is reported only at comarca level, which averages out within-comarca damage.

## Discussion points

- **How should the autumn finding be reported?** I think it should be presented as a case description of 2018 and 2024, not as a regression coefficient. Treating two extreme events as a population-level effect overstates what the data can support.
- **The winter signal is the part that can carry weight.** It is stable across every robustness check and is the candidate transferable result. The Argentine results are also consistent with the stable positive S_pre coefficient across our moderate-rainfall years (2016, 2020, 2021 at 40–70 mm autumn rain, normal winter rainfall): under the shoot-elongation pathway, even sub-extreme winter recharge should support fruit number the following season.
- **Storm-event predictors are mechanistically more honest than window-mean rainfall**, but the current data cannot resolve them. In the literature, intensity drives mechanical stripping (a per-event threshold), while wetness duration drives disease (hours of free water on the fruit). In our panel they are too correlated to separate: across the 90 comarca-years, daily-intensity and longest wet-run correlate r ≈ 0.5; total rainfall and storm-day counts correlate r ≈ 0.97; in the two DANA years they are essentially the same variable. A meaningful intensity-vs-duration test would require a longer record with mixed event types (long soft rains as well as DANAs) — which we don't have, and can't get from Catalonia within the project timeline.
- **Given the Catalonia-only constraint,** the practical implication is that the autumn pathway will remain underdetermined for this thesis. The contribution should sit on the winter signal and on the methodological framework (scalar-on-function regression on calendar phenology), with autumn treated as a documented case rather than an open analytical question we plan to close.
## Sanity check — OLS by phenological stage (CWB-state means)
To address the request for a method-of-comparison that does not rely on the Bayesian scalar-on-function model, the same data was re-analysed by collapsing the daily CWB-state series into **five stage means per comarca-year** (S_pre, S5, S6, S7, S8) and fitting plain OLS, separately for the Arbequina and all-olive cohorts. Yield and lag_yield were centred; the five stage predictors were z-scored so coefficients are directly comparable in standardised units. Standard errors are HC3 robust (heteroskedasticity only — see autocorrelation note below).
**Results — Arbequina (n = 90; 10 comarques)** — R² = 0.264, joint F-test of the 5 stage coefficients: F = 7.30, p = 1.0 × 10⁻⁵
- S_pre: β = **+0.204**, SE = 0.039, p < **0.001**, 95% CI [+0.128, +0.280]
- S5:    β = −0.130, SE = 0.083, p = 0.117
- S6:    β = +0.101, SE = 0.067, p = 0.131
- S7:    β = −0.168, SE = 0.073, p = 0.021
- S8:    β = +0.047, SE = 0.071, p = 0.503
- lag_yield: β ≈ −0.014, p = 0.92
**Results — All olive (n = 162; 18 comarques)** — R² = 0.290, joint F-test: F = 15.01, p = 5 × 10⁻¹²
- S_pre: β = **+0.200**, SE = 0.028, p < **0.001**, 95% CI [+0.146, +0.255]
- S5:    β = −0.153, SE = 0.062, p = 0.014
- S6:    β = +0.120, SE = 0.053, p = 0.024
- S7:    β = −0.099, SE = 0.052, p = 0.056
- S8:    β = −0.047, SE = 0.053, p = 0.372
Figure: `results/figures/beta_by_season_ols_stepband.png` (piecewise-constant β with 90% CI band, same calendar-DOY axis and stage shading as the Bayesian β(t) figure).
### What this confirms
- **Winter recharge is the only large, stable signal.** S_pre is the strongest coefficient in both cohorts, with the smallest standard error and a p-value under 10⁻³ in both. It does not depend on the Bayesian machinery, the smoothness prior, or the choice of τ = 45 d. Strip everything back to a 5-predictor OLS and S_pre still dominates.
- **The S_pre effect size is essentially identical across cohorts** (β ≈ +0.20 in both Arbequina and the 18-comarca all-olive panel). A one-standard-deviation increase in winter CWB-state is associated with a 0.20-SD increase in yield. The agreement between the cultivar-restricted headline and the breadth check is the strongest evidence that this is a real, transferable signal rather than a cultivar-specific or panel-specific artefact.
- **Joint significance of the seasonal predictors is overwhelming** in both cohorts (joint F p ≈ 10⁻⁵ Arbequina, 10⁻¹² all-olive), so the climate signal is not a fluke of one stage.
### Why this is *not* a smooth transition between seasons
The Bayesian β(t) curve descends smoothly from a clear positive peak at S_pre, crosses zero around DOY ~150, and trends mildly negative by harvest. The OLS-by-stage picture is **not a smoother version of the same monotone story** — the stage coefficients oscillate in sign:
```
S_pre   S5      S6      S7      S8
+0.20   −0.13   +0.10   −0.17   +0.05   (Arbequina)
+0.20   −0.15   +0.12   −0.10   −0.05   (all olive)
```
This is the signature of **collinearity between adjacent stage means** combined with the τ = 45 d leaky-integrator that defines CWB-state. The stage-mean correlation matrix:
```
         S_pre   S5     S6     S7     S8
S_pre    1.00    0.43   0.29   0.13  −0.15
S5       0.43    1.00   0.72   0.46   0.16
S6       0.29    0.72   1.00   0.74   0.36
S7       0.13    0.46   0.74   1.00   0.64
S8      −0.15    0.16   0.36   0.64   1.00
```
Adjacent stages are correlated r = 0.6–0.7. When OLS is forced to partition a shared signal into discrete buckets, it allocates positive weight to one bucket and pulls the next negative to compensate. **The sign flips between S5/S6 and S6/S7 are a statistical artefact of binning a slowly varying continuous time series, not five distinct phenological responses.** This is exactly the problem the smoothness prior in the Bayesian functional regression is designed to address — it tells the model that β(t) cannot change discontinuously day-to-day, which forces OLS-style oscillations to collapse into the underlying smooth trend.
S_pre is the exception: it has the lowest correlation with the other stages (r ≤ 0.43) because winter precedes the others by several months, so the leaky-integrator memory has decayed. Its coefficient is therefore the one most resistant to collinearity-driven sign-flipping, and it is the one stage where OLS and the smooth model agree both in sign and in magnitude.
### Why the OLS p-values shown above are *not valid* as reported
Classical and HC3-robust OLS inference both assume residuals are **independent across observations**. In this panel they are not:
1. **Within-comarca serial autocorrelation.** Each comarca contributes up to 9 years; yield at year *t* is correlated with yield at *t−1* (which is partly why `lag_yield` is in the model) and unobserved comarca characteristics (soil, cultivar mix, elevation) persist across years. Two observations from Garrigues 2019 and Garrigues 2020 are not statistically independent.
2. **Cross-comarca spatial autocorrelation within a year.** Neighbouring comarques share weather systems; a wet 2024 winter is wet across most of inland Catalonia simultaneously.
3. **Predictor autocorrelation.** The CWB-state predictor is a leaky-integrator with τ = 45 d, so the regressor itself is serially correlated by construction.
OLS point estimates β̂ remain unbiased and consistent under all three conditions (Gauss–Markov for coefficients survives), but the **variance estimate** σ²(X'X)⁻¹ assumes iid residuals. With positive autocorrelation it understates the true variance, so the SEs above are too small, the CIs too narrow, and the reported p-values are anti-conservative. HC3 fixes heteroskedasticity but not autocorrelation.
The minimum acceptable fix for a short-T, multi-comarca panel is **comarca-clustered SEs** (`cov_type="cluster"`, clustering on comarca), which allows arbitrary within-comarca correlation in residuals. Adding two-way clustering on year handles the spatial-in-year component. With these corrections the S_pre p-value will widen but remain comfortably significant given its t ≈ 5–7; the marginal S5/S6/S7 results (p in the 0.01–0.13 range) will inflate and likely lose individual significance — which is consistent with the interpretation above that they are not separate effects to begin with.
The Bayesian hierarchical model handles these dependencies more transparently: the year random effect absorbs the year-level shocks, the lag_yield term absorbs persistence, and posterior uncertainty is propagated through the basis coefficients rather than being read off a sandwich estimator that was never designed for this data structure.
### Bottom line for the meeting
- **OLS and the Bayesian functional regression agree on the substantive finding**: winter water surplus (S_pre, DOY 1–71) is the largest, most stable, most robust climate effect on rainfed olive yield in Catalonia, β ≈ +0.20 SD per SD of CWB-state in both cohorts.
- **OLS by stage is a useful sanity check but a poor inferential tool here**: its sign-flipping between adjacent stages exposes the collinearity in the design, and its standard errors do not account for the panel autocorrelation that is unavoidable with comarca-year data. The reported p-values understate uncertainty.
- **The smoothness assumption in the Bayesian model is not a stylistic choice** — it is the statistical mechanism that prevents collinearity-driven OLS oscillations from being mistaken for true stage-to-stage biological switches. The two methods, taken together, are more convincing than either alone.
