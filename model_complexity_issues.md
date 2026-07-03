# Model Complexity vs Available Data — Issues Summary
**Date:** June 2026
**Purpose:** Discussion note for supervisor meeting
**Core problem in one line:** the model has more factors that can be tuned than the data can constrain.

---

## 1. The central issue: degrees of freedom vs effective sample size

The analysis has many researcher-tunable factors, but the data contain only a small amount of independent information to pin them down. The result is that several defensible parameter choices can move the secondary findings — there are more knobs than the data can support.

### Effective sample size is ~9, not ~90
- The Arbequina panel is 90 comarca-years (10 comarcas × 9 years after lag).
- Comarca replication is **spatial replication of the same annual climate** — all comarcas share each year's drought or wet winter.
- The climate→yield relationship is therefore identified from **year-to-year variation only ≈ 9 independent observations**.
- Any method, however sophisticated, is bound by this. More machinery cannot manufacture statistical power.

---

## 2. The factors that can be "wiggled" (researcher degrees of freedom)

Each is individually defensible, but together they define a large space, and some combinations change the secondary conclusions:

- Memory timescale of the water predictor (how far back water "counts")
- Number of basis functions / stages used to describe the season
- Strength of the smoothness assumption
- Phenological stage-window boundaries (±10 days is within stated uncertainty)
- Choice of water predictor (climatic water balance vs VPD vs precipitation)
- Cohort definition (minimum rainfed area threshold; cultivar restriction)
- Whether a year random effect is included
- Harvest-truncation date and lag-yield handling

With ~9 effective years these cannot all be identified jointly.

---

## 3. What is robust (survives essentially any reasonable choice)

**Winter / pre-flowering water is positively associated with rainfed olive yield.** This holds independently of the modelling machinery:

- Model-free correlation: Pearson r = 0.375, **p = 0.0003** (Arbequina, winter water vs centred yield).
- Plain OLS by stage: winter coefficient β ≈ **+0.20 SD per SD**, **p < 0.001**, in **both** cohorts (Arbequina and the broader ≥500 ha panel).
- Hierarchical Bayesian functional model: winter stage credibly positive.

The agreement in **sign and magnitude across cohorts and across three different methods** is the load-bearing result. It does **not** depend on the tunable parts of the model.

---

## 4. What is fragile (depends on choices the data can't resolve)

- **Ranking stages against each other** (e.g. fruit development vs flowering): adjacent stage predictors are strongly correlated (r ≈ 0.6–0.7). They cannot be cleanly separated; ordinary regression coefficients oscillate in sign, which is a **collinearity artefact, not biology**. A stage-by-stage sensitivity ranking is not reliably identified at this sample size.
- **Autumn (maturation) signal** rests almost entirely on two extreme storm years (2018, 2024). It should be reported as a **case description of two events**, not as a coefficient.
- **Leave-one-year-out sensitivity:** some stage contrasts flip sign when a single year is removed.
- **Representation dependence:** the winter signal is strong when water is encoded as a recency-weighted running state, but weak when encoded as a simple raw seasonal total. The result depends partly on how the predictor is constructed — which is itself one of the free factors.
- **Unpropagated assumption error (over-confident intervals):** the memory timescale, stage-window boundaries, and smoothness strength are all fixed at *guessed* values. The reported credible intervals are computed *conditional* on those guesses being correct, so they exclude this source of uncertainty. The true uncertainty is therefore **wider than reported** — the intervals are over-confident. The only honest fixes are to propagate the guesses (put priors on them and marginalise — costly and hard to identify at n ≈ 9) or to **eliminate** them (fewer assumptions = fewer uncounted error sources).

---

## 5. Caveat on the current sensitivity analysis

The existing sweep varies several parameters together (memory timescale × basis size × year-effect) as a full grid, so it does capture interactions among those — good. **But** it holds the stage-window boundaries, the predictor form, and the cohort threshold fixed at their baseline values. Sensitivity to *those* choices, and their interactions, is not assessed. Expanding the sweep further is double-edged: with ~9 years, exploring more knobs risks over-fitting the robustness analysis rather than resolving anything.

---

## 6. Interpretation: rain is not irrigation, and its harm needs fruit to be present

This is the physiological backbone that ties the winter and autumn findings together and reconciles them with the irrigation literature.

**Rain ≠ irrigation.** Drip irrigation delivers water to the root zone with the canopy and fruit kept dry, at controlled timing and amount. Rain delivers the same water *plus* a wet canopy, wet fruit, days of raised humidity, and sometimes wind and hail. The deficit-irrigation literature (e.g. Hueso et al. 2019; Gucci et al. 2019) measures only the **supply** channel. Rain additionally activates **disease, mechanical fruit loss, and harvest disruption** — channels that do not exist in a drip trial. "Irrigation helps in autumn" and "rain hurts in autumn" are therefore not contradictory; they measure different things.

**The harm channels require fruit to be present.** Disease (anthracnose, peacock spot), olive fruit fly, and mechanical stripping all need developing/ripening fruit as a target, and susceptibility rises as the fruit softens and accumulates oil. So:

- **Early season (winter / pre-flowering):** no fruit to damage → rain acts almost purely through the positive **supply** channel → positive effect (the robust winter signal).
- **Fruit development → maturation:** fruit is present, soft, and oil-rich → the **damage** channels dominate → negative effect (the S7–S8 decline).

This gives a single coherent reason for the monotone-declining β(t): the harm channel switches on only once fruit exists. It is not a statistical accident.

**Implications and honest caveats:**

- The autumn finding is not *purely* two freak DANA years — there is a plausible *continuous* damage mechanism, and the DANA years are its extreme, visible tail. But the data can only quantify the extreme; the mild version stays buried in noise. Honest line: *coherent mechanism, extreme years carry the estimate.*
- A water-balance predictor (P−ET0) is tuned to the **supply** channel and is largely **blind to the disease channel**, which is driven by wetness *duration/frequency* (rain days, wet spells, humidity) rather than net balance. So the current predictor is biased toward detecting the positive channel; a rain-day/wet-spell metric would be needed to probe the damage channel directly (under-powered here, but the right tool).
- There is also a separate early-negative wrinkle: rain *at flowering* can disrupt wind pollination and reduce fruit set. So "early always good" is not perfectly clean either.
- **This physiology justifies the K=2 split.** "Early rain = supply benefit, late rain = fruit damage" is a before/after-fruit-set structure. The two-window model is therefore motivated by biology, not imposed for statistical convenience — a stronger footing for the deparametrised model.

---

## 7. Proposed direction: match the model to the claim, and deparametrise

The cleanest way out of "too many knobs" is to retreat to the claim the data can actually carry and use the simplest valid model for it.

- **The data robustly support:** *wetter winters → higher rainfed olive yield.*
- **The data do not robustly support:** *a stage-by-stage ranking of water sensitivity.*

Implications for the model:

- **Headline:** a 1–2 predictor regression — winter water alone, or a pre-flowering vs post-flowering split — with comarca/year-clustered standard errors (or a year random effect). Few parameters; honest at n ≈ 9.
- **Any cross-stage statement must use a collinearity-aware method** — orthogonal-component regression (PCA/FPCA, with coefficients back-transformed to the original scale) or partial least squares, or a smoothness-prior functional model. Ordinary regression with correlated stage predictors is **not valid** and should not be used for stage comparisons.
- **Daily-resolution FPCA/PLS removes several guesses at once.** Running an orthogonal decomposition directly on the *daily* water curves replaces the guessed memory timescale, stage boundaries, and smoothness prior with a single data-driven basis. The only remaining choice — how many components to keep (~2–3 at this sample size) — can be made by cross-validation rather than guessed, and the coefficient curve β(t) is recovered by back-transformation. This does not reduce the *effective* component count much (the current model already resolves only ~2–3 shape features), but it makes the components orthogonal and cuts the number of fixed assumptions, so the resulting intervals are wider but more honest.
- **Frame the whole analysis as exploratory / hypothesis-generating** — appropriate and unavoidable at this sample size.

---

## 8. Decision needed at the meeting

Which claim do we stand behind? The model complexity should follow this decision, not precede it.

1. **"Winter water matters"** — robust, simple model, strong evidence. *Recommended headline.*
2. **"Winter matters more than later stages"** — requires a collinearity-aware method, carries more caveats, and is only weakly identified at n ≈ 9.

Choosing (1) lets us strip out most of the tunable factors and present a transparent, defensible result, with the more elaborate functional model demoted to a supporting/robustness role rather than the main event.

---

## 9. Update (July 2026) — Tau (memory timescale) sensitivity: resolved at the OLS level

Section 2 listed the CWB_state memory timescale (τ) as one of the free/tunable factors that
could be driving the S_pre result. This has now been tested directly rather than assumed.

**Method:** `src/tau_sensitivity_sweep.py` recomputes CWB_state from scratch and refits the
OLS stage-mean model at each τ independently — no coefficients are frozen or carried between
τ values, and S_pre is not computed once and reused. Grid: τ = 15–90 days, step 5 (16 values),
Arbequina cohort (n = 90).

**Result:**

| τ (days) | β_S_pre | 95% CI | R² |
|---|---|---|---|
| 15 | 0.171 | [0.066, 0.276] | 0.214 |
| 30 | 0.193 | [0.112, 0.274] | 0.256 |
| 45 (baseline) | 0.204 | [0.128, 0.280] | 0.264 |
| 60 | 0.217 | [0.133, 0.300] | 0.264 |
| 75 | 0.234 | [0.131, 0.336] | 0.263 |
| 90 | 0.253 | [0.124, 0.383] | 0.261 |

- **Sign:** positive at all 16 τ values tested; 95% CI excludes zero at every τ.
- **Coefficient of variation:** 11.2% across the full range (β_S_pre 0.171–0.253) — well
  inside the "stable" band, not the narrow-collapse pattern that would invalidate a
  single-τ result.
- **R²** is essentially flat (0.21–0.26), peaking mildly around τ = 45–55, i.e. the
  baseline choice happens to sit near the OLS optimum rather than being an outlier pick.

**Conclusion:** at the OLS level, the S_pre winter-water effect is **not** an artefact of
the τ = 45 d choice. This removes "memory timescale" from the list of unresolved
researcher degrees of freedom in Section 2, and directly supports the Section 3 robustness
claim (winter water effect holds independently of the modelling machinery).

**Not yet closed:** this sweep used the OLS stage-mean model only (fast, ~1 minute for the
full grid). The stricter version of this check — refitting the actual reported Bayesian
scalar-on-function model at each τ and comparing leave-one-year-out (LOYO) skill against
effect size — has not been run (estimated ~12–24 h with LOYO at all 16 τ; ~2–3 h for
Bayesian point estimates without LOYO). Per the standing rule for this analysis, if a
headline τ is ever reported explicitly (rather than the current practice of citing a
range), it must be selected by out-of-sample LOYO skill under the functional model, not by
OLS effect size. See `TAU_SWEEP_README.md` for how to run the remaining Bayesian/LOYO leg.

Artifacts: `src/tau_sensitivity_sweep.py`,
`results/tables/tau_sensitivity_sweep_arbequina.csv`,
`results/figures/tau_sensitivity_sweep_arbequina.png`.
