# Catalan Olive Analysis: Supervisor Update

> **DEPRECATED — do not cite (May 2026).** The analysis described in this
> document is based on sliding-window regressions over overlapping calendar
> intervals. That methodology is invalid because adjacent windows share most
> of their daily observations, producing strong serial autocorrelation in
> both predictors and residuals. The resulting coefficients, R² values, and
> "peak" window dates are statistical artifacts of window overlap rather
> than real phenological signals, and the implied significance is overstated.
>
> This document is retained for project history only. The current olive
> analysis uses a hierarchical Bayesian scalar-on-function regression with a
> B-spline basis on fixed calendar phenological windows; see
> `olive_water_sensitivity_plan.md` for the active methodology and results.
> Findings stated below (Tmin as primary heat signal, summer-rain damage,
> three-channel model, 49 % within-comarca R², etc.) should be re-derived on
> the current pipeline before they are cited in the thesis or in any
> supervisor-facing document.
>
> *Note added May 2026.*

**Project:** PhD Research — AI in Horticultural Irrigation  
**Document purpose:** Update on progress since the April 2026 CORINE-based briefing. Describes what changed methodologically, what was found, and what it means for the thesis.  
**Audience:** Supervisors with different backgrounds. Does not assume specialist knowledge of statistics or agronomy.  
**Preceding document:** *Olive Climate Stress and Irrigation Timing: A Briefing for Supervisors* (April 2026)  
**Date:** April 2026

---

## 1. What Changed Since the Previous Briefing

The previous briefing described an analysis built on a hybrid climate dataset: weather was matched to olive farm locations identified from CORINE Land Cover 2018 satellite imagery, with a centroid-based fallback for comarques where CORINE did not map enough farms. That pipeline covered 36 comarques and produced a panel of around 320 comarca-year observations.

Since that briefing, three things changed.

**The spatial matching source was replaced.** CORINE has a 25-hectare minimum mapping unit, which meant it missed small and dispersed olive groves entirely — including most of the flat interior Lleida comarques where some of Catalonia's most important Arbequina olive production occurs. The CORINE-based hybrid was replaced with the DUN parcel registry (Declaració Única Agrícola), which records every individual agricultural parcel declared to the Catalan government. This gives near-complete olive coverage at ~0.5 ha resolution, with each parcel tagged by crop, irrigation regime (rainfed or irrigated), and cultivar. Climate is now aggregated per comarca by weighting each AgERA5 grid cell by the DUN olive area it contains, rather than by CORINE polygon geometry. This is more accurate and covers all 42 Catalan comarques.

**The analysis panel was restricted to a single cultivar.** The previous analysis treated all 36 olive comarques as a single population. Catalan olive production spans many cultivars — Arbequina in the west and south, Morruda and Empeltre in southern Tarragona, others elsewhere — with baseline yields differing by a factor of five across comarques for reasons unrelated to climate. Those structural differences were absorbed by comarca fixed effects in the regression, which consumed most of the available variance and left very little for climate signals to explain. Using DUN cultivar shares, the panel was restricted to 10 comarques where Arbequina accounts for at least 90% of rainfed olive area and at least 500 ha of rainfed Arbequina is recorded. This reduced the sample from 174 to 100 comarca-year observations, but the climate signal strengthened by a factor of nearly four.

**Additional climate variables were tested.** The previous analysis focused on maximum temperature (Tmax ≥ 32°C day counts) and vapour pressure deficit (VPD). The updated analysis also tested minimum temperature (Tmin), precipitation, solar radiation, cumulative water balance (P − ET₀), and growing degree days.

---

## 2. What Was Found

### 2.1 Minimum temperature, not maximum temperature, is the primary heat signal

The previous analysis identified a late July to late August window where high maximum temperatures were negatively associated with olive yield. That result holds on the cultivar-restricted panel, but minimum temperature tells a substantially stronger story.

When mean Tmin over the 21 days ending 17 August is used as a predictor instead of mean Tmax over the same window, the within-comarca explanatory power rises from 32% to 42% of available variance. When both temperature variables are included in the same model, Tmax loses statistical significance while Tmin remains strongly negative. Tmax was a noisy proxy for a signal that Tmin carries more directly.

The mechanism is consistent with night-time physiology. During the day, photosynthesis produces carbohydrates that are needed for fruit development and oil accumulation. At night, respiration consumes some of those carbohydrates. When nights are warm, respiration runs faster, consuming more of the carbon that would otherwise go into yield. Additionally, overnight cooling normally allows the tree to rehydrate and recover from the daytime heat load; warm nights eliminate that recovery window. Both pathways are documented in the olive physiology literature, particularly in relation to the ongoing decline in Andalusian olive yields over the past decade as Mediterranean nights have warmed.

The quantified effect is approximately **−21% yield per °C of mean August minimum temperature** (β = −0.43 t/ha per °C at a panel mean yield of around 2 t/ha), making this the strongest single climate signal identified in this dataset to date.

### 2.2 Summer rain is also damaging, and the mechanism is not what would be expected

A separate scan of precipitation revealed a strong negative association between June–July rainfall and olive yield. More summer rain predicts lower yield: approximately **−1.5% yield per millimetre** of rain in the 30 days ending around 19 July.

This is initially counterintuitive — olive is a drought-tolerant tree grown primarily without irrigation, so summer rainfall might be expected to help. Four independent tests were run to identify what mechanism is actually operating. Tests 2, 3 and 4 are shown as Panels A, B and D of Figure 4; Panel C plots the full solar-radiation scan as supporting context for the PAR-loss test in Panel D. Test 1 (the irrigated comparison) is shown separately in Figure 3.

**Test 1 — Irrigated comparison (Figure 3).** The same models were refit on a panel of irrigated Arbequina comarques. If the summer rain signal were primarily a hidden water-supply effect — that is, dry years are bad for yield and rainy years happen to also be wetter years — then irrigation, which eliminates the water-supply deficit, should make the rain signal disappear. It did not. On the irrigated panel, the rain coefficient kept the same negative sign and was larger in absolute magnitude, not smaller. Water supply does not explain the result.

**Test 2 — Phenological alignment (Figure 4, Panel A).** The calendar window (30 days ending 19 July) was replaced with a growing degree day anchor, so that each comarca-year was compared to rain during the same physiological development stage rather than the same calendar period. If the rain signal were driven by a phenological sensitivity window — a specific moment in fruit development when the tree is vulnerable — the GDD-anchored version should perform at least as well as the calendar version. The full GDD scan was run across endpoints from roughly 600 to 2200 cumulative growing degree days, and at no endpoint did within-comarca R² reach the calendar peak (0.254 best GDD vs 0.389 best calendar). The signal is tied to a calendar period (the convective thunderstorm season), not to a developmental stage.

**Test 3 — Seasonal water balance (Figure 4, Panel B).** Cumulative rainfall minus evapotranspiration (P − ET₀) accumulated from the start of the agronomic year was tested as a predictor. If the rain signal were a proxy for overall seasonal water availability, the running balance should be a stronger predictor than any short calendar window. It explained only about 4% as much within-comarca variance as the 30-day calendar window.

**Test 4 — PAR-loss / contemporaneous solar radiation (Figure 4, Panel D).** One remaining possibility was that summer cloud cover reduces the light available for photosynthesis (PAR loss), and that rain days are a proxy for cloudy days. The clean test is to add solar radiation *measured in the same window as precipitation* (genuinely contemporaneous, not at a later timing) to a model that already contains precipitation, and to ask whether the precipitation coefficient shrinks. Panel D scans the precipitation window across summer and at each end-date refits three nested fixed-effects regressions: precipitation alone, with night-time minimum temperature added, and then with contemporaneous solar radiation added on top. Adding Tmin raises within-comarca R² substantially (by about +0.08 at the precipitation peak); adding contemporaneous solar radiation on top adds essentially nothing (about +0.008) — the two upper lines lie on top of each other across the whole scan. The contemporaneous solar radiation coefficient itself loses statistical significance once Tmin is in the model. PAR loss does not explain the precipitation signal.

For context, the supporting scan in Panel C shows that solar radiation as a univariate predictor is significantly *positive* in early July (peak β = +0.32 around 13 July) and significantly *negative* in early August (peak β = −0.21 around 3 August). This is consistent with solar radiation acting as an apparent PAR resource during the rainfall window and as a heat-load proxy afterward, but Panel D shows that neither effect, once Tmin is controlled, absorbs the precipitation signal — i.e. these two univariate sign patterns are themselves largely captured by the heat channel.

After four failed alternative explanations, the rain signal is attributed to **convective-storm damage** — hail, wind, and mechanical fruit drop — and **humidity-driven disease pressure**, particularly anthracnose, peacock spot, and olive fruit fly, which are activated by water on and around the trees rather than by any absence of a resource.

### 2.3 A three-channel model

The two heat and rain findings together, plus the solar radiation channel, form a three-predictor model:

- **Mean Tmin, 21 days ending 17 August** — night-warmth / oil-deposition channel
- **Total precipitation, 30 days ending 19 July** — storm and disease channel
- **Mean solar radiation, 30 days ending 10 August** — radiation-load channel

This three-channel model explains 49% of the within-comarca variance in yield (after removing the structural differences between comarques) and 66% of total variance including the structural differences. For context, on the previous mixed-cultivar panel the climate variables added only 5–10% of explanatory power over the comarca fixed effects alone; on the cultivar-restricted panel, the three climate channels (adding 34 percentage points) outperform the site dummies (32 percentage points). Restricting to a single cultivar transferred variance from geography to climate.

---

## 3. What This Means for the Thesis

The previous briefing concluded that the analysis was confirmatory — it verified that weather signals correlate with patterns that practitioners already know, without generating new agronomic knowledge. The updated analysis has gone further.

The displacement of Tmax by Tmin as the primary heat predictor was not expected at the outset and changes the mechanistic interpretation. The four-test mechanism triangulation for the rain channel — ruling out water supply, phenological alignment, seasonal balance, and PAR loss in sequence — is a structured inference rather than a simple correlation report. The irrigated comparison in particular provides a controlled test that is not available in most crop-climate datasets: the same model fit to two panels (rainfed and irrigated, same comarques, same years) with an interpretable prediction about how the coefficient should change under irrigation, and that prediction confirmed.

For the thesis framing, the key implication is that the climate-yield relationship in this crop and region is mechanistically specific enough to be a meaningful transfer-learning target. The signals are not generic correlations with annual average temperature or rainfall; they point to specific biological pathways operating in specific calendar windows. A model trained to capture these relationships has a defined physiological interpretation, which matters both for the scientific contribution and for any eventual application to irrigation scheduling.

---

## 4. Limitations

The panel remains small (100 comarca-year observations) and in-sample. R² values will be somewhat overstated relative to out-of-sample performance. Standard errors may be slightly underestimated because the 10 Arbequina-dominant comarques cluster geographically in western and southern Catalonia, and climate conditions across neighbouring comarques are not fully independent. The DUN parcel data are from 2024 only, though for a perennial crop with slow area change this is acceptable. Coefficient sensitivity across individual comarques is large for the temperature channel, indicating that the pooled slope is an average over sites with quite different responses — a finding that warrants further investigation once more years of data become available.

---

*Document prepared as part of PhD candidature in AI in Horticultural Irrigation, Deakin University.*

*This document was produced by Augment (Augment Code) under the direction of the candidate. The analysis it describes was conducted by the candidate using Python code developed with Augment assistance.*
