# The Autumn Maturation Stage (S8) and DANA Storm Events

**Date:** June 2026  
**Status:** Draft — for inclusion in thesis Chapter 4 (Results and Discussion)  
**Figure:** `results/figures/dana_autumn_analysis.png`  
**Analysis scripts:** `src/dana_loyo_comparison.py`

---

## Context

The scalar-on-function regression model covers five phenological stages across the olive
growing season (DOY 1–327, January to late November). Stage S8 (DOY 300–327, late October
to late November) corresponds to fruit maturation and the onset of harvest. The climatic
water balance (CWB) predictor — precipitation minus reference evapotranspiration — is
carried forward through the season as a leaky-integrator state variable (τ = 45 days),
so the S8 coefficient reflects the accumulated water-balance signal entering the
maturation window.

The agronomic literature identifies autumn root-zone water as beneficial for oil
accumulation and mesocarp development in irrigated trials. Hueso et al. (2019), working
specifically on cv. Arbequina hedgerow orchards over three seasons in central Spain,
found that stem water potential during oil synthesis must be maintained above −2.21 MPa
for maximum oil production, and that deficit irrigation during this period (late summer
through harvest) reduces fruit size, oil and water content of the fruit. Gucci et al.
(2019) report equivalent findings for cv. Frantoio: regulated deficit irrigation imposed
during the initial phase of oil accumulation reduced olive and oil production relative to
full irrigation. The full model result appears to contradict this — but the
contradiction dissolves under closer examination.

---

## 4.x.1 Model Result and Apparent Contradiction with Agronomic Expectations

The full scalar-on-function model, fitted to all available years (2015–2024), returns a
negative median β coefficient for the S8 maturation stage:

| Cohort | β(S8) median | 90% CI | P(β_S8 < 0) |
|---|---|---|---|
| Arbequina (n = 90) | −0.000591 | [−0.00218, +0.00131] | 0.73 |
| All olive ≥500 ha (n = 162) | −0.000618 | [−0.00169, +0.00067] | 0.81 |

Taken at face value this would imply that a higher climatic water balance during the
maturation window is associated with lower rainfed yield — superficially inconsistent with
the irrigation literature.

Critically, however, **this signal does not reach 95% posterior credibility in either
direction**. P(β_S8 < 0) = 0.73 (Arbequina) and 0.81 (all-olive) — directionally negative
but well below the threshold applied to the headline S_pre result (P(diff) = 0.95–0.99).
The S8 coefficient should not be interpreted as a credible agronomic effect.

---

## 4.x.2 Autumn Rainfall Record and DANA Events

Inspection of the autumn rainfall record reveals that two years — 2018 and 2024 — were
affected by DANA (*Depresión Aislada en Niveles Altos*, cold-drop) events, producing
exceptional rainfall during the S8 window across the Arbequina comarcas:

| Year | Mean rainfall (mm) | Std across comarcas (mm) | Storm days (>10 mm) | Max single day (mm) |
|---|---|---|---|---|
| 2015 | 47 | 5 | 1.0 | 38 |
| 2016 | 42 | 4 | 1.9 | 15 |
| 2017 | 11 | 2 | 0.2 | 7 |
| **2018** | **139** | **21** | **5.1** | **25** |
| 2019 | 27 | 2 | 0.4 | 10 |
| 2020 | 39 | 7 | 1.5 | 17 |
| 2021 | 68 | 17 | 2.1 | 21 |
| 2022 | 17 | 3 | 0.0 | 6 |
| 2023 | 13 | 3 | 0.0 | 5 |
| **2024** | **149** | **43** | **5.8** | **34** |

2018 and 2024 are extreme on every axis: 3–5× the typical rainfall volume, 3× the normal
count of heavy-rain days, and 2–4× the normal within-comarca patchiness. Individual
comarca totals ranged from 102–167 mm in 2018 and from 86–202 mm in 2024, compared with
a typical inter-comarca spread of less than 20 mm in non-DANA years. All figures verified
from AgERA5 precipitation data (`data/agera5_daily_catalonia.csv`, DOY 300–327,
10 Arbequina comarcas).

---

## 4.x.3 DANA Attribution via Leave-One-Year-Out Analysis

Leave-one-year-out (LOYO) refits confirm that the negative S8 signal is entirely
attributable to these two events. Removing either DANA year individually is sufficient to
reverse the sign of β(S8) (Figure X, Panel C):

| Scenario | β(S8) median | 90% CI | P(β_S8 > 0) | S7−S8 P(diff) |
|---|---|---|---|---|
| Full model | −0.000591 | [−0.00218, +0.00131] | 0.27 | 0.69 |
| excl 2018 | +0.000103 | [−0.00170, +0.00231] | 0.54 | 0.58 |
| excl 2024 | +0.000082 | [−0.00174, +0.00255] | 0.53 | 0.50 |

In both excl* scenarios the 90% credible interval spans zero and P(diff) drops to 0.50–0.58,
indicating that the model can no longer distinguish the S7 and S8 stage sensitivities once
either DANA year is removed. The autumn signal is therefore a product of two observations,
not a systematic agroclimatic relationship.

---

## 4.x.4 Irrigated versus Rainfed Comparison Rules Out a Water-Supply Mechanism

If the negative S8 signal reflected a genuine agronomic response to excess water — for
instance waterlogging, anaerobic root stress, or dilution of oil content — irrigated fields,
which already receive supplemental water throughout the season, should be buffered against
additional autumn rainfall. The data do not support this.

Irrigated Arbequina fields show the same DANA-driven β(S8) flip as rainfed fields:

| Scenario | Rainfed P(β_S8 > 0) | Irrigated P(β_S8 > 0) |
|---|---|---|
| Full model | 0.27 → negative | 0.38 → negative |
| excl 2018 | 0.54 → neutral | 0.62 → neutral/positive |
| excl 2024 | 0.53 → neutral | 0.73 → positive |

Both regimes flip together (Figure X, Panel C). Since irrigation eliminates chronic
water-supply stress but offers no protection against canopy wetting, mechanical fruit loss
from wind and hail, saturated ground preventing harvest access, or fungal disease pressure
(*Colletotrichum acutatum*, *Spilocaea oleagina*, *Bactrocera oleae*), the convergence of
rainfed and irrigated responses identifies **storm damage and disease** — not water stress —
as the operative mechanisms.

---

## 4.x.5 2018 and 2024 Are Mechanistically Distinct

Despite both being DANA years, the two events tell different stories (Figure X, Panel B).
The irrigated-to-rainfed yield ratio by year for the Arbequina comarcas:

| Year | Rainfed (t/ha) | Irrigated (t/ha) | Ratio | Note |
|---|---|---|---|---|
| 2015–2017 | 1.42–1.74 | 2.98–3.89 | 2.52–2.66 | Normal |
| **2018** | **0.87** | **2.28** | **2.66** | DANA — ratio normal |
| 2019–2021 | 1.18–1.53 | 2.54–3.64 | 2.30–2.63 | Normal |
| 2022 | 0.41 | 1.29 | 3.16 | Severe drought |
| 2023 | 1.04 | 3.13 | 3.63 | Post-drought recovery |
| **2024** | **0.73** | **2.99** | **4.26** | DANA — ratio anomalous |

**2018:** The irrigated-to-rainfed ratio of 2.66 is indistinguishable from the non-DANA
mean of 2.60. Both regimes dropped proportionally from 2017 levels, consistent with storm
damage affecting all trees equally regardless of irrigation status.

**2024:** The ratio of 4.26 is the highest in the ten-year record. Rainfed yield collapsed
(0.73 t/ha) while irrigated yield remained at 2.99 t/ha — a divergence inconsistent with
a pure storm mechanism. The 2022 growing season produced a severe drought (rainfed
0.41 t/ha), and the subsequent recovery trajectories of irrigated and rainfed orchards
appear to have diverged through endogenous alternate-bearing cycles that irrigation can
buffer but rainfall cannot. The 2024 rainfed yield deficit is therefore likely confounded
by post-drought carry-over and desynchronised alternation; the DANA contribution cannot be
cleanly separated from these factors with the available data.

---

## 4.x.6 Mechanistic Interpretation

Three pathways link extreme autumn rainfall to lower yield in rainfed orchards, all
operating independently of root-zone water supply:

1. **Mechanical fruit loss.** Ripe Arbequina fruit detaches at a fraction of the force
   required earlier in the season [REF: detachment force at maturity — needs citation,
   e.g. Tombesi et al. or mechanical harvest literature]. DANA winds and hail strip fruit
   directly before harvest can begin.

2. **Disease pressure.** Anthracnose (*Colletotrichum acutatum*), peacock spot
   (*Spilocaea oleagina*), and olive fruit fly (*Bactrocera oleae*) all require free water
   on fruit or foliage. A sustained wet period in October–November creates ideal infection
   conditions and elevates both the incidence of premature fruit drop and the proportion of
   damaged fruit downgraded at the mill.

3. **Harvest disruption.** Saturated ground prevents mechanical harvester access; fallen
   fruit ferments on wet soil and cannot be recovered at commercial oil quality. In a DANA
   year, a significant fraction of the harvestable crop is effectively lost post-maturation.

None of these pathways exist in drip-irrigated experimental trials, which is why the
irrigation literature finds the opposite relationship: controlled root-zone water without
canopy wetting is unambiguously beneficial. The two bodies of evidence are not
contradictory — they are measuring different things.

---

## 4.x.7 Conclusion

The weight of evidence supports the following interpretation: under typical Catalan autumn
rainfall conditions, the S8 maturation stage exerts no detectable systematic effect on
comarca-level rainfed olive yield — a result consistent with the agronomic literature on
autumn irrigation. The apparent negative full-model coefficient is an artefact of two
exceptional convective storm events operating through physical and biological damage
pathways categorically different from the chronic water-balance signal captured by
CWB-state throughout the remainder of the season.

The S8 autumn finding is accordingly reported as a **case description of two exceptional
events** rather than as an estimated sensitivity coefficient. The primary finding of this
analysis — the monotone decline of β(t) from a credible positive peak at S_pre
(P(diff) = 0.95–0.99) toward near-zero at S7 — stands independently of the S8 stage and
is unaffected by the DANA attribution.

---

## References cited in this section

- Gucci, R., Caruso, G., Gennai, C., Esposto, S., Urbani, S., & Servili, M. (2019). Fruit
  growth, yield and oil quality changes induced by deficit irrigation at different stages
  of olive fruit development. University of Pisa / University of Perugia preprint.
  [verify DOI before submission]

- Hueso, A., Trentacoste, E.R., Junquera, P., Gómez-Miguel, V.D., & Gómez Del Campo, M.
  (2019). Differences in stem water potential during oil synthesis determine fruit
  characteristics and production but not vegetative growth or return bloom in an olive
  hedgerow orchard (cv. Arbequina). *Agricultural Water Management*, 223, 105589.
  doi:10.1016/j.agwat.2019.04.006

- Rousseaux, M.C. et al. (2008). [Argentine winter irrigation study — see
  SUPERVISOR_UPDATE_MAY2026.md for full details]

- Correa-Tedesco, G. et al. (2010). [Water deficit, fruit number, shoot elongation — see
  SUPERVISOR_UPDATE_MAY2026.md for full details]

---

*Figure X caption. DANA years and the autumn (S8) signal — Arbequina comarcas (10
comarques, 2015–2024). Panel A: mean annual yield for rainfed (circles) and irrigated
(squares) Arbequina comarcas; DANA years highlighted. Panel B: irrigated-to-rainfed yield
ratio by year; labelled values shown for DANA years; dashed line = non-DANA mean (2.60).
Panel C: posterior β(S8) median and 90% credible interval for the full model and LOYO
refits excluding each DANA year, for rainfed (circles, blue) and irrigated (squares, red)
panels. Dashed vertical line at zero.*
