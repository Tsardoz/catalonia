# Olive yield timing analysis in Catalonia

## Slide 1 — Question

- Goal: identify **when** olive yield is most sensitive to climate conditions.
- Focus shifted from broad annual bins to **timing scans on hybrid daily climate**.
- Current question: which metric gives the clearest **exploratory timing signal**: **heat timing**, **dry-air timing**, or **water-balance context**?

---

## Slide 2 — Why the exposure method changed

- Centroid-only climate is a coarse exposure definition for olive.
- Olive groves are not necessarily located near comarca centroids, especially in topographically varied areas.
- Current daily climate is **hybrid**:
  - area-weighted farm-based climate where olive-grove coverage exists
  - centroid fallback elsewhere
- Daily olive panel available for timing work:
  - **36 olive comarques**, **131,508 daily rows**
- The analysis dataset had to be **constructed by linking separate sources**:
  - administrative olive yield (Gencat - Generalitat de Catalunya)
  - external daily weather (agERA5)
  - farm/location matching before aggregation (CORINE Land Cover)

![Weighted vs centroid comparison](figures/climate_comparison_weighted_vs_centroid.png)

**Note:** The numerical differences were not huge, which is reassuring, but the hybrid exposure is still more defensible spatially.

![Matching farms to weather](figures/olive_agera5_matching_topo.png)

---

## Slide 3 — Current analysis setup

- Daily source: `data/agera5_daily_hybrid_olive.csv`
- Yield table: `data/catalan_olive_yield_climate_hybrid.csv`
- Unit of analysis: **comarca-year**, with **comarca fixed effects**
- Each OLS uses **one climate-window predictor at a time**, plus comarca fixed effects
- Current screening variables:
  - `Tmax >= 32°C` hot-day counts
  - `VPD >= 3.0 kPa` high-demand day counts
  - `P - ET₀` as climatic water-balance context
- Timing scan: **14-day rolling windows** with weekly anchor dates

**Note:** This is an exploratory screening framework built on a relatively hard-won panel, not a mechanistic water-stress model.

---

## Slide 4 — Main summary figure

![Olive timing summary](figures/olive_timing_summary.png)

### Main takeaways

- The clearest current pattern is a **late July to late August** candidate sensitive period.
- `Tmax` and `VPD` point to a similar timing pattern.
- `P - ET₀` is more useful as **winter/background context** than as a clean summer timing variable.

---

## Slide 5 — Strongest current pattern: heat timing

![Tmax timing scan](figures/olive_tmax_timing_scan.png)

- Most informative simple screening variable: **14-day count of days with `Tmax >= 32°C`**
- The strongest negative associations cluster in **late July through late August**
- The lowest single-window coefficient occurs for the fortnight ending around **21 August**

**Speaker note:** This is the cleanest current screening answer to the timing question, not a final causal statement.

---

## Slide 6 — VPD broadly agrees on timing, but is not cleaner

![VPD timing scan](figures/olive_vpd_timing_scan.png)

- `VPD >= 3.0 kPa` produces a **similar late-summer timing peak**
- Early July windows are positive in both scans
- Those early-July points represent the **preceding fortnight** and likely overlap the **onset of pit hardening**
- For olive, `Tmax` remains the **simpler and cleaner practical screening variable**

**Speaker note:** I am not claiming VPD is irrelevant; only that it does not improve the olive timing story in the current data.

---

## Slide 7 — Window-length sensitivity check

![Tmax window comparison](figures/olive_tmax_window_compare.png)

- Compared **7-day**, **14-day**, and **21-day** hot-day windows
- All three point to the same broad candidate sensitive period:
  - **late July to late August**
- Longer windows mechanically broaden and lag the apparent peak because they are backward-looking
- Best compromise for presentation: **14-day window**

---

## Slide 8 — Water balance looks more like context than a clean timing driver

![CWB timing scan](figures/olive_cwb_timing_scan.png)

- Full agronomic-year scan of **`P - ET₀`** (`Dec 1 → Nov 30`)
- Shows a plausible **positive winter signal** around late December / early January
- Summer windows are mixed-sign and harder to interpret cleanly

**Working interpretation:**
- `Tmax` captures the clearest main summer timing signal
- `P - ET₀` may capture **background recharge / water-balance context**

---

## Slide 9 — Working interpretation

- The strongest current olive signal is **timing**, not annual-bin model fit.
- The main candidate sensitive period appears to be **late July to late August**.
- `Tmax >= 32°C` is the cleanest current screening metric.
- `VPD >= 3.0 kPa` supports the same broad timing window.
- `P - ET₀` is more useful as a **background-context** variable than as the primary summer signal.

---

## Slide 10 — Caution and next steps

### Caution

- These are still **exploratory fixed-effects screening models** at comarca-year scale.
- They should not yet be described as a full mechanistic stress model for deep-rooted olives.
- Raw p-values exist, but because the scan evaluates **many correlated windows**, they should be treated as **screening statistics**, not strict confirmatory significance tests.

### Next steps

- Look at other measures such as Tmin and standard deviations as well as min, max, mean
- Investigate multivariate regression methods to combine different parameters T, RH and P
- Try the GDD thermal index method to synchronise phases across farms

**Closing line:**

> The current evidence suggests a **late-July to late-August candidate sensitive period** for olive, with `Tmax` giving the clearest screening signal in the present data.
