# Olive timing update for supervisors

## Headline

The clearest current pattern from the hybrid daily climate analysis is a **late-July to late-August candidate sensitive period** for hot / high-demand atmospheric conditions, rather than a uniform signal across the year.

## What we changed

- Moved from centroid-only climate to a **hybrid daily climate** dataset:
  - area-weighted farm-based climate where olive farm coverage exists
  - centroid fallback elsewhere
- Daily hybrid table now contains **36 olive comarques only** and is the current recommended daily source for olive timing work.
- The analysis dataset had to be **constructed**, not simply downloaded:
  - olive yield from administrative records
  - daily weather from a separate AgERA5 source
  - spatial matching to olive-growing locations / farms before aggregation back to comarca-year

## Main timing result

- A timing scan using **14-day rolling counts of days with `Tmax >= 32°C`** gives the clearest current screening signal.
- The strongest negative associations cluster in **late July through late August**, with the lowest single-window coefficient for the fortnight ending around **21 August**.
- The same broad pattern appears for **7-day**, **14-day**, and **21-day** windows, which suggests the timing signal is reasonably stable even though longer windows mechanically broaden the peak.
- Positive early-July windows are now better interpreted as the **late-June / early-July transition into pit hardening**, which makes them more consistent with established olive RDI timing than the earlier coarse stage labels implied.

## Comparison variables

- **`VPD >= 3.0 kPa`** produces a similar timing pattern, but not a cleaner one than `Tmax`.
- **Longest consecutive runs** of high-VPD days were weaker than **total counts** of high-VPD days.
- Full-year **`P - ET₀`** scans are less clean in summer, but they do show a plausible **positive winter signal** in late December / early January, which may reflect recharge/background context rather than the main summer damage window.

## Current interpretation

- For olive, the most useful current screening signal is **timing of hot days**, not annual-bin climate summaries.
- `Tmax` currently looks like the cleanest practical screening variable.
- `P - ET₀` may still be useful, but mainly as **background context**, not as the primary summer timing metric.

## Caution

These are still **exploratory fixed-effects screening models** at comarca-year scale. They are useful for identifying candidate sensitive periods, but they should not yet be described as a full mechanistic stress model for deep-rooted olives.

Each regression is still simple (**one climate-window regressor + comarca fixed effects**), but the raw p-values should be treated cautiously because the timing scan evaluates **many correlated windows** rather than one pre-specified hypothesis.