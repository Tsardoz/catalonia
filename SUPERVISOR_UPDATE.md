# Olive timing update for supervisors

## Headline

The clearest current result from the new hybrid daily climate analysis is that olive yield appears most sensitive to hot / high-demand atmospheric conditions in **late July to late August**, rather than uniformly across the year.

## What we changed

- Moved from centroid-only climate to a **hybrid daily climate** dataset:
  - area-weighted farm-based climate where olive farm coverage exists
  - centroid fallback elsewhere
- Daily hybrid table now contains **36 olive comarques only** and is the current recommended daily source for olive timing work.

## Main timing result

- A simple timing scan using **14-day rolling counts of days with `Tmax >= 32°C`** gives the cleanest current olive signal.
- The strongest negative windows cluster in **late July through late August**, with the clearest single window ending around **21 August**.
- The same broad result appears for **7-day**, **14-day**, and **21-day** windows, suggesting the timing pattern is stable even though longer windows mechanically broaden the peak.

## Comparison variables

- **`VPD >= 3.0 kPa`** produces a very similar timing pattern, but not a cleaner one than `Tmax`.
- **Longest consecutive runs** of high-VPD days were weaker than **total counts** of high-VPD days.
- Full-year **`P - ET₀`** scans are less clean in summer, but they do show a plausible **positive winter signal** in late December / early January, which may reflect recharge/background context rather than the main summer damage window.

## Current interpretation

- For olive, the most useful current screening result is **timing of hot days**, not annual-bin climate summaries.
- `Tmax` currently looks like the cleanest practical screening variable.
- `P - ET₀` may still be useful, but mainly as **background context**, not as the primary summer timing metric.

## Caution

These are still **exploratory fixed-effects screening models** at comarca-year scale. They are useful for identifying candidate sensitive periods, but they should not yet be described as a full mechanistic stress model for deep-rooted olives.