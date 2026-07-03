"""One scatter: Arbequina olive yield vs. WINTER water balance (S_pre, DOY 1-71).

The May 2026 supervisor update identifies winter recharge (DOY 1-71, ~Jan-mid-Mar)
as the stable climate signal: more water -> higher yield, robust to LOYO. Agro-year
was hiding this by averaging it with noisy summer rain. This plot isolates it.

- Cohort: Arbequina rainfed whitelist (10 comarques)
- Window: S_pre = DOY 1 - 71 (Jan 1 - Mar 12) of the harvest year
- Predictor: sum of daily (P - ET0) over the window
- Years: harvest years 2015-2024 with DANA harvest years (2018, 2024) dropped
- Both axes are within-comarca anomalies
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
WB_PATH = ROOT / "data" / "agera5_water_balance_catalonia.csv"
YIELD_PATH = ROOT / "data" / "catalan_woody_yield_raw.csv"
WHITELIST = ROOT / "data" / "dun" / "comarca_arbequina_whitelist.csv"
OUT_FIG = ROOT / "figures" / "olive_wb_simple.png"

DANA_YEARS = {2018, 2024}

# Winter recharge window: DOY 1-71 of each calendar (harvest) year.
wb = pd.read_csv(WB_PATH, usecols=["comarca", "date", "p_minus_et0_daily"],
                  parse_dates=["date"])
wb["doy"] = wb["date"].dt.dayofyear
wb["year"] = wb["date"].dt.year
winter = wb[(wb["doy"] >= 1) & (wb["doy"] <= 71)]
season = (winter.groupby(["comarca", "year"], as_index=False)["p_minus_et0_daily"]
                .sum().rename(columns={"p_minus_et0_daily": "wb_mm"}))

# olive yield, Arbequina cohort. Need lag_yield BEFORE dropping DANA years.
y = pd.read_csv(YIELD_PATH)
arbe = set(pd.read_csv(WHITELIST)["comarca"])
oli = (y[(y["crop_group"] == "olive") & (y["yield_tha"] > 0) & y["comarca"].isin(arbe)]
       [["year", "comarca", "yield_tha"]]
       .sort_values(["comarca", "year"]).reset_index(drop=True))
oli["lag_yield"] = oli.groupby("comarca")["yield_tha"].shift(1)

df = oli.merge(season, on=["comarca", "year"]).query("year not in @DANA_YEARS").copy()
df = df.dropna(subset=["lag_yield"])

# Lag-yield residualised olive yield: per comarca regress yield ~ lag_yield,
# take the residual. This removes alternate-bearing oscillation.
def lag_resid(g):
    b = np.polyfit(g["lag_yield"], g["yield_tha"], 1)
    return g["yield_tha"] - np.polyval(b, g["lag_yield"])
df["y_resid"] = (df.groupby("comarca", group_keys=False)
                   .apply(lag_resid, include_groups=False))
df["wb_anom"] = df["wb_mm"] - df.groupby("comarca")["wb_mm"].transform("mean")

r, p = stats.pearsonr(df["wb_anom"], df["y_resid"])
slope, intercept, *_ = stats.linregress(df["wb_anom"], df["y_resid"])

fig, ax = plt.subplots(figsize=(7.5, 5.2))
sc = ax.scatter(df["wb_anom"], df["y_resid"], c=df["year"], cmap="viridis", s=40, alpha=0.85)
xs = np.linspace(df["wb_anom"].min(), df["wb_anom"].max(), 100)
ax.plot(xs, intercept + slope * xs, "k--", lw=1)
ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
ax.set_xlabel("Winter (DOY 1−71) water balance anomaly: P − ET₀ (mm)")
ax.set_ylabel("Yield residual after removing alternate bearing (t/ha)")
ax.set_title(
    f"Arbequina rainfed olive: winter water balance vs. yield (alt-bearing removed)\n"
    f"r = {r:.2f}  (n = {len(df)}, p = {p:.2g})  |  DANA harvest years 2018 & 2024 dropped"
)
fig.colorbar(sc, ax=ax, label="year")
fig.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
print(f"Wrote {OUT_FIG}  r={r:.3f}  n={len(df)}")
