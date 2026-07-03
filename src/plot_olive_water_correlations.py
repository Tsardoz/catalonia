"""Olive rainfed yield vs. seasonal water variables — clean focused view.

Restricted to water-related predictors only:
  - sum_precip   total precipitation in window (mm)
  - sum_cwb      total climatic water balance, P - ET0 (mm)
  - sum_et0      total ET0 in window (mm)
  - ratio_p_et0  precipitation / ET0

Cohort: Arbequina rainfed whitelist (10 comarques).

Two passes are shown side-by-side:

  RAW anomaly view:   y_anom_raw = yield_tha - comarca_mean
  LAG-ADJUSTED view:  y_anom_lag = residual of (yield_tha) after regressing on
                      lag-1 yield within each comarca.  Removes most of the
                      alternate-bearing oscillation so what is left is the
                      year-to-year weather signal.

x-axis is the within-comarca predictor anomaly in both passes.
DANA years (2018, 2024) are dropped.

A single r is shown per panel and it is the Pearson r of the points actually
plotted. The line is the OLS fit to those same points.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DAILY_PATH = ROOT / "data" / "agera5_daily_catalonia.csv"
YIELD_PATH = ROOT / "data" / "catalan_woody_yield_raw.csv"
WHITELIST_DIR = ROOT / "data" / "dun"
OUT_FIG = ROOT / "figures" / "olive_water_correlations.png"

DANA_YEARS = {2018, 2024}

WINDOWS = {
    "agro_year":        ((10, 1, -1), (9, 30, 0)),
    "winter_recharge":  ((11, 1, -1), (3, 31, 0)),
    "spring":           ((3, 1, 0),   (5, 31, 0)),
    "bloom":            ((3, 13, 0),  (5, 18, 0)),
    "fruit_set":        ((5, 19, 0),  (6, 22, 0)),
    "fruit_dev":        ((6, 23, 0),  (10, 26, 0)),
    "summer":           ((6, 1, 0),   (8, 31, 0)),
    "calendar_year":    ((1, 1, 0),   (12, 31, 0)),
}

PREDICTORS = ["sum_precip", "sum_cwb", "ratio_p_et0", "sum_et0"]
PRED_LABELS = {
    "sum_precip":  "Precipitation (mm)",
    "sum_cwb":     "P − ET₀ (mm)",
    "ratio_p_et0": "P / ET₀",
    "sum_et0":     "ET₀ (mm)",
}


def daterange_for_year(H, win):
    (mo_lo, d_lo, off_lo), (mo_hi, d_hi, off_hi) = win
    return (pd.Timestamp(year=H + off_lo, month=mo_lo, day=d_lo),
            pd.Timestamp(year=H + off_hi, month=mo_hi, day=d_hi))


def aggregate_window(daily, H, win):
    lo, hi = daterange_for_year(H, win)
    sub = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
    g = sub.groupby("comarca", as_index=False).agg(
        sum_precip=("precip_mean", "sum"),
        sum_et0=("et0_mean", "sum"),
    )
    g["sum_cwb"] = g["sum_precip"] - g["sum_et0"]
    g["ratio_p_et0"] = g["sum_precip"] / g["sum_et0"].replace(0, np.nan)
    g["year"] = H
    return g


def main():
    daily = pd.read_csv(
        DAILY_PATH, parse_dates=["date"],
        usecols=["date", "comarca", "precip_mean", "et0_mean"],
    )
    y = pd.read_csv(YIELD_PATH)
    olive = (y[(y["crop_group"] == "olive") & (y["yield_tha"] > 0)]
             [["year", "comarca", "yield_tha"]]
             .sort_values(["comarca", "year"]).reset_index(drop=True))
    arbe = pd.read_csv(WHITELIST_DIR / "comarca_arbequina_whitelist.csv")
    olive = olive[olive["comarca"].isin(arbe["comarca"])].copy()

    olive["lag_yield"] = olive.groupby("comarca")["yield_tha"].shift(1)
    years = sorted(olive["year"].unique())

    agg = {w: pd.concat([aggregate_window(daily, H, win) for H in years],
                       ignore_index=True)
           for w, win in WINDOWS.items()}

    # ------------------------------------------------------------------
    # Figure: rows = windows, cols = (raw anomaly | lag-adjusted) for each
    # predictor.  Keep it manageable: one figure per predictor.
    # ------------------------------------------------------------------
    nrows = len(WINDOWS)
    fig, axes = plt.subplots(nrows, 2 * len(PREDICTORS),
                             figsize=(2.0 * 2 * len(PREDICTORS), 1.7 * nrows),
                             squeeze=False)

    for ri, (wname, _) in enumerate(WINDOWS.items()):
        df = olive.merge(agg[wname], on=["comarca", "year"], how="inner")
        df = df[~df["year"].isin(DANA_YEARS)].copy()

        # within-comarca raw yield anomaly
        df["y_anom_raw"] = df["yield_tha"] - df.groupby("comarca")["yield_tha"].transform("mean")

        # lag-adjusted yield anomaly: residual of yield_tha ~ lag_yield, per comarca
        def lag_resid(g):
            m = g["lag_yield"].notna()
            if m.sum() < 3:
                return pd.Series(np.nan, index=g.index)
            b = np.polyfit(g.loc[m, "lag_yield"], g.loc[m, "yield_tha"], 1)
            pred = np.polyval(b, g["lag_yield"])
            return g["yield_tha"] - pred
        df["y_anom_lag"] = (df.groupby("comarca", group_keys=False)
                              .apply(lag_resid, include_groups=False))

        for ci, pcol in enumerate(PREDICTORS):
            df["x_anom"] = df[pcol] - df.groupby("comarca")[pcol].transform("mean")

            for kk, (ycol, ylabel) in enumerate(
                    [("y_anom_raw", "raw yield anom"),
                     ("y_anom_lag", "lag-adj yield anom")]):
                ax = axes[ri, 2 * ci + kk]
                m = df["x_anom"].notna() & df[ycol].notna()
                d = df.loc[m]
                sc = ax.scatter(d["x_anom"], d[ycol], c=d["year"],
                                cmap="viridis", s=16, alpha=0.85)
                if len(d) > 3:
                    r, p = stats.pearsonr(d["x_anom"], d[ycol])
                    slope, intercept, _, _, _ = stats.linregress(d["x_anom"], d[ycol])
                    xs = np.linspace(d["x_anom"].min(), d["x_anom"].max(), 100)
                    ax.plot(xs, intercept + slope * xs, "k--", lw=1)
                    ax.text(0.04, 0.96, f"r={r:.2f}\np={p:.2g}  n={len(d)}",
                            transform=ax.transAxes, va="top", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.25",
                                      fc="white", ec="0.8", alpha=0.85))
                ax.axhline(0, color="0.7", lw=0.5)
                ax.axvline(0, color="0.7", lw=0.5)
                if ri == 0:
                    ax.set_title(f"{PRED_LABELS[pcol]}\n{ylabel}", fontsize=9)
                if ci == 0 and kk == 0:
                    ax.set_ylabel(f"{wname}\n(t/ha)", fontsize=9)
                ax.tick_params(labelsize=7)

    fig.suptitle(
        "Arbequina rainfed olive: yield anomaly vs. seasonal water variables\n"
        "(within-comarca anomalies, DANA 2018 & 2024 dropped, n=10 comarques × 8 yrs)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.4, pad=0.01, label="year")
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=110, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
