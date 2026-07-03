"""Explore olive yield vs. climate correlations across many window × predictor combos.

For each (cohort, window, predictor) compute Pearson r with rainfed yield_tha
(and a partial r controlling for lagged yield, since olive alternate bearing
is dominant). Print a ranked table; plot the top-N as a scatter grid.

Cohorts:
  - all_olive       : every olive×year row (n=354)
  - min500ha        : comarques with mean rainfed area >= 500 ha
  - arbequina       : Arbequina-dominant whitelist (10 comarques)

Windows (calendar DOY ranges; lo/hi inclusive):
  - agro_year       Oct prev -> Sep curr
  - calendar_year   Jan -> Dec
  - winter_recharge Nov prev -> Mar curr   (recharge before bloom)
  - pre_bloom       DOY 1-71    (S_pre)
  - bloom           DOY 72-138  (S5)
  - fruit_set       DOY 139-173 (S6)
  - fruit_dev       DOY 174-299 (S7)
  - maturation      DOY 300-327 (S8)
  - bloom_to_set    DOY 72-173
  - set_to_dev      DOY 139-299
  - spring          Mar-May
  - summer          Jun-Aug

Predictors (per window):
  - sum_precip      sum precip_mean (mm)
  - sum_et0         sum et0_mean (mm)
  - sum_cwb         sum (precip_mean - et0_mean) (mm)
  - ratio_p_et0     sum_precip / sum_et0
  - mean_vpd        mean vpd_mean (kPa)
  - mean_tmax       mean tmax_mean (°C)
  - days_vpd_gt2    count days vpd_mean > 2 kPa
  - days_tmax_gt35  count days tmax_mean > 35 °C
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
OUT_TABLE = ROOT / "figures" / "olive_correlation_ranking.csv"
OUT_FIG = ROOT / "figures" / "olive_top_correlations.png"

DANA_YEARS = {2018, 2024}

# -----------------------------------------------------------------------------
# Windows  (date_lo, date_hi) inclusive, relative to harvest year H.
# A tuple ((month_lo, day_lo, year_offset_lo), (month_hi, day_hi, year_offset_hi))
# year_offset -1 means previous calendar year, 0 means harvest year.
# -----------------------------------------------------------------------------
WINDOWS = {
    "agro_year":        ((10, 1, -1), (9, 30, 0)),
    "calendar_year":    ((1, 1, 0),   (12, 31, 0)),
    "winter_recharge":  ((11, 1, -1), (3, 31, 0)),
    "pre_bloom":        ((1, 1, 0),   (3, 12, 0)),    # DOY 1-71
    "bloom":            ((3, 13, 0),  (5, 18, 0)),    # DOY 72-138
    "fruit_set":        ((5, 19, 0),  (6, 22, 0)),    # DOY 139-173
    "fruit_dev":        ((6, 23, 0),  (10, 26, 0)),   # DOY 174-299
    "maturation":       ((10, 27, 0), (11, 23, 0)),   # DOY 300-327
    "bloom_to_set":     ((3, 13, 0),  (6, 22, 0)),
    "set_to_dev":       ((5, 19, 0),  (10, 26, 0)),
    "spring":           ((3, 1, 0),   (5, 31, 0)),
    "summer":           ((6, 1, 0),   (8, 31, 0)),
    # Carry-over from previous season
    "prev_summer":      ((6, 1, -1),  (8, 31, -1)),
    "prev_fruit_dev":   ((6, 23, -1), (10, 26, -1)),
}


def daterange_for_year(H: int, win):
    (mo_lo, d_lo, off_lo), (mo_hi, d_hi, off_hi) = win
    lo = pd.Timestamp(year=H + off_lo, month=mo_lo, day=d_lo)
    hi = pd.Timestamp(year=H + off_hi, month=mo_hi, day=d_hi)
    return lo, hi


def aggregate_window(daily: pd.DataFrame, H: int, win) -> pd.DataFrame:
    lo, hi = daterange_for_year(H, win)
    sub = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
    g = sub.groupby("comarca", as_index=False).agg(
        sum_precip=("precip_mean", "sum"),
        sum_et0=("et0_mean", "sum"),
        mean_vpd=("vpd_mean", "mean"),
        mean_tmax=("tmax_mean", "mean"),
        days_vpd_gt2=("vpd_mean", lambda s: int((s > 2).sum())),
        days_tmax_gt35=("tmax_mean", lambda s: int((s > 35).sum())),
    )
    g["sum_cwb"] = g["sum_precip"] - g["sum_et0"]
    g["ratio_p_et0"] = g["sum_precip"] / g["sum_et0"].replace(0, np.nan)
    g["year"] = H
    return g


def partial_corr(x, y, z):
    """Partial Pearson r of (x, y) controlling for z."""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 4:
        return np.nan, np.nan, len(x)
    # residualise
    bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
    by = np.polyfit(z, y, 1); ry = y - np.polyval(by, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(x)


def main():
    print("Loading daily climate...")
    daily = pd.read_csv(
        DAILY_PATH,
        parse_dates=["date"],
        usecols=["date", "comarca", "precip_mean", "et0_mean", "vpd_mean", "tmax_mean"],
    )

    print("Loading yields...")
    y = pd.read_csv(YIELD_PATH)
    olive = y[(y["crop_group"] == "olive") & (y["yield_tha"] > 0)].copy()
    olive = olive[["year", "comarca", "yield_tha", "seca_ha"]]
    print(f"olive rows: {len(olive)}  comarques: {olive.comarca.nunique()}")

    arbe = pd.read_csv(WHITELIST_DIR / "comarca_arbequina_whitelist.csv")
    arbe_set = set(arbe["comarca"])

    # cohort: min500ha by mean rainfed area across years
    area_mean = olive.groupby("comarca")["seca_ha"].mean()
    min500 = set(area_mean[area_mean >= 500].index)

    cohorts = {
        "all_olive": set(olive["comarca"].unique()),
        "min500ha":  min500,
        "arbequina": arbe_set,
    }
    for k, v in cohorts.items():
        print(f"  cohort {k}: {len(v)} comarques")

    # Lagged yield for partial correlation
    olive = olive.sort_values(["comarca", "year"])
    olive["lag_yield"] = olive.groupby("comarca")["yield_tha"].shift(1)

    # ------------------------------------------------------------------
    # Build (cohort, window, predictor) ranking
    # ------------------------------------------------------------------
    predictor_cols = [
        "sum_precip", "sum_et0", "sum_cwb", "ratio_p_et0",
        "mean_vpd", "mean_tmax", "days_vpd_gt2", "days_tmax_gt35",
    ]
    years = sorted(olive["year"].unique())

    # Precompute window aggregates per (window, year)
    print("Aggregating windows...")
    agg_cache = {}
    for wname, win in WINDOWS.items():
        chunks = [aggregate_window(daily, H, win) for H in years]
        agg_cache[wname] = pd.concat(chunks, ignore_index=True)

    # ------------------------------------------------------------------
    # Modes:
    #   raw         : Pearson on raw values (cross-comarca + temporal)
    #   demeaned    : within-comarca anomalies (yield and predictor both
    #                 centred by comarca mean) -> temporal weather signal
    #   demeaned_nodana : same as demeaned, but with DANA years dropped
    # ------------------------------------------------------------------
    modes = ["raw", "demeaned", "demeaned_nodana"]
    rows = []
    for cohort_name, comarques in cohorts.items():
        oli_c = olive[olive["comarca"].isin(comarques)].copy()
        for wname in WINDOWS:
            ag = agg_cache[wname]
            df = oli_c.merge(ag, on=["comarca", "year"], how="inner")
            for mode in modes:
                d = df.copy()
                if mode == "demeaned_nodana":
                    d = d[~d["year"].isin(DANA_YEARS)]
                if mode.startswith("demeaned"):
                    # subtract comarca means from yield and every predictor
                    d["yield_tha"] = d["yield_tha"] - d.groupby("comarca")["yield_tha"].transform("mean")
                    for pcol in predictor_cols:
                        d[pcol] = d[pcol] - d.groupby("comarca")[pcol].transform("mean")
                    if "lag_yield" in d.columns:
                        d["lag_yield"] = d["lag_yield"] - d.groupby("comarca")["lag_yield"].transform("mean")
                for pcol in predictor_cols:
                    x = d[pcol].to_numpy(float)
                    yv = d["yield_tha"].to_numpy(float)
                    mask = ~(np.isnan(x) | np.isnan(yv))
                    if mask.sum() < 5 or np.nanstd(x[mask]) == 0 or np.nanstd(yv[mask]) == 0:
                        continue
                    r, p = stats.pearsonr(x[mask], yv[mask])
                    lag = d["lag_yield"].to_numpy(float)
                    pr, pp, n_pr = partial_corr(x, yv, lag)
                    rows.append({
                        "mode": mode,
                        "cohort": cohort_name,
                        "window": wname,
                        "predictor": pcol,
                        "n": int(mask.sum()),
                        "r": r,
                        "p": p,
                        "partial_r_lag": pr,
                        "partial_p_lag": pp,
                        "n_partial": n_pr,
                    })

    rank = pd.DataFrame(rows)
    rank["abs_r"] = rank["r"].abs()
    rank["abs_partial_r"] = rank["partial_r_lag"].abs()
    rank = rank.sort_values("abs_partial_r", ascending=False).reset_index(drop=True)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    rank.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote ranking -> {OUT_TABLE}")

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    cols = ["mode", "cohort", "window", "predictor", "n", "r", "p",
            "partial_r_lag", "partial_p_lag"]
    for mode in modes:
        sub = rank[rank["mode"] == mode].sort_values("abs_r", ascending=False)
        print(f"\n=== Top 15 by |r| — mode = {mode} ===")
        print(sub.head(15).to_string(index=False, columns=cols))

    # ------------------------------------------------------------------
    # Scatter the top-9 from the within-comarca (no-DANA) view —
    # that's the cleanest temporal weather-vs-yield comparison.
    # ------------------------------------------------------------------
    target_mode = "demeaned_nodana"
    top = (rank[rank["mode"] == target_mode]
           .sort_values("abs_r", ascending=False)
           .head(9)
           .reset_index(drop=True))

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    sc = None
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        cohort = cohorts[row["cohort"]]
        oli_c = olive[olive["comarca"].isin(cohort)].copy()
        df = oli_c.merge(agg_cache[row["window"]], on=["comarca", "year"], how="inner")
        df = df[~df["year"].isin(DANA_YEARS)].copy()
        df["y_anom"] = df["yield_tha"] - df.groupby("comarca")["yield_tha"].transform("mean")
        df["x_anom"] = df[row["predictor"]] - df.groupby("comarca")[row["predictor"]].transform("mean")
        sc = ax.scatter(df["x_anom"], df["y_anom"], c=df["year"],
                        cmap="viridis", s=22, alpha=0.8)
        mask = df["x_anom"].notna() & df["y_anom"].notna()
        if mask.sum() > 2:
            slope, intercept, _, _, _ = stats.linregress(df.loc[mask, "x_anom"], df.loc[mask, "y_anom"])
            xs = np.linspace(df["x_anom"].min(), df["x_anom"].max(), 100)
            ax.plot(xs, intercept + slope * xs, "k--", lw=1)
        ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
        ax.set_title(
            f"{row['cohort']} | {row['window']} | {row['predictor']}\n"
            f"r={row['r']:.2f}  partial_r={row['partial_r_lag']:.2f}  n={row['n']}",
            fontsize=9,
        )
        ax.set_xlabel(f"{row['predictor']} anomaly")
        ax.set_ylabel("yield anomaly (t/ha)")
    fig.suptitle(
        "Olive: top 9 climate–yield correlations (within-comarca anomalies, DANA years 2018/2024 dropped)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02, label="year")
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Wrote figure -> {OUT_FIG}")


if __name__ == "__main__":
    main()
