"""
trivariate_arbequina_par.py
Additive trivariate FE-OLS on the Arbequina-restricted olive panel:
  yield ~ FE_c + b1 * summer_Tmin + b2 * summer_precip + b3 * summer_srad

Tests whether the negative summer-radiation signal (peak 30d ending 10-Aug)
is independent of the headline two-channel model (Tmin 21d / 08-17 +
precip 30d / 07-19), or whether srad absorbs / is absorbed by either of
the heat or rain channels. Discriminates the PAR-loss subhypothesis for
the rainfed precip signal:
  * If srad enters POSITIVELY and absorbs part of the precip negative -> PAR-loss.
  * If srad enters NEGATIVELY (alongside Tmin) -> srad is acting as a heat /
    radiation-load indicator, NOT as a PAR resource; the precip channel is
    not PAR-loss-driven.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent.parent / "data"
CLIMATE = DATA / "agera5_daily_catalonia.csv"
YIELD = DATA / "catalan_woody_yield_raw.csv"
WHITELIST = DATA / "dun" / "comarca_arbequina_whitelist.csv"

# Headline summer heat: 21-day mean Tmin ending 17 August
TMIN_END_MMDD = "08-17"
TMIN_WINDOW = 21
# Headline summer rain: 30-day precip sum ending 19 July
PRECIP_END_MMDD = "07-19"
PRECIP_WINDOW = 30
# Strongest univariate srad: 30-day mean ending 10 August
SRAD_END_MMDD = "08-10"
SRAD_WINDOW = 30


def load_inputs():
    wl = pd.read_csv(WHITELIST)["comarca"].str.strip().unique().tolist()
    y = pd.read_csv(YIELD)
    y = y.loc[(y["pheno_key"] == "olive") & y["comarca"].isin(wl),
              ["year", "comarca", "yield_tha"]].copy()
    cols = ["date", "comarca", "tmin_mean", "precip_mean", "srad_mean"]
    c = pd.read_csv(CLIMATE, usecols=cols, parse_dates=["date"])
    c = c.loc[c["comarca"].isin(wl)].copy()
    c["doy"] = c["date"].dt.dayofyear
    c["year"] = c["date"].dt.year
    return y, c


def window_feature(c: pd.DataFrame, var: str, end_mmdd: str, window: int, agg: str) -> pd.DataFrame:
    end_doy = pd.Timestamp(f"2021-{end_mmdd}").dayofyear
    start_doy = end_doy - window + 1
    sub = c[(c["doy"] >= start_doy) & (c["doy"] <= end_doy)]
    if agg == "mean":
        out = sub.groupby(["comarca", "year"], as_index=False)[var].mean()
    elif agg == "sum":
        out = sub.groupby(["comarca", "year"], as_index=False)[var].sum()
    else:
        raise ValueError(f"Unsupported agg: {agg}")
    return out


def fit(df, predictors):
    sub = df.dropna(subset=["yield_tha"] + predictors).copy() if predictors else df.dropna(subset=["yield_tha"]).copy()
    dummies = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float)
    y = sub["yield_tha"].values
    X_fe = sm.add_constant(dummies.reset_index(drop=True))
    res_fe = sm.OLS(y, X_fe).fit()
    if predictors:
        X = sm.add_constant(pd.concat([sub[predictors].reset_index(drop=True),
                                        dummies.reset_index(drop=True)], axis=1))
        res = sm.OLS(y, X).fit()
        within = 1.0 - res.ssr / res_fe.ssr if res_fe.ssr > 0 else np.nan
    else:
        res = res_fe
        within = 0.0
    return {
        "n": len(sub),
        "r2": float(res.rsquared),
        "r2_fe_only": float(res_fe.rsquared),
        "within_r2": float(within),
        "params": {p: (float(res.params[p]), float(res.bse[p]),
                       float(res.tvalues[p]), float(res.pvalues[p]))
                   for p in predictors},
    }


def main():
    y, c = load_inputs()
    tf = window_feature(c, "tmin_mean", TMIN_END_MMDD, TMIN_WINDOW, "mean") \
            .rename(columns={"tmin_mean": "summer_tmin"})
    pf = window_feature(c, "precip_mean", PRECIP_END_MMDD, PRECIP_WINDOW, "sum") \
            .rename(columns={"precip_mean": "summer_precip"})
    sf = window_feature(c, "srad_mean", SRAD_END_MMDD, SRAD_WINDOW, "mean") \
            .rename(columns={"srad_mean": "summer_srad"})
    df = (y.merge(tf, on=["comarca", "year"], how="inner")
            .merge(pf, on=["comarca", "year"], how="inner")
            .merge(sf, on=["comarca", "year"], how="inner"))

    print(f"Panel: {len(df)} obs, {df['comarca'].nunique()} comarques, "
          f"years {df['year'].min()}-{df['year'].max()}")
    print(f"Tmin window:   {TMIN_WINDOW}d mean ending {TMIN_END_MMDD}")
    print(f"Precip window: {PRECIP_WINDOW}d sum ending {PRECIP_END_MMDD}")
    print(f"Srad window:   {SRAD_WINDOW}d mean ending {SRAD_END_MMDD}")
    print()
    print("Predictor descriptive statistics:")
    print(df[["summer_tmin", "summer_precip", "summer_srad"]].describe().round(2).to_string())
    print()
    print("Predictor correlations (Pearson, raw):")
    print(df[["summer_tmin", "summer_precip", "summer_srad"]].corr().round(3).to_string())

    centred = df.copy()
    for col in ("summer_tmin", "summer_precip", "summer_srad"):
        centred[col] = centred[col] - centred.groupby("comarca")[col].transform("mean")
    print("\nWithin-comarca correlations (after subtracting comarca means):")
    print(centred[["summer_tmin", "summer_precip", "summer_srad"]].corr().round(3).to_string())

    specs = [
        ("FE only (comarca dummies)",          []),
        ("Univariate: Tmin",                   ["summer_tmin"]),
        ("Univariate: precip",                 ["summer_precip"]),
        ("Univariate: srad",                   ["summer_srad"]),
        ("Bivariate:  Tmin + precip",          ["summer_tmin", "summer_precip"]),
        ("Bivariate:  Tmin + srad",            ["summer_tmin", "summer_srad"]),
        ("Bivariate:  precip + srad",          ["summer_precip", "summer_srad"]),
        ("Trivariate: Tmin + precip + srad",   ["summer_tmin", "summer_precip", "summer_srad"]),
    ]
    summary = []
    for name, preds in specs:
        r = fit(df, preds)
        print(f"\n=== {name} ===")
        print(f"n={r['n']}  total_R2={r['r2']:.4f}  within_R2={r['within_r2']:.4f}  "
              f"FE-only_R2={r['r2_fe_only']:.4f}")
        for p, (coef, se, t, pv) in r["params"].items():
            print(f"  {p:18s}  beta={coef:+.5f}  SE={se:.5f}  t={t:+.2f}  p={pv:.3e}")
        summary.append((name, r["r2"], r["within_r2"], r["r2_fe_only"]))

    print("\n=== Summary table (R² with vs. without comarca FEs) ===")
    print(f"{'spec':40s}  {'total_R2':>9s}  {'within_R2':>9s}  {'FE-only_R2':>10s}  {'Δ_vs_FE':>8s}")
    fe_baseline = summary[0][3]
    for name, r2, within, r2_fe in summary:
        delta = r2 - fe_baseline
        print(f"{name:40s}  {r2:9.4f}  {within:9.4f}  {r2_fe:10.4f}  {delta:+8.4f}")


if __name__ == "__main__":
    main()
