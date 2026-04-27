"""
temperature_decomp_arbequina.py
FE-OLS decomposition on the Arbequina-restricted olive panel:
  yield ~ FE_c + {summer_Tmin, summer_Tmax, summer_precip}

Tests whether mean Tmin (best univariate predictor on this panel) carries
information independent of mean Tmax and of summer precipitation, fitting
all 7 univariate / bivariate / trivariate specifications at the headline
windows.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent.parent / "data"
CLIMATE = DATA / "agera5_daily_catalonia.csv"
YIELD = DATA / "catalan_woody_yield_raw.csv"
WHITELIST = DATA / "dun" / "comarca_arbequina_whitelist.csv"

# Headline windows (all 21d for temperature, 30d for precip; same end as before)
SPECS = [
    ("summer_tmin",   "tmin_mean",   "08-17", 21, "mean"),
    ("summer_tmax",   "tmax_mean",   "08-17", 21, "mean"),
    ("summer_precip", "precip_mean", "07-19", 30, "sum"),
]


def load_inputs():
    wl = pd.read_csv(WHITELIST)["comarca"].str.strip().unique().tolist()
    y = pd.read_csv(YIELD)
    y = y.loc[(y["pheno_key"] == "olive") & y["comarca"].isin(wl),
              ["year", "comarca", "yield_tha"]].copy()
    cols = ["date", "comarca", "tmin_mean", "tmax_mean", "precip_mean"]
    c = pd.read_csv(CLIMATE, usecols=cols, parse_dates=["date"])
    c = c.loc[c["comarca"].isin(wl)].copy()
    c["doy"] = c["date"].dt.dayofyear
    c["year"] = c["date"].dt.year
    return y, c


def window_feature(c, var, end_mmdd, window, agg):
    end_doy = pd.Timestamp(f"2021-{end_mmdd}").dayofyear
    start_doy = end_doy - window + 1
    sub = c[(c["doy"] >= start_doy) & (c["doy"] <= end_doy)]
    if agg == "mean":
        out = sub.groupby(["comarca", "year"], as_index=False)[var].mean()
    else:
        out = sub.groupby(["comarca", "year"], as_index=False)[var].sum()
    return out


def fit(df, predictors):
    sub = df.dropna(subset=["yield_tha"] + predictors).copy()
    dummies = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float)
    y = sub["yield_tha"].values
    X_fe = sm.add_constant(dummies.reset_index(drop=True))
    res_fe = sm.OLS(y, X_fe).fit()
    X = sm.add_constant(pd.concat([sub[predictors].reset_index(drop=True),
                                    dummies.reset_index(drop=True)], axis=1))
    res = sm.OLS(y, X).fit()
    within = 1.0 - res.ssr / res_fe.ssr if res_fe.ssr > 0 else np.nan
    return {
        "n": len(sub),
        "r2": float(res.rsquared),
        "within_r2": float(within),
        "params": {p: (float(res.params[p]), float(res.bse[p]),
                       float(res.tvalues[p]), float(res.pvalues[p]))
                   for p in predictors},
    }


def main():
    y, c = load_inputs()
    feats = []
    for name, var, end, w, agg in SPECS:
        f = window_feature(c, var, end, w, agg).rename(columns={var: name})
        feats.append(f)
    df = y
    for f in feats:
        df = df.merge(f, on=["comarca", "year"], how="inner")

    pred_names = [s[0] for s in SPECS]
    print(f"Panel: {len(df)} obs, {df['comarca'].nunique()} comarques, "
          f"years {df['year'].min()}-{df['year'].max()}")
    for name, var, end, w, agg in SPECS:
        print(f"  {name:14s} = {var} {agg} over {w}d ending {end}")
    print()
    print("Predictor descriptive statistics:")
    print(df[pred_names].describe().round(2).to_string())
    print()
    print("Predictor correlation (raw):")
    print(df[pred_names].corr().round(3).to_string())
    centred = df.copy()
    for col in pred_names:
        centred[col] = centred[col] - centred.groupby("comarca")[col].transform("mean")
    print("\nWithin-comarca correlation (after subtracting comarca means):")
    print(centred[pred_names].corr().round(3).to_string())

    specs = []
    for k in (1, 2, 3):
        for combo in combinations(pred_names, k):
            specs.append((", ".join(combo), list(combo)))

    print("\n" + "=" * 78)
    for name, preds in specs:
        r = fit(df, preds)
        print(f"\n=== {name} ===")
        print(f"n={r['n']}  within_R2={r['within_r2']:.4f}  total_R2={r['r2']:.4f}")
        for p, (coef, se, t, pv) in r["params"].items():
            print(f"  {p:14s}  beta={coef:+.5f}  SE={se:.5f}  t={t:+.2f}  p={pv:.3e}")


if __name__ == "__main__":
    main()
