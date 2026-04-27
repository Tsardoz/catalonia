"""
bivariate_summer_arbequina.py
Additive bivariate FE-OLS on the Arbequina-restricted olive panel:
  yield ~ FE_c + beta1 * summer_Tmax + beta2 * summer_precip

Tests whether the negative summer-precipitation signal (peak 30d ending 19-Jul)
is independent of the negative summer-Tmax signal (peak 21d ending 17-Aug),
or whether one absorbs the other.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent.parent / "data"
CLIMATE = DATA / "agera5_daily_catalonia.csv"
YIELD = DATA / "catalan_woody_yield_raw.csv"
WHITELIST = DATA / "dun" / "comarca_arbequina_whitelist.csv"

# Summer heat: 21-day mean Tmax ending 17 August
TMAX_END_MMDD = "08-17"
TMAX_WINDOW = 21
# Summer rain: 30-day precip sum ending 19 July (strongest univariate)
PRECIP_END_MMDD = "07-19"
PRECIP_WINDOW = 30


def load_inputs():
    wl = pd.read_csv(WHITELIST)["comarca"].str.strip().unique().tolist()
    y = pd.read_csv(YIELD)
    y = y.loc[(y["pheno_key"] == "olive") & y["comarca"].isin(wl),
              ["year", "comarca", "yield_tha"]].copy()
    cols = ["date", "comarca", "tmax_mean", "precip_mean"]
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
    tf = window_feature(c, "tmax_mean", TMAX_END_MMDD, TMAX_WINDOW, "mean") \
            .rename(columns={"tmax_mean": "summer_tmax"})
    pf = window_feature(c, "precip_mean", PRECIP_END_MMDD, PRECIP_WINDOW, "sum") \
            .rename(columns={"precip_mean": "summer_precip"})
    df = (y.merge(tf, on=["comarca", "year"], how="inner")
            .merge(pf, on=["comarca", "year"], how="inner"))

    print(f"Panel: {len(df)} obs, {df['comarca'].nunique()} comarques, "
          f"years {df['year'].min()}-{df['year'].max()}")
    print(f"Tmax window: {TMAX_WINDOW}d mean ending {TMAX_END_MMDD}")
    print(f"Precip window: {PRECIP_WINDOW}d sum ending {PRECIP_END_MMDD}")
    print()
    print("Predictor descriptive statistics:")
    print(df[["summer_tmax", "summer_precip"]].describe().round(2).to_string())
    print()
    print("Predictor correlation:")
    print(df[["summer_tmax", "summer_precip"]].corr().round(3).to_string())

    # Within-comarca correlation: subtract comarca means before correlating
    centred = df.copy()
    for col in ("summer_tmax", "summer_precip"):
        centred[col] = centred[col] - centred.groupby("comarca")[col].transform("mean")
    print("\nWithin-comarca correlation (after subtracting comarca means):")
    print(centred[["summer_tmax", "summer_precip"]].corr().round(3).to_string())

    specs = [
        ("Univariate: summer Tmax",   ["summer_tmax"]),
        ("Univariate: summer precip", ["summer_precip"]),
        ("Bivariate:  Tmax + precip", ["summer_tmax", "summer_precip"]),
    ]
    for name, preds in specs:
        r = fit(df, preds)
        print(f"\n=== {name} ===")
        print(f"n={r['n']}  within_R2={r['within_r2']:.4f}  total_R2={r['r2']:.4f}")
        for p, (coef, se, t, pv) in r["params"].items():
            print(f"  {p:18s}  beta={coef:+.5f}  SE={se:.5f}  t={t:+.2f}  p={pv:.3e}")


if __name__ == "__main__":
    main()
