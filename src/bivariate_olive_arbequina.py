"""
bivariate_olive_arbequina.py
Additive bivariate FE-OLS on the Arbequina-restricted olive panel:
  yield ~ FE_c + β1 * summer_Tmax + β2 * winter_recharge

Summer window (fixed): 21-day mean Tmax ending 17 August (headline spec).
Winter window (fixed): 1 Dec (Y-1) through 31 Mar (Y), agronomic recharge period.

Runs 5 specifications and prints a comparison table:
  1. Univariate: summer Tmax
  2. Univariate: winter precipitation
  3. Univariate: winter (P - ET0)
  4. Bivariate: summer Tmax + winter precipitation
  5. Bivariate: summer Tmax + winter (P - ET0)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent.parent / "data"
CLIMATE = DATA / "agera5_daily_catalonia.csv"
YIELD = DATA / "catalan_woody_yield_raw.csv"
WHITELIST = DATA / "dun" / "comarca_arbequina_whitelist.csv"

SUMMER_END_MMDD = "08-17"
SUMMER_WINDOW = 21
WINTER_START_MMDD = "12-01"   # of prior calendar year (Y-1)
WINTER_END_MMDD   = "03-31"   # of harvest year (Y)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    wl = pd.read_csv(WHITELIST)["comarca"].str.strip().unique().tolist()
    y = pd.read_csv(YIELD)
    y = y.loc[(y["pheno_key"] == "olive") & y["comarca"].isin(wl),
              ["year", "comarca", "yield_tha"]].copy()
    cols = ["date", "comarca", "tmax_mean", "precip_mean", "et0_mean"]
    c = pd.read_csv(CLIMATE, usecols=cols, parse_dates=["date"])
    c = c.loc[c["comarca"].isin(wl)].copy()
    return y, c, wl


def summer_feature(c: pd.DataFrame) -> pd.DataFrame:
    end_doy = pd.Timestamp(f"2021-{SUMMER_END_MMDD}").dayofyear
    start_doy = end_doy - SUMMER_WINDOW + 1
    cy = c.assign(doy=c["date"].dt.dayofyear, year=c["date"].dt.year)
    sub = cy[(cy["doy"] >= start_doy) & (cy["doy"] <= end_doy)]
    out = sub.groupby(["comarca", "year"], as_index=False)["tmax_mean"].mean()
    return out.rename(columns={"tmax_mean": "summer_tmax"})


def winter_features(c: pd.DataFrame) -> pd.DataFrame:
    min_date = c["date"].min()
    max_date = c["date"].max()
    years = sorted(c["date"].dt.year.unique())
    rows = []
    for hy in years:
        start = pd.Timestamp(f"{hy - 1}-{WINTER_START_MMDD}")
        end   = pd.Timestamp(f"{hy}-{WINTER_END_MMDD}")
        if start < min_date or end > max_date:
            continue   # skip incomplete winter (e.g. 2015 lacks Dec-2014)
        sub = c[(c["date"] >= start) & (c["date"] <= end)]
        agg = sub.groupby("comarca").agg(
            winter_precip=("precip_mean", "sum"),
            winter_et0=("et0_mean", "sum"),
        ).reset_index()
        agg["year"] = hy
        agg["winter_pminuse"] = agg["winter_precip"] - agg["winter_et0"]
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)[
        ["comarca", "year", "winter_precip", "winter_et0", "winter_pminuse"]
    ]


def fit(df: pd.DataFrame, predictors: list[str]) -> dict:
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


def main() -> None:
    y, c, wl = load_inputs()
    sumf = summer_feature(c)
    winf = winter_features(c)
    df = (y.merge(sumf, on=["comarca", "year"], how="inner")
            .merge(winf, on=["comarca", "year"], how="inner"))

    print(f"Panel: {len(df)} obs, {df['comarca'].nunique()} comarques, "
          f"years {df['year'].min()}-{df['year'].max()}")
    print(f"Summer feature: mean Tmax, {SUMMER_WINDOW}-day window ending {SUMMER_END_MMDD}")
    print(f"Winter feature: sum over (Y-1)-{WINTER_START_MMDD} -> (Y)-{WINTER_END_MMDD}")
    print()
    print("Predictor descriptive statistics:")
    print(df[["summer_tmax", "winter_precip", "winter_et0", "winter_pminuse"]]
          .describe().round(2).to_string())
    print()
    print("Predictor correlations (Pearson):")
    print(df[["summer_tmax", "winter_precip", "winter_pminuse"]].corr().round(3).to_string())

    specs = [
        ("Univariate: summer Tmax",                 ["summer_tmax"]),
        ("Univariate: winter precip",               ["winter_precip"]),
        ("Univariate: winter (P-ET0)",              ["winter_pminuse"]),
        ("Bivariate:  summer Tmax + winter precip", ["summer_tmax", "winter_precip"]),
        ("Bivariate:  summer Tmax + winter P-ET0",  ["summer_tmax", "winter_pminuse"]),
    ]
    for name, preds in specs:
        r = fit(df, preds)
        print(f"\n=== {name} ===")
        print(f"n={r['n']}  within_R2={r['within_r2']:.4f}  total_R2={r['r2']:.4f}")
        for p, (coef, se, t, pv) in r["params"].items():
            print(f"  {p:18s}  beta={coef:+.5f}  SE={se:.5f}  t={t:+.2f}  p={pv:.3e}")


if __name__ == "__main__":
    main()
