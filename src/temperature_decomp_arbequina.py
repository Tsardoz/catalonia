"""
temperature_decomp_arbequina.py
FE-OLS decomposition on a (cultivar-restricted) olive panel:
  target ~ FE_c + {summer_Tmin, summer_Tmax, summer_precip}

Tests whether mean Tmin (best univariate predictor on the rainfed Arbequina
panel) carries information independent of mean Tmax and of summer
precipitation, fitting all 7 univariate / bivariate / trivariate specifications
at the headline windows.

Defaults to the rainfed Arbequina panel; override --whitelist / --target-col
to run on the irrigated Arbequina panel (or any other restriction).
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = Path(__file__).parent.parent / "data"
CLIMATE_DEFAULT = DATA / "agera5_daily_catalonia.csv"
YIELD = DATA / "catalan_woody_yield_raw.csv"
WHITELIST_DEFAULT = DATA / "dun" / "comarca_arbequina_whitelist.csv"

# Headline windows (all 21d for temperature, 30d for precip; same end as before)
SPECS = [
    ("summer_tmin",   "tmin_mean",   "08-17", 21, "mean"),
    ("summer_tmax",   "tmax_mean",   "08-17", 21, "mean"),
    ("summer_precip", "precip_mean", "07-19", 30, "sum"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--whitelist", type=Path, default=WHITELIST_DEFAULT,
                   help=f"Comarca whitelist CSV (default {WHITELIST_DEFAULT.name})")
    p.add_argument("--target-col", default="yield_tha",
                   help="Yield column to model (default yield_tha; use yield_irrig_tha for irrigated)")
    p.add_argument("--climate-csv", type=Path, default=CLIMATE_DEFAULT,
                   help=f"Daily climate CSV (default {CLIMATE_DEFAULT.name})")
    p.add_argument("--tag", default="", help="Optional label printed in the header")
    return p.parse_args()


def load_inputs(whitelist: Path, target_col: str, climate_csv: Path):
    wl = pd.read_csv(whitelist)["comarca"].str.strip().unique().tolist()
    y = pd.read_csv(YIELD)
    y = y.loc[(y["pheno_key"] == "olive") & y["comarca"].isin(wl),
              ["year", "comarca", target_col]].copy()
    y = y.dropna(subset=[target_col])
    cols = ["date", "comarca", "tmin_mean", "tmax_mean", "precip_mean"]
    c = pd.read_csv(climate_csv, usecols=cols, parse_dates=["date"])
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


def fit(df, predictors, target_col):
    sub = df.dropna(subset=[target_col] + predictors).copy()
    dummies = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float)
    y = sub[target_col].values
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
    args = parse_args()
    y, c = load_inputs(args.whitelist, args.target_col, args.climate_csv)
    feats = []
    for name, var, end, w, agg in SPECS:
        f = window_feature(c, var, end, w, agg).rename(columns={var: name})
        feats.append(f)
    df = y
    for f in feats:
        df = df.merge(f, on=["comarca", "year"], how="inner")

    pred_names = [s[0] for s in SPECS]
    tag_suffix = f"  [{args.tag}]" if args.tag else ""
    print(f"Panel{tag_suffix}: target={args.target_col}, whitelist={args.whitelist.name}")
    print(f"  {len(df)} obs, {df['comarca'].nunique()} comarques, "
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
        r = fit(df, preds, args.target_col)
        print(f"\n=== {name} ===")
        print(f"n={r['n']}  within_R2={r['within_r2']:.4f}  total_R2={r['r2']:.4f}")
        for p, (coef, se, t, pv) in r["params"].items():
            print(f"  {p:14s}  beta={coef:+.5f}  SE={se:.5f}  t={t:+.2f}  p={pv:.3e}")


if __name__ == "__main__":
    main()
