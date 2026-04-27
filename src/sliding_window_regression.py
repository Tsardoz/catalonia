"""
sliding_window_regression.py
One-predictor sliding-window OLS for crop yield vs daily climate.

For each weekly anchor date in [--scan-start, --scan-end] a rolling window of
length --window ends at that date (inclusive). The within-window climate is
reduced to a single per-(comarca, year) feature using --agg (count of days
above --threshold, or mean/sum of the variable). One OLS is fitted:

    target ~ feature + comarca_FE

where target defaults to yield_tha (rainfed) and can be set with --target-col
(e.g. yield_irrig_tha for irrigated panels). Rows with a NaN target are dropped.

Outputs the coefficient table as CSV and a coefficient-vs-window-end plot.

Usage:
  python src/sliding_window_regression.py --crop olive --var tmax_mean \\
      --threshold 32 --agg count --window 14 \\
      --scan-start 06-01 --scan-end 08-31 \\
      --comarca-whitelist data/dun/comarca_olive_whitelist.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
OUT = DATA / "sliding_window"

YIELD_CSV = DATA / "catalan_woody_yield_raw.csv"
CLIMATE_CSV = DATA / "agera5_daily_catalonia.csv"
CLIMATE_CSV_ELEVCOR = DATA / "agera5_daily_catalonia_elevcor.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop", required=True, help="pheno_key value, e.g. olive, almond, apricot")
    p.add_argument("--var", required=True, help="Daily climate column, e.g. tmax_mean, vpd_mean, precip_mean")
    p.add_argument("--threshold", type=float, default=None, help="Threshold for --agg count")
    p.add_argument("--agg", choices=["count", "mean", "sum"], default="count")
    p.add_argument("--window", type=int, default=14, help="Window length in days (default 14)")
    p.add_argument("--scan-start", default="06-01", help="MM-DD anchor-end start (default 06-01)")
    p.add_argument("--scan-end", default="08-31", help="MM-DD anchor-end stop (default 08-31)")
    p.add_argument("--anchor-step", type=int, default=7, help="Days between anchor ends (default 7)")
    p.add_argument("--comarca-whitelist", type=Path, default=None,
                   help="CSV with a 'comarca' column to restrict the panel")
    p.add_argument("--climate-csv", type=Path, default=None,
                   help="Override daily climate CSV path (default agera5_daily_catalonia.csv)")
    p.add_argument("--elevation-correct", action="store_true",
                   help="Use agera5_daily_catalonia_elevcor.csv (ignored if --climate-csv set)")
    p.add_argument("--target-col", default="yield_tha",
                   help="Yield column to model (default yield_tha; use yield_irrig_tha for irrigated)")
    p.add_argument("--tag", default=None, help="Optional suffix for output filenames")
    return p.parse_args()


def load_yield(crop: str, whitelist: Path | None, target_col: str) -> pd.DataFrame:
    y = pd.read_csv(YIELD_CSV)
    y = y.loc[y["pheno_key"] == crop, ["year", "comarca", target_col]].copy()
    y = y.dropna(subset=[target_col])
    if whitelist is not None:
        wl = pd.read_csv(whitelist)["comarca"].str.strip().unique()
        y = y.loc[y["comarca"].isin(wl)].copy()
    return y.reset_index(drop=True)


def load_climate(var: str, comarcas: list[str], path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, usecols=["date", "comarca", var], parse_dates=["date"])
    c = c.loc[c["comarca"].isin(comarcas)].copy()
    c["year"] = c["date"].dt.year
    c["doy"] = c["date"].dt.dayofyear
    return c


def aggregate_window(c: pd.DataFrame, var: str, agg: str, threshold: float | None,
                     end_doy: int, window: int) -> pd.DataFrame:
    start_doy = end_doy - window + 1
    mask = (c["doy"] >= start_doy) & (c["doy"] <= end_doy)
    sub = c.loc[mask, ["comarca", "year", var]]
    if agg == "count":
        if threshold is None:
            raise ValueError("--threshold is required for --agg count")
        sub = sub.assign(_x=(sub[var] >= threshold).astype(float))
        feat = sub.groupby(["comarca", "year"], as_index=False)["_x"].sum().rename(columns={"_x": "feature"})
    elif agg == "mean":
        feat = sub.groupby(["comarca", "year"], as_index=False)[var].mean().rename(columns={var: "feature"})
    else:  # sum
        feat = sub.groupby(["comarca", "year"], as_index=False)[var].sum().rename(columns={var: "feature"})
    return feat


def fit_one(yield_df: pd.DataFrame, feat: pd.DataFrame, target_col: str) -> dict:
    df = yield_df.merge(feat, on=["comarca", "year"], how="inner").dropna()
    if df["comarca"].nunique() < 2 or len(df) < 10:
        return {"n": len(df), "coef": np.nan, "stderr": np.nan, "t": np.nan, "p": np.nan,
                "r2": np.nan, "within_r2": np.nan}
    dummies = pd.get_dummies(df["comarca"], drop_first=True, dtype=float)
    y = df[target_col].values
    X_fe = sm.add_constant(dummies.reset_index(drop=True))
    res_fe = sm.OLS(y, X_fe).fit()
    X_full = sm.add_constant(pd.concat([df[["feature"]].reset_index(drop=True),
                                         dummies.reset_index(drop=True)], axis=1))
    res = sm.OLS(y, X_full).fit()
    ssr_fe = float(res_fe.ssr)
    ssr_full = float(res.ssr)
    within_r2 = 1.0 - ssr_full / ssr_fe if ssr_fe > 0 else np.nan
    return {
        "n": len(df),
        "coef": float(res.params["feature"]),
        "stderr": float(res.bse["feature"]),
        "t": float(res.tvalues["feature"]),
        "p": float(res.pvalues["feature"]),
        "r2": float(res.rsquared),
        "within_r2": float(within_r2),
    }


def main() -> None:
    args = parse_args()
    yield_df = load_yield(args.crop, args.comarca_whitelist, args.target_col)
    if yield_df.empty:
        raise SystemExit(f"No yield rows for crop={args.crop} target={args.target_col} "
                         f"(whitelist={args.comarca_whitelist})")
    comarcas = sorted(yield_df["comarca"].unique())
    print(f"Crop {args.crop} target={args.target_col}: {len(yield_df)} obs, "
          f"{len(comarcas)} comarques, years {yield_df['year'].min()}-{yield_df['year'].max()}")

    climate_path = (args.climate_csv if args.climate_csv is not None
                    else (CLIMATE_CSV_ELEVCOR if args.elevation_correct else CLIMATE_CSV))
    print(f"Climate file: {climate_path}")
    climate = load_climate(args.var, comarcas, climate_path)

    ref = pd.Timestamp(2021, 1, 1)  # non-leap reference for MM-DD parsing
    end0 = (pd.to_datetime(f"2021-{args.scan_start}") - ref).days + 1
    end1 = (pd.to_datetime(f"2021-{args.scan_end}") - ref).days + 1
    end_doys = list(range(end0, end1 + 1, args.anchor_step))

    rows = []
    for end_doy in end_doys:
        feat = aggregate_window(climate, args.var, args.agg, args.threshold, end_doy, args.window)
        res = fit_one(yield_df, feat, args.target_col)
        res["window_end_doy"] = end_doy
        res["window_end_date"] = (ref + pd.Timedelta(days=end_doy - 1)).strftime("%m-%d")
        rows.append(res)

    out = pd.DataFrame(rows)[["window_end_doy", "window_end_date", "n", "coef", "stderr",
                                "t", "p", "r2", "within_r2"]]
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    thr = f"_ge{args.threshold:g}" if args.agg == "count" else ""
    stem = f"{args.crop}_{args.var}_{args.agg}{thr}_w{args.window}{tag}"
    csv_path = OUT / f"{stem}.csv"
    fig_path = FIG / f"{stem}.png"
    out.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.errorbar(out["window_end_doy"], out["coef"], yerr=1.96 * out["stderr"],
                fmt="o-", color="#c0392b", ecolor="#c0392b55", capsize=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"Window-end day of year ({args.window}-day rolling)")
    ax.set_ylabel(f"OLS coefficient on {args.agg}({args.var})")
    ax.set_title(f"{args.crop} — {stem}  (n={int(out['n'].max())}, comarques={len(comarcas)})")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")
    print("\nTop 5 most negative coefficients:")
    print(out.sort_values("coef").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
