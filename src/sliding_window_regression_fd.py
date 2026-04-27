"""
sliding_window_regression_fd.py
Within-comarca first-difference sliding-window OLS for crop yield vs daily climate.

Companion to sliding_window_regression.py. Replaces the comarca fixed-effects
specification with year-on-year first differences:

    d_log_yield[c, t] = log(y[c, t]) - log(y[c, t-1])
    d_feature[c, t]   = feature[c, t] - feature[c, t-1]
    d_log_yield ~ d_feature

Year-on-year pairs are restricted to consecutive years per comarca; missing
years are skipped, not jumped. R^2 is honest (no FE share inflating it).

Usage:
  python src/sliding_window_regression_fd.py --crop olive --var tmax_mean \\
      --threshold 32 --agg count --window 14 \\
      --scan-start 06-01 --scan-end 09-15 \\
      --comarca-whitelist data/dun/comarca_olive_whitelist.csv \\
      --climate-csv data/agera5_daily_catalonia_oliveweighted.csv --tag oliveweighted
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
    p.add_argument("--no-log", action="store_true", help="Use Delta yield_tha instead of Delta log(yield_tha)")
    p.add_argument("--tag", default=None, help="Optional suffix for output filenames")
    return p.parse_args()


def load_yield(crop: str, whitelist: Path | None) -> pd.DataFrame:
    y = pd.read_csv(YIELD_CSV)
    y = y.loc[y["pheno_key"] == crop, ["year", "comarca", "yield_tha"]].copy()
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
    else:
        feat = sub.groupby(["comarca", "year"], as_index=False)[var].sum().rename(columns={var: "feature"})
    return feat


def fit_one(yield_df: pd.DataFrame, feat: pd.DataFrame, use_log: bool) -> dict:
    df = yield_df.merge(feat, on=["comarca", "year"], how="inner").dropna()
    df = df.sort_values(["comarca", "year"]).reset_index(drop=True)
    df["y_t"] = np.log(df["yield_tha"]) if use_log else df["yield_tha"]
    df["year_prev"] = df.groupby("comarca")["year"].shift(1)
    df["dy"] = df.groupby("comarca")["y_t"].diff()
    df["dx"] = df.groupby("comarca")["feature"].diff()
    fd = df.loc[(df["year"] - df["year_prev"] == 1)].dropna(subset=["dy", "dx"])
    if len(fd) < 10:
        return {"n": len(fd), "n_comarcas": fd["comarca"].nunique(),
                "coef": np.nan, "stderr": np.nan, "t": np.nan, "p": np.nan, "r2": np.nan}
    X = sm.add_constant(fd[["dx"]].values)
    res = sm.OLS(fd["dy"].values, X).fit()
    return {
        "n": len(fd),
        "n_comarcas": int(fd["comarca"].nunique()),
        "coef": float(res.params[1]),
        "stderr": float(res.bse[1]),
        "t": float(res.tvalues[1]),
        "p": float(res.pvalues[1]),
        "r2": float(res.rsquared),
    }


def main() -> None:
    args = parse_args()
    use_log = not args.no_log
    yield_df = load_yield(args.crop, args.comarca_whitelist)
    if yield_df.empty:
        raise SystemExit(f"No yield rows for crop={args.crop} (whitelist={args.comarca_whitelist})")
    comarcas = sorted(yield_df["comarca"].unique())
    print(f"Crop {args.crop}: {len(yield_df)} obs, {len(comarcas)} comarques, "
          f"years {yield_df['year'].min()}-{yield_df['year'].max()}, "
          f"target = {'Delta log(yield_tha)' if use_log else 'Delta yield_tha'}")

    climate_path = (args.climate_csv if args.climate_csv is not None
                    else (CLIMATE_CSV_ELEVCOR if args.elevation_correct else CLIMATE_CSV))
    print(f"Climate file: {climate_path}")
    climate = load_climate(args.var, comarcas, climate_path)

    ref = pd.Timestamp(2021, 1, 1)
    end0 = (pd.to_datetime(f"2021-{args.scan_start}") - ref).days + 1
    end1 = (pd.to_datetime(f"2021-{args.scan_end}") - ref).days + 1
    end_doys = list(range(end0, end1 + 1, args.anchor_step))

    rows = []
    for end_doy in end_doys:
        feat = aggregate_window(climate, args.var, args.agg, args.threshold, end_doy, args.window)
        res = fit_one(yield_df, feat, use_log)
        res["window_end_doy"] = end_doy
        res["window_end_date"] = (ref + pd.Timedelta(days=end_doy - 1)).strftime("%m-%d")
        rows.append(res)

    out = pd.DataFrame(rows)[["window_end_doy", "window_end_date", "n", "n_comarcas",
                               "coef", "stderr", "t", "p", "r2"]]
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    thr = f"_ge{args.threshold:g}" if args.agg == "count" else ""
    log_tag = "_logy" if use_log else "_rawy"
    stem = f"{args.crop}_{args.var}_{args.agg}{thr}_w{args.window}{log_tag}_fd{tag}"
    csv_path = OUT / f"{stem}.csv"
    fig_path = FIG / f"{stem}.png"
    out.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.errorbar(out["window_end_doy"], out["coef"], yerr=1.96 * out["stderr"],
                fmt="o-", color="#2c3e50", ecolor="#2c3e5055", capsize=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"Window-end day of year ({args.window}-day rolling)")
    ylab_y = "Delta log(yield)" if use_log else "Delta yield_tha"
    ax.set_ylabel(f"OLS coefficient: {ylab_y} per Delta {args.agg}({args.var})")
    ax.set_title(f"{args.crop} FD - {stem}  (n={int(out['n'].max())} pairs, comarques={len(comarcas)})")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")
    print("\nTop 5 most negative coefficients:")
    print(out.sort_values("coef").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
