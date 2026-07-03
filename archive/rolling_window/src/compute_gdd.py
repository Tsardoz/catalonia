"""
compute_gdd.py
Per-(comarca, date) cumulative growing-degree-days from daily AgERA5 climate.

GDD definition (single-base method, base 10 C, standard for olive; see PLAN.md):
    tmean      = 0.5 * (tmax_mean + tmin_mean)
    gdd_daily  = max(tmean - 10, 0)
    cumgdd     = cumulative sum of gdd_daily within each (comarca, year),
                 reset on 1 January.

Outputs a long-format CSV with one row per (comarca, date):
    comarca, date, year, tmean, gdd_daily, cumgdd

Used downstream by sliding_window_regression.py --anchor gdd, where each
(comarca, year) row uses its OWN cumgdd-vs-date mapping to locate the
window-end date corresponding to a given cumGDD value (per-comarca-year
local anchoring, not panel-median).

Usage:
    python src/compute_gdd.py
    python src/compute_gdd.py --input data/agera5_daily_catalonia_oliveweighted.csv \\
                              --output data/agera5_gdd_catalonia_oliveweighted.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DEFAULT_INPUT = DATA / "agera5_daily_catalonia.csv"
DEFAULT_OUTPUT = DATA / "agera5_gdd_catalonia.csv"

BASE_TEMP = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Daily climate CSV (default agera5_daily_catalonia.csv)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path (default mirrors --input filename with _gdd_ instead of _daily_)")
    p.add_argument("--base", type=float, default=BASE_TEMP,
                   help=f"Base temperature in C (default {BASE_TEMP})")
    return p.parse_args()


def derive_default_output(input_path: Path) -> Path:
    name = input_path.name
    if "_daily_" in name:
        return input_path.with_name(name.replace("_daily_", "_gdd_"))
    return input_path.with_name(f"{input_path.stem}_gdd.csv")


def main() -> None:
    args = parse_args()
    out_path = args.output if args.output is not None else derive_default_output(args.input)

    print(f"Reading {args.input}")
    df = pd.read_csv(args.input, usecols=["date", "comarca", "tmax_mean", "tmin_mean"],
                     parse_dates=["date"])
    df = df.sort_values(["comarca", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year

    df["tmean"] = 0.5 * (df["tmax_mean"] + df["tmin_mean"])
    df["gdd_daily"] = np.maximum(df["tmean"] - args.base, 0.0)
    df["cumgdd"] = (df.groupby(["comarca", "year"])["gdd_daily"].cumsum())

    out = df[["comarca", "date", "year", "tmean", "gdd_daily", "cumgdd"]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out):,} rows, {out['comarca'].nunique()} comarques, "
          f"years {out['year'].min()}-{out['year'].max()}, base={args.base} C)")

    # Quick per-year summary of end-of-year cumGDD across comarques
    eoy = (out.groupby(["comarca", "year"])["cumgdd"].max()
              .reset_index()
              .groupby("year")["cumgdd"]
              .agg(["min", "median", "max"])
              .round(0))
    print("\nEnd-of-year cumGDD across comarques (min / median / max):")
    print(eoy.to_string())


if __name__ == "__main__":
    main()
