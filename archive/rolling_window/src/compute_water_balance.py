"""
compute_water_balance.py
Per-(comarca, date) daily and cumulative P - ET0 water balance.

Cumulative balance resets on 1 October Y-1 (start of the Mediterranean
agronomic year for perennial woody crops; cf. CLAUDE.md). Each calendar day
d is assigned an agronomic year:

    agro_year(d) = d.year + 1   if d.month >= 10
                   d.year       otherwise

so that 1 Oct 2023 .. 30 Sep 2024 all share agro_year = 2024 (the harvest
year for olive). The cumulative balance is then a chronic water-supply
indicator usable as a yield predictor at any in-season endpoint date.

Output columns (long format, one row per (comarca, date)):
    comarca, date, year, agro_year, p_minus_et0_daily, cum_p_minus_et0

Used downstream by sliding_window_regression.py via:
    --climate-csv data/agera5_water_balance_catalonia.csv \\
    --var cum_p_minus_et0 --agg mean --window 1
which selects the cumulative balance on each scan endpoint date for each
(comarca, year) row (window = 1 day reduces the rolling mean to that
endpoint's value).

Usage:
    python src/compute_water_balance.py
    python src/compute_water_balance.py --input data/agera5_daily_catalonia_oliveweighted.csv \\
                                        --output data/agera5_water_balance_catalonia_oliveweighted.csv
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DEFAULT_INPUT = DATA / "agera5_daily_catalonia.csv"
DEFAULT_OUTPUT = DATA / "agera5_water_balance_catalonia.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Daily climate CSV (default agera5_daily_catalonia.csv)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path (default mirrors --input filename with _water_balance_)")
    return p.parse_args()


def derive_default_output(input_path: Path) -> Path:
    name = input_path.name
    if "_daily_" in name:
        return input_path.with_name(name.replace("_daily_", "_water_balance_"))
    return input_path.with_name(f"{input_path.stem}_water_balance.csv")


def main() -> None:
    args = parse_args()
    out_path = args.output if args.output is not None else derive_default_output(args.input)

    print(f"Reading {args.input}")
    df = pd.read_csv(args.input, usecols=["date", "comarca", "precip_mean", "et0_mean"],
                     parse_dates=["date"])
    df = df.sort_values(["comarca", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["agro_year"] = df["year"].where(df["date"].dt.month < 10, df["year"] + 1)

    df["p_minus_et0_daily"] = df["precip_mean"] - df["et0_mean"]
    df["cum_p_minus_et0"] = df.groupby(["comarca", "agro_year"])["p_minus_et0_daily"].cumsum()

    out = df[["comarca", "date", "year", "agro_year",
              "p_minus_et0_daily", "cum_p_minus_et0"]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out):,} rows, {out['comarca'].nunique()} comarques, "
          f"agro_years {out['agro_year'].min()}-{out['agro_year'].max()})")

    eoy = (out.loc[(out["date"].dt.month == 9) & (out["date"].dt.day == 30)]
              .groupby("agro_year")["cum_p_minus_et0"]
              .agg(["min", "median", "max"])
              .round(0))
    print("\nEnd-of-agro-year (30 Sep) cumulative P - ET0 (mm) across comarques:")
    print(eoy.to_string())


if __name__ == "__main__":
    main()
