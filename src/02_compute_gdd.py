#!/usr/bin/env python3
"""
02_compute_gdd.py — Re-express daily climate on a common GDD axis.

Uses existing GDD data (T_b=10°C) to map each day to cumulative GDD,
then interpolates daily climate variables onto a common GDD grid
(0 to GDD_UPPER in GDD_STEP increments).

Supports configurable T_b via --base flag for sensitivity analysis.
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, GDD_UPPER, GDD_STEP, GDD_GRID_MIN, BASE_TEMP,
    DEFAULT_COHORT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def build_gdd_grid(gdd_upper: float = GDD_UPPER) -> np.ndarray:
    """Common GDD grid: 0 to gdd_upper in GDD_STEP increments."""
    return np.arange(GDD_GRID_MIN, gdd_upper + GDD_STEP, GDD_STEP, dtype=float)


def recompute_gdd(climate: pd.DataFrame, base_temp: float) -> pd.DataFrame:
    """Recompute cumulative GDD from daily T_mean with given base temperature."""
    climate = climate.copy()
    climate["tmean"] = (climate["tmax_mean"] + climate["tmin_mean"]) / 2.0
    climate["gdd_daily"] = np.maximum(climate["tmean"] - base_temp, 0.0)
    climate = climate.sort_values(["comarca", "year", "date"])
    climate["cumgdd"] = climate.groupby(["comarca", "year"])["gdd_daily"].cumsum()
    return climate


def interpolate_to_gdd_grid(
    group: pd.DataFrame, gdd_grid: np.ndarray, variables: list[str]
) -> dict | None:
    """
    Interpolate daily climate variables from calendar-day to GDD axis
    for a single comarca-year. Returns dict with interpolated arrays,
    or None if the comarca-year doesn't reach GDD_UPPER.
    """
    cumgdd = group["cumgdd"].values
    max_gdd = cumgdd[-1]

    if max_gdd < gdd_grid[-1]:
        # This comarca-year doesn't accumulate enough GDD
        return None

    result = {}
    for var in variables:
        vals = group[var].values
        # Linear interpolation from cumulative GDD to common grid
        result[var] = np.interp(gdd_grid, cumgdd, vals)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=float, default=BASE_TEMP,
                        help=f"Base temperature °C (default {BASE_TEMP})")
    parser.add_argument("--gdd-upper", type=float, default=GDD_UPPER,
                        help=f"GDD upper bound (default {GDD_UPPER})")
    args = parser.parse_args()

    gdd_grid = build_gdd_grid(gdd_upper=args.gdd_upper)
    n_grid = len(gdd_grid)
    log.info(f"GDD grid: {gdd_grid[0]:.0f} to {gdd_grid[-1]:.0f}, "
             f"{n_grid} points, step={GDD_STEP}")

    # Load processed yield to know which cohort-comarca-years we need
    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")
    if "cohort" not in yield_df.columns:
        yield_df["cohort"] = DEFAULT_COHORT
    needed = set(zip(yield_df["comarca"], yield_df["year"]))

    # Load daily climate + GDD
    if args.base == BASE_TEMP:
        # Use pre-computed GDD
        climate = pd.read_csv(PROCESSED / "daily_climate.csv", parse_dates=["date"])
        gdd = pd.read_csv(PROCESSED / "gdd.csv", parse_dates=["date"])
        climate = climate.merge(
            gdd[["comarca", "date", "cumgdd"]], on=["comarca", "date"], how="left"
        )
    else:
        # Recompute GDD with different base temperature
        climate = pd.read_csv(PROCESSED / "daily_climate.csv", parse_dates=["date"])
        climate = recompute_gdd(climate, args.base)
        log.info(f"Recomputed GDD with T_b = {args.base}°C")

    # Variables to interpolate onto GDD axis
    interp_vars = [
        "precip_mean", "et0_mean", "vpd_mean", "tmax_mean", "tmin_mean", "srad_mean"
    ]

    rows = []
    skipped = 0
    for (comarca, year), grp in climate.groupby(["comarca", "year"]):
        if (comarca, year) not in needed:
            continue

        grp = grp.sort_values("date").reset_index(drop=True)
        result = interpolate_to_gdd_grid(grp, gdd_grid, interp_vars)

        if result is None:
            log.warning(f"  {comarca} {year}: max GDD={grp['cumgdd'].max():.0f} "
                        f"< {GDD_UPPER}, skipping")
            skipped += 1
            continue

        row = {"comarca": comarca, "year": year}
        for var in interp_vars:
            for i, g in enumerate(gdd_grid):
                row[f"{var}_gdd{int(g)}"] = result[var][i]
        rows.append(row)
    base_aligned = pd.DataFrame(rows)
    if base_aligned.empty:
        raise ValueError("No comarca-year reached GDD upper bound; climate_gdd_aligned is empty.")

    cohort_keys = yield_df[["cohort", "comarca", "year"]].drop_duplicates()
    df_gdd_aligned = cohort_keys.merge(
        base_aligned, on=["comarca", "year"], how="inner"
    )
    df_gdd_aligned = df_gdd_aligned.sort_values(
        ["cohort", "comarca", "year"]
    ).reset_index(drop=True)
    log.info(
        f"GDD-aligned: {len(df_gdd_aligned)} cohort-comarca-years "
        f"({skipped} comarca-years skipped for insufficient GDD)"
    )

    # Save GDD grid metadata and aligned data
    np.save(PROCESSED / "gdd_grid.npy", gdd_grid)
    df_gdd_aligned.to_csv(PROCESSED / "climate_gdd_aligned.csv", index=False)
    log.info(f"Saved gdd_grid.npy ({n_grid} points) and climate_gdd_aligned.csv")


if __name__ == "__main__":
    main()
