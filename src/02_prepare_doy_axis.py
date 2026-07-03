#!/usr/bin/env python3
"""
02_prepare_doy_axis.py — Prepare fixed calendar-DOY climate curves for Step 2.

Builds a common non-leap DOY grid spanning Jan 1 (DOY 1, pre-inflorescence)
through harvest (DOY 327, S8 end). Drops Feb 29 to keep calendar dates aligned
across years, derives T_mean, and writes wide-format daily climate curves for
each cohort-comarca-year needed by the olive panel.
"""
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import DOY_END, DOY_START, DOY_STEP, PROCESSED, DAILY_CLIMATE_CSV, DEFAULT_COHORT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ALIGN_VARS = [
    "precip_mean",
    "et0_mean",
    "vpd_mean",
    "tmax_mean",
    "tmin_mean",
    "tmean_mean",
    "srad_mean",
]


def build_doy_grid() -> np.ndarray:
    """Common non-leap DOY grid used by downstream calendar-axis steps."""
    return np.arange(DOY_START, DOY_END + DOY_STEP, DOY_STEP, dtype=int)


def add_calendar_doy(climate: pd.DataFrame) -> pd.DataFrame:
    """Add a non-leap DOY column and drop Feb 29 so dates align across years."""
    climate = climate.copy()
    month = climate["date"].dt.month
    day = climate["date"].dt.day
    is_feb29 = (month == 2) & (day == 29)
    shift = climate["date"].dt.is_leap_year & (month > 2)
    climate = climate.loc[~is_feb29].copy()
    climate["calendar_doy"] = (
        climate["date"].dt.dayofyear.astype(int) - shift.loc[~is_feb29].astype(int)
    )
    return climate


def align_group(group: pd.DataFrame, doy_grid: np.ndarray) -> dict | None:
    """Extract one wide-format DOY-aligned row for a single comarca-year."""
    group = group.sort_values("calendar_doy").drop_duplicates("calendar_doy", keep="last")
    have = group["calendar_doy"].to_numpy(dtype=int)
    if not np.array_equal(have, doy_grid):
        missing = sorted(set(doy_grid) - set(have))
        log.warning(
            "Skipping %s %s: expected %d DOYs, found %d (missing first few: %s)",
            group["comarca"].iloc[0],
            int(group["year"].iloc[0]),
            len(doy_grid),
            len(have),
            missing[:8],
        )
        return None

    row: dict[str, float | int | str] = {
        "comarca": group["comarca"].iloc[0],
        "year": int(group["year"].iloc[0]),
    }
    for var in ALIGN_VARS:
        vals = group[var].to_numpy(dtype=float)
        for i, doy in enumerate(doy_grid):
            row[f"{var}_doy{int(doy)}"] = float(vals[i])
    return row


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    doy_grid = build_doy_grid()
    log.info(
        "Calendar DOY grid: %d to %d, %d points, step=%d",
        int(doy_grid[0]), int(doy_grid[-1]), len(doy_grid), int(DOY_STEP)
    )

    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")
    if "cohort" not in yield_df.columns:
        yield_df["cohort"] = DEFAULT_COHORT
    cohort_keys = yield_df[["cohort", "comarca", "year"]].drop_duplicates()
    needed = set(zip(cohort_keys["comarca"], cohort_keys["year"]))

    # Read the canonical climate file directly — never a filtered copy
    climate = pd.read_csv(DAILY_CLIMATE_CSV, parse_dates=["date"])
    if "year" not in climate.columns:
        climate["year"] = climate["date"].dt.year

    # Derive T_mean if not already in the data
    if "tmean_mean" not in climate.columns:
        climate["tmean_mean"] = (climate["tmax_mean"] + climate["tmin_mean"]) / 2.0

    missing_cols = sorted(set(ALIGN_VARS) - set(climate.columns))
    if missing_cols:
        raise ValueError(f"agera5_daily_catalonia.csv missing required columns: {missing_cols}")

    climate = add_calendar_doy(climate)
    climate = climate[climate["calendar_doy"].between(DOY_START, DOY_END)].copy()

    rows = []
    skipped = 0
    for (comarca, year), group in climate.groupby(["comarca", "year"]):
        if (comarca, year) not in needed:
            continue
        row = align_group(group, doy_grid)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    base_aligned = pd.DataFrame(rows)
    if base_aligned.empty:
        raise ValueError("No comarca-years could be aligned to the calendar DOY grid.")

    aligned = cohort_keys.merge(base_aligned, on=["comarca", "year"], how="inner")
    aligned = aligned.sort_values(["cohort", "comarca", "year"]).reset_index(drop=True)

    np.save(PROCESSED / "doy_grid.npy", doy_grid)
    aligned.to_csv(PROCESSED / "climate_doy_aligned.csv", index=False)
    log.info(
        "Saved doy_grid.npy (%d points) and climate_doy_aligned.csv (%d rows; %d skipped)",
        len(doy_grid), len(aligned), skipped,
    )


if __name__ == "__main__":
    main()