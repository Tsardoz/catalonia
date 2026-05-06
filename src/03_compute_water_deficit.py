#!/usr/bin/env python3
"""
03_compute_water_deficit.py — Build WD and VPD predictor matrices on GDD axis.

Primary predictor: WD(t) = P(t) - ET0(t) on the GDD axis.
Secondary predictor: VPD curve on GDD axis.
Comparison predictor: cumulative WD over a trailing 30-GDD window.

Output: numpy arrays of shape (n_obs, n_gdd_grid) saved to data/processed/.
"""
import logging
import sys

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, DEFAULT_COHORT, WD_ROLLING_WINDOW_GDD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def extract_matrix(aligned_df: pd.DataFrame, var_prefix: str,
                   gdd_grid: np.ndarray) -> np.ndarray:
    """Extract a (n_obs, n_grid) matrix from wide-format GDD-aligned data."""
    cols = [f"{var_prefix}_gdd{int(g)}" for g in gdd_grid]
    return aligned_df[cols].values


def align_to_yield(
    aligned: pd.DataFrame, yield_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align GDD-aligned data rows to yield ordering using shared key columns."""
    if "cohort" not in aligned.columns:
        aligned["cohort"] = DEFAULT_COHORT
    if "cohort" not in yield_df.columns:
        yield_df["cohort"] = DEFAULT_COHORT

    key_cols = ["cohort", "comarca", "year"]
    keys = yield_df[key_cols].drop_duplicates().copy()
    aligned = keys.merge(aligned, on=key_cols, how="inner")
    matched_keys = aligned[key_cols].drop_duplicates()
    yield_df = matched_keys.merge(yield_df, on=key_cols, how="left")
    aligned = aligned.sort_values(key_cols).reset_index(drop=True)
    yield_df = yield_df.sort_values(key_cols).reset_index(drop=True)
    return aligned, yield_df, key_cols


def rolling_window_cumulative(
    matrix: np.ndarray, gdd_grid: np.ndarray, window_gdd: float
) -> np.ndarray:
    """Compute a trailing cumulative integral over a rolling GDD window."""
    starts = np.maximum(gdd_grid - window_gdd, gdd_grid[0])
    out = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        cum = cumulative_trapezoid(matrix[i], gdd_grid, initial=0.0)
        out[i] = cum - np.interp(starts, gdd_grid, cum)
    return out


def main():
    gdd_grid = np.load(PROCESSED / "gdd_grid.npy")
    aligned = pd.read_csv(PROCESSED / "climate_gdd_aligned.csv")
    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")

    # Ensure aligned data matches yield data ordering
    aligned, yield_df, key_cols = align_to_yield(aligned, yield_df)
    assert len(aligned) == len(yield_df), (
        f"Mismatch: {len(aligned)} aligned vs {len(yield_df)} yield rows"
    )

    n_obs = len(aligned)
    n_grid = len(gdd_grid)
    log.info(f"Building predictor matrices: {n_obs} obs × {n_grid} grid points")

    # Primary: Water deficit WD = P - ET0
    P = extract_matrix(aligned, "precip_mean", gdd_grid)
    ET0 = extract_matrix(aligned, "et0_mean", gdd_grid)
    WD = P - ET0
    log.info(f"WD matrix: shape {WD.shape}, "
             f"mean={WD.mean():.4f}, range=[{WD.min():.3f}, {WD.max():.3f}]")

    # Comparison: rolling cumulative WD over trailing 30 GDD
    WD30 = rolling_window_cumulative(WD, gdd_grid, window_gdd=WD_ROLLING_WINDOW_GDD)
    log.info(
        f"WD30 matrix (window={WD_ROLLING_WINDOW_GDD} GDD): shape {WD30.shape}, "
        f"mean={WD30.mean():.4f}, range=[{WD30.min():.3f}, {WD30.max():.3f}]"
    )

    # Secondary: VPD
    VPD = extract_matrix(aligned, "vpd_mean", gdd_grid)
    log.info(f"VPD matrix: shape {VPD.shape}, "
             f"mean={VPD.mean():.4f}")

    # Save matrices and observation index
    np.save(PROCESSED / "WD_matrix.npy", WD)
    np.save(PROCESSED / "WD30_matrix.npy", WD30)
    np.save(PROCESSED / "VPD_matrix.npy", VPD)

    # Save the observation index (cohort, comarca, year) for later joining
    obs_index = aligned[key_cols].copy()
    obs_index.to_csv(PROCESSED / "obs_index.csv", index=False)

    log.info("Saved WD_matrix.npy, WD30_matrix.npy, VPD_matrix.npy, obs_index.csv")


if __name__ == "__main__":
    main()
