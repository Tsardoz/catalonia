#!/usr/bin/env python3
"""
03_compute_water_deficit.py — Build WD, VPD, Tmin, and Tmax predictor matrices on GDD axis.

Primary predictor: WD(t) = P(t) - ET0(t) on the GDD axis.
Secondary predictor: VPD curve on GDD axis.
Additional predictor: Tmin curve on GDD axis.
Additional predictor: Tmax curve on GDD axis.
Comparison predictor: cumulative WD over a trailing 30-GDD window.

Output: numpy arrays of shape (n_obs, n_gdd_grid) saved to data/processed/.
"""
import logging
import sys
import argparse

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, DEFAULT_COHORT, WD_ROLLING_WINDOW_GDD, WD_STATE_TAU_GDD,
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

def leaky_integrator_state(
    matrix: np.ndarray, dt: float, tau_gdd: float
) -> tuple[np.ndarray, float]:
    """Compute first-order leaky-integrator state proxy along the GDD axis."""
    if tau_gdd <= 0:
        raise ValueError(f"tau_gdd must be > 0, got {tau_gdd}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    alpha = 1.0 - np.exp(-dt / tau_gdd)
    state = np.empty_like(matrix, dtype=float)
    state[:, 0] = matrix[:, 0]
    for j in range(1, matrix.shape[1]):
        state[:, j] = alpha * matrix[:, j] + (1.0 - alpha) * state[:, j - 1]
    return state, float(alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau-gdd",
        type=float,
        default=WD_STATE_TAU_GDD,
        help=f"Leaky-integrator tau in GDD (default {WD_STATE_TAU_GDD})",
    )
    args = parser.parse_args()
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
    dt = float(np.median(np.diff(gdd_grid)))
    WD_state, alpha = leaky_integrator_state(WD, dt=dt, tau_gdd=args.tau_gdd)
    log.info(
        f"WD_state matrix (tau={args.tau_gdd} GDD, alpha={alpha:.3f}, dt={dt:.1f}): "
        f"shape {WD_state.shape}, mean={WD_state.mean():.4f}, "
        f"range=[{WD_state.min():.3f}, {WD_state.max():.3f}]"
    )

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
    # Additional: Tmin
    TMIN = extract_matrix(aligned, "tmin_mean", gdd_grid)
    log.info(f"TMIN matrix: shape {TMIN.shape}, "
             f"mean={TMIN.mean():.4f}")
    # Additional: Tmax
    TMAX = extract_matrix(aligned, "tmax_mean", gdd_grid)
    log.info(f"TMAX matrix: shape {TMAX.shape}, "
             f"mean={TMAX.mean():.4f}")

    # Save matrices and observation index
    np.save(PROCESSED / "WD_matrix.npy", WD)
    np.save(PROCESSED / "WD_state_matrix.npy", WD_state)
    np.save(PROCESSED / "WD30_matrix.npy", WD30)
    np.save(PROCESSED / "VPD_matrix.npy", VPD)
    np.save(PROCESSED / "TMIN_matrix.npy", TMIN)
    np.save(PROCESSED / "TMAX_matrix.npy", TMAX)

    # Save the observation index (cohort, comarca, year) for later joining
    obs_index = aligned[key_cols].copy()
    obs_index.to_csv(PROCESSED / "obs_index.csv", index=False)
    log.info(
        "Saved WD_matrix.npy, WD_state_matrix.npy, WD30_matrix.npy, "
        "VPD_matrix.npy, TMIN_matrix.npy, TMAX_matrix.npy, obs_index.csv"
    )


if __name__ == "__main__":
    main()
