#!/usr/bin/env python3
"""
03_compute_water_deficit.py — Build predictor matrices on the calendar DOY axis.

Primary predictor: CWB(t) = P(t) - ET0(t) (climatic water balance) on the DOY axis.
  Positive = water surplus, negative = water deficit.
  CWB_state: leaky integrator of CWB (tau in days), accumulates the water balance signal.
Secondary predictor: VPD curve on DOY axis.
Additional predictors: Tmin, Tmax curves on DOY axis.
Comparison predictor: cumulative CWB over a trailing 30-day window.

Output: numpy arrays of shape (n_obs, n_doy_grid) saved to data/processed/.
Note: internal file names use WD/CWB interchangeably; WD_state_matrix = CWB_state_matrix.
"""
import logging
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, DEFAULT_COHORT, WD_ROLLING_WINDOW_DAYS, WD_STATE_TAU_DAYS,
    SOIL_FIELD_CAPACITY_MM, SOIL_KC, SOIL_INIT_FRAC,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def extract_matrix(aligned_df: pd.DataFrame, var_prefix: str,
                   doy_grid: np.ndarray) -> np.ndarray:
    """Extract a (n_obs, n_grid) matrix from wide-format DOY-aligned data."""
    cols = [f"{var_prefix}_doy{int(d)}" for d in doy_grid]
    return aligned_df[cols].values


def align_to_yield(
    aligned: pd.DataFrame, yield_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align DOY-aligned data rows to yield ordering using shared key columns."""
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


def rolling_window_cumulative(matrix: np.ndarray, window_days: int) -> np.ndarray:
    """Compute a trailing cumulative sum over a rolling day window."""
    cum = np.cumsum(matrix, axis=1)
    out = cum.copy()
    out[:, window_days:] = cum[:, window_days:] - cum[:, :-window_days]
    return out


def leaky_integrator_state(
    matrix: np.ndarray, dt: float, tau_days: float
) -> tuple[np.ndarray, float]:
    """Compute first-order leaky-integrator state proxy along the DOY axis."""
    if tau_days <= 0:
        raise ValueError(f"tau_days must be > 0, got {tau_days}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    alpha = 1.0 - np.exp(-dt / tau_days)
    state = np.empty_like(matrix, dtype=float)
    state[:, 0] = matrix[:, 0]
    for j in range(1, matrix.shape[1]):
        state[:, j] = alpha * matrix[:, j] + (1.0 - alpha) * state[:, j - 1]
    return state, float(alpha)


def soil_bucket_state(
    P: np.ndarray, ET0: np.ndarray,
    field_capacity: float, kc: float, init_frac: float,
) -> np.ndarray:
    """Single-layer soil water reservoir along the DOY axis.

    Physical alternative to the leaky integrator: water accumulates from
    precipitation, is capped at field capacity (excess drains below the root
    zone and is lost), and is drawn down by actual ET = min(Kc*ET0, available).
    Storage is floored at zero (wilting). Returns soil storage S(t) in mm,
    bounded [0, field_capacity], shape (n_obs, n_grid).
    """
    if field_capacity <= 0:
        raise ValueError(f"field_capacity must be > 0, got {field_capacity}")
    if not (0.0 <= init_frac <= 1.0):
        raise ValueError(f"init_frac must be in [0, 1], got {init_frac}")

    n_obs, n_grid = P.shape
    S = np.empty((n_obs, n_grid), dtype=float)
    s_prev = np.full(n_obs, init_frac * field_capacity, dtype=float)
    for j in range(n_grid):
        s_tmp = s_prev + P[:, j]                          # add rain
        s_tmp -= np.maximum(s_tmp - field_capacity, 0.0)  # drain excess > FC
        aet = np.minimum(kc * ET0[:, j], s_tmp)           # ET limited by supply
        s_cur = np.maximum(s_tmp - aet, 0.0)              # floor at wilting
        S[:, j] = s_cur
        s_prev = s_cur
    return S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau-days",
        type=float,
        default=WD_STATE_TAU_DAYS,
        help=f"Leaky-integrator tau in days (default {WD_STATE_TAU_DAYS})",
    )
    parser.add_argument(
        "--field-capacity",
        type=float,
        default=SOIL_FIELD_CAPACITY_MM,
        help=f"Soil-bucket field capacity in mm (default {SOIL_FIELD_CAPACITY_MM})",
    )
    parser.add_argument(
        "--kc",
        type=float,
        default=SOIL_KC,
        help=f"Soil-bucket crop coefficient for ET drawdown (default {SOIL_KC})",
    )
    parser.add_argument(
        "--init-frac",
        type=float,
        default=SOIL_INIT_FRAC,
        help=f"Initial soil storage as fraction of field capacity (default {SOIL_INIT_FRAC})",
    )
    args = parser.parse_args()
    doy_grid = np.load(PROCESSED / "doy_grid.npy")
    aligned = pd.read_csv(PROCESSED / "climate_doy_aligned.csv")
    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")

    # Ensure aligned data matches yield data ordering
    aligned, yield_df, key_cols = align_to_yield(aligned, yield_df)
    assert len(aligned) == len(yield_df), (
        f"Mismatch: {len(aligned)} aligned vs {len(yield_df)} yield rows"
    )

    n_obs = len(aligned)
    n_grid = len(doy_grid)
    log.info(f"Building predictor matrices: {n_obs} obs × {n_grid} grid points")

    # Primary: Climatic water balance CWB = P - ET0 (positive = surplus)
    P = extract_matrix(aligned, "precip_mean", doy_grid)
    ET0 = extract_matrix(aligned, "et0_mean", doy_grid)
    WD = P - ET0  # CWB; internal variable name kept as WD for file compatibility
    log.info(f"CWB matrix: shape {WD.shape}, "
             f"mean={WD.mean():.4f}, range=[{WD.min():.3f}, {WD.max():.3f}]")
    dt = float(np.median(np.diff(doy_grid)))
    WD_state, alpha = leaky_integrator_state(WD, dt=dt, tau_days=args.tau_days)
    log.info(
        f"CWB_state matrix (tau={args.tau_days} days, alpha={alpha:.3f}, dt={dt:.1f}): "
        f"shape {WD_state.shape}, mean={WD_state.mean():.4f}, "
        f"range=[{WD_state.min():.3f}, {WD_state.max():.3f}]"
    )

    # Comparison: rolling cumulative CWB over trailing 30 days
    WD30 = rolling_window_cumulative(WD, window_days=WD_ROLLING_WINDOW_DAYS)
    log.info(
        f"CWB-30d matrix (window={WD_ROLLING_WINDOW_DAYS} days): shape {WD30.shape}, "
        f"mean={WD30.mean():.4f}, range=[{WD30.min():.3f}, {WD30.max():.3f}]"
    )

    # Physical alternative: single-layer soil bucket with drainage at field capacity
    WD_bucket = soil_bucket_state(
        P, ET0,
        field_capacity=args.field_capacity,
        kc=args.kc,
        init_frac=args.init_frac,
    )
    log.info(
        f"Soil-bucket matrix (FC={args.field_capacity} mm, Kc={args.kc}, "
        f"init={args.init_frac}): shape {WD_bucket.shape}, "
        f"mean={WD_bucket.mean():.4f}, range=[{WD_bucket.min():.3f}, {WD_bucket.max():.3f}]"
    )

    # Secondary: VPD
    VPD = extract_matrix(aligned, "vpd_mean", doy_grid)
    log.info(f"VPD matrix: shape {VPD.shape}, "
             f"mean={VPD.mean():.4f}")
    # Additional: Tmin
    TMIN = extract_matrix(aligned, "tmin_mean", doy_grid)
    log.info(f"TMIN matrix: shape {TMIN.shape}, "
             f"mean={TMIN.mean():.4f}")
    # Additional: Tmax
    TMAX = extract_matrix(aligned, "tmax_mean", doy_grid)
    log.info(f"TMAX matrix: shape {TMAX.shape}, "
             f"mean={TMAX.mean():.4f}")

    # Save matrices and observation index
    np.save(PROCESSED / "WD_matrix.npy", WD)
    np.save(PROCESSED / "WD_state_matrix.npy", WD_state)
    np.save(PROCESSED / "WD30_matrix.npy", WD30)
    np.save(PROCESSED / "WD_bucket_matrix.npy", WD_bucket)
    np.save(PROCESSED / "VPD_matrix.npy", VPD)
    np.save(PROCESSED / "TMIN_matrix.npy", TMIN)
    np.save(PROCESSED / "TMAX_matrix.npy", TMAX)

    # Save the observation index (cohort, comarca, year) for later joining
    obs_index = aligned[key_cols].copy()
    obs_index.to_csv(PROCESSED / "obs_index.csv", index=False)
    log.info(
        "Saved WD_matrix.npy, WD_state_matrix.npy, WD30_matrix.npy, "
        "WD_bucket_matrix.npy, VPD_matrix.npy, TMIN_matrix.npy, TMAX_matrix.npy, "
        "obs_index.csv"
    )


if __name__ == "__main__":
    main()
