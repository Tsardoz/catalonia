#!/usr/bin/env python3
"""
05_build_basis.py — Construct B-spline basis on the calendar DOY grid.

Builds cubic B-splines with quantile-placed knots on the DOY grid.
Pre-computes reduced functional predictors = matrix @ B * dt for the model.

Output: B matrix (n_grid × n_basis), *_reduced (n_obs × n_basis).
"""
import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import PROCESSED, N_BASIS, DOY_STEP
from utils.splines import build_bspline_basis

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-basis", type=int, default=N_BASIS,
                        help=f"Number of basis functions (default {N_BASIS})")
    args = parser.parse_args()

    doy_grid = np.load(PROCESSED / "doy_grid.npy")
    WD = np.load(PROCESSED / "WD_matrix.npy")
    WD_STATE = np.load(PROCESSED / "WD_state_matrix.npy")
    WD30 = np.load(PROCESSED / "WD30_matrix.npy")
    WD_BUCKET = np.load(PROCESSED / "WD_bucket_matrix.npy")
    VPD = np.load(PROCESSED / "VPD_matrix.npy")
    TMIN = np.load(PROCESSED / "TMIN_matrix.npy")
    TMAX = np.load(PROCESSED / "TMAX_matrix.npy")

    n_grid = len(doy_grid)
    dt = float(DOY_STEP)  # day step size — essential for prior interpretability

    # Build basis
    B = build_bspline_basis(doy_grid, n_basis=args.n_basis)
    log.info(f"B-spline basis: {B.shape} (n_grid={n_grid}, n_basis={args.n_basis})")
    log.info(f"dt (DOY step) = {dt}")

    # Pre-compute integrated functional predictors
    WD_reduced = WD @ B * dt
    WD_state_reduced = WD_STATE @ B * dt
    WD30_reduced = WD30 @ B * dt
    WD_bucket_reduced = WD_BUCKET @ B * dt
    VPD_reduced = VPD @ B * dt
    TMIN_reduced = TMIN @ B * dt
    TMAX_reduced = TMAX @ B * dt
    log.info(f"WD_reduced: {WD_reduced.shape}, "
             f"mean={WD_reduced.mean():.4f}")
    log.info(f"WD_state_reduced: {WD_state_reduced.shape}, "
             f"mean={WD_state_reduced.mean():.4f}")
    log.info(f"WD30_reduced: {WD30_reduced.shape}, "
             f"mean={WD30_reduced.mean():.4f}")
    log.info(f"WD_bucket_reduced: {WD_bucket_reduced.shape}, "
             f"mean={WD_bucket_reduced.mean():.4f}")
    log.info(f"VPD_reduced: {VPD_reduced.shape}, "
             f"mean={VPD_reduced.mean():.4f}")
    log.info(f"TMIN_reduced: {TMIN_reduced.shape}, "
             f"mean={TMIN_reduced.mean():.4f}")
    log.info(f"TMAX_reduced: {TMAX_reduced.shape}, "
             f"mean={TMAX_reduced.mean():.4f}")

    # Save
    np.save(PROCESSED / "B_basis.npy", B)
    np.save(PROCESSED / "WD_reduced.npy", WD_reduced)
    np.save(PROCESSED / "WD_state_reduced.npy", WD_state_reduced)
    np.save(PROCESSED / "WD30_reduced.npy", WD30_reduced)
    np.save(PROCESSED / "WD_bucket_reduced.npy", WD_bucket_reduced)
    np.save(PROCESSED / "VPD_reduced.npy", VPD_reduced)
    np.save(PROCESSED / "TMIN_reduced.npy", TMIN_reduced)
    np.save(PROCESSED / "TMAX_reduced.npy", TMAX_reduced)
    log.info(
        "Saved B_basis.npy, WD_reduced.npy, WD_state_reduced.npy, WD30_reduced.npy, "
        "WD_bucket_reduced.npy, VPD_reduced.npy, TMIN_reduced.npy, TMAX_reduced.npy"
    )


if __name__ == "__main__":
    main()
