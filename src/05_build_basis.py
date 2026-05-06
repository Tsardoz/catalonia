#!/usr/bin/env python3
"""
05_build_basis.py — Construct B-spline basis on the common GDD grid.

Builds cubic B-splines with quantile-placed knots on the GDD grid.
Pre-computes reduced functional predictors = matrix @ B * dt for the model.

Output: B matrix (n_grid × n_basis), *_reduced (n_obs × n_basis).
"""
import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import PROCESSED, N_BASIS, GDD_STEP
from utils.splines import build_bspline_basis

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-basis", type=int, default=N_BASIS,
                        help=f"Number of basis functions (default {N_BASIS})")
    args = parser.parse_args()

    gdd_grid = np.load(PROCESSED / "gdd_grid.npy")
    WD = np.load(PROCESSED / "WD_matrix.npy")
    WD30 = np.load(PROCESSED / "WD30_matrix.npy")
    VPD = np.load(PROCESSED / "VPD_matrix.npy")

    n_grid = len(gdd_grid)
    dt = float(GDD_STEP)  # GDD step size — essential for prior interpretability

    # Build basis
    B = build_bspline_basis(gdd_grid, n_basis=args.n_basis)
    log.info(f"B-spline basis: {B.shape} (n_grid={n_grid}, n_basis={args.n_basis})")
    log.info(f"dt (GDD step) = {dt}")

    # Pre-compute integrated functional predictors
    WD_reduced = WD @ B * dt
    WD30_reduced = WD30 @ B * dt
    VPD_reduced = VPD @ B * dt
    log.info(f"WD_reduced: {WD_reduced.shape}, "
             f"mean={WD_reduced.mean():.4f}")
    log.info(f"WD30_reduced: {WD30_reduced.shape}, "
             f"mean={WD30_reduced.mean():.4f}")
    log.info(f"VPD_reduced: {VPD_reduced.shape}, "
             f"mean={VPD_reduced.mean():.4f}")

    # Save
    np.save(PROCESSED / "B_basis.npy", B)
    np.save(PROCESSED / "WD_reduced.npy", WD_reduced)
    np.save(PROCESSED / "WD30_reduced.npy", WD30_reduced)
    np.save(PROCESSED / "VPD_reduced.npy", VPD_reduced)
    log.info("Saved B_basis.npy, WD_reduced.npy, WD30_reduced.npy, VPD_reduced.npy")


if __name__ == "__main__":
    main()
