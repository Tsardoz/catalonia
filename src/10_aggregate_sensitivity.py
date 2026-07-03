#!/usr/bin/env python3
"""
10_aggregate_sensitivity.py — Rebuild sensitivity_summary.csv from per-row files.

Reads every results/tables/sens_rows/*.csv (each contains exactly one
parameter combination × window-shift row), deduplicates by key columns,
applies baseline-comparison flags, and writes the aggregated summary.

This is the ONLY script that writes results/tables/sensitivity_summary.csv,
so it must NOT be run concurrently with itself.  It is fast (<1s) and
intended to run once after a batch of 10_sensitivity_analysis.py invocations.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import TABLES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

KEY_COLS = [
    "cohort", "predictor", "tau_days",
    "basis_k", "year_re_mode", "window_shift_days",
]
# Baseline row used as reference for sign-flip and p_diff flags
BASELINE = {
    "predictor": "wd_state",
    "tau_days": 45.0,
    "basis_k": 5,
    "year_re_mode": "with_year",
    "window_shift_days": 0,
}
HEADLINE_CONTRAST = "delta_S7_S6"  # median column used for sign-flip check


def main():
    sens_rows_dir = TABLES / "sens_rows"
    if not sens_rows_dir.exists():
        raise FileNotFoundError(
            f"{sens_rows_dir} not found. Run 10_sensitivity_analysis.py first."
        )

    row_files = sorted(sens_rows_dir.glob("*.csv"))
    if not row_files:
        raise ValueError(f"No per-row CSVs found in {sens_rows_dir}.")

    log.info(f"Reading {len(row_files)} per-row CSVs from {sens_rows_dir}")
    frames = []
    for f in row_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            log.warning(f"Skipping unreadable row file {f}: {exc}")

    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    df = df.drop_duplicates(subset=KEY_COLS, keep="last").reset_index(drop=True)
    if len(df) != n_raw:
        log.info(f"Deduplicated {n_raw} → {len(df)} rows on {KEY_COLS}")

    # Per-cohort baseline-comparison flags
    headline_col = f"{HEADLINE_CONTRAST}_median"
    pdiff_col    = f"{HEADLINE_CONTRAST}_p_diff"
    flagged = []
    for cohort in df["cohort"].unique():
        mask = pd.Series([True] * len(df))
        for col, val in BASELINE.items():
            if col in df.columns:
                mask &= (df[col] == val)
        mask &= (df["cohort"] == cohort)
        baseline = df[mask]
        d = df[df["cohort"] == cohort].copy()
        if baseline.empty or headline_col not in df.columns:
            d["headline_sign_flip"] = False
            d["headline_p_diff_lt_0p7"] = False
        else:
            b_median = baseline.iloc[0][headline_col]
            d["headline_sign_flip"]     = np.sign(d[headline_col]) != np.sign(b_median)
            d["headline_p_diff_lt_0p7"] = d[pdiff_col] < 0.7 if pdiff_col in d.columns else False
        flagged.append(d)
    df = pd.concat(flagged, ignore_index=True)

    out_csv = TABLES / "sensitivity_summary.csv"
    df.sort_values(KEY_COLS).to_csv(out_csv, index=False)
    log.info(f"Wrote {len(df)} rows → {out_csv}")

    # Brief stability report
    flips = int(df["headline_sign_flip"].sum()) if "headline_sign_flip" in df.columns else 0
    weak  = int(df["headline_p_diff_lt_0p7"].sum()) if "headline_p_diff_lt_0p7" in df.columns else 0
    log.info(
        f"Stability ({HEADLINE_CONTRAST}): sign-flip rows = {flips}, "
        f"P(diff)<0.7 rows = {weak} (out of {len(df)})"
    )


if __name__ == "__main__":
    main()
