#!/usr/bin/env python3
"""
10_aggregate_sensitivity.py — Rebuild sensitivity_summary.csv from per-row files.

Reads every results/tables/sens_rows/*.csv (each contains exactly one
parameter combination × window-shift row), deduplicates by key columns,
applies baseline-comparison flags, and writes the aggregated summary.

This is the ONLY script that writes results/tables/sensitivity_summary.csv,
so it must NOT be run concurrently with itself.  It is fast (<1s) and
intended to run once at the end of 10_run_sweeps.sh.
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
    "cohort", "predictor", "base_temp", "tau_gdd",
    "basis_k", "year_re_mode", "window_shift_gdd",
]


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
    flagged = []
    for cohort in df["cohort"].unique():
        baseline = df[
            (df["cohort"] == cohort)
            & (df["predictor"] == "wd_state")
            & (df["base_temp"] == 10.0)
            & (df["tau_gdd"] == 45.0)
            & (df["basis_k"] == 5)
            & (df["year_re_mode"] == "with_year")
            & (df["window_shift_gdd"] == 0)
        ]
        d = df[df["cohort"] == cohort].copy()
        if baseline.empty:
            d["delta_sign_flip_vs_baseline"] = False
        else:
            b = baseline.iloc[0]
            d["delta_sign_flip_vs_baseline"] = (
                np.sign(d["delta_median"]) != np.sign(b["delta_median"])
            )
        d["prob_lt_0p7"] = d["prob_delta_gt_0"] < 0.7
        flagged.append(d)
    df = pd.concat(flagged, ignore_index=True)

    out_csv = TABLES / "sensitivity_summary.csv"
    df.sort_values(KEY_COLS).to_csv(out_csv, index=False)
    log.info(f"Wrote {len(df)} rows → {out_csv}")

    # Brief stability report
    flips = df["delta_sign_flip_vs_baseline"].sum()
    weak = df["prob_lt_0p7"].sum()
    log.info(
        f"Stability: sign-flip rows = {flips}, P(Δ>0)<0.7 rows = {weak} "
        f"(out of {len(df)})"
    )


if __name__ == "__main__":
    main()
