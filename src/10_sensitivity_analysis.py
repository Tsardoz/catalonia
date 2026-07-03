#!/usr/bin/env python3
"""
10_sensitivity_analysis.py — Calendar-axis sensitivity sweeps for the Step 7 model.

Sweep dimensions (calendar DOY axis only; GDD axis is retired):
  tau_days   : leaky-integrator memory in days {15, 30, 45, 60, 120}
               (only meaningful for wd_state; VPD runs use baseline tau=45)
  basis_k    : B-spline basis dimension {4, 5, 6, 8}
  predictor  : {wd_state, vpd}
  year_re    : {with_year, no_year}
  window_shift_days : DOY shift applied at contrast-summary time {-10, 0, 10}
               (uses existing posterior — no Step 7 refit needed)

Irrigated yield negative control runs Step 1 --irrigated, rebuilds matrices,
fits arbequina wd_state, then restores rainfed data.

Outputs: per-row CSVs in results/tables/sens_rows/
         run src/10_aggregate_sensitivity.py to produce sensitivity_summary.csv
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, POSTERIORS, TABLES, PLAN_COHORTS, STAGES_RESOLVED,
    WD_STATE_TAU_DAYS, N_BASIS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calendar-axis Step 10 sensitivity sweeps."
    )
    parser.add_argument(
        "--cohorts",
        type=str,
        default="arbequina",
        help=f"Comma list from {{{','.join(PLAN_COHORTS)}}}. Default: arbequina.",
    )
    parser.add_argument(
        "--tau-days",
        type=str,
        default="15,30,45,60,120",
        help="Comma list of WD-state tau values in days (wd_state only).",
    )
    parser.add_argument(
        "--basis-k",
        type=str,
        default="4,5,6,8",
        help="Comma list of basis dimensions for Step 5.",
    )
    parser.add_argument(
        "--predictors",
        type=str,
        default="wd_state,vpd",
        help="Comma list of Step 7 predictors.",
    )
    parser.add_argument(
        "--year-re-modes",
        type=str,
        default="with_year,no_year",
        help="Comma list from {with_year,no_year}.",
    )
    parser.add_argument(
        "--window-shifts",
        type=str,
        default="-10,0,10",
        help="Comma list of DOY window shifts in days (applied at contrast time; no refit).",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument(
        "--chain-method",
        choices=["sequential", "parallel", "vectorized"],
        default="sequential",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--hdi-prob", type=float, default=0.90)
    parser.add_argument(
        "--max-runs", type=int, default=0,
        help="Cap total Step 7 invocations (0 = unlimited).",
    )
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Skip Step 7 fit when expected beta_t artifact already exists.",
    )
    parser.add_argument(
        "--irrigated", action="store_true",
        help="Run irrigated yield negative control (arbequina wd_state only).",
    )
    return parser.parse_args()


def parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def sanitize(v) -> str:
    return str(v).replace(".", "p").replace("-", "m")


def run_cmd(cmd: list[str]):
    log.info("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def stem_name(cohort: str, predictor: str, no_year_re: bool, suffix: str) -> str:
    stem = f"numpyro_main_{cohort}_{predictor}"
    if no_year_re:
        stem += "_noyearre"
    if suffix:
        stem += f"_{suffix}"
    return stem


def row_key(cohort: str, predictor: str, tau_days: float,
           basis_k: int, year_re_mode: str, window_shift_days: int) -> str:
    """Stable filename per (run × window-shift) for atomic per-row CSV writes."""
    return "__".join([
        f"cohort-{cohort}",
        f"pred-{predictor}",
        f"tau{sanitize(tau_days)}d",
        f"k{int(basis_k)}",
        year_re_mode,
        f"shift{sanitize(window_shift_days)}d",
    ])


def write_row_atomic(row: dict, sens_rows_dir: Path) -> None:
    """Write one row atomically via tmp+rename."""
    sens_rows_dir.mkdir(parents=True, exist_ok=True)
    key = row_key(
        cohort=row["cohort"],
        predictor=row["predictor"],
        tau_days=row["tau_days"],
        basis_k=row["basis_k"],
        year_re_mode=row["year_re_mode"],
        window_shift_days=row["window_shift_days"],
    )
    final = sens_rows_dir / f"{key}.csv"
    tmp   = sens_rows_dir / f".{key}.csv.tmp.{os.getpid()}"
    pd.DataFrame([row]).to_csv(tmp, index=False)
    os.replace(tmp, final)


def window_masks_doy(doy_grid: np.ndarray, shift: int) -> dict[str, np.ndarray]:
    """Return per-stage boolean masks on *doy_grid* with a uniform DOY shift."""
    masks = {}
    for stage, (s, e) in STAGES_RESOLVED.items():
        mask = (doy_grid >= s + shift) & (doy_grid <= e + shift)
        if not np.any(mask):
            raise ValueError(
                f"Stage '{stage}' window ({s+shift}–{e+shift}) has no grid points "
                f"after shift={shift} days."
            )
        masks[stage] = mask
    return masks


def summarize_contrasts(
    beta_t: np.ndarray,
    doy_grid: np.ndarray,
    shift: int,
    hdi_prob: float,
) -> dict:
    """Compute all S7-referenced contrasts + P(diff) for one posterior."""
    masks = window_masks_doy(doy_grid, shift)
    stage_means = {
        s: beta_t[:, m].mean(axis=1) for s, m in masks.items()
    }
    lo = 0.5 * (1.0 - hdi_prob)
    hi = 1.0 - lo
    out = {}
    for ref in ["S6", "S5", "S_pre", "S8"]:
        delta = stage_means["S7"] - stage_means[ref]
        p_gt0 = float((delta > 0).mean())
        p_diff = float(max(p_gt0, 1.0 - p_gt0))
        label = f"delta_S7_{ref}"
        out.update({
            f"{label}_median": float(np.median(delta)),
            f"{label}_q_lo":    float(np.quantile(delta, lo)),
            f"{label}_q_hi":    float(np.quantile(delta, hi)),
            f"{label}_prob_gt0": p_gt0,
            f"{label}_p_diff":  p_diff,
        })
    return out


def _max_runs_hit(n_runs: int, max_runs: int) -> bool:
    return max_runs > 0 and n_runs >= max_runs


def main():
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    POSTERIORS.mkdir(parents=True, exist_ok=True)

    cohorts    = parse_list(args.cohorts,      str)
    taus       = parse_list(args.tau_days,     float)
    basis_ks   = parse_list(args.basis_k,      int)
    predictors = parse_list(args.predictors,   str)
    year_modes = parse_list(args.year_re_modes, str)
    shifts     = parse_list(args.window_shifts, int)
    if any(m not in {"with_year", "no_year"} for m in year_modes):
        raise ValueError("--year-re-modes must be from {with_year, no_year}.")
    for c in cohorts:
        if c not in PLAN_COHORTS:
            raise ValueError(f"Unknown cohort '{c}'. Choose from {PLAN_COHORTS}.")

    SRC = Path(__file__).resolve().parent
    step1 = str(SRC / "01_load_data.py")
    step2 = str(SRC / "02_prepare_doy_axis.py")
    step3 = str(SRC / "03_compute_water_deficit.py")
    step5 = str(SRC / "05_build_basis.py")
    step7 = str(SRC / "07_fit_numpyro_main.py")

    doy_grid = np.load(PROCESSED / "doy_grid.npy")
    sens_rows_dir = TABLES / "sens_rows"
    sens_rows_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_runs = 0
    current_tau = None   # track which tau is loaded in the matrices

    for tau in taus:
        for k in basis_ks:
            for predictor in predictors:
                # tau only affects wd_state; skip non-baseline tau for vpd
                effective_tau = tau if predictor == "wd_state" else WD_STATE_TAU_DAYS
                if predictor == "vpd" and tau != WD_STATE_TAU_DAYS:
                    continue

                # Rebuild WD matrices only when tau changes
                if predictor == "wd_state" and effective_tau != current_tau:
                    run_cmd([sys.executable, step3,
                             "--tau-days", str(effective_tau)])
                    current_tau = effective_tau

                run_cmd([sys.executable, step5, "--n-basis", str(k)])

                for mode in year_modes:
                    no_year_re = (mode == "no_year")
                    for cohort in cohorts:
                        suffix = (
                            f"sens_tau{sanitize(effective_tau)}d"
                            f"_k{k}_{mode}"
                        )
                        stem = stem_name(
                            cohort=cohort, predictor=predictor,
                            no_year_re=no_year_re, suffix=suffix,
                        )
                        beta_t_path = POSTERIORS / f"{stem}_beta_t_draws.npy"

                        if not (args.reuse_existing and beta_t_path.exists()):
                            cmd = [
                                sys.executable, step7,
                                "--cohort", cohort,
                                "--predictor", predictor,
                                "--draws", str(args.draws),
                                "--tune", str(args.tune),
                                "--chains", str(args.chains),
                                "--chain-method", args.chain_method,
                                "--seed", str(args.seed),
                                "--target-accept", str(args.target_accept),
                                "--stem-suffix", suffix,
                            ]
                            if no_year_re:
                                cmd.append("--no-year-re")
                            run_cmd(cmd)
                            n_runs += 1

                        beta_t = np.load(beta_t_path)
                        if beta_t.shape[1] != doy_grid.size:
                            log.error(
                                f"Axis mismatch for {stem}: "
                                f"beta_t.shape[1]={beta_t.shape[1]} ≠ "
                                f"doy_grid.size={doy_grid.size}. Skipping."
                            )
                            continue

                        for shift in shifts:
                            contrasts = summarize_contrasts(
                                beta_t=beta_t, doy_grid=doy_grid,
                                shift=shift, hdi_prob=args.hdi_prob,
                            )
                            row = {
                                "cohort": cohort,
                                "predictor": predictor,
                                "tau_days": effective_tau,
                                "basis_k": int(k),
                                "year_re_mode": mode,
                                "window_shift_days": int(shift),
                                "stem": stem,
                                **contrasts,
                            }
                            rows.append(row)
                            write_row_atomic(row, sens_rows_dir)

                        if _max_runs_hit(n_runs, args.max_runs):
                            log.warning(f"Reached max-runs={args.max_runs}; stopping.")
                            break
                    if _max_runs_hit(n_runs, args.max_runs): break
                if _max_runs_hit(n_runs, args.max_runs): break
            if _max_runs_hit(n_runs, args.max_runs): break
        if _max_runs_hit(n_runs, args.max_runs): break

    # ── Irrigated yield negative control ────────────────────────────────────
    if args.irrigated:
        log.info("=== Irrigated yield negative control ===")
        rainfed_yield = PROCESSED / "olive_yield.csv"
        rainfed_means = PROCESSED / "comarca_yield_means.csv"
        backup_yield  = PROCESSED / "olive_yield_rainfed_backup.csv"
        backup_means  = PROCESSED / "comarca_yield_means_rainfed_backup.csv"
        shutil.copy2(rainfed_yield, backup_yield)
        if rainfed_means.exists():
            shutil.copy2(rainfed_means, backup_means)
        try:
            run_cmd([sys.executable, step1, "--irrigated"])
            # Step 2 must follow Step 1: regenerates climate_doy_aligned.csv
            # and obs_index.csv for the new (irrigated) comarca-year set
            run_cmd([sys.executable, step2])
            run_cmd([sys.executable, step3, "--tau-days", str(WD_STATE_TAU_DAYS)])
            run_cmd([sys.executable, step5, "--n-basis",  str(N_BASIS)])
            suffix_irr = f"sens_irrigated_tau{sanitize(WD_STATE_TAU_DAYS)}d_k{N_BASIS}_with_year"
            stem_irr   = stem_name("arbequina", "wd_state",
                                   no_year_re=False, suffix=suffix_irr)
            beta_t_path_irr = POSTERIORS / f"{stem_irr}_beta_t_draws.npy"
            if not (args.reuse_existing and beta_t_path_irr.exists()):
                run_cmd([
                    sys.executable, step7,
                    "--cohort", "arbequina",
                    "--predictor", "wd_state",
                    "--draws", str(args.draws),
                    "--tune", str(args.tune),
                    "--chains", str(args.chains),
                    "--chain-method", args.chain_method,
                    "--seed", str(args.seed),
                    "--target-accept", str(args.target_accept),
                    "--stem-suffix", suffix_irr,
                ])
                n_runs += 1
            beta_t_irr = np.load(beta_t_path_irr)
            for shift in shifts:
                contrasts = summarize_contrasts(
                    beta_t=beta_t_irr, doy_grid=doy_grid,
                    shift=shift, hdi_prob=args.hdi_prob,
                )
                row = {
                    "cohort": "arbequina_irrigated",
                    "predictor": "wd_state",
                    "tau_days": float(WD_STATE_TAU_DAYS),
                    "basis_k": int(N_BASIS),
                    "year_re_mode": "with_year",
                    "window_shift_days": int(shift),
                    "stem": stem_irr,
                    **contrasts,
                }
                rows.append(row)
                write_row_atomic(row, sens_rows_dir)
        finally:
            shutil.copy2(backup_yield, rainfed_yield)
            if backup_means.exists():
                shutil.copy2(backup_means, rainfed_means)
            backup_yield.unlink(missing_ok=True)
            backup_means.unlink(missing_ok=True)
            log.info("Restored rainfed yield data.")
            # Restore all derived files in pipeline order
            run_cmd([sys.executable, step2])
            run_cmd([sys.executable, step3, "--tau-days", str(WD_STATE_TAU_DAYS)])
            run_cmd([sys.executable, step5, "--n-basis",  str(N_BASIS)])

    if not rows:
        log.warning("No sensitivity rows produced in this run.")
        return
    log.info(
        f"Wrote {len(rows)} per-row CSVs to {sens_rows_dir}. "
        "Run src/10_aggregate_sensitivity.py to rebuild sensitivity_summary.csv."
    )


if __name__ == "__main__":
    main()
