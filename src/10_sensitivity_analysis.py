#!/usr/bin/env python3
"""
10_sensitivity_analysis.py — Sensitivity sweeps for the Step 7 model.

Runs combinations across preprocessing/model settings, then summarizes
Δ(pit − flower) robustness from posterior beta(t) draws.
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

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, TABLES, GDD_UPPER,
    PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD,
    ALT_BASE_TEMP, ALT_PRE_FLOWERING_GDD, ALT_FLOWERING_GDD,
    ALT_PIT_HARDENING_GDD, ALT_GDD_UPPER,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 10 sensitivity sweeps and summarize contrast stability."
    )
    parser.add_argument(
        "--cohorts",
        type=str,
        default="arbequina,morruda,all_olive",
        help="Comma list from {arbequina,morruda,all_olive,all}.",
    )
    parser.add_argument(
        "--base-temps",
        type=str,
        default="10",
        help="Comma list of T_b values for Step 2 (e.g. 6.5,7,8.5,10,12.5).",
    )
    parser.add_argument(
        "--taus",
        type=str,
        default="15,30,45,60,120",
        help="Comma list of WD-state tau values (GDD).",
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
        default="-50,0,50",
        help="Comma list of stage-window GDD shifts for contrast summaries.",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--hdi-prob", type=float, default=0.90)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum Step 7 runs (0 = unlimited). Useful for quick checks.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip rerunning Step 7 if expected beta_t artifact exists.",
    )
    parser.add_argument(
        "--irrigated",
        action="store_true",
        help="Run irrigated yield negative control after rainfed sweeps.",
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


def row_key(
    cohort: str, predictor: str, base_temp: float, tau_gdd: float,
    basis_k: int, year_re_mode: str, window_shift: int,
) -> str:
    """Stable filename per (run, window-shift) for atomic per-row CSV writes."""
    parts = [
        f"cohort-{cohort}",
        f"pred-{predictor}",
        f"tb{sanitize(base_temp)}",
        f"tau{sanitize(tau_gdd)}",
        f"k{int(basis_k)}",
        year_re_mode,
        f"shift{sanitize(window_shift)}",
    ]
    return "__".join(parts)


def write_row_atomic(row: dict, sens_rows_dir: Path) -> None:
    """Write one row to its own CSV via tmp+rename so partial writes are
    never observed by the aggregator and concurrent writers cannot collide
    on the same file."""
    sens_rows_dir.mkdir(parents=True, exist_ok=True)
    key = row_key(
        cohort=row["cohort"],
        predictor=row["predictor"],
        base_temp=row["base_temp"],
        tau_gdd=row["tau_gdd"],
        basis_k=row["basis_k"],
        year_re_mode=row["year_re_mode"],
        window_shift=row["window_shift_gdd"],
    )
    final = sens_rows_dir / f"{key}.csv"
    tmp = sens_rows_dir / f".{key}.csv.tmp.{os.getpid()}"
    pd.DataFrame([row]).to_csv(tmp, index=False)
    os.replace(tmp, final)


def get_anchors(base_temp: float):
    """Return (pre_flowering, flowering, pit_hardening) GDD anchors for *base_temp*."""
    if base_temp == ALT_BASE_TEMP:
        return ALT_PRE_FLOWERING_GDD, ALT_FLOWERING_GDD, ALT_PIT_HARDENING_GDD
    return PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD


def window_masks(
    gdd_grid: np.ndarray,
    shift: int,
    anchors: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if anchors is None:
        anchors = (PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD)
    pre_a, flo_a, pit_a = anchors
    pre = (pre_a[0] + shift, pre_a[1] + shift)
    flo = (flo_a[0] + shift, flo_a[1] + shift)
    pit = (pit_a[0] + shift, pit_a[1] + shift)
    m_pre = (gdd_grid >= pre[0]) & (gdd_grid <= pre[1])
    m_flo = (gdd_grid >= flo[0]) & (gdd_grid <= flo[1])
    m_pit = (gdd_grid >= pit[0]) & (gdd_grid <= pit[1])
    if not (np.any(m_pre) and np.any(m_flo) and np.any(m_pit)):
        raise ValueError(f"Empty stage mask for shift={shift}.")
    return m_pre, m_flo, m_pit


def summarize_delta(
    beta_t: np.ndarray,
    gdd_grid: np.ndarray,
    shift: int,
    hdi_prob: float,
    anchors: tuple | None = None,
) -> dict:
    m_pre, m_flo, m_pit = window_masks(gdd_grid, shift, anchors=anchors)
    beta_pre = beta_t[:, m_pre].mean(axis=1)
    beta_flo = beta_t[:, m_flo].mean(axis=1)
    beta_pit = beta_t[:, m_pit].mean(axis=1)
    delta = beta_pit - beta_flo
    lo = 0.5 * (1.0 - hdi_prob)
    hi = 1.0 - lo
    return {
        "delta_median": float(np.median(delta)),
        "delta_q_lo": float(np.quantile(delta, lo)),
        "delta_q_hi": float(np.quantile(delta, hi)),
        "prob_delta_gt_0": float((delta > 0).mean()),
    }


def main():
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)

    cohorts = parse_list(args.cohorts, str)
    base_temps = parse_list(args.base_temps, float)
    taus = parse_list(args.taus, float)
    basis_ks = parse_list(args.basis_k, int)
    predictors = parse_list(args.predictors, str)
    year_modes = parse_list(args.year_re_modes, str)
    shifts = parse_list(args.window_shifts, int)
    if any(m not in {"with_year", "no_year"} for m in year_modes):
        raise ValueError("year-re-modes must contain only with_year/no_year.")

    step1 = "/home/tsardoz/phd/catalonia/src/01_load_data.py"
    step2 = "/home/tsardoz/phd/catalonia/src/02_compute_gdd.py"
    step3 = "/home/tsardoz/phd/catalonia/src/03_compute_water_deficit.py"
    step5 = "/home/tsardoz/phd/catalonia/src/05_build_basis.py"
    step7 = "/home/tsardoz/phd/catalonia/src/07_fit_numpyro_main.py"

    rows = []
    n_runs = 0
    sens_rows_dir = TABLES / "sens_rows"
    sens_rows_dir.mkdir(parents=True, exist_ok=True)

    for base in base_temps:
        gdd_upper = ALT_GDD_UPPER if base == ALT_BASE_TEMP else GDD_UPPER
        run_cmd([sys.executable, step2, "--base", str(base),
                 "--gdd-upper", str(gdd_upper)])
        gdd_grid = np.load(PROCESSED / "gdd_grid.npy")
        anchors = get_anchors(base)
        for tau in taus:
            run_cmd([sys.executable, step3, "--tau-gdd", str(tau)])
            for k in basis_ks:
                run_cmd([sys.executable, step5, "--n-basis", str(k)])
                for predictor in predictors:
                    for mode in year_modes:
                        no_year_re = (mode == "no_year")
                        for cohort in cohorts:
                            suffix = (
                                f"sens_tb{sanitize(base)}_tau{sanitize(tau)}"
                                f"_k{k}_{mode}"
                            )
                            stem = stem_name(
                                cohort=cohort,
                                predictor=predictor,
                                no_year_re=no_year_re,
                                suffix=suffix,
                            )
                            beta_t_path = (
                                f"/home/tsardoz/phd/catalonia/results/posteriors/"
                                f"{stem}_beta_t_draws.npy"
                            )
                            if not (args.reuse_existing and __import__("pathlib").Path(beta_t_path).exists()):
                                cmd = [
                                    sys.executable, step7,
                                    "--cohort", cohort,
                                    "--predictor", predictor,
                                    "--draws", str(args.draws),
                                    "--tune", str(args.tune),
                                    "--chains", str(args.chains),
                                    "--seed", str(args.seed),
                                    "--target-accept", str(args.target_accept),
                                    "--stem-suffix", suffix,
                                ]
                                if no_year_re:
                                    cmd.append("--no-year-re")
                                run_cmd(cmd)
                                n_runs += 1

                            beta_t = np.load(beta_t_path)
                            for shift in shifts:
                                s = summarize_delta(
                                    beta_t=beta_t,
                                    gdd_grid=gdd_grid,
                                    shift=shift,
                                    hdi_prob=args.hdi_prob,
                                    anchors=anchors,
                                )
                                row = {
                                    "cohort": cohort,
                                    "predictor": predictor,
                                    "base_temp": base,
                                    "tau_gdd": tau,
                                    "basis_k": int(k),
                                    "year_re_mode": mode,
                                    "window_shift_gdd": int(shift),
                                    "stem": stem,
                                    **s,
                                }
                                rows.append(row)
                                write_row_atomic(row, sens_rows_dir)
                            if args.max_runs > 0 and n_runs >= args.max_runs:
                                log.warning(
                                    f"Reached max-runs={args.max_runs}; stopping early."
                                )
                                break
                        if args.max_runs > 0 and n_runs >= args.max_runs:
                            break
                    if args.max_runs > 0 and n_runs >= args.max_runs:
                        break
                if args.max_runs > 0 and n_runs >= args.max_runs:
                    break
            if args.max_runs > 0 and n_runs >= args.max_runs:
                break
        if args.max_runs > 0 and n_runs >= args.max_runs:
            break

    # ── Irrigated yield negative control ────────────────────────────────
    if args.irrigated:
        log.info("=== Irrigated yield negative control ===")
        rainfed_yield = PROCESSED / "olive_yield.csv"
        rainfed_means = PROCESSED / "comarca_yield_means.csv"
        backup_yield = PROCESSED / "olive_yield_rainfed_backup.csv"
        backup_means = PROCESSED / "comarca_yield_means_rainfed_backup.csv"

        shutil.copy2(rainfed_yield, backup_yield)
        if rainfed_means.exists():
            shutil.copy2(rainfed_means, backup_means)

        try:
            run_cmd([sys.executable, step1, "--irrigated"])

            # Baseline preprocessing for irrigated yield
            irr_base, irr_tau, irr_k = 10.0, 45.0, 5
            run_cmd([sys.executable, step2, "--base", str(irr_base)])
            run_cmd([sys.executable, step3, "--tau-gdd", str(irr_tau)])
            run_cmd([sys.executable, step5, "--n-basis", str(irr_k)])

            gdd_grid_irr = np.load(PROCESSED / "gdd_grid.npy")
            anchors_irr = get_anchors(irr_base)

            for cohort in ["arbequina"]:
                suffix = (
                    f"sens_irrigated_tb{sanitize(irr_base)}"
                    f"_tau{sanitize(irr_tau)}_k{irr_k}_with_year"
                )
                stem = stem_name(cohort, "wd_state", no_year_re=False,
                                 suffix=suffix)
                beta_t_path = (
                    f"/home/tsardoz/phd/catalonia/results/posteriors/"
                    f"{stem}_beta_t_draws.npy"
                )
                if not (args.reuse_existing and Path(beta_t_path).exists()):
                    cmd = [
                        sys.executable, step7,
                        "--cohort", cohort,
                        "--predictor", "wd_state",
                        "--draws", str(args.draws),
                        "--tune", str(args.tune),
                        "--chains", str(args.chains),
                        "--seed", str(args.seed),
                        "--target-accept", str(args.target_accept),
                        "--stem-suffix", suffix,
                    ]
                    run_cmd(cmd)
                    n_runs += 1

                beta_t = np.load(beta_t_path)
                for shift in shifts:
                    s = summarize_delta(
                        beta_t=beta_t, gdd_grid=gdd_grid_irr,
                        shift=shift, hdi_prob=args.hdi_prob,
                        anchors=anchors_irr,
                    )
                    row = {
                        "cohort": f"{cohort}_irrigated",
                        "predictor": "wd_state",
                        "base_temp": irr_base,
                        "tau_gdd": irr_tau,
                        "basis_k": int(irr_k),
                        "year_re_mode": "with_year",
                        "window_shift_gdd": int(shift),
                        "stem": stem,
                        **s,
                    }
                    rows.append(row)
                    write_row_atomic(row, sens_rows_dir)
        finally:
            # Restore rainfed data
            shutil.copy2(backup_yield, rainfed_yield)
            if backup_means.exists():
                shutil.copy2(backup_means, rainfed_means)
            backup_yield.unlink(missing_ok=True)
            backup_means.unlink(missing_ok=True)
            log.info("Restored rainfed yield data")
            # Restore rainfed matrices
            run_cmd([sys.executable, step2, "--base", "10.0"])
            run_cmd([sys.executable, step3, "--tau-gdd", "45.0"])
            run_cmd([sys.executable, step5, "--n-basis", "5"])

    # ── Per-row CSVs already written atomically above ────────────────────────
    # Aggregation into sensitivity_summary.csv is the aggregator's job
    # (src/10_aggregate_sensitivity.py).  This script intentionally does
    # NOT touch the shared summary file so that concurrent or interleaved
    # runs cannot lose data.
    if not rows:
        log.warning("No sensitivity rows produced in this run.")
        return
    log.info(
        f"Wrote {len(rows)} per-row CSVs to {sens_rows_dir}. "
        "Run src/10_aggregate_sensitivity.py to rebuild sensitivity_summary.csv."
    )


if __name__ == "__main__":
    main()
