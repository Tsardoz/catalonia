#!/usr/bin/env python3
"""
10_sensitivity_analysis.py — Sensitivity sweeps for the Step 7 model.

Runs combinations across preprocessing/model settings, then summarizes
Δ(pit − flower) robustness from posterior beta(t) draws.
"""
import argparse
import logging
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, TABLES,
    PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD,
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


def window_masks(gdd_grid: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = (PRE_FLOWERING_GDD[0] + shift, PRE_FLOWERING_GDD[1] + shift)
    flo = (FLOWERING_GDD[0] + shift, FLOWERING_GDD[1] + shift)
    pit = (PIT_HARDENING_GDD[0] + shift, PIT_HARDENING_GDD[1] + shift)
    m_pre = (gdd_grid >= pre[0]) & (gdd_grid <= pre[1])
    m_flo = (gdd_grid >= flo[0]) & (gdd_grid <= flo[1])
    m_pit = (gdd_grid >= pit[0]) & (gdd_grid <= pit[1])
    if not (np.any(m_pre) and np.any(m_flo) and np.any(m_pit)):
        raise ValueError(f"Empty stage mask for shift={shift}.")
    return m_pre, m_flo, m_pit


def summarize_delta(beta_t: np.ndarray, gdd_grid: np.ndarray, shift: int, hdi_prob: float) -> dict:
    m_pre, m_flo, m_pit = window_masks(gdd_grid, shift)
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

    step2 = "/home/tsardoz/phd/catalonia/src/02_compute_gdd.py"
    step3 = "/home/tsardoz/phd/catalonia/src/03_compute_water_deficit.py"
    step5 = "/home/tsardoz/phd/catalonia/src/05_build_basis.py"
    step7 = "/home/tsardoz/phd/catalonia/src/07_fit_numpyro_main.py"

    rows = []
    n_runs = 0
    gdd_grid = np.load(PROCESSED / "gdd_grid.npy")

    for base in base_temps:
        run_cmd([sys.executable, step2, "--base", str(base)])
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
                                )
                                rows.append({
                                    "cohort": cohort,
                                    "predictor": predictor,
                                    "base_temp": base,
                                    "tau_gdd": tau,
                                    "basis_k": int(k),
                                    "year_re_mode": mode,
                                    "window_shift_gdd": int(shift),
                                    "stem": stem,
                                    **s,
                                })
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

    if not rows:
        raise ValueError("No sensitivity rows produced.")
    df = pd.DataFrame(rows)

    # Baseline comparison flags (per cohort if baseline exists)
    flags = []
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
        if baseline.empty:
            continue
        b = baseline.iloc[0]
        d = df[df["cohort"] == cohort].copy()
        d["delta_sign_flip_vs_baseline"] = (
            np.sign(d["delta_median"]) != np.sign(b["delta_median"])
        )
        d["prob_lt_0p7"] = d["prob_delta_gt_0"] < 0.7
        flags.append(d)
    if flags:
        df = pd.concat(flags, ignore_index=True)
    else:
        df["delta_sign_flip_vs_baseline"] = False
        df["prob_lt_0p7"] = df["prob_delta_gt_0"] < 0.7

    out_csv = TABLES / "sensitivity_summary.csv"
    df.sort_values(
        ["cohort", "predictor", "base_temp", "tau_gdd", "basis_k", "year_re_mode", "window_shift_gdd"]
    ).to_csv(out_csv, index=False)
    log.info(f"Saved sensitivity summary: {out_csv}")


if __name__ == "__main__":
    main()
