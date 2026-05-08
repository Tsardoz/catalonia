#!/usr/bin/env python3
"""
09_stability_loyo.py — Leave-one-year-out stability check for Step 7 model.

For each held-out year:
  1) Refit Step 7 excluding that year
  2) Recompute beta(t) and Δ contrast summaries
  3) Compare fold medians to full-data posterior
"""
import argparse
import logging
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, TABLES, FIGURES,
    PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LOYO stability using Step 7 refits."
    )
    parser.add_argument(
        "--cohort",
        choices=["arbequina", "morruda", "all_olive", "all"],
        default="arbequina",
    )
    parser.add_argument("--predictor", default="wd_state")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--hdi-prob", type=float, default=0.90)
    parser.add_argument("--window-shift", type=int, default=0)
    parser.add_argument("--no-year-re", action="store_true")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Limit number of held-out years (0 = all years).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip Step 7 run when expected artifacts already exist.",
    )
    return parser.parse_args()


def stem_name(
    cohort: str,
    predictor: str,
    no_year_re: bool,
    exclude_year: int | None = None,
    suffix: str = "",
) -> str:
    stem = f"numpyro_main_{cohort}_{predictor}"
    if no_year_re:
        stem += "_noyearre"
    if exclude_year is not None:
        stem += f"_excl{int(exclude_year)}"
    if suffix:
        stem += f"_{suffix}"
    return stem


def _build_step7_cmd(args: argparse.Namespace, exclude_year: int | None, suffix: str) -> list[str]:
    """Build the Step 7 subprocess command list."""
    step7_path = "/home/tsardoz/phd/catalonia/src/07_fit_numpyro_main.py"
    cmd = [
        sys.executable,
        step7_path,
        "--cohort", args.cohort,
        "--predictor", args.predictor,
        "--draws", str(args.draws),
        "--tune", str(args.tune),
        "--chains", str(args.chains),
        "--seed", str(args.seed),
        "--target-accept", str(args.target_accept),
    ]
    if args.no_year_re:
        cmd.append("--no-year-re")
    if exclude_year is not None:
        cmd.extend(["--exclude-year", str(int(exclude_year))])
    if suffix:
        cmd.extend(["--stem-suffix", suffix])
    return cmd


def run_step7(args: argparse.Namespace, exclude_year: int | None, suffix: str = "") -> str:
    """Run a Step 7 fit blockingly (used for the full-data reference fit)."""
    stem = stem_name(
        cohort=args.cohort,
        predictor=args.predictor,
        no_year_re=args.no_year_re,
        exclude_year=exclude_year,
        suffix=suffix,
    )
    beta_t_path = __import__("pathlib").Path(
        f"/home/tsardoz/phd/catalonia/results/posteriors/{stem}_beta_t_draws.npy"
    )
    if args.reuse_existing and beta_t_path.exists():
        log.info(f"Reusing existing Step 7 artifacts for stem={stem}")
        return stem

    log.info(f"Running Step 7 fit (blocking) for stem={stem}")
    subprocess.run(_build_step7_cmd(args, exclude_year, suffix), check=True)
    return stem


def launch_step7(
    args: argparse.Namespace, exclude_year: int, suffix: str = ""
) -> tuple[str, "subprocess.Popen | None"]:
    """Launch a Step 7 fold fit non-blockingly. Returns (stem, Popen) or (stem, None) if reused."""
    stem = stem_name(
        cohort=args.cohort,
        predictor=args.predictor,
        no_year_re=args.no_year_re,
        exclude_year=exclude_year,
        suffix=suffix,
    )
    beta_t_path = __import__("pathlib").Path(
        f"/home/tsardoz/phd/catalonia/results/posteriors/{stem}_beta_t_draws.npy"
    )
    if args.reuse_existing and beta_t_path.exists():
        log.info(f"Reusing existing Step 7 artifacts for stem={stem}")
        return stem, None

    log.info(f"Launching Step 7 fold fit (non-blocking) for stem={stem}")
    proc = subprocess.Popen(_build_step7_cmd(args, exclude_year, suffix))
    return stem, proc


def window_masks(gdd_grid: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = (PRE_FLOWERING_GDD[0] + shift, PRE_FLOWERING_GDD[1] + shift)
    flo = (FLOWERING_GDD[0] + shift, FLOWERING_GDD[1] + shift)
    pit = (PIT_HARDENING_GDD[0] + shift, PIT_HARDENING_GDD[1] + shift)
    m_pre = (gdd_grid >= pre[0]) & (gdd_grid <= pre[1])
    m_flo = (gdd_grid >= flo[0]) & (gdd_grid <= flo[1])
    m_pit = (gdd_grid >= pit[0]) & (gdd_grid <= pit[1])
    if not (np.any(m_pre) and np.any(m_flo) and np.any(m_pit)):
        raise ValueError("At least one stage window has no grid points.")
    return m_pre, m_flo, m_pit


def summarize_stem(stem: str, shift: int, hdi_prob: float) -> dict:
    beta_t = np.load(f"/home/tsardoz/phd/catalonia/results/posteriors/{stem}_beta_t_draws.npy")
    gdd_grid = np.load(PROCESSED / "gdd_grid.npy")
    m_pre, m_flo, m_pit = window_masks(gdd_grid, shift)
    beta_pre = beta_t[:, m_pre].mean(axis=1)
    beta_flo = beta_t[:, m_flo].mean(axis=1)
    beta_pit = beta_t[:, m_pit].mean(axis=1)
    delta = beta_pit - beta_flo

    lo = 0.5 * (1.0 - hdi_prob)
    hi = 1.0 - lo
    return {
        "stem": stem,
        "beta_t": beta_t,
        "delta_draws": delta,
        "delta_median": float(np.median(delta)),
        "delta_q_lo": float(np.quantile(delta, lo)),
        "delta_q_hi": float(np.quantile(delta, hi)),
        "prob_delta_gt_0": float((delta > 0).mean()),
    }


def plot_beta_overlay(
    cohort: str,
    predictor: str,
    full_beta_t: np.ndarray,
    fold_beta_medians: dict[int, np.ndarray],
    gdd_grid: np.ndarray,
):
    full_median = np.median(full_beta_t, axis=0)
    full_lo = np.quantile(full_beta_t, 0.05, axis=0)
    full_hi = np.quantile(full_beta_t, 0.95, axis=0)

    plt.figure(figsize=(9, 5))
    plt.fill_between(gdd_grid, full_lo, full_hi, alpha=0.25, label="Full-data 90% band")
    plt.plot(gdd_grid, full_median, lw=2.0, label="Full-data median")
    for year, med in sorted(fold_beta_medians.items()):
        plt.plot(gdd_grid, med, lw=1.2, alpha=0.85, label=f"LOYO excl {year}")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("GDD")
    plt.ylabel("β(t)")
    plt.title(f"LOYO β(t) stability — {cohort} / {predictor}")
    plt.legend(ncol=2, fontsize=8)
    out = FIGURES / f"loyo_beta_overlay_{cohort}_{predictor}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_delta_strip(
    cohort: str,
    predictor: str,
    fold_medians: dict[int, float],
    full_delta_draws: np.ndarray,
):
    years = sorted(fold_medians.keys())
    vals = np.array([fold_medians[y] for y in years])
    x = np.arange(len(years))

    plt.figure(figsize=(8, 4))
    plt.boxplot(vals, positions=[x.mean()], widths=0.5, vert=True)
    plt.scatter(x, vals, color="tab:blue", alpha=0.9, label="LOYO Δ medians")
    for xi, y, v in zip(x, years, vals):
        plt.text(xi, v, str(y), fontsize=8, ha="center", va="bottom")
    full_med = np.median(full_delta_draws)
    full_lo = np.quantile(full_delta_draws, 0.05)
    full_hi = np.quantile(full_delta_draws, 0.95)
    plt.axhline(full_med, color="tab:red", lw=2, label="Full-data Δ median")
    plt.axhspan(full_lo, full_hi, color="tab:red", alpha=0.15, label="Full-data Δ 90% CI")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(x, years, rotation=45)
    plt.ylabel("Δ (pit − flower)")
    plt.title(f"LOYO Δ stability — {cohort} / {predictor}")
    plt.legend(fontsize=8)
    out = FIGURES / f"loyo_delta_strip_{cohort}_{predictor}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def main():
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")
    if "cohort" not in yield_df.columns:
        raise ValueError("olive_yield.csv missing cohort column; rerun Step 1.")
    if args.cohort != "all":
        yield_df = yield_df[yield_df["cohort"] == args.cohort].copy()
    years = sorted(yield_df["year"].astype(int).unique().tolist())
    if args.max_folds > 0:
        years = years[:args.max_folds]
    if len(years) < 2:
        raise ValueError("Need at least two years for LOYO.")
    log.info(f"LOYO years for cohort={args.cohort}: {years}")

    full_stem = run_step7(args=args, exclude_year=None, suffix="full")
    full = summarize_stem(stem=full_stem, shift=args.window_shift, hdi_prob=args.hdi_prob)

    # Launch all fold fits simultaneously
    fold_launches: list[tuple[int, str, "subprocess.Popen | None"]] = []
    for y in years:
        fold_stem, proc = launch_step7(args=args, exclude_year=y)
        fold_launches.append((int(y), fold_stem, proc))

    # Wait for all folds; collect failures
    failed: list[int] = []
    for y, fold_stem, proc in fold_launches:
        if proc is None:
            log.info(f"Fold {y}: reused existing artifacts (stem={fold_stem})")
            continue
        rc = proc.wait()
        if rc != 0:
            failed.append(y)
            log.error(f"Fold {y} failed (stem={fold_stem}, return code={rc})")
        else:
            log.info(f"Fold {y} complete (stem={fold_stem})")
    if failed:
        raise RuntimeError(f"LOYO fold(s) failed for years: {failed}. Check stderr above.")

    # Summarise
    rows = []
    fold_beta_medians: dict[int, np.ndarray] = {}
    fold_delta_medians: dict[int, float] = {}
    for y, fold_stem, _ in fold_launches:
        fold = summarize_stem(stem=fold_stem, shift=args.window_shift, hdi_prob=args.hdi_prob)
        fold_beta_medians[int(y)] = np.median(fold["beta_t"], axis=0)
        fold_delta_medians[int(y)] = fold["delta_median"]
        rows.append({
            "cohort": args.cohort,
            "predictor": args.predictor,
            "held_out_year": int(y),
            "stem": fold_stem,
            "delta_median": fold["delta_median"],
            "delta_q_lo": fold["delta_q_lo"],
            "delta_q_hi": fold["delta_q_hi"],
            "prob_delta_gt_0": fold["prob_delta_gt_0"],
        })

    out_df = pd.DataFrame(rows).sort_values("held_out_year").reset_index(drop=True)
    out_csv = TABLES / f"loyo_summary_{args.cohort}_{args.predictor}.csv"
    out_df.to_csv(out_csv, index=False)

    full_row = pd.DataFrame([{
        "cohort": args.cohort,
        "predictor": args.predictor,
        "held_out_year": "full",
        "stem": full_stem,
        "delta_median": full["delta_median"],
        "delta_q_lo": full["delta_q_lo"],
        "delta_q_hi": full["delta_q_hi"],
        "prob_delta_gt_0": full["prob_delta_gt_0"],
    }])
    full_csv = TABLES / f"loyo_full_{args.cohort}_{args.predictor}.csv"
    full_row.to_csv(full_csv, index=False)

    gdd_grid = np.load(PROCESSED / "gdd_grid.npy")
    fig1 = plot_beta_overlay(
        cohort=args.cohort,
        predictor=args.predictor,
        full_beta_t=full["beta_t"],
        fold_beta_medians=fold_beta_medians,
        gdd_grid=gdd_grid,
    )
    fig2 = plot_delta_strip(
        cohort=args.cohort,
        predictor=args.predictor,
        fold_medians=fold_delta_medians,
        full_delta_draws=full["delta_draws"],
    )

    log.info(f"Saved LOYO summary: {out_csv}")
    log.info(f"Saved LOYO full summary: {full_csv}")
    log.info(f"Saved LOYO figures: {fig1}, {fig2}")


if __name__ == "__main__":
    main()
