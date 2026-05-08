#!/usr/bin/env python3
"""
08_posterior_contrast.py — Window-integrated posterior contrasts from Step 7 outputs.

Reads physical-scale beta(t) draws saved by Step 7 and computes:
  - β_preflower, β_flowering, β_pit summaries
  - Δ(pit − flower) headline contrast
  - Δ(flower − preflower) secondary contrast
"""
import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, POSTERIORS, TABLES,
    PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute posterior stage-window contrasts from Step 7 beta(t) draws."
    )
    parser.add_argument(
        "--stems",
        type=str,
        default="",
        help="Comma-separated output stems from Step 7 (without file extension).",
    )
    parser.add_argument(
        "--cohort",
        choices=["arbequina", "morruda", "all_olive", "both", "all"],
        default="both",
        help="Used only when --stems is not provided (default: both).",
    )
    parser.add_argument(
        "--predictor",
        default="wd_state",
        help="Used only when --stems is not provided (default: wd_state).",
    )
    parser.add_argument(
        "--no-year-re",
        action="store_true",
        help="Used only when --stems is not provided; discover no-year-re stems.",
    )
    parser.add_argument(
        "--window-shift",
        type=int,
        default=0,
        help="Uniform GDD shift applied to all window bounds (e.g. ±50).",
    )
    parser.add_argument("--hdi-prob", type=float, default=0.90)
    parser.add_argument(
        "--output-name",
        type=str,
        default="posterior_contrast_summary.csv",
        help="Filename under results/tables/ for combined summary output.",
    )
    return parser.parse_args()


def quant_summary(arr: np.ndarray, hdi_prob: float) -> dict:
    lo = 0.5 * (1.0 - hdi_prob)
    hi = 1.0 - lo
    return {
        "median": float(np.median(arr)),
        "q_lo": float(np.quantile(arr, lo)),
        "q_hi": float(np.quantile(arr, hi)),
        "prob_gt_0": float((arr > 0).mean()),
    }


def discover_stems(args: argparse.Namespace) -> list[str]:
    if args.stems.strip():
        return [s.strip() for s in args.stems.split(",") if s.strip()]

    suffix = "_noyearre" if args.no_year_re else ""
    if args.cohort == "both":
        cohorts = ["arbequina", "morruda", "all_olive"]
    elif args.cohort == "all":
        cohorts = ["all"]
    else:
        cohorts = [args.cohort]

    stems = []
    for cohort in cohorts:
        base = f"numpyro_main_{cohort}_{args.predictor}{suffix}"
        meta = TABLES / f"{base}_meta.json"
        if meta.exists():
            stems.append(base)
            continue
        matches = sorted(TABLES.glob(f"{base}*_meta.json"))
        stems.extend([m.name.replace("_meta.json", "") for m in matches])
    stems = sorted(set(stems))
    if not stems:
        raise FileNotFoundError(
            "No Step 7 stems found. Pass --stems explicitly or run Step 7 first."
        )
    return stems


def window_masks(gdd_grid: np.ndarray, shift: int) -> dict[str, np.ndarray]:
    pre = (PRE_FLOWERING_GDD[0] + shift, PRE_FLOWERING_GDD[1] + shift)
    flo = (FLOWERING_GDD[0] + shift, FLOWERING_GDD[1] + shift)
    pit = (PIT_HARDENING_GDD[0] + shift, PIT_HARDENING_GDD[1] + shift)
    masks = {
        "pre_flowering": (gdd_grid >= pre[0]) & (gdd_grid <= pre[1]),
        "flowering": (gdd_grid >= flo[0]) & (gdd_grid <= flo[1]),
        "pit_hardening": (gdd_grid >= pit[0]) & (gdd_grid <= pit[1]),
    }
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"No grid points in window '{name}' after shift={shift}.")
    return masks


def summarize_one_stem(stem: str, hdi_prob: float, shift: int) -> dict:
    beta_t_path = POSTERIORS / f"{stem}_beta_t_draws.npy"
    meta_path = TABLES / f"{stem}_meta.json"
    if not beta_t_path.exists():
        raise FileNotFoundError(f"Missing beta(t) draws: {beta_t_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    beta_t = np.load(beta_t_path)  # (n_draws, n_grid)
    gdd_grid = np.load(PROCESSED / "gdd_grid.npy")
    masks = window_masks(gdd_grid=gdd_grid, shift=shift)

    beta_pre = beta_t[:, masks["pre_flowering"]].mean(axis=1)
    beta_flo = beta_t[:, masks["flowering"]].mean(axis=1)
    beta_pit = beta_t[:, masks["pit_hardening"]].mean(axis=1)
    delta_pf = beta_flo - beta_pre
    delta_pit_flo = beta_pit - beta_flo

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    out = {
        "stem": stem,
        "cohort": meta.get("cohort"),
        "predictor": meta.get("predictor"),
        "include_year_re": meta.get("include_year_re"),
        "window_shift_gdd": int(shift),
        "n_draws": int(beta_t.shape[0]),
    }

    s_pre = quant_summary(beta_pre, hdi_prob)
    s_flo = quant_summary(beta_flo, hdi_prob)
    s_pit = quant_summary(beta_pit, hdi_prob)
    s_delta = quant_summary(delta_pit_flo, hdi_prob)
    s_delta_pf = quant_summary(delta_pf, hdi_prob)

    out.update({
        "beta_pre_median": s_pre["median"],
        "beta_pre_q_lo": s_pre["q_lo"],
        "beta_pre_q_hi": s_pre["q_hi"],
        "beta_flowering_median": s_flo["median"],
        "beta_flowering_q_lo": s_flo["q_lo"],
        "beta_flowering_q_hi": s_flo["q_hi"],
        "beta_pit_median": s_pit["median"],
        "beta_pit_q_lo": s_pit["q_lo"],
        "beta_pit_q_hi": s_pit["q_hi"],
        "delta_pit_minus_flower_median": s_delta["median"],
        "delta_pit_minus_flower_q_lo": s_delta["q_lo"],
        "delta_pit_minus_flower_q_hi": s_delta["q_hi"],
        "prob_delta_pit_minus_flower_gt_0": s_delta["prob_gt_0"],
        "delta_flower_minus_pre_median": s_delta_pf["median"],
        "delta_flower_minus_pre_q_lo": s_delta_pf["q_lo"],
        "delta_flower_minus_pre_q_hi": s_delta_pf["q_hi"],
        "prob_delta_flower_minus_pre_gt_0": s_delta_pf["prob_gt_0"],
    })

    per_stem_json = TABLES / f"{stem}_contrast_summary.json"
    with open(per_stem_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    stems = discover_stems(args)
    log.info(f"Summarizing {len(stems)} stem(s): {', '.join(stems)}")

    rows = [summarize_one_stem(s, args.hdi_prob, args.window_shift) for s in stems]
    df = pd.DataFrame(rows).sort_values(["cohort", "predictor", "stem"]).reset_index(drop=True)
    out_csv = TABLES / args.output_name
    df.to_csv(out_csv, index=False)
    log.info(f"Saved combined contrast summary: {out_csv}")


if __name__ == "__main__":
    main()
