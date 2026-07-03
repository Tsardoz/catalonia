#!/usr/bin/env python3
"""
08_posterior_contrast.py — Window-integrated posterior contrasts from Step 7 outputs.

Reads physical-scale beta(t) draws saved by Step 7 and computes:
  - β_S_pre, β_S5, β_S6, β_S7, β_S8 summaries
  - All contrasts use S7 (fruit development) as reference:
      Δ(S7 − S6): headline (fruit dev vs flowering)
      Δ(S7 − S5): fruit dev vs inflorescence
      Δ(S7 − S_pre): fruit dev vs winter rest
      Δ(S7 − S8): fruit dev vs maturation
  - For each contrast: median, 90% CI, prob_gt_0, and p_diff = max(p, 1-p)
    where p_diff answers "how likely are these two stages different?"

Operates on the calendar DOY-axis pipeline (S_pre/S5/S6/S7/S8 windows).
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
    PROCESSED, POSTERIORS, TABLES, STAGES_RESOLVED,
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
        choices=["arbequina", "all_olive"],
        default="arbequina",
        help="Used only when --stems is not provided (default: arbequina).",
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
        help="Uniform DOY shift applied to all window bounds (e.g. ±10 days).",
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
    p_gt0 = float((arr > 0).mean())
    return {
        "median": float(np.median(arr)),
        "q_lo": float(np.quantile(arr, lo)),
        "q_hi": float(np.quantile(arr, hi)),
        "prob_gt_0": p_gt0,
        "p_diff": float(max(p_gt0, 1.0 - p_gt0)),  # P(stages differ)
    }


def discover_stems(args: argparse.Namespace) -> list[str]:
    if args.stems.strip():
        return [s.strip() for s in args.stems.split(",") if s.strip()]

    suffix = "_noyearre" if args.no_year_re else ""
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


def load_calendar_anchors() -> dict[str, tuple[int, int]]:
    """Load resolved calendar DOY stage boundaries from phenology_anchors.csv."""
    path = PROCESSED / "phenology_anchors.csv"
    if not path.exists():
        log.warning(f"{path} not found; using default resolved windows from config.")
        return {name: (int(a), int(b)) for name, (a, b) in STAGES_RESOLVED.items()}

    anchors = pd.read_csv(path)
    needed = set(STAGES_RESOLVED)
    found: dict[str, tuple[int, int]] = {}
    for stage, grp in anchors.groupby("stage"):
        if stage not in needed:
            continue
        row = grp.iloc[0]
        # Prefer resolved (non-overlapping) boundaries
        for col_start, col_end in (("doy_start", "doy_end"), ("doy_start_raw", "doy_end_raw")):
            if col_start in row and col_end in row:
                found[stage] = (int(row[col_start]), int(row[col_end]))
                break

    for stage, fallback in STAGES_RESOLVED.items():
        if stage not in found:
            log.warning(f"Stage '{stage}' missing in phenology_anchors.csv; using default {fallback}.")
            found[stage] = (int(fallback[0]), int(fallback[1]))
    return found


def window_masks(doy_grid: np.ndarray, shift: int) -> dict[str, np.ndarray]:
    anchors = load_calendar_anchors()
    masks = {}
    for stage, (start, end) in anchors.items():
        s = start + shift
        e = end + shift
        mask = (doy_grid >= s) & (doy_grid <= e)
        if not np.any(mask):
            raise ValueError(f"No grid points in window '{stage}' ({s}-{e} DOY) after shift={shift}.")
        masks[stage] = mask
    return masks


def summarize_one_stem(stem: str, hdi_prob: float, shift: int) -> dict:
    beta_t_path = POSTERIORS / f"{stem}_beta_t_draws.npy"
    meta_path = TABLES / f"{stem}_meta.json"
    if not beta_t_path.exists():
        raise FileNotFoundError(f"Missing beta(t) draws: {beta_t_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    beta_t = np.load(beta_t_path)  # (n_draws, n_grid)
    doy_grid = np.load(PROCESSED / "doy_grid.npy")
    if beta_t.shape[1] != doy_grid.size:
        raise ValueError(
            f"Axis mismatch for stem '{stem}': beta_t.shape[1]={beta_t.shape[1]} "
            f"but doy_grid.size={doy_grid.size}. This file was produced on a "
            "different (e.g. retired GDD) axis. Delete the stale artifact and rerun Step 7."
        )
    masks = window_masks(doy_grid=doy_grid, shift=shift)

    # Stage means
    beta_S_pre = beta_t[:, masks["S_pre"]].mean(axis=1)
    beta_S5    = beta_t[:, masks["S5"]].mean(axis=1)
    beta_S6    = beta_t[:, masks["S6"]].mean(axis=1)
    beta_S7    = beta_t[:, masks["S7"]].mean(axis=1)
    beta_S8    = beta_t[:, masks["S8"]].mean(axis=1)

    # All contrasts use S7 as reference (matches figure convention)
    delta_S7_S6   = beta_S7 - beta_S6
    delta_S7_S5   = beta_S7 - beta_S5
    delta_S7_Spre = beta_S7 - beta_S_pre
    delta_S7_S8   = beta_S7 - beta_S8

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    out = {
        "stem": stem,
        "cohort": meta.get("cohort"),
        "predictor": meta.get("predictor"),
        "include_year_re": meta.get("include_year_re"),
        "window_shift_doy": int(shift),
        "n_draws": int(beta_t.shape[0]),
    }

    s_pre = quant_summary(beta_S_pre, hdi_prob)
    s5    = quant_summary(beta_S5,    hdi_prob)
    s6    = quant_summary(beta_S6,    hdi_prob)
    s7    = quant_summary(beta_S7,    hdi_prob)
    s8    = quant_summary(beta_S8,    hdi_prob)
    c_s7_s6   = quant_summary(delta_S7_S6,   hdi_prob)
    c_s7_s5   = quant_summary(delta_S7_S5,   hdi_prob)
    c_s7_spre = quant_summary(delta_S7_Spre, hdi_prob)
    c_s7_s8   = quant_summary(delta_S7_S8,   hdi_prob)

    def _stage_cols(prefix: str, q: dict) -> dict:
        return {
            f"{prefix}_median": q["median"],
            f"{prefix}_q_lo":   q["q_lo"],
            f"{prefix}_q_hi":   q["q_hi"],
        }

    def _contrast_cols(prefix: str, q: dict) -> dict:
        return {
            f"{prefix}_median":   q["median"],
            f"{prefix}_q_lo":     q["q_lo"],
            f"{prefix}_q_hi":     q["q_hi"],
            f"{prefix}_prob_gt0": q["prob_gt_0"],
            f"{prefix}_p_diff":   q["p_diff"],
        }

    out.update(_stage_cols("beta_S_pre", s_pre))
    out.update(_stage_cols("beta_S5",    s5))
    out.update(_stage_cols("beta_S6",    s6))
    out.update(_stage_cols("beta_S7",    s7))
    out.update(_stage_cols("beta_S8",    s8))
    # S7-referenced contrasts
    out.update(_contrast_cols("delta_S7_minus_S6",   c_s7_s6))
    out.update(_contrast_cols("delta_S7_minus_S5",   c_s7_s5))
    out.update(_contrast_cols("delta_S7_minus_S_pre", c_s7_spre))
    out.update(_contrast_cols("delta_S7_minus_S8",   c_s7_s8))

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
