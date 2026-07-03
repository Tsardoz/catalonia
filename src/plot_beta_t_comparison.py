#!/usr/bin/env python3
"""
plot_beta_t_comparison.py — β(t) posterior curves across cohorts on the calendar DOY axis.

Produces a single figure:
  - β(t) posterior median + 90% band for each PLAN_COHORT
  - Calendar-stage shading (S_pre, S5, S6, S7, S8) with DOY labels
  - Reads NumPyro beta(t) draws from Step 7 (07_fit_numpyro_main.py)

Output: results/figures/beta_t_cohort_comparison.png
"""
import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import POSTERIORS, PROCESSED, FIGURES, PLAN_COHORTS, STAGES_RESOLVED

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Per-cohort line colours
COHORT_COLOR = {
    "arbequina": "#2166ac",
    "all_olive":  "#4dac26",
}

COHORT_DISPLAY_NAME = {
    "arbequina": "Arbequina",
    "all_olive":  "All olive",
}

PREDICTOR_DISPLAY = {
    "wd_state": "CWB-state",
    "wd":       "CWB",
    "wd30":     "CWB-30d",
    "vpd":      "VPD",
    "tmax":     "Tmax",
    "tmin":     "Tmin",
}

# Stage shading: DOY boundaries come from config; colours defined here
STAGE_COLORS = {
    "S_pre": "#aec7e8",
    "S5":    "#98df8a",
    "S6":    "#ffbb78",
    "S7":    "#ff9896",
    "S8":    "#c5b0d5",
}

STAGE_LABELS = {
    "S_pre": "S_pre\n(DOY 1–71)",
    "S5":    "S5 inflorescence\n(72–138)",
    "S6":    "S6 flowering\n(139–173)",
    "S7":    "S7 fruit dev\n(174–299)",
    "S8":    "S8 maturation\n(300–327)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot β(t) cohort comparison on calendar DOY axis."
    )
    parser.add_argument("--predictor", default="wd_state",
                        help="Predictor key used in Step 7 stem (default: wd_state).")
    parser.add_argument("--hdi-prob",  type=float, default=0.90)
    parser.add_argument("--stem-suffix", default="",
                        help="Optional Step 7 stem suffix (e.g. 'full' for LOYO reference fits).")
    parser.add_argument("--irrigated", action="store_true",
                        help="Overlay irrigated arbequina posterior as a dashed comparison line.")
    return parser.parse_args()


def cohort_label(cohort: str, predictor: str, stem_suffix: str) -> str:
    """Build legend label from Step 7 metadata (n_obs, n_comarcas)."""
    import json
    stem = f"numpyro_main_{cohort}_{predictor}"
    if stem_suffix:
        stem += f"_{stem_suffix}"
    meta_path = __import__("pathlib").Path(
        f"/home/tsardoz/phd/catalonia/results/tables/{stem}_meta.json"
    )
    name = COHORT_DISPLAY_NAME.get(cohort, cohort)
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        return f"{name} ({m['n_comarcas']} com., {m['n_obs']} obs)"
    return name


def load_beta_t(cohort: str, predictor: str, stem_suffix: str) -> np.ndarray:
    stem = f"numpyro_main_{cohort}_{predictor}"
    if stem_suffix:
        stem += f"_{stem_suffix}"
    path = POSTERIORS / f"{stem}_beta_t_draws.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing beta(t) draws: {path}. Run Step 7 first.")
    beta_t = np.load(path)
    doy_grid = np.load(PROCESSED / "doy_grid.npy")
    if beta_t.shape[1] != doy_grid.size:
        raise ValueError(
            f"Axis mismatch for {stem}: beta_t.shape[1]={beta_t.shape[1]} "
            f"but doy_grid.size={doy_grid.size}. Stale retired-GDD-axis file — delete and rerun Step 7."
        )
    log.info(f"Loaded {beta_t.shape} from {path}")
    return beta_t


def main() -> None:
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doy_grid = np.load(PROCESSED / "doy_grid.npy")
    lo_q = 0.5 * (1.0 - args.hdi_prob)
    hi_q = 1.0 - lo_q

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # ── Stage shading ────────────────────────────────────────────────────
    for stage, (s, e) in STAGES_RESOLVED.items():
        ax.axvspan(s, e, color=STAGE_COLORS[stage], alpha=0.22, lw=0)

    # ── Posterior curves ─────────────────────────────────────────────────
    for cohort in PLAN_COHORTS:
        if cohort not in COHORT_COLOR:
            log.warning(f"No colour defined for cohort '{cohort}'; skipping.")
            continue
        color = COHORT_COLOR[cohort]
        label = cohort_label(cohort, args.predictor, args.stem_suffix)
        beta_t = load_beta_t(cohort, args.predictor, args.stem_suffix)
        median = np.median(beta_t, axis=0)
        lo     = np.quantile(beta_t, lo_q, axis=0)
        hi     = np.quantile(beta_t, hi_q, axis=0)
        ax.plot(doy_grid, median, color=color, lw=2.0, label=label)
        ax.fill_between(doy_grid, lo, hi, color=color, alpha=0.15)

    # ── Stage labels at top ──────────────────────────────────────────────
    ylim = ax.get_ylim()
    for stage, (s, e) in STAGES_RESOLVED.items():
        ax.text(
            (s + e) / 2, ylim[1],
            STAGE_LABELS[stage],
            ha="center", va="top", fontsize=6.5, color="0.35",
        )

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("Day of year (DOY)", fontsize=12)
    pred_disp = PREDICTOR_DISPLAY.get(args.predictor, args.predictor.upper())
    ax.set_ylabel(f"β(t) — {pred_disp} effect on yield", fontsize=12)
    ax.set_title(
        f"β(t) comparison across cohorts — {pred_disp} predictor\n"
        f"(calendar DOY axis, Garrido et al. 2021 windows, "
        f"{int(args.hdi_prob * 100)}% posterior band)",
        fontsize=12, fontweight="bold",
    )
    # Optional irrigated overlay
    if args.irrigated:
        try:
            beta_irr = load_beta_t("arbequina", args.predictor, "irrigated")
            median_irr = np.median(beta_irr, axis=0)
            irr_label = cohort_label("arbequina", args.predictor, "irrigated")
            ax.plot(doy_grid, median_irr, color=COHORT_COLOR["arbequina"],
                    lw=1.5, ls="--", alpha=0.7,
                    label=f"{irr_label} [irrigated, median only]")
        except FileNotFoundError:
            log.warning("Irrigated beta_t not found; skipping overlay.")

    ax.set_xlim(int(doy_grid[0]), int(doy_grid[-1]))
    ax.legend(fontsize=10, framealpha=0.9)

    FIGURES.mkdir(parents=True, exist_ok=True)
    suffix_tag = f"_{args.stem_suffix}" if args.stem_suffix else ""
    out = FIGURES / f"beta_t_cohort_comparison_{args.predictor}{suffix_tag}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    log.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
