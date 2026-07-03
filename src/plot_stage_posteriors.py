#!/usr/bin/env python3
"""
plot_stage_posteriors.py — Visualize posterior β distributions by calendar stage.

Produces a two-panel figure:
  Top:    KDE + HDI bars for β_S_pre, β_S5, β_S6, β_S7, β_S8
  Bottom: KDE + HDI bars for headline and secondary contrasts
          (Δ S7−S6, Δ S6−S5, Δ S5−S_pre)

Reads the NumPyro beta(t) draws saved by Step 7 (07_fit_numpyro_main.py).
Stage windows are the resolved calendar DOY boundaries from config.STAGES_RESOLVED.
"""
import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import POSTERIORS, PROCESSED, FIGURES, STAGES_RESOLVED

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Display labels and DOY annotations for each calendar stage
STAGE_META = {
    "S_pre": ("S_pre\n(DOY 1–71)",   "#4575b4"),
    "S5":    ("S5 inflorescence\n(DOY 72–138)",  "#74add1"),
    "S6":    ("S6 flowering\n(DOY 139–173)",     "#fee090"),
    "S7":    ("S7 fruit dev\n(DOY 174–299)",     "#f46d43"),
    "S8":    ("S8 maturation\n(DOY 300–327)",    "#d73027"),
}

CONTRAST_META = [
    ("Δ S7 − S6 (fruit dev − flowering)",        "S7", "S6",    "#762a83"),
    ("Δ S7 − S5 (fruit dev − inflorescence)",    "S7", "S5",    "#1b7837"),
    ("Δ S7 − S_pre (fruit dev − winter)",        "S7", "S_pre", "#8c510a"),
    ("Δ S7 − S8 (fruit dev − maturation)",      "S7", "S8",    "#d6604d"),
]


def predictor_label(predictor: str) -> str:
    labels = {
        "wd_state": "CWB-state",
        "wd":       "CWB",
        "wd30":     "CWB-30d",
        "vpd":      "VPD",
        "tmax":     "Tmax",
        "tmin":     "Tmin",
    }
    return labels.get(predictor, predictor.upper())


def load_stage_draws(cohort: str, predictor: str, stem_suffix: str = "") -> dict[str, np.ndarray]:
    """Load beta_t draws, compute per-stage window means."""
    stem = f"numpyro_main_{cohort}_{predictor}"
    if stem_suffix:
        stem += f"_{stem_suffix}"
    beta_t_path = POSTERIORS / f"{stem}_beta_t_draws.npy"
    if not beta_t_path.exists():
        raise FileNotFoundError(
            f"beta(t) draws not found: {beta_t_path}. Run Step 7 first."
        )
    beta_t = np.load(beta_t_path)  # (n_draws, n_doy)
    doy_grid = np.load(PROCESSED / "doy_grid.npy")

    if beta_t.shape[1] != doy_grid.size:
        raise ValueError(
            f"Axis mismatch: beta_t.shape[1]={beta_t.shape[1]} "
            f"but doy_grid.size={doy_grid.size}."
        )

    stage_draws = {}
    for stage, (start, end) in STAGES_RESOLVED.items():
        mask = (doy_grid >= start) & (doy_grid <= end)
        stage_draws[stage] = beta_t[:, mask].mean(axis=1)
    log.info(f"Loaded {beta_t.shape[0]} draws for {len(stage_draws)} stages from {beta_t_path}")
    return stage_draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort",    default="arbequina",
                        choices=["arbequina", "all_olive"])
    parser.add_argument("--predictor", default="wd_state")
    parser.add_argument("--hdi",       type=float, default=0.90)
    parser.add_argument("--stem-suffix", default="",
                        help="Suffix appended to stem name (e.g. 'irrigated').")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    stage_draws = load_stage_draws(args.cohort, args.predictor, args.stem_suffix)
    pred_label  = predictor_label(args.predictor)
    lo_q = 0.5 * (1 - args.hdi)
    hi_q = 1 - lo_q

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )

    # ── Top panel: per-stage β posteriors ───────────────────────────────
    for i, (stage, (label, color)) in enumerate(STAGE_META.items()):
        arr = stage_draws[stage]
        span = arr.max() - arr.min()
        pad  = max(span * 0.1, 1e-6)
        x    = np.linspace(arr.min() - pad, arr.max() + pad, 400)
        y    = gaussian_kde(arr)(x)
        ax1.fill_between(x, y, alpha=0.20, color=color)
        ax1.plot(x, y, color=color, lw=1.5, label=label)
        q_lo, med, q_hi = np.quantile(arr, [lo_q, 0.5, hi_q])
        bar_y = -0.65 * (i + 1)
        ax1.plot([q_lo, q_hi], [bar_y, bar_y], color=color, lw=3,
                 solid_capstyle="round")
        ax1.plot(med, bar_y, "o", color=color, ms=6, zorder=5)

    ax1.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax1.set_xlabel(f"β ({pred_label} window-mean coefficient)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    tag_label = f" [{args.stem_suffix}]" if args.stem_suffix else ""
    ax1.set_title(
        f"Posterior {pred_label} β by calendar stage — "
        f"{args.cohort.replace('_', ' ').title()}{tag_label}",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_yticks([])

    # ── Bottom panel: contrasts ──────────────────────────────────────────
    for i, (clabel, s_hi, s_lo, ccolor) in enumerate(CONTRAST_META):
        delta = stage_draws[s_hi] - stage_draws[s_lo]
        span  = delta.max() - delta.min()
        pad   = max(span * 0.1, 1e-6)
        x     = np.linspace(delta.min() - pad, delta.max() + pad, 400)
        y     = gaussian_kde(delta)(x)
        ax2.fill_between(x, y, alpha=0.20, color=ccolor)
        ax2.plot(x, y, color=ccolor, lw=1.5, label=clabel)
        q_lo, med, q_hi = np.quantile(delta, [lo_q, 0.5, hi_q])
        p_gt0 = float((delta > 0).mean())
        p_diff = max(p_gt0, 1.0 - p_gt0)   # two-sided: prob stages differ
        direction = ">" if p_gt0 >= 0.5 else "<"
        bar_y = -0.65 * (i + 1)
        ax2.plot([q_lo, q_hi], [bar_y, bar_y], color=ccolor, lw=3,
                 solid_capstyle="round")
        ax2.plot(med, bar_y, "o", color=ccolor, ms=6, zorder=5)
        # Annotate to the left of and below the KDE peak
        peak_x = x[np.argmax(y)]
        ax2.text(
            peak_x, y.max() * 0.60,
            f"P(diff) = {p_diff:.2f}  ({s_hi}{direction}{s_lo})",
            fontsize=8, color=ccolor,
            ha="right", va="top",
            fontweight="bold",
        )

    ax2.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax2.set_xlabel("Δβ", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Stage contrasts (S7 fruit dev as reference)",
                  fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.set_yticks([])

    FIGURES.mkdir(parents=True, exist_ok=True)
    suffix_tag = f"_{args.stem_suffix}" if args.stem_suffix else ""
    out = FIGURES / f"stage_posteriors_{args.cohort}_{args.predictor}{suffix_tag}.png"
    fig.savefig(out, dpi=180)
    log.info(f"Saved figure: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
