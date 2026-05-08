#!/usr/bin/env python3
"""
plot_stage_posteriors.py — Visualize posterior distributions for stage coefficients.

Produces a two-panel figure:
  Top:    KDE + HDI bars for β_preflower, β_flowering, β_pit
  Bottom: KDE + HDI bars for the two contrasts (pit−flower, flower−preflower)

Reads the saved Bambi InferenceData (.nc) from results/posteriors/.
"""
import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import POSTERIORS, FIGURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def predictor_labels(predictor: str) -> tuple[str, str]:
    """Return short title label and x-axis label for a predictor key."""
    labels = {
        "wd": ("WD", "β (WD stage-mean coefficient)"),
        "vpd": ("VPD", "β (VPD stage-mean coefficient)"),
        "wd30": ("WD30", "β (WD30 stage-mean coefficient)"),
    }
    if predictor in labels:
        return labels[predictor]
    upper = predictor.upper()
    return upper, f"β ({upper} stage-mean coefficient)"


def load_posterior(cohort: str, predictor: str):
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError("arviz required: pip install arviz") from exc

    path = POSTERIORS / f"bambi_sanity_{cohort}_{predictor}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Posterior not found: {path}. Run Step 6 first.")
    idata = az.from_netcdf(path)
    log.info(f"Loaded posterior from {path}")
    return idata


def extract_draws(idata, predictor: str) -> dict[str, np.ndarray]:
    """Extract flattened posterior draws for the three stage terms."""
    terms = {
        "Pre-flowering\n(0–350 GDD)": f"{predictor}_preflower_mean",
        "Flowering\n(350–450 GDD)": f"{predictor}_flowering_mean",
        "Pit hardening\n(900–1200 GDD)": f"{predictor}_pit_mean",
    }
    draws = {}
    for label, var in terms.items():
        if var in idata.posterior:
            draws[label] = idata.posterior[var].values.reshape(-1)
        else:
            log.warning(f"Variable '{var}' not found in posterior; skipping.")
    return draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="arbequina")
    parser.add_argument("--predictor", default="wd")
    parser.add_argument("--hdi", type=float, default=0.90)
    args = parser.parse_args()
    predictor_short, beta_label = predictor_labels(args.predictor)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    idata = load_posterior(args.cohort, args.predictor)
    draws = extract_draws(idata, args.predictor)

    if len(draws) < 3:
        raise ValueError("Need all three stage posteriors to plot.")

    labels = list(draws.keys())
    arrays = [draws[l] for l in labels]

    # Colours: cool→warm gradient matching phenological progression
    stage_colors = ["#4575b4", "#fee090", "#d73027"]  # blue, amber, red
    contrast_colors = ["#762a83", "#1b7837"]  # purple, green

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 2]},
        constrained_layout=True,
    )

    lo = 0.5 * (1 - args.hdi)
    hi = 1 - lo

    # ── Top panel: stage coefficient posteriors ──────────────────────────
    for i, (label, arr, color) in enumerate(zip(labels, arrays, stage_colors)):
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(arr)
        x = np.linspace(arr.min() - 0.02, arr.max() + 0.02, 300)
        y = kde(x)
        ax1.fill_between(x, y, alpha=0.25, color=color)
        ax1.plot(x, y, color=color, lw=1.5, label=label)

        # HDI bar
        q_lo, med, q_hi = np.quantile(arr, [lo, 0.5, hi])
        bar_y = -0.6 * (i + 1)
        ax1.plot([q_lo, q_hi], [bar_y, bar_y], color=color, lw=3, solid_capstyle="round")
        ax1.plot(med, bar_y, "o", color=color, ms=6, zorder=5)

    ax1.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax1.set_xlabel(beta_label, fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        f"Posterior {predictor_short} coefficients by phenological stage — {args.cohort.title()}",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.set_yticks([])

    # ── Bottom panel: contrasts ──────────────────────────────────────────
    contrast_defs = [
        ("Δ pit − flower", arrays[2] - arrays[1], contrast_colors[0]),
        ("Δ flower − preflower", arrays[1] - arrays[0], contrast_colors[1]),
    ]

    for i, (clabel, delta, ccolor) in enumerate(contrast_defs):
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(delta)
        x = np.linspace(delta.min() - 0.02, delta.max() + 0.02, 300)
        y = kde(x)
        ax2.fill_between(x, y, alpha=0.25, color=ccolor)
        ax2.plot(x, y, color=ccolor, lw=1.5, label=clabel)

        q_lo, med, q_hi = np.quantile(delta, [lo, 0.5, hi])
        p_gt0 = float((delta > 0).mean())
        bar_y = -0.8 * (i + 1)
        ax2.plot([q_lo, q_hi], [bar_y, bar_y], color=ccolor, lw=3, solid_capstyle="round")
        ax2.plot(med, bar_y, "o", color=ccolor, ms=6, zorder=5)

        # Annotate P(Δ>0)
        ax2.annotate(
            f"P(Δ>0) = {p_gt0:.2f}",
            xy=(q_hi + 0.005, bar_y), fontsize=9, color=ccolor,
            va="center",
        )

    ax2.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax2.set_xlabel("Δβ (contrast)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Stage contrasts", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.set_yticks([])

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"stage_posteriors_{args.cohort}_{args.predictor}.png"
    fig.savefig(out, dpi=180)
    log.info(f"Saved figure: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
