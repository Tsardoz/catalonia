"""Create a polished summary figure for olive timing results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

TMAX_CSV = DATA / "olive_tmax_timing_scan.csv"
VPD_CSV = DATA / "olive_vpd_timing_scan.csv"
CWB_CSV = DATA / "olive_cwb_timing_scan.csv"
OUT_FIG = FIG / "olive_timing_summary.png"


def add_phase_bands(ax, df: pd.DataFrame) -> None:
    phase_colors = {
        "winter": "#ebedef",
        "pre_flower": "#e8daef",
        "flower": "#d6eaf8",
        "fruit_set": "#fdebd0",
        "maturation": "#d5f5e3",
    }
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    phases = df["phase"].tolist()
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            phase = phases[start]
            x0, x1 = start - 0.5, i - 0.5
            ax.axvspan(x0, x1, color=phase_colors.get(phase, "#eeeeee"), alpha=0.16, zorder=0)
            ax.text((x0 + x1) / 2, 0.97, phase.replace("_", " "), transform=trans,
                    ha="center", va="top", fontsize=8, color="#34495e", fontweight="bold")
            start = i


def add_sig_stars(ax, df: pd.DataFrame) -> None:
    for i, row in df.reset_index(drop=True).iterrows():
        if row["pvalue"] < 0.001:
            star = "***"
        elif row["pvalue"] < 0.01:
            star = "**"
        elif row["pvalue"] < 0.05:
            star = "*"
        else:
            continue
        yoff = 0.012 if row["coef"] >= 0 else -0.014
        ax.text(i, row["coef"] + yoff, star, ha="center",
                va="bottom" if row["coef"] >= 0 else "top", fontsize=7)


def plot_panel(ax, df: pd.DataFrame, title: str, color: str, ylabel: str, highlight=None) -> None:
    x = np.arange(len(df))
    add_phase_bands(ax, df)
    if highlight is not None:
        x0, x1, label = highlight
        ax.axvspan(x0 - 0.5, x1 + 0.5, color="#f5b7b1", alpha=0.22, zorder=1)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text((x0 + x1) / 2, 0.08, label, transform=trans, ha="center", va="bottom",
                fontsize=8, color="#922b21", fontweight="bold")
    bar_colors = np.where(df["pvalue"] < 0.05, color, "#cfd8dc")
    ax.bar(x, df["coef"], color=bar_colors, edgecolor="none", width=0.82, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(x)
    step = 1 if len(df) <= 24 else 2
    keep = x[::step]
    ax.set_xticks(keep)
    ax.set_xticklabels(df.loc[keep, "anchor_label"], rotation=40, ha="right", fontsize=8)
    add_sig_stars(ax, df)


def main() -> None:
    tmax = pd.read_csv(TMAX_CSV)
    vpd = pd.read_csv(VPD_CSV)
    cwb = pd.read_csv(CWB_CSV)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    plot_panel(
        axes[0],
        tmax,
        "Heat timing scan: 14-day count of Tmax ≥ 32°C",
        color="#c0392b",
        ylabel="OLS coefficient",
        highlight=(10, 14, "late Jul–late Aug peak"),
    )
    plot_panel(
        axes[1],
        vpd,
        "Dry-air timing scan: 14-day count of VPD ≥ 3.0 kPa",
        color="#af601a",
        ylabel="OLS coefficient",
        highlight=(10, 14, "late Jul–late Aug peak"),
    )
    plot_panel(
        axes[2],
        cwb,
        "Background context: 14-day sum of P − ET₀ across agronomic year",
        color="#1f618d",
        ylabel="OLS coefficient",
        highlight=(2, 5, "winter recharge signal"),
    )
    axes[2].set_xlabel("Window end date")

    fig.suptitle(
        "Olive timing summary on hybrid daily climate\n"
        "Heat and dry-air signals peak in late July–late August; P − ET₀ is more useful as winter context",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()