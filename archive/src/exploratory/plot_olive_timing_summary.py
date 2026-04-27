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


def add_phase_bands(ax, df: pd.DataFrame, compact_labels: bool = False) -> None:
    phase_colors = {
        "winter": "#ebedef",     # Light gray - Winter/pre-growth
        "pre_flower": "#e8daef", # Light purple - Spring build-up
        "sp1": "#d6eaf8",        # Light blue - SP1: Flowering to pit hardening
        "np": "#fdebd0",         # Light orange - NP: Pit hardening
        "sp2": "#d5f5e3",        # Light green - SP2: Oil synthesis to harvest
    }
    phase_labels = {
        "winter": "Winter / pre-growth",
        "pre_flower": "Spring build-up",
        "sp1": "SP1: Flowering → pit hardening",
        "np": "NP: Pit hardening",
        "sp2": "SP2: Oil synthesis → harvest",
    }
    if compact_labels:
        phase_labels = {
            "winter": "Winter",
            "pre_flower": "Spring",
            "sp1": "SP1",
            "np": "NP",
            "sp2": "SP2",
            "other": "Other",
        }
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    phases = df["phase"].tolist()
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            phase = phases[start]
            x0, x1 = start - 0.5, i - 0.5
            ax.axvspan(x0, x1, color=phase_colors.get(phase, "#eeeeee"), alpha=0.16, zorder=0)
            ax.text(
                (x0 + x1) / 2,
                0.97,
                phase_labels.get(phase, phase.replace("_", " ")),
                transform=trans,
                ha="center",
                va="top",
                fontsize=7 if compact_labels else 8,
                color="#34495e",
                fontweight="bold",
            )
            # Add vertical dashed line at phase boundary
            if i < len(phases):
                ax.axvline(i - 0.5, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            start = i


def set_y_limits(ax, values: pd.Series) -> None:
    coef_abs = max(float(np.abs(values).max()), 1e-6)
    pad = max(coef_abs * 0.28, 0.0015 if coef_abs < 0.02 else 0.01)
    ax.set_ylim(-(coef_abs + pad), coef_abs + pad)


def add_sig_stars(ax, df: pd.DataFrame) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    y_pad = 0.035 * y_span
    y_cap = 0.06 * y_span
    for i, row in df.reset_index(drop=True).iterrows():
        if row["pvalue"] < 0.001:
            star = "***"
        elif row["pvalue"] < 0.01:
            star = "**"
        elif row["pvalue"] < 0.05:
            star = "*"
        else:
            continue
        if row["coef"] >= 0:
            y = min(row["coef"] + y_pad, y_max - y_cap)
            va = "bottom"
        else:
            y = max(row["coef"] - y_pad, y_min + y_cap)
            va = "top"
        ax.text(
            i,
            y,
            star,
            ha="center",
            va=va,
            fontsize=7,
            color="#2c3e50",
            fontweight="bold",
            zorder=3,
        )


def plot_panel(
    ax,
    df: pd.DataFrame,
    title: str,
    color: str,
    ylabel: str,
    compact_phase_labels: bool = False,
    tick_step: int | None = None,
) -> None:
    x = np.arange(len(df))
    add_phase_bands(ax, df, compact_labels=compact_phase_labels)
    bar_colors = np.where(df["pvalue"] < 0.05, color, "#cfd8dc")
    ax.bar(x, df["coef"], color=bar_colors, edgecolor="none", width=0.82, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8)
    set_y_limits(ax, df["coef"])
    add_sig_stars(ax, df)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(x)
    step = tick_step if tick_step is not None else (1 if len(df) <= 24 else 2)
    keep = x[::step]
    ax.set_xticks(keep)
    ax.set_xticklabels(df.loc[keep, "anchor_label"], rotation=40, ha="right", fontsize=8)


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
    )
    plot_panel(
        axes[1],
        vpd,
        "Dry-air timing scan: 14-day count of VPD ≥ 3.0 kPa",
        color="#af601a",
        ylabel="OLS coefficient",
    )
    plot_panel(
        axes[2],
        cwb,
        "Background context: 14-day sum of P − ET₀ across agronomic year",
        color="#1f618d",
        ylabel="OLS coefficient",
        compact_phase_labels=True,
        tick_step=3,
    )
    axes[2].set_xlabel("Fortnight ending on")

    fig.suptitle(
        "Olive timing summary on hybrid daily climate\n"
        "Heat and dry-air signals peak in midsummer; P − ET₀ is more useful as winter context",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.985, 0.012, "* p<0.05   ** p<0.01   *** p<0.001", ha="right", fontsize=9, color="#34495e")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()