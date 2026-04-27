"""
plot_scatter.py  —  Step 6
Generate scatter plots for cherry period analysis and cross-crop VPD comparison.

Figure 1: cherry_period_scatter.png
    4-panel grid: rows = VPD / ET0, columns = pre-flower / flower-set / fruit-dev / full-season
    Each point = one comarca-year, coloured by year, OLS line + R²

Figure 2: cross_crop_vpd_yield.png  (placeholder until CropClimateX data is available)
    Panel A: Catalan cherry fruit_dev VPD vs yield_tha
    Panel B: placeholder — replace with CropClimateX soybean pod-fill VPD data
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR    = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

VPD_XLIM  = (0, 4.0)   # kPa — adjust if data range differs
ET0_XLIM  = (0, None)  # mm — auto upper
YIELD_YLIM = (0, None) # t/ha — auto upper


def _ols_line(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r_squared, x_range) for an OLS fit."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    slope, intercept, r, *_ = stats.linregress(x[mask], y[mask])
    x_range = np.linspace(x[mask].min(), x[mask].max(), 100)
    return slope, intercept, r**2, x_range


def _scatter_panel(ax, df: pd.DataFrame, x_col: str, y_col: str = "yield_tha",
                   title: str = "", xlabel: str = "", ylabel: str = "Yield (t/ha)"):
    years = sorted(df["year"].unique())
    cmap  = cm.get_cmap("viridis", len(years))
    year_colour = {y: cmap(i) for i, y in enumerate(years)}

    for _, row in df.iterrows():
        ax.scatter(row[x_col], row[y_col],
                   color=year_colour[row["year"]], s=30, alpha=0.8, zorder=3)

    # OLS fit
    fit = _ols_line(df[x_col].values, df[y_col].values)
    if fit:
        slope, intercept, r2, x_range = fit
        ax.plot(x_range, slope * x_range + intercept, "k--", lw=1.2, zorder=4)
        ax.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes,
                va="top", fontsize=9)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def plot_cherry_periods(df: pd.DataFrame) -> None:
    cherry = df[df["pheno_key"] == "cherry"].copy()
    if cherry.empty:
        print("No cherry data — skipping cherry_period_scatter.png")
        return

    # Full-season VPD = mean of the three period means
    vpd_cols = ["pre_flower_mean_vpd", "flower_set_mean_vpd", "fruit_dev_mean_vpd"]
    et0_cols = ["pre_flower_cum_et0",  "flower_set_cum_et0",  "fruit_dev_cum_et0"]
    cherry["fullseason_mean_vpd"] = cherry[vpd_cols].mean(axis=1)
    cherry["fullseason_cum_et0"]  = cherry[et0_cols].sum(axis=1)

    periods_vpd = [
        ("pre_flower_mean_vpd",  "Pre-flowering (Jan–Feb)"),
        ("flower_set_mean_vpd",  "Flowering–fruit set (Mar–Apr)"),
        ("fruit_dev_mean_vpd",   "Fruit development (May–Jun)"),
        ("fullseason_mean_vpd",  "Full-season mean"),
    ]
    periods_et0 = [
        ("pre_flower_cum_et0",   "Pre-flowering (Jan–Feb)"),
        ("flower_set_cum_et0",   "Flowering–fruit set (Mar–Apr)"),
        ("fruit_dev_cum_et0",    "Fruit development (May–Jun)"),
        ("fullseason_cum_et0",   "Full-season cumulative"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("Cherry (rainfed) — VPD and ET₀ vs yield by phenological period",
                 fontsize=12, fontweight="bold")

    for col_idx, (col, label) in enumerate(periods_vpd):
        ax = axes[0, col_idx]
        _scatter_panel(ax, cherry, col, title=label,
                       xlabel="Mean VPD at Tmax (kPa)",
                       ylabel="Yield (t/ha)" if col_idx == 0 else "")

    for col_idx, (col, label) in enumerate(periods_et0):
        ax = axes[1, col_idx]
        _scatter_panel(ax, cherry, col, title=label,
                       xlabel="Cumulative ET₀ (mm)",
                       ylabel="Yield (t/ha)" if col_idx == 0 else "")

    # Year colourbar
    years = sorted(cherry["year"].unique())
    cmap  = cm.get_cmap("viridis", len(years))
    sm    = cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=years[0] - 0.5, vmax=years[-1] + 0.5))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Year", shrink=0.6, pad=0.01)

    out = FIGURES_DIR / "cherry_period_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_cross_crop(df: pd.DataFrame) -> None:
    cherry = df[(df["pheno_key"] == "cherry") & df["fruit_dev_mean_vpd"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Cross-crop VPD–yield stress response", fontsize=12, fontweight="bold")

    # Panel A: Catalan cherry
    _scatter_panel(axes[0], cherry, "fruit_dev_mean_vpd",
                   title="A: Cherry (Catalonia) — fruit development",
                   xlabel="VPD at Tmax (kPa)", ylabel="Yield (t/ha)")

    # Panel B: placeholder for CropClimateX soybean
    axes[1].text(0.5, 0.5,
                 "Panel B\nCropClimateX soybean\n(pod fill VPD vs yield)\nTo be added",
                 ha="center", va="center", transform=axes[1].transAxes,
                 fontsize=11, color="grey",
                 bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="grey"))
    axes[1].set_title("B: Soybean (US) — pod fill", fontsize=10)
    axes[1].set_xlabel("VPD at Tmax (kPa)", fontsize=9)
    axes[1].set_ylabel("Yield (t/ha)", fontsize=9)

    out = FIGURES_DIR / "cross_crop_vpd_yield.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    data_path = DATA_DIR / "catalan_woody_yield_climate.csv"
    if not data_path.exists():
        raise FileNotFoundError("Run join_dataset.py first to generate catalan_woody_yield_climate.csv")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records")

    plot_cherry_periods(df)
    plot_cross_crop(df)
    print("Done.")
