"""Scatter: rainfed yield vs. seasonal water balance (P - ET0).

Seasonal window = agro_year (Oct prev year -> Sep current year), summed daily
p_minus_et0 from data/agera5_water_balance_catalonia.csv. Joined to rainfed
yields in data/catalan_woody_yield_raw.csv on (comarca, year == agro_year).

One panel per crop_group, OLS line + R^2 per panel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
WB_PATH = ROOT / "data" / "agera5_water_balance_catalonia.csv"
YIELD_PATH = ROOT / "data" / "catalan_woody_yield_raw.csv"
OUT_PATH = ROOT / "figures" / "yield_vs_seasonal_water_balance.png"


def main() -> None:
    wb = pd.read_csv(WB_PATH, usecols=["comarca", "agro_year", "p_minus_et0_daily"])
    seasonal = (
        wb.groupby(["comarca", "agro_year"], as_index=False)["p_minus_et0_daily"]
        .sum()
        .rename(columns={"p_minus_et0_daily": "wb_season_mm", "agro_year": "year"})
    )

    y = pd.read_csv(YIELD_PATH, usecols=["year", "comarca", "crop_group", "crop_en", "yield_tha"])
    y = y[y["yield_tha"] > 0].copy()

    df = y.merge(seasonal, on=["comarca", "year"], how="inner")
    print(f"Merged rows: {len(df):,}  comarcas: {df.comarca.nunique()}  years: {sorted(df.year.unique())}")

    groups = sorted(df["crop_group"].dropna().unique())
    ncols = 3
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, g in zip(axes.ravel(), groups):
        sub = df[df["crop_group"] == g]
        sc = ax.scatter(
            sub["wb_season_mm"], sub["yield_tha"],
            c=sub["year"], cmap="viridis", s=22, alpha=0.75, edgecolor="none",
        )
        if len(sub) > 2:
            slope, intercept, r, p, _ = stats.linregress(sub["wb_season_mm"], sub["yield_tha"])
            xs = np.linspace(sub["wb_season_mm"].min(), sub["wb_season_mm"].max(), 100)
            ax.plot(xs, intercept + slope * xs, "k--", lw=1)
            ax.text(0.04, 0.96, f"R²={r**2:.2f}  p={p:.2g}\nn={len(sub)}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
        ax.set_title(g)
        ax.set_xlabel("Seasonal P − ET₀ (mm, Oct–Sep)")
        ax.set_ylabel("Rainfed yield (t/ha)")
        ax.axvline(0, color="0.7", lw=0.8)

    # Hide unused panels
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)

    # Shared colourbar for year
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Year")

    fig.suptitle("Rainfed yield vs. seasonal climatic water balance (P − ET₀)", fontsize=13)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
