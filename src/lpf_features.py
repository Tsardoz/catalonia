"""
lpf_features.py  —  Step 6b
IIR low-pass filter feature extraction for yield prediction.

For each daily climate variable (precip_mean, et0_mean, vpd_mean, vpd_max, tmax_mean) and each
τ in [3, 7, 14, 30] days, applies a first-order IIR filter continuously per
comarca, then extracts:

  Peak value  — max of the filtered signal within each phenological window
                (per window × variable × τ × pheno_key)

  Peak timing — DOY at which the filtered signal peaks across the full
                Oct(Y-1)–Sep(Y) agronomic year (per variable × τ)

Features are joined with rainfed yield data, Pearson R² is computed for each
feature vs yield_tha (per crop), ranked, and the top-5 scatter plots saved.

Output:
  data/lpf_features.csv          — wide feature table joined with yield
  figures/lpf_top5_scatter.png   — scatter plots of top-5 features vs yield
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR  = Path(__file__).parent.parent / "figures"

# Import PHENO_WINDOWS from aggregate_seasonal — single source of truth
sys.path.insert(0, str(Path(__file__).parent))
from aggregate_seasonal import PHENO_WINDOWS  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────
VARIABLES = ["precip_mean", "et0_mean", "vpd_mean", "vpd_max", "tmax_mean"]
TAUS      = [3, 7, 14, 30]



# ── IIR filter ─────────────────────────────────────────────────────────────
def iir_filter(x: np.ndarray, tau: float) -> np.ndarray:
    """First-order IIR low-pass filter applied causally."""
    alpha = 1.0 - np.exp(-1.0 / tau)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def apply_filters(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IIR filter per comarca for all variables × τ combinations.
    Returns daily DataFrame with additional columns {var}_lpf{tau}.
    """
    out_frames = []
    for comarca, grp in daily.groupby("comarca"):
        grp = grp.sort_values("date").copy()
        for var in VARIABLES:
            x = grp[var].fillna(0.0).values
            for tau in TAUS:
                grp[f"{var}_lpf{tau}"] = iir_filter(x, tau)
        out_frames.append(grp)
    return pd.concat(out_frames, ignore_index=True)


# ── Feature extraction ──────────────────────────────────────────────────────
def extract_peak_value_features(daily: pd.DataFrame, pheno_key: str) -> pd.DataFrame:
    """Peak of filtered signal within each phenological window."""
    windows = PHENO_WINDOWS[pheno_key]
    rows = []
    for (comarca, year), grp in daily.groupby(["comarca", "year"]):
        row = {"comarca": comarca, "year": year}
        for window, months in windows.items():
            mask = grp["month"].isin(months)
            sub  = grp[mask]
            for var in VARIABLES:
                for tau in TAUS:
                    col = f"{var}_lpf{tau}"
                    vals = sub[col].values
                    row[f"{window}_{var}_lpf{tau}_peak"] = (
                        float(vals.max()) if len(vals) > 0 else np.nan
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def extract_timing_features(daily: pd.DataFrame) -> pd.DataFrame:
    """DOY of filtered signal peak across the Oct(Y-1)–Sep(Y) agronomic year."""
    rows = []
    years = sorted(daily["year"].unique())
    for comarca, grp in daily.groupby("comarca"):
        grp = grp.sort_values("date")
        for year in years:
            # Oct Y-1 → Sep Y
            start = pd.Timestamp(year - 1, 10, 1)
            end   = pd.Timestamp(year, 9, 30)
            mask  = (grp["date"] >= start) & (grp["date"] <= end)
            sub   = grp[mask]
            if len(sub) == 0:
                continue
            row = {"comarca": comarca, "year": year}
            for var in VARIABLES:
                for tau in TAUS:
                    col  = f"{var}_lpf{tau}"
                    vals = sub[col].values
                    doys = sub["date"].dt.dayofyear.values
                    row[f"{var}_lpf{tau}_peak_doy"] = (
                        int(doys[np.argmax(vals)]) if len(vals) > 0 else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows)

# ── Correlation analysis ────────────────────────────────────────────────────
def compute_r2_table(features: pd.DataFrame, yield_df: pd.DataFrame,
                     pheno_key: str) -> pd.DataFrame:
    """Pearson R² for every feature column vs yield_tha."""
    crop_yield = yield_df[yield_df["pheno_key"] == pheno_key].copy()
    merged = crop_yield.merge(features, on=["comarca", "year"], how="inner")

    feat_cols = [c for c in features.columns if c not in ("comarca", "year")]
    records = []
    for col in feat_cols:
        y = merged["yield_tha"].values
        x = merged[col].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
        records.append({"feature": col, "r2": r**2, "slope": slope,
                        "n": int(mask.sum()), "p": p})
    return (pd.DataFrame(records)
              .sort_values("r2", ascending=False)
              .reset_index(drop=True))


def plot_top5(features: pd.DataFrame, yield_df: pd.DataFrame,
              pheno_key: str, r2_table: pd.DataFrame) -> Path:
    """Scatter plots of top-5 features vs yield, coloured by year."""
    crop_yield = yield_df[yield_df["pheno_key"] == pheno_key].copy()
    merged = crop_yield.merge(features, on=["comarca", "year"], how="inner")

    top5 = r2_table.head(5)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
    fig.suptitle(f"Top-5 LPF features vs rainfed {pheno_key} yield\n"
                 f"(Catalonia comarques, 2015–2024)",
                 fontsize=11, fontweight="bold")

    years = sorted(merged["year"].unique())
    cmap  = plt.get_cmap("viridis", len(years))
    yc    = {yr: cmap(i) for i, yr in enumerate(years)}

    for ax, (_, row) in zip(axes, top5.iterrows()):
        col  = row["feature"]
        x    = merged[col].values
        y    = merged["yield_tha"].values
        mask = np.isfinite(x) & np.isfinite(y)

        ax.scatter(x, y, c=[yc[yr] for yr in merged["year"]],
                   s=20, alpha=0.75, zorder=3)

        slope, intercept, *_ = stats.linregress(x[mask], y[mask])
        xl = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xl, slope * xl + intercept, "k--", lw=1.5, zorder=4)

        ax.set_title(col.replace("_", "\n"), fontsize=7)
        ax.set_ylabel("Yield (t/ha)" if ax is axes[0] else "")
        ax.text(0.05, 0.95, f"R²={row['r2']:.3f}  n={row['n']}",
                transform=ax.transAxes, va="top", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap,
         norm=plt.Normalize(vmin=years[0]-0.5, vmax=years[-1]+0.5))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Year", shrink=0.7)

    out = FIG_DIR / f"lpf_top5_{pheno_key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading daily climate data...")
    daily = pd.read_csv(DATA_DIR / "agera5_daily_catalonia.csv",
                        parse_dates=["date"])
    daily["year"]  = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    print(f"  {len(daily):,} rows, {daily['comarca'].nunique()} comarques")

    print("Applying IIR filters (all variables × τ values per comarca)...")
    daily_f = apply_filters(daily)
    print("  Done.")

    print("Loading yield data...")
    yield_df = pd.read_csv(DATA_DIR / "catalan_woody_yield_climate.csv")

    print("Extracting peak timing features (full Oct–Sep year)...")
    timing = extract_timing_features(daily_f)

    all_r2 = []
    for pheno_key in PHENO_WINDOWS:
        crops_in_yield = yield_df[yield_df["pheno_key"] == pheno_key]
        if len(crops_in_yield) == 0:
            continue
        print(f"\n── {pheno_key} ({len(crops_in_yield)} yield records) ──")

        pv    = extract_peak_value_features(daily_f, pheno_key)
        feats = pv.merge(timing, on=["comarca", "year"], how="left")

        r2_tab = compute_r2_table(feats, yield_df, pheno_key)
        if r2_tab.empty:
            continue

        r2_tab["pheno_key"] = pheno_key
        all_r2.append(r2_tab)
        print(r2_tab.head(10).to_string(index=False))

        out_png = plot_top5(feats, yield_df, pheno_key, r2_tab)
        print(f"  Saved {out_png}")

    if all_r2:
        full_table = (pd.concat(all_r2, ignore_index=True)
                        .sort_values("r2", ascending=False))
        out_csv = DATA_DIR / "lpf_r2_table.csv"
        full_table.to_csv(out_csv, index=False)
        print(f"\nFull R² table saved to {out_csv}")
        print("\nTop 20 features across all crops:")
        print(full_table.head(20).to_string(index=False))
