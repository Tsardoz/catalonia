"""Timing scan for olive yield vs rolling deficit-day counts from P - ET0."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).parent))
from features import phase_for_agroyear_mmdd


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

CLIMATE_CSV = DATA / "agera5_daily_hybrid_olive.csv"
YIELD_CSV = DATA / "catalan_olive_yield_climate_hybrid.csv"

DEFICIT_THRESHOLD_MM = 0.0
WINDOW_DAYS = 14
REF_YEAR_DEC = 2000
REF_YEAR_JAN = 2001
SCAN_START = "12-01"
SCAN_END = "11-30"


def load_yield() -> pd.DataFrame:
    cols = ["comarca", "year", "yield_tha"]
    return pd.read_csv(YIELD_CSV, usecols=cols).drop_duplicates().copy()


def load_climate() -> pd.DataFrame:
    df = pd.read_csv(
        CLIMATE_CSV,
        parse_dates=["date"],
        usecols=["date", "comarca", "precip_mean", "et0_mean"],
    )
    df["agro_year"] = np.where(df["date"].dt.month == 12, df["date"].dt.year + 1, df["date"].dt.year)
    df["anchor_key"] = df["date"].dt.strftime("%m-%d")
    df["cwb"] = df["precip_mean"] - df["et0_mean"]
    df["deficit_day"] = (df["cwb"] < DEFICIT_THRESHOLD_MM).astype(float)
    return df


def weekly_anchor_dates() -> list[pd.Timestamp]:
    return list(pd.date_range(f"{REF_YEAR_DEC}-12-01", f"{REF_YEAR_DEC}-12-31", freq="7D")) + list(
        pd.date_range(f"{REF_YEAR_JAN}-01-01", f"{REF_YEAR_JAN}-11-30", freq="7D")
    )


def phase_for_mmdd(mmdd: str) -> str:
    """Convert mm-dd to olive phenological phase for agronomic year scan."""
    return phase_for_agroyear_mmdd(mmdd)


def build_feature_table(climate: pd.DataFrame) -> pd.DataFrame:
    anchors = weekly_anchor_dates()
    rows = []
    for (comarca, agro_year), grp in climate.groupby(["comarca", "agro_year"], sort=True):
        grp = grp.sort_values("date").copy()
        grp["deficit_14d"] = grp["deficit_day"].rolling(WINDOW_DAYS, min_periods=WINDOW_DAYS).sum()
        row = {"comarca": comarca, "year": agro_year}
        for anchor in anchors:
            key = anchor.strftime("%m%d")
            sub = grp.loc[grp["anchor_key"] == anchor.strftime("%m-%d"), "deficit_14d"]
            row[f"def14_{key}"] = float(sub.iloc[0]) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fit_single_feature(df: pd.DataFrame, feature_col: str) -> dict:
    sub = df[["comarca", "yield_tha", feature_col]].dropna().copy()
    if sub.empty or sub[feature_col].nunique() < 2:
        return {}
    comarca_dummies = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float)
    X = pd.concat([sub[[feature_col]].reset_index(drop=True), comarca_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(sub["yield_tha"].to_numpy(), X).fit()
    anchor_mmdd = feature_col.split("_")[-1]
    anchor_year = REF_YEAR_DEC if anchor_mmdd.startswith("12") else REF_YEAR_JAN
    anchor_date = pd.to_datetime(f"{anchor_year}-{anchor_mmdd[:2]}-{anchor_mmdd[2:]}")
    return {
        "feature": feature_col,
        "anchor_mmdd": anchor_mmdd,
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "anchor_label": anchor_date.strftime("%b %d"),
        "phase": phase_for_mmdd(anchor_mmdd),
        "coef": float(model.params[feature_col]),
        "stderr": float(model.bse[feature_col]),
        "pvalue": float(model.pvalues[feature_col]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "n": int(len(sub)),
        "mean_deficit_days_14d": float(sub[feature_col].mean()),
        "sd_deficit_days_14d": float(sub[feature_col].std()),
    }


def fit_scan(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c.startswith("def14_")]
    records = [fit_single_feature(df, col) for col in sorted(feature_cols)]
    records = [r for r in records if r]
    return pd.DataFrame(records).sort_values("anchor_date").reset_index(drop=True)


def add_phase_bands(ax, results: pd.DataFrame) -> None:
    phase_colors = {
        "winter": "#ebedef",     # Light gray - Winter/pre-growth
        "pre_flower": "#e8daef", # Light purple - Spring build-up
        "sp1": "#d6eaf8",        # Light blue - SP1: Flowering to pit hardening
        "np": "#fdebd0",         # Light orange - NP: Pit hardening
        "sp2": "#d5f5e3",        # Light green - SP2: Oil synthesis to harvest
        "other": "#f5f5f5",
    }
    phase_labels = {
        "winter": "Winter / pre-growth",
        "pre_flower": "Spring build-up",
        "sp1": "SP1: Flowering → pit hardening",
        "np": "NP: Pit hardening",
        "sp2": "SP2: Oil synthesis → harvest",
    }
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    phases = results["phase"].tolist()
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            phase = phases[start]
            x0, x1 = start - 0.5, i - 0.5
            ax.axvspan(x0, x1, color=phase_colors.get(phase, "#eeeeee"), alpha=0.18, zorder=0)
            ax.text((x0 + x1) / 2, 0.97, phase_labels.get(phase, phase.replace("_", " ")), transform=trans,
                    ha="center", va="top", fontsize=8, color="#34495e", fontweight="bold")
            # Add vertical dashed line at phase boundary
            if i < len(phases):
                ax.axvline(i - 0.5, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
            start = i


def set_y_limits(ax, values: pd.Series) -> None:
    coef_abs = max(float(np.abs(values).max()), 1e-6)
    pad = max(coef_abs * 0.28, 0.01)
    ax.set_ylim(-(coef_abs + pad), coef_abs + pad)


def add_sig_stars(ax, results: pd.DataFrame) -> None:
    for i, row in results.reset_index(drop=True).iterrows():
        if row["pvalue"] < 0.001:
            star = "***"
        elif row["pvalue"] < 0.01:
            star = "**"
        elif row["pvalue"] < 0.05:
            star = "*"
        else:
            continue
        ax.annotate(
            star,
            xy=(i, row["coef"]),
            xytext=(0, 5 if row["coef"] >= 0 else -7),
            textcoords="offset points",
            ha="center",
            va="bottom" if row["coef"] >= 0 else "top",
            fontsize=8,
            clip_on=True,
            annotation_clip=True,
        )


def plot_results(results: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.2))
    x = np.arange(len(results))
    add_phase_bands(ax, results)
    colours = np.where(results["pvalue"] < 0.05, "#1f618d", "#bfc9ca")
    ax.bar(x, results["coef"], color=colours, edgecolor="none", width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    set_y_limits(ax, results["coef"])
    ax.set_ylabel("OLS coefficient")
    ax.set_xlabel("Fortnight ending on")
    ax.set_title(
        f"Olive yield vs 14-day count of days with P - ET₀ < {DEFICIT_THRESHOLD_MM:.0f}\n"
        "Hybrid climate, weekly agronomic-year timing scan, comarca FE"
    )
    ax.spines[["top", "right"]].set_visible(False)
    tick_idx = x if len(results) <= 30 else x[::2]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(results.loc[tick_idx, "anchor_label"], rotation=40, ha="right", fontsize=8)
    add_sig_stars(ax, results)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    yield_df = load_yield()
    climate_df = load_climate()
    features = build_feature_table(climate_df)
    sample = yield_df.merge(features, on=["comarca", "year"], how="inner")

    print(f"Sample: {len(sample)} rows, {sample['comarca'].nunique()} comarques, agronomic years {sample['year'].min()}-{sample['year'].max()}")
    print(f"Scan: weekly anchors across agronomic year {SCAN_START} to {SCAN_END}")
    print(f"Feature: {WINDOW_DAYS}-day rolling count of days with P - ET0 < {DEFICIT_THRESHOLD_MM:.0f}")

    results = fit_scan(sample)
    out_csv = DATA / "olive_cwb_deficit_timing_scan.csv"
    out_fig = FIG / "olive_cwb_deficit_timing_scan.png"
    results.to_csv(out_csv, index=False)
    plot_results(results, out_fig)

    print(f"Saved {out_csv}")
    print(f"Saved {out_fig}")
    print("\nTop windows by p-value:")
    print(results.sort_values(["pvalue", "coef"], ascending=[True, False]).head(12).to_string(index=False))