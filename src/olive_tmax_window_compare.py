"""Compare 7/14/21-day Tmax timing scans for olive yield using hybrid daily climate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.transforms import blended_transform_factory


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

CLIMATE_CSV = DATA / "agera5_daily_hybrid_olive.csv"
YIELD_CSV = DATA / "catalan_olive_yield_climate_hybrid.csv"

HOT_THRESHOLD_C = 32.0
WINDOW_DAYS_LIST = [7, 14, 21]
REF_YEAR = 2001
SCAN_START = "05-01"
SCAN_END = "09-30"


def load_yield() -> pd.DataFrame:
    return pd.read_csv(YIELD_CSV, usecols=["comarca", "year", "yield_tha"]).drop_duplicates()


def load_climate() -> pd.DataFrame:
    df = pd.read_csv(CLIMATE_CSV, parse_dates=["date"], usecols=["date", "comarca", "tmax_mean"])
    df["year"] = df["date"].dt.year
    df["anchor_key"] = df["date"].dt.strftime("%m-%d")
    df["hot_day"] = (df["tmax_mean"] >= HOT_THRESHOLD_C).astype(float)
    return df


def weekly_anchor_dates() -> list[pd.Timestamp]:
    return list(pd.date_range(f"{REF_YEAR}-{SCAN_START}", f"{REF_YEAR}-{SCAN_END}", freq="7D"))


def phase_for_mmdd(mmdd: str) -> str:
    month = int(mmdd[:2])
    if month in {5, 6}:
        return "flower"
    if month in {7, 8}:
        return "fruit_set"
    return "maturation"


def build_scan_table(climate: pd.DataFrame, yield_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    anchors = weekly_anchor_dates()
    rows = []
    for (comarca, year), grp in climate.groupby(["comarca", "year"], sort=True):
        grp = grp.sort_values("date").copy()
        grp["hot_roll"] = grp["hot_day"].rolling(window_days, min_periods=window_days).sum()
        row = {"comarca": comarca, "year": year}
        for anchor in anchors:
            mmdd = anchor.strftime("%m-%d")
            key = anchor.strftime("%m%d")
            sub = grp.loc[grp["anchor_key"] == mmdd, "hot_roll"]
            row[f"hot_{key}"] = float(sub.iloc[0]) if len(sub) else np.nan
        rows.append(row)
    sample = yield_df.merge(pd.DataFrame(rows), on=["comarca", "year"], how="inner")

    records = []
    for col in sorted(c for c in sample.columns if c.startswith("hot_")):
        sub = sample[["comarca", "yield_tha", col]].dropna().copy()
        if sub.empty or sub[col].nunique() < 2:
            continue
        X = pd.concat([sub[[col]].reset_index(drop=True), pd.get_dummies(sub["comarca"], drop_first=True, dtype=float).reset_index(drop=True)], axis=1)
        model = sm.OLS(sub["yield_tha"].to_numpy(), sm.add_constant(X)).fit()
        mmdd = col.split("_")[-1]
        anchor = pd.to_datetime(f"{REF_YEAR}-{mmdd[:2]}-{mmdd[2:]}")
        records.append({
            "window_days": window_days,
            "anchor_mmdd": mmdd,
            "anchor_date": anchor.strftime("%Y-%m-%d"),
            "anchor_label": anchor.strftime("%b %d"),
            "phase": phase_for_mmdd(mmdd),
            "coef": float(model.params[col]),
            "stderr": float(model.bse[col]),
            "pvalue": float(model.pvalues[col]),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "mean_hot_days": float(sub[col].mean()),
        })
    return pd.DataFrame(records).sort_values(["window_days", "anchor_date"]).reset_index(drop=True)


def add_phase_bands(ax, sub: pd.DataFrame) -> None:
    colors = {"flower": "#d6eaf8", "fruit_set": "#fdebd0", "maturation": "#d5f5e3"}
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    phases = sub["phase"].tolist()
    start = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            phase = phases[start]
            x0, x1 = start - 0.5, i - 0.5
            ax.axvspan(x0, x1, color=colors[phase], alpha=0.18, zorder=0)
            ax.text((x0 + x1) / 2, 0.97, phase.replace("_", " "), transform=trans,
                    ha="center", va="top", fontsize=8, color="#34495e", fontweight="bold")
            start = i


def plot_compare(results: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(len(WINDOW_DAYS_LIST), 1, figsize=(12, 10), sharex=True)
    for ax, window_days in zip(axes, WINDOW_DAYS_LIST):
        sub = results.loc[results["window_days"] == window_days].reset_index(drop=True)
        add_phase_bands(ax, sub)
        x = np.arange(len(sub))
        ax.bar(x, sub["coef"], color=np.where(sub["pvalue"] < 0.05, "#c0392b", "#bfc9ca"), edgecolor="none", width=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("OLS coef")
        ax.set_title(f"{window_days}-day count of Tmax ≥ {HOT_THRESHOLD_C:.0f}°C", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for i, row in sub.iterrows():
            if row["pvalue"] < 0.001: star = "***"
            elif row["pvalue"] < 0.01: star = "**"
            elif row["pvalue"] < 0.05: star = "*"
            else: continue
            yoff = 0.015 if row["coef"] >= 0 else -0.02
            ax.text(i, row["coef"] + yoff, star, ha="center", va="bottom" if row["coef"] >= 0 else "top", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["anchor_label"], rotation=40, ha="right", fontsize=8)
    fig.suptitle("Olive yield vs rolling hot-day counts across timing windows\nHybrid climate, comarca FE", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    climate = load_climate()
    yield_df = load_yield()
    results = pd.concat([build_scan_table(climate, yield_df, w) for w in WINDOW_DAYS_LIST], ignore_index=True)
    out_csv = DATA / "olive_tmax_window_compare.csv"
    out_fig = FIG / "olive_tmax_window_compare.png"
    results.to_csv(out_csv, index=False)
    plot_compare(results, out_fig)
    print(f"Saved {out_csv}")
    print(f"Saved {out_fig}")
    for w in WINDOW_DAYS_LIST:
        print(f"\nTop negative windows for {w}d:")
        print(results.loc[results['window_days'] == w].sort_values(['pvalue', 'coef']).head(6).to_string(index=False))