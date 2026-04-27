"""Simple summer VPD consecutive-run screen for olive yield using hybrid daily climate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

CLIMATE_CSV = DATA / "agera5_daily_hybrid_olive.csv"
YIELD_CSV = DATA / "catalan_olive_yield_climate_hybrid.csv"

SUMMER_START = "06-20"
SUMMER_END = "08-15"
THRESHOLDS = [2.0, 2.5, 3.0, 3.5]


def load_yield() -> pd.DataFrame:
    cols = ["comarca", "year", "yield_tha"]
    return pd.read_csv(YIELD_CSV, usecols=cols).drop_duplicates().copy()


def load_climate() -> pd.DataFrame:
    df = pd.read_csv(CLIMATE_CSV, parse_dates=["date"], usecols=["date", "comarca", "vpd_mean"])
    df["year"] = df["date"].dt.year
    mmdd = df["date"].dt.strftime("%m-%d")
    return df.loc[(mmdd >= SUMMER_START) & (mmdd <= SUMMER_END)].copy()


def longest_run(mask: pd.Series) -> int:
    arr = mask.to_numpy(dtype=int)
    best = cur = 0
    for val in arr:
        if val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def build_feature_table(climate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comarca, year), grp in climate.groupby(["comarca", "year"], sort=True):
        grp = grp.sort_values("date")
        row = {"comarca": comarca, "year": year}
        for thr in THRESHOLDS:
            key = f"vpd_run_ge_{str(thr).replace('.', 'p')}"
            row[key] = longest_run(grp["vpd_mean"] >= thr)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_single_feature(df: pd.DataFrame, feature_col: str) -> dict:
    sub = df[["comarca", "yield_tha", feature_col]].dropna().copy()
    comarca_dummies = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float)
    X = pd.concat([sub[[feature_col]].reset_index(drop=True), comarca_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(sub["yield_tha"].to_numpy(), X).fit()
    return {
        "feature": feature_col,
        "threshold_kpa": float(feature_col.split("_")[-1].replace("p", ".")),
        "coef": float(model.params[feature_col]),
        "stderr": float(model.bse[feature_col]),
        "pvalue": float(model.pvalues[feature_col]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "n": int(len(sub)),
        "mean_run_days": float(sub[feature_col].mean()),
        "sd_run_days": float(sub[feature_col].std()),
    }


def plot_results(results: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(results))
    colours = np.where(results["pvalue"] < 0.05, "#c0392b", "#95a5a6")
    ax.bar(x, results["coef"], color=colours, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f">= {t:.1f}" for t in results["threshold_kpa"]])
    ax.set_ylabel("OLS coefficient")
    ax.set_xlabel("Daily VPD threshold (kPa)")
    ax.set_title(f"Olive yield vs longest summer VPD run\n{SUMMER_START} to {SUMMER_END}, comarca FE")
    ax.spines[["top", "right"]].set_visible(False)
    for i, row in results.reset_index(drop=True).iterrows():
        if row["pvalue"] < 0.001:
            star = "***"
        elif row["pvalue"] < 0.01:
            star = "**"
        elif row["pvalue"] < 0.05:
            star = "*"
        else:
            star = ""
        if star:
            yoff = 0.01 if row["coef"] >= 0 else -0.01
            ax.text(i, row["coef"] + yoff, star, ha="center",
                    va="bottom" if row["coef"] >= 0 else "top", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    yield_df = load_yield()
    climate_df = load_climate()
    features = build_feature_table(climate_df)
    sample = yield_df.merge(features, on=["comarca", "year"], how="inner")

    print(f"Sample: {len(sample)} rows, {sample['comarca'].nunique()} comarques, years {sample['year'].min()}-{sample['year'].max()}")
    print(f"Summer window: {SUMMER_START} to {SUMMER_END}")

    results = []
    for thr in THRESHOLDS:
        col = f"vpd_run_ge_{str(thr).replace('.', 'p')}"
        results.append(fit_single_feature(sample, col))
    results = pd.DataFrame(results).sort_values("threshold_kpa").reset_index(drop=True)

    out_csv = DATA / "olive_summer_vpd_run_screen.csv"
    out_fig = FIG / "olive_summer_vpd_run_screen.png"
    results.to_csv(out_csv, index=False)
    plot_results(results, out_fig)

    print(f"Saved {out_csv}")
    print(f"Saved {out_fig}")
    print("\nResults:")
    print(results.to_string(index=False))