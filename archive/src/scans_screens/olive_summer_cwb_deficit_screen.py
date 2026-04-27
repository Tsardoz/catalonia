"""Simple summer P - ET0 deficit-day screens for olive yield using hybrid daily climate."""

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
THRESHOLDS_MM = [0.0, -2.0, -4.0, -6.0]


def load_yield() -> pd.DataFrame:
    cols = ["comarca", "year", "yield_tha"]
    return pd.read_csv(YIELD_CSV, usecols=cols).drop_duplicates().copy()


def load_climate() -> pd.DataFrame:
    df = pd.read_csv(
        CLIMATE_CSV,
        parse_dates=["date"],
        usecols=["date", "comarca", "precip_mean", "et0_mean"],
    )
    df["year"] = df["date"].dt.year
    mmdd = df["date"].dt.strftime("%m-%d")
    df = df.loc[(mmdd >= SUMMER_START) & (mmdd <= SUMMER_END)].copy()
    df["cwb"] = df["precip_mean"] - df["et0_mean"]
    return df


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
        for thr in THRESHOLDS_MM:
            mask = grp["cwb"] < thr
            key = str(thr).replace("-", "neg").replace(".", "p")
            row[f"cwb_days_lt_{key}"] = int(mask.sum())
            row[f"cwb_run_lt_{key}"] = longest_run(mask)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_single_feature(df: pd.DataFrame, feature_col: str) -> dict:
    sub = df[["comarca", "yield_tha", feature_col]].dropna().copy()
    comarca_dummies = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float)
    X = pd.concat([sub[[feature_col]].reset_index(drop=True), comarca_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(sub["yield_tha"].to_numpy(), X).fit()
    threshold = feature_col.split("_")[-1].replace("neg", "-").replace("p", ".")
    metric = "count" if "_days_" in feature_col else "run"
    return {
        "feature": feature_col,
        "metric": metric,
        "threshold_mm": float(threshold),
        "coef": float(model.params[feature_col]),
        "stderr": float(model.bse[feature_col]),
        "pvalue": float(model.pvalues[feature_col]),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "n": int(len(sub)),
        "mean_value": float(sub[feature_col].mean()),
        "sd_value": float(sub[feature_col].std()),
    }


def plot_panel(ax, results: pd.DataFrame, title: str) -> None:
    x = np.arange(len(results))
    colours = np.where(results["pvalue"] < 0.05, "#1f618d", "#bfc9ca")
    ax.bar(x, results["coef"], color=colours, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"< {t:.0f}" for t in results["threshold_mm"]])
    ax.set_ylabel("OLS coefficient")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(results: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    plot_panel(axes[0], results.loc[results["metric"] == "count"].reset_index(drop=True), "Count of deficit days")
    plot_panel(axes[1], results.loc[results["metric"] == "run"].reset_index(drop=True), "Longest consecutive deficit run")
    fig.suptitle(f"Olive yield vs summer deficit-day screens from P - ET₀\n{SUMMER_START} to {SUMMER_END}, comarca FE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    yield_df = load_yield()
    climate_df = load_climate()
    features = build_feature_table(climate_df)
    sample = yield_df.merge(features, on=["comarca", "year"], how="inner")

    records = []
    for thr in THRESHOLDS_MM:
        key = str(thr).replace("-", "neg").replace(".", "p")
        records.append(fit_single_feature(sample, f"cwb_days_lt_{key}"))
        records.append(fit_single_feature(sample, f"cwb_run_lt_{key}"))
    results = pd.DataFrame(records).sort_values(["metric", "threshold_mm"]).reset_index(drop=True)

    out_csv = DATA / "olive_summer_cwb_deficit_screen.csv"
    out_fig = FIG / "olive_summer_cwb_deficit_screen.png"
    results.to_csv(out_csv, index=False)
    plot_results(results, out_fig)

    print(f"Sample: {len(sample)} rows, {sample['comarca'].nunique()} comarques, years {sample['year'].min()}-{sample['year'].max()}")
    print(f"Summer window: {SUMMER_START} to {SUMMER_END}")
    print("Thresholds (mm day-1):", THRESHOLDS_MM)
    print(f"Saved {out_csv}")
    print(f"Saved {out_fig}")
    print("\nResults:")
    print(results.to_string(index=False))