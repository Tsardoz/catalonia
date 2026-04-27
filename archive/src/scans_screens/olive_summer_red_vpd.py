"""
Pit-hardening rolling red-zone VPD analysis for rainfed olive yield.

Defines red-zone days as comarca-mean daily VPD >= 3.0 kPa, scans weekly anchor
dates through an approximate pit-hardening window, and fits separate OLS
regressions with comarca fixed effects. Pre- and post-summer climatic water
balance controls (P - ET0) are added so the rolling red-day signal is estimated
conditional on spring and late-season water status.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

CLIMATE_CSV = DATA / "agera5_daily_catalonia.csv"
YIELD_CSV = DATA / "catalan_woody_yield_raw.csv"

RED_VPD_KPA = 3.0
ROLLING_WINDOWS = [7, 14, 21]
REF_YEAR = 2001
PRE_CTRL_START = "04-15"
PRE_CTRL_END = "06-15"
PIT_START = "06-20"
PIT_END = "08-15"
POST_CTRL_START = "08-16"
POST_CTRL_END = "10-15"


def load_olive_yield() -> pd.DataFrame:
    df = pd.read_csv(YIELD_CSV)
    return df.loc[df["crop_group"] == "olive", ["comarca", "year", "yield_tha"]].copy()


def load_climate() -> pd.DataFrame:
    df = pd.read_csv(
        CLIMATE_CSV,
        parse_dates=["date"],
        usecols=["date", "comarca", "vpd_mean", "precip_mean", "et0_mean"],
    )
    df["year"] = df["date"].dt.year
    df["red_day"] = (df["vpd_mean"] >= RED_VPD_KPA).astype(float)
    df["cwb"] = df["precip_mean"] - df["et0_mean"]
    return df


def weekly_anchor_dates() -> list[pd.Timestamp]:
    return list(pd.date_range(f"{REF_YEAR}-{PIT_START}", f"{REF_YEAR}-{PIT_END}", freq="7D"))


def seasonal_mask(date_series: pd.Series, start_mmdd: str, end_mmdd: str) -> pd.Series:
    mmdd = date_series.dt.strftime("%m-%d")
    return (mmdd >= start_mmdd) & (mmdd <= end_mmdd)


def build_feature_table(climate: pd.DataFrame) -> pd.DataFrame:
    anchors = weekly_anchor_dates()
    rows = []

    for (comarca, year), grp in climate.groupby(["comarca", "year"]):
        grp = grp.sort_values("date").copy()
        grp["anchor_key"] = grp["date"].dt.strftime("%m-%d")
        for window in ROLLING_WINDOWS:
            grp[f"red_{window}d"] = grp["red_day"].rolling(window, min_periods=window).sum()

        pre = grp.loc[seasonal_mask(grp["date"], PRE_CTRL_START, PRE_CTRL_END)]
        pit = grp.loc[seasonal_mask(grp["date"], PIT_START, PIT_END)]
        post = grp.loc[seasonal_mask(grp["date"], POST_CTRL_START, POST_CTRL_END)]

        row = {"comarca": comarca, "year": year}
        row["pre_cwb_sum"] = float(pre["cwb"].sum())
        row["post_cwb_sum"] = float(post["cwb"].sum())
        row["pit_red_total"] = float(pit["red_day"].sum())
        row["pit_mean_vpd"] = float(pit["vpd_mean"].mean()) if len(pit) else np.nan
        red = pit["red_day"].to_numpy()
        if len(red):
            starts = np.where(np.r_[True, red[1:] != red[:-1]] & (red == 1))[0]
            ends = np.where(np.r_[red[1:] != red[:-1], True] & (red == 1))[0]
            row["pit_longest_red_run"] = int((ends - starts + 1).max()) if len(starts) else 0
        else:
            row["pit_longest_red_run"] = 0

        for anchor in anchors:
            sub = grp.loc[grp["anchor_key"] == anchor.strftime("%m-%d")]
            for window in ROLLING_WINDOWS:
                col = f"red_{window}d_{anchor.strftime('%m%d')}"
                row[col] = float(sub[f"red_{window}d"].iloc[0]) if len(sub) else np.nan
        if len(pit):
            rows.append(row)

    return pd.DataFrame(rows)


def fit_single_feature(df: pd.DataFrame, feature_col: str, control_cols: list[str]) -> dict:
    sub = df[["comarca", "yield_tha", feature_col] + control_cols].dropna().copy()
    if sub.empty or sub[feature_col].nunique() < 2:
        return {}
    comarca_dummies = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float)
    X = pd.concat(
        [sub[[feature_col] + control_cols].reset_index(drop=True), comarca_dummies.reset_index(drop=True)],
        axis=1,
    )
    X = sm.add_constant(X)
    model = sm.OLS(sub["yield_tha"].to_numpy(), X).fit()
    out = {
        "feature": feature_col,
        "coef": float(model.params[feature_col]),
        "stderr": float(model.bse[feature_col]),
        "pvalue": float(model.pvalues[feature_col]),
        "r2": float(model.rsquared),
        "n": int(len(sub)),
        "coef_pre_cwb": float(model.params.get("pre_cwb_sum", np.nan)),
        "pvalue_pre_cwb": float(model.pvalues.get("pre_cwb_sum", np.nan)),
        "coef_post_cwb": float(model.params.get("post_cwb_sum", np.nan)),
        "pvalue_post_cwb": float(model.pvalues.get("post_cwb_sum", np.nan)),
    }
    return out


def fit_feature_scan(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c.startswith("red_") and c[4].isdigit()]
    control_cols = ["pre_cwb_sum", "post_cwb_sum"]
    records = []
    for col in sorted(feature_cols):
        rec = fit_single_feature(df, col, control_cols)
        if not rec:
            continue
        parts = col.split("_")
        rec["window_days"] = int(parts[1].replace("d", ""))
        rec["anchor_mmdd"] = parts[2]
        rec["anchor_date"] = pd.to_datetime(f"{REF_YEAR}-{parts[2][:2]}-{parts[2][2:]}")
        rec["anchor_label"] = rec["anchor_date"].strftime("%b %d")
        records.append(rec)
    return pd.DataFrame(records).sort_values(["window_days", "anchor_date"]).reset_index(drop=True)


def plot_scan(results: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(len(ROLLING_WINDOWS), 1, figsize=(13, 8), sharex=True)
    xlabels = None

    for ax, window in zip(axes, ROLLING_WINDOWS):
        sub = results.loc[results["window_days"] == window].copy()
        x = np.arange(len(sub))
        colours = np.where(sub["pvalue"] < 0.05, "#c0392b", "#bfc9ca")
        ax.bar(x, sub["coef"], color=colours, edgecolor="none", width=0.82)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel("OLS coef")
        ax.set_title(
            f"{window}-day rolling red-zone count (VPD ≥ {RED_VPD_KPA:.1f} kPa)",
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)

        for i, row in sub.reset_index(drop=True).iterrows():
            p = row["pvalue"]
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                continue
            yoff = 0.02 if row["coef"] >= 0 else -0.03
            ax.text(i, row["coef"] + yoff, star, ha="center",
                    va="bottom" if row["coef"] >= 0 else "top", fontsize=8)

        xlabels = sub["anchor_label"].tolist()

    axes[-1].set_xticks(np.arange(len(xlabels)))
    axes[-1].set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
    axes[-1].set_xlabel(f"Weekly window end date within pit hardening ({PIT_START} to {PIT_END})")
    fig.suptitle(
        "Olive yield vs pit-hardening red-zone VPD days\n"
        "Separate fixed-effect OLS with spring and late-season P - ET₀ controls",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    olives = load_olive_yield()
    climate = load_climate()
    features = build_feature_table(climate)
    sample = olives.merge(features, on=["comarca", "year"], how="inner")

    print(f"Sample: {len(sample)} rows, {sample['comarca'].nunique()} comarcas, years {sample['year'].min()}-{sample['year'].max()}")
    print(f"Pit-hardening window: {PIT_START} to {PIT_END}")
    print(f"Controls: pre={PRE_CTRL_START} to {PRE_CTRL_END}, post={POST_CTRL_START} to {POST_CTRL_END}")
    print(f"Mean pit-hardening red days (VPD >= {RED_VPD_KPA:.1f} kPa): {sample['pit_red_total'].mean():.2f}")
    print(f"Mean longest pit-hardening red spell: {sample['pit_longest_red_run'].mean():.2f} days")

    results = fit_feature_scan(sample)
    out_csv = DATA / "olive_pithardening_vpd_red_scan.csv"
    results.to_csv(out_csv, index=False)
    out_fig = FIG / "olive_pithardening_vpd_red_scan.png"
    plot_scan(results, out_fig)

    print(f"Saved {out_csv}")
    print(f"Saved {out_fig}")
    print("\nTop negative windows:")
    print(results.sort_values(["pvalue", "coef"]).head(12).to_string(index=False))