"""
plot_mechanism_tests.py
Four-panel figure summarising the four mechanism tests for the rainfed-Arbequina
summer precipitation channel (Tests 1-4 in SUPERVISOR_UPDATE_APR2026.md §2.2).

Panel A is the corrected version: paired within-comarca R^2 bars for calendar
vs GDD anchoring (Test 2). The earlier draft showed coefficient bars, which
made the actual finding ("GDD anchor weakens fit") invisible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCAN = DATA / "sliding_window"
FIG = ROOT / "figures" / "fig4_mechanism_tests.png"
WHITELIST = DATA / "dun" / "comarca_arbequina_whitelist.csv"
CLIMATE = DATA / "agera5_daily_catalonia.csv"
YIELD_CSV = DATA / "catalan_woody_yield_raw.csv"

TMIN_END_MMDD = "08-17"
TMIN_WINDOW = 21
PRECIP_WINDOW = 30
SRAD_WINDOW = 30

NEG_COLOR = "#e08533"
POS_COLOR = "#3d7fb8"
NS_COLOR = "#bdbdbd"
BIVAR_COLOR = "#3d7fb8"
TRIVAR_COLOR = "#c66464"


def best_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.loc[df["within_r2"].idxmax()]


def _plot_signed_scan(ax: plt.Axes, df: pd.DataFrame, x_col: str, ls: str,
                      label: str, marker: str = "o", base_lw: float = 2.0,
                      sig_lw: float = 2.6, marker_size: float = 4.5) -> None:
    """Draw a within-R² scan with segments coloured by sign of β when p<0.05.

    A neutral grey backbone carries the trajectory and the panel-level label
    (so the legend stays single-entry per series); blue/orange overlays mark
    the statistically significant positive / negative segments respectively.
    """
    d = df.sort_values(x_col).reset_index(drop=True)
    ax.plot(d[x_col], d["within_r2"], color=NS_COLOR, lw=base_lw, ls=ls, label=label)
    pos = d[(d["p"] < 0.05) & (d["coef"] > 0)]
    neg = d[(d["p"] < 0.05) & (d["coef"] < 0)]
    if not pos.empty:
        eff_ls = "None" if len(pos) == 1 else ls
        ax.plot(pos[x_col], pos["within_r2"], color=POS_COLOR, lw=sig_lw,
                marker=marker, markersize=marker_size, linestyle=eff_ls)
    if not neg.empty:
        eff_ls = "None" if len(neg) == 1 else ls
        ax.plot(neg[x_col], neg["within_r2"], color=NEG_COLOR, lw=sig_lw,
                marker=marker, markersize=marker_size, linestyle=eff_ls)


def panel_a(ax: plt.Axes) -> None:
    cal = pd.read_csv(SCAN / "olive_precip_mean_sum_w30_arbequina.csv")
    gdd = pd.read_csv(SCAN / "olive_precip_mean_sum_w30_gdd_arbequina.csv")
    cal["d"] = pd.to_datetime("2021-" + cal["window_end_date"])
    gdd["d"] = pd.to_datetime("2021-" + gdd["median_end_date"])
    gdd = gdd.sort_values("d").reset_index(drop=True)

    _plot_signed_scan(ax, cal, "d", ls="-", label="Calendar anchor (window-end date)")
    ax.plot(gdd["d"], gdd["within_r2"], color=NS_COLOR, lw=2, ls="--",
            label="GDD anchor (median end date per cumGDD)")

    cpk = cal.loc[cal["within_r2"].idxmax()]
    gpk = gdd.loc[gdd["within_r2"].idxmax()]
    ax.scatter([cpk["d"]], [cpk["within_r2"]], color=NEG_COLOR, s=55, zorder=5,
               edgecolor="black", lw=0.5)
    ax.scatter([gpk["d"]], [gpk["within_r2"]], color=NS_COLOR, s=55, zorder=5,
               edgecolor="black", lw=0.5, marker="s")
    ax.annotate(f"calendar peak\n{cpk['window_end_date']}\nR²={cpk['within_r2']:.3f}",
                (cpk["d"], cpk["within_r2"]), xytext=(8, 4), textcoords="offset points",
                fontsize=9, color="#3a3a3a", fontweight="bold")
    ax.annotate(f"best GDD\n{gpk['median_end_date']}  ({int(gpk['cumgdd_end'])} GDD)\nR²={gpk['within_r2']:.3f}",
                (gpk["d"], gpk["within_r2"]), xytext=(-95, -45), textcoords="offset points",
                fontsize=9, color="#3a3a3a",
                arrowprops=dict(arrowstyle="->", color="#3a3a3a", lw=1))
    ax.axhline(cpk["within_r2"], color="#3a3a3a", lw=0.7, ls=":", alpha=0.6)

    ax.set_ylabel("Within-comarca R²")
    ax.set_ylim(0, max(cpk["within_r2"], gpk["within_r2"]) * 1.18)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.set_title("A.  Test 2: GDD anchor never matches calendar fit\n"
                 "(30-day precip sum)",
                 loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)


def panel_b(ax: plt.Axes) -> None:
    pr = pd.read_csv(SCAN / "olive_precip_mean_sum_w30_arbequina.csv")
    pet = pd.read_csv(SCAN / "olive_cum_p_minus_et0_mean_w1_arbequina_rainfed.csv")
    pr["d"] = pd.to_datetime("2021-" + pr["window_end_date"])
    pet["d"] = pd.to_datetime("2021-" + pet["window_end_date"])
    pet = pet.sort_values("d").reset_index(drop=True)

    _plot_signed_scan(ax, pr, "d", ls="-", label="30d precip sum")
    ax.plot(pet["d"], pet["within_r2"], color=NS_COLOR, lw=2, ls="--",
            label="Cum. P − ET₀ from agronomic-year start")

    peak = pr.loc[pr["within_r2"].idxmax()]
    ax.scatter([peak["d"]], [peak["within_r2"]], color=NEG_COLOR, s=55, zorder=5,
               edgecolor="black", lw=0.5)
    ax.annotate(f"precip peak\n{peak['window_end_date']}\nR²={peak['within_r2']:.3f}",
                (peak["d"], peak["within_r2"]), xytext=(8, -2), textcoords="offset points",
                fontsize=9, color="#3a3a3a", fontweight="bold")

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel("Within-comarca R²"); ax.set_xlabel("")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.set_title("B.  Test 3: Cumulative P − ET₀ explains < 5%\n"
                 "of the precipitation window variance",
                 loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25)


def panel_c(ax: plt.Axes) -> None:
    s = pd.read_csv(SCAN / "olive_srad_mean_mean_w21_arbequina_rainfed_w21.csv")
    s["d"] = pd.to_datetime("2021-" + s["window_end_date"])
    _plot_signed_scan(ax, s, "d", ls="-", label="21-day mean solar radiation")

    pos = s[(s["p"] < 0.05) & (s["coef"] > 0)]
    neg = s[(s["p"] < 0.05) & (s["coef"] < 0)]
    if not pos.empty:
        pk = pos.loc[pos["within_r2"].idxmax()]
        ax.annotate(f"{pk['window_end_date']}\nβ=+{pk['coef']:.2f}\nR²={pk['within_r2']:.3f}",
                    (pk["d"], pk["within_r2"]), xytext=(-58, -8),
                    textcoords="offset points", fontsize=8.5, color=POS_COLOR,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=POS_COLOR, lw=0.8))
    if not neg.empty:
        pk = neg.loc[neg["within_r2"].idxmax()]
        ax.annotate(f"{pk['window_end_date']}\nβ={pk['coef']:.2f}\nR²={pk['within_r2']:.3f}",
                    (pk["d"], pk["within_r2"]), xytext=(8, -2),
                    textcoords="offset points", fontsize=8.5, color=NEG_COLOR,
                    fontweight="bold")

    ax.set_ylabel("Within-comarca R²")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.set_title("C.  Solar radiation context: PAR-resource sign in early\n"
                 "July (blue), heat-load sign in August (orange)",
                 loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25)


def _window_feature(c: pd.DataFrame, var: str, end_doy: int, window: int, agg: str) -> pd.DataFrame:
    sub = c[(c["doy"] >= end_doy - window + 1) & (c["doy"] <= end_doy)]
    if agg == "mean":
        out = sub.groupby(["comarca", "year"], as_index=False)[var].mean()
    else:
        out = sub.groupby(["comarca", "year"], as_index=False)[var].sum()
    return out


def _within_r2(df: pd.DataFrame, predictors: list[str]) -> float:
    sub = df.dropna(subset=["yield_tha"] + predictors).copy()
    D = pd.get_dummies(sub["comarca"], drop_first=True, dtype=float).reset_index(drop=True)
    y = sub["yield_tha"].values
    res_fe = sm.OLS(y, sm.add_constant(D)).fit()
    if not predictors:
        return 0.0
    X = sm.add_constant(pd.concat([sub[predictors].reset_index(drop=True), D], axis=1))
    res = sm.OLS(y, X).fit()
    return float(1.0 - res.ssr / res_fe.ssr) if res_fe.ssr > 0 else float("nan")


def _scan_par_loss() -> pd.DataFrame:
    wl = pd.read_csv(WHITELIST)["comarca"].str.strip().unique().tolist()
    y = pd.read_csv(YIELD_CSV)
    y = y.loc[(y["pheno_key"] == "olive") & y["comarca"].isin(wl),
              ["year", "comarca", "yield_tha"]].copy()
    cols = ["date", "comarca", "tmin_mean", "precip_mean", "srad_mean"]
    c = pd.read_csv(CLIMATE, usecols=cols, parse_dates=["date"])
    c = c.loc[c["comarca"].isin(wl)].copy()
    c["doy"] = c["date"].dt.dayofyear
    c["year"] = c["date"].dt.year

    tmin_end = pd.Timestamp(f"2021-{TMIN_END_MMDD}").dayofyear
    tmin = _window_feature(c, "tmin_mean", tmin_end, TMIN_WINDOW, "mean").rename(columns={"tmin_mean": "tmin"})

    end_dates = pd.read_csv(SCAN / "olive_precip_mean_sum_w30_arbequina.csv")["window_end_date"].tolist()
    rows = []
    for end_mmdd in end_dates:
        end_doy = pd.Timestamp(f"2021-{end_mmdd}").dayofyear
        prec = _window_feature(c, "precip_mean", end_doy, PRECIP_WINDOW, "sum").rename(columns={"precip_mean": "precip"})
        srad = _window_feature(c, "srad_mean", end_doy, SRAD_WINDOW, "mean").rename(columns={"srad_mean": "srad"})
        df = (y.merge(tmin, on=["comarca", "year"]).merge(prec, on=["comarca", "year"])
                .merge(srad, on=["comarca", "year"]))
        rows.append({
            "end_date": end_mmdd,
            "precip_only":     _within_r2(df, ["precip"]),
            "plus_tmin":       _within_r2(df, ["tmin", "precip"]),
            "plus_tmin_srad":  _within_r2(df, ["tmin", "precip", "srad"]),
        })
    out = pd.DataFrame(rows)
    out["d"] = pd.to_datetime("2021-" + out["end_date"])
    return out.sort_values("d").reset_index(drop=True)


def panel_d(ax: plt.Axes) -> None:
    s = _scan_par_loss()
    ax.plot(s["d"], s["precip_only"], color="#7d8a90", lw=2.0, ls="-",
            label="Precip alone")
    ax.plot(s["d"], s["plus_tmin"], color=BIVAR_COLOR, lw=2.2, ls="-",
            label="+ T$_{min}$")
    ax.plot(s["d"], s["plus_tmin_srad"], color=TRIVAR_COLOR, lw=2.2, ls="--",
            label="+ T$_{min}$ + contemporaneous srad")

    pk = s.loc[s["precip_only"].idxmax()]
    ax.scatter([pk["d"]], [pk["plus_tmin"]], color=BIVAR_COLOR, s=40, zorder=5)
    ax.scatter([pk["d"]], [pk["plus_tmin_srad"]], color=TRIVAR_COLOR, s=40, zorder=5)
    gain_tmin = pk["plus_tmin"] - pk["precip_only"]
    gain_srad = pk["plus_tmin_srad"] - pk["plus_tmin"]
    ax.annotate(f"T$_{{min}}$ adds\n+{gain_tmin:.3f}",
                (pk["d"], (pk["precip_only"] + pk["plus_tmin"]) / 2),
                xytext=(10, 0), textcoords="offset points", fontsize=9,
                color=BIVAR_COLOR, fontweight="bold")
    ax.annotate(f"srad adds\n+{gain_srad:.3f}",
                (pk["d"], (pk["plus_tmin"] + pk["plus_tmin_srad"]) / 2),
                xytext=(10, 18), textcoords="offset points", fontsize=9,
                color=TRIVAR_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TRIVAR_COLOR, lw=0.8))

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel("Within-comarca R²")
    ax.set_ylim(0, 0.65)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.set_title("D.  Test 4: PAR-loss test — adding contemporaneous srad\n"
                 "adds essentially zero variance once T$_{min}$ is in",
                 loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    panel_a(axes[0, 0]); panel_b(axes[0, 1]); panel_c(axes[1, 0]); panel_d(axes[1, 1])
    fig.suptitle("Four mechanism tests for the summer precipitation channel\n"
                 "Rainfed Arbequina olive panel, Catalonia 2015–2024  (n = 100)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
