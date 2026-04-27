"""
ElasticNet distributed-lag impulse response for rainfed olive yield.
HYBRID VERSION: Uses area-weighted climate where available, centroid fallback.

Recovers the weekly sensitivity of olive yield to VPD, precipitation,
and ET₀ across the agronomic year (Dec–Nov) using ElasticNet with
comarca fixed effects.

Uses: agera5_daily_hybrid_olive.csv
  - 23 comarques: area-weighted from actual farm locations (elevation-corrected)
  - 13 comarques: centroid-based (fallback for small/dispersed farms)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# CHANGED: Use hybrid daily climate (area-weighted + centroid)
CLIMATE_CSV = DATA / "agera5_daily_hybrid_olive.csv"
YIELD_CSV = DATA / "catalan_woody_yield_raw.csv"

# Climate variables to bin weekly
VARS = ["vpd_mean", "et0_mean", "precip_mean"]

# VPD exceedance threshold (kPa)
VPD_THRESHOLD = 2.0

# ── 1. Data Preparation ─────────────────────────────────────────────


N_BINS = 13  # 4-week bins (28 days each; last bin absorbs remainder)
LP_TAU_DAYS = 45.0
BUCKET_CAPACITY_MM = 120.0
CWB_TAU_DAYS = 45.0


def add_agronomic_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add agronomic-year labels, day-of-year, and 4-week bins."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["cal_year"] = df["date"].dt.year
    df["agro_year"] = np.where(df["month"] == 12, df["cal_year"] + 1, df["cal_year"])

    agro_start = np.where(
        df["month"] == 12,
        pd.to_datetime(df["cal_year"].astype(str) + "-12-01"),
        pd.to_datetime((df["cal_year"] - 1).astype(str) + "-12-01"),
    )
    agro_start = pd.to_datetime(pd.Series(agro_start, index=df.index))
    df["agro_day"] = (df["date"] - agro_start).dt.days
    df["agro_bin"] = np.clip(df["agro_day"] // 28 + 1, 1, N_BINS)
    return df


def load_climate_biweekly() -> pd.DataFrame:
    """Bin daily climate data into 13 four-week agronomic-year bins (Dec 1 – Nov 30)."""
    df = pd.read_csv(CLIMATE_CSV, parse_dates=["date"])

    # Print climate method breakdown
    print("\nClimate data sources:")
    method_counts = df.groupby('climate_method')['comarca'].nunique()
    for method, count in method_counts.items():
        print(f"  {method:15s}: {count} comarques")

    df = add_agronomic_time(df)

    # VPD exceedance flag
    df["vpd_exceed"] = (df["vpd_mean"] > VPD_THRESHOLD).astype(float)

    # Bin means for climate vars + sum for exceedance days
    grouped_mean = df.groupby(["comarca", "agro_year", "agro_bin"])[VARS].mean()
    grouped_ndays = df.groupby(["comarca", "agro_year", "agro_bin"])["vpd_exceed"].sum()
    grouped_max = df.groupby(["comarca", "agro_year", "agro_bin"])[["vpd_mean", "et0_mean"]].max()

    # Pivot to wide
    records = []
    for (comarca, year), grp in grouped_mean.groupby(level=[0, 1]):
        row = {"comarca": comarca, "agro_year": year}
        ndays_grp = grouped_ndays.loc[(comarca, year)]
        max_grp = grouped_max.loc[(comarca, year)]
        for bw, vals in grp.droplevel([0, 1]).iterrows():
            for var in VARS:
                short = var.replace("_mean", "")  # vpd, et0, precip
                row[f"{short}_w{int(bw):02d}"] = vals[var]
            row[f"nvpd_w{int(bw):02d}"] = ndays_grp.loc[bw]
            row[f"vmax_w{int(bw):02d}"] = max_grp.loc[bw, "vpd_mean"]
            row[f"emax_w{int(bw):02d}"] = max_grp.loc[bw, "et0_mean"]
        records.append(row)

    wide = pd.DataFrame(records)
    return wide


def load_olive_yield() -> pd.DataFrame:
    """Load olive yield data: comarca, year, yield_tha."""
    df = pd.read_csv(YIELD_CSV)
    olives = df.loc[df["crop_group"] == "olive", ["comarca", "year", "yield_tha"]].copy()
    olives = olives.rename(columns={"year": "agro_year"})
    return olives


def build_dataset() -> pd.DataFrame:
    """Join biweekly climate features to olive yield; drop incomplete years.
    Also adds yield_lag1 (previous year's yield for same comarca)."""
    climate = load_climate_biweekly()
    olives = load_olive_yield()

    merged = olives.merge(climate, on=["comarca", "agro_year"], how="inner")

    # Drop 2015 — no Dec 2014 daily data available
    merged = merged.loc[merged["agro_year"] >= 2016].reset_index(drop=True)

    # Sanity: drop rows with any missing weekly columns
    week_cols = [c for c in merged.columns if "_w" in c]
    merged = merged.dropna(subset=week_cols).reset_index(drop=True)

    # Add lagged yield: yield_{t-1} for the same comarca
    # 2015 yield is available even though agro_year 2015 climate is incomplete
    yield_lag = olives.copy()
    yield_lag["agro_year"] = yield_lag["agro_year"] + 1
    yield_lag = yield_lag.rename(columns={"yield_tha": "yield_lag1"})
    merged = merged.merge(yield_lag[["comarca", "agro_year", "yield_lag1"]],
                          on=["comarca", "agro_year"], how="left")

    return merged


def get_sample_daily_climate(sample_rows: pd.DataFrame) -> pd.DataFrame:
    """Return daily mean precip and ET₀ for the olive sample across agronomic years."""
    climate = pd.read_csv(CLIMATE_CSV, parse_dates=["date"])
    climate = add_agronomic_time(climate)

    sample_keys = sample_rows[["comarca", "agro_year"]].drop_duplicates()
    climate = climate.merge(sample_keys, on=["comarca", "agro_year"], how="inner")
    climate = climate.loc[climate["agro_year"] >= 2016].copy()

    daily = (climate.groupby("agro_day", as_index=False)[["precip_mean", "et0_mean"]]
             .mean()
             .sort_values("agro_day"))
    return daily


def compute_precip_proxy(sample_rows: pd.DataFrame, tau_days: float = LP_TAU_DAYS) -> pd.DataFrame:
    """Return a low-pass filtered antecedent precipitation proxy for the olive sample."""
    daily = get_sample_daily_climate(sample_rows)

    decay = np.exp(-1.0 / tau_days)
    proxy = np.empty(len(daily), dtype=float)
    proxy[0] = daily["precip_mean"].iloc[0]
    for i in range(1, len(daily)):
        proxy[i] = decay * proxy[i - 1] + daily["precip_mean"].iloc[i]

    proxy_scaled = proxy - proxy.min()
    if proxy_scaled.max() > 0:
        proxy_scaled = proxy_scaled / proxy_scaled.max()

    x = 1 + (daily["agro_day"].to_numpy() - 13.5) / 28.0
    return pd.DataFrame({
        "agro_day": daily["agro_day"].to_numpy(),
        "x": x,
        "precip_lpf": proxy,
        "precip_lpf_scaled": proxy_scaled,
    })


def compute_et0_proxy(sample_rows: pd.DataFrame, tau_days: float = LP_TAU_DAYS) -> pd.DataFrame:
    """Return a low-pass filtered ET₀ proxy for the olive sample."""
    daily = get_sample_daily_climate(sample_rows)

    decay = np.exp(-1.0 / tau_days)
    proxy = np.empty(len(daily), dtype=float)
    proxy[0] = daily["et0_mean"].iloc[0]
    for i in range(1, len(daily)):
        proxy[i] = decay * proxy[i - 1] + daily["et0_mean"].iloc[i]

    proxy_scaled = proxy - proxy.min()
    if proxy_scaled.max() > 0:
        proxy_scaled = proxy_scaled / proxy_scaled.max()

    x = 1 + (daily["agro_day"].to_numpy() - 13.5) / 28.0
    return pd.DataFrame({
        "agro_day": daily["agro_day"].to_numpy(),
        "x": x,
        "et0_lpf": proxy,
        "et0_lpf_scaled": proxy_scaled,
    })


def compute_bucket_proxy(sample_rows: pd.DataFrame,
                         capacity_mm: float = BUCKET_CAPACITY_MM) -> pd.DataFrame:
    """Return a simple daily bucket water-balance proxy for the olive sample."""
    daily = get_sample_daily_climate(sample_rows)
    net = daily["precip_mean"].to_numpy() - daily["et0_mean"].to_numpy()

    storage = np.empty(len(daily), dtype=float)
    storage[0] = capacity_mm
    for i in range(1, len(daily)):
        storage[i] = np.clip(storage[i - 1] + net[i], 0.0, capacity_mm)

    x = 1 + (daily["agro_day"].to_numpy() - 13.5) / 28.0
    return pd.DataFrame({
        "agro_day": daily["agro_day"].to_numpy(),
        "x": x,
        "bucket_mm": storage,
        "bucket_scaled": storage / capacity_mm,
        "net_mm": net,
    })


def compute_cwb_proxy(sample_rows: pd.DataFrame,
                      tau_days: float = CWB_TAU_DAYS) -> pd.DataFrame:
    """Return a low-pass filtered climatic water-balance proxy: P - ET₀."""
    daily = get_sample_daily_climate(sample_rows)
    net = daily["precip_mean"].to_numpy() - daily["et0_mean"].to_numpy()

    decay = np.exp(-1.0 / tau_days)
    filt = np.empty(len(daily), dtype=float)
    filt[0] = net[0]
    for i in range(1, len(daily)):
        filt[i] = decay * filt[i - 1] + net[i]

    x = 1 + (daily["agro_day"].to_numpy() - 13.5) / 28.0
    return pd.DataFrame({
        "agro_day": daily["agro_day"].to_numpy(),
        "x": x,
        "cwb_lpf": filt,
        "net_mm": net,
    })


# ── 2. Model Specification ───────────────────────────────────────────

L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]


def fit_elasticnet(df: pd.DataFrame, lag_cols: list[str], label: str):
    """Fit ElasticNetCV with comarca fixed effects.

    Returns (model, scaler, lag_cols, comarca_cols, r2_train, r2_cv, coef_lag).
    """
    # Comarca dummies (drop_first to avoid collinearity)
    comarca_dummies = pd.get_dummies(df["comarca"], drop_first=True, dtype=float)
    comarca_cols = list(comarca_dummies.columns)

    # Scale lag features only (not comarca dummies)
    scaler = StandardScaler()
    X_lag = scaler.fit_transform(df[lag_cols])
    X = np.hstack([X_lag, comarca_dummies.values])
    y = df["yield_tha"].values

    model = ElasticNetCV(
        l1_ratio=L1_RATIOS,
        alphas=100,
        cv=5,
        fit_intercept=True,
        max_iter=10_000,
    )
    model.fit(X, y)

    r2_train = model.score(X, y)
    coef_lag = model.coef_[: len(lag_cols)]  # lag coefficients only
    n_nonzero = np.sum(coef_lag != 0)

    # CV R²: derive from best CV MSE vs variance of y
    # mse_path_ shape: (n_l1_ratio, n_alphas, n_folds)
    l1_idx = L1_RATIOS.index(model.l1_ratio_)
    alpha_idx = np.argmin(np.abs(model.alphas_[l1_idx] - model.alpha_))
    best_cv_mse = model.mse_path_[l1_idx][alpha_idx].mean()
    r2_cv = 1 - best_cv_mse / np.var(y)

    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"  Best alpha:    {model.alpha_:.6f}")
    print(f"  Best l1_ratio: {model.l1_ratio_}")
    print(f"  Training R²:   {r2_train:.4f}")
    print(f"  CV R²:         {r2_cv:.4f}")
    print(f"  Nonzero lag coefficients: {n_nonzero} / {len(lag_cols)}")

    return model, scaler, lag_cols, comarca_cols, r2_train, r2_cv, coef_lag


def fit_ols(df: pd.DataFrame, lag_cols: list[str], label: str):
    """Fit OLS with comarca fixed effects. Prints coefficient p-values."""
    comarca_dummies = pd.get_dummies(df["comarca"], drop_first=True, dtype=float)
    X = pd.concat([df[lag_cols].reset_index(drop=True),
                   comarca_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = df["yield_tha"].values

    model = sm.OLS(y, X).fit()

    print(f"\n{'='*60}")
    print(f"OLS: {label}")
    print(f"  R²:      {model.rsquared:.4f}")
    print(f"  Adj R²:  {model.rsquared_adj:.4f}")
    print(f"  F-stat:  {model.fvalue:.2f}  (p = {model.f_pvalue:.2e})")
    print(f"\n  Lag coefficients:")
    print(f"  {'Period':<12} {'Coef':>8} {'Std Err':>8} {'t':>7} {'p':>10} {'Sig'}")
    print(f"  {'-'*55}")
    for col in lag_cols:
        c = model.params[col]
        se = model.bse[col]
        t = model.tvalues[col]
        p = model.pvalues[col]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<12} {c:>8.4f} {se:>8.4f} {t:>7.2f} {p:>10.4f} {sig}")

    return model


# ── 3. Impulse Response Plots

# Phenological period bands mapped to 4-week bins
# (label, start_bin, end_bin, colour)
PHENO_BANDS = [
    ("Dormancy",    1,  3, "#d0e1f9"),   # Dec–Feb
    ("Vegetative",  4,  5, "#d4edda"),   # Mar–Apr
    ("Flowering",   6,  7, "#fff3cd"),   # May–Jun
    ("Fruit set",   8, 10, "#fde2e2"),   # Jul–Aug (includes pit hardening)
    ("Maturation", 11, 13, "#e8daef"),   # Sep–Nov
]

# Schematic olive RDI guide from phenology / fruit development literature.
RDI_GUIDE_BANDS = [
    ("Sensitive\nflower / set", 6, 7, "#f5b7b1"),
    ("Tolerant\npit hardening", 8, 9, "#d5f5e3"),
    ("Sensitive\noil accumulation", 10, 13, "#f5b7b1"),
]

# Month tick positions (approximate 4-week bin of each month start)
MONTH_TICKS = [
    (1, "Dec"), (2, "Jan"), (3, "Feb–Mar"),
    (4, "Mar–Apr"), (5, "Apr–May"), (6, "May–Jun"),
    (7, "Jun–Jul"), (8, "Jul–Aug"), (9, "Aug"),
    (10, "Aug–Sep"), (11, "Sep–Oct"), (12, "Oct–Nov"), (13, "Nov"),
]

# Short month labels for compact x-axis
MONTH_LABELS = ["Dec", "Jan", "Feb\u2013Mar", "Mar\u2013Apr", "Apr\u2013May",
                "May\u2013Jun", "Jun\u2013Jul", "Jul\u2013Aug", "Aug",
                "Aug\u2013Sep", "Sep\u2013Oct", "Oct\u2013Nov", "Nov"]


def _add_pheno_bands(ax):
    """Add shaded phenological background bands and month x-ticks."""
    for label, b0, b1, colour in PHENO_BANDS:
        ax.axvspan(b0 - 0.5, b1 + 0.5, alpha=0.25, color=colour, zorder=0)
        mid = (b0 + b1) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95, label,
                ha="center", va="top", fontsize=7, fontstyle="italic", alpha=0.7)
    ax.set_xticks([b for b, _ in MONTH_TICKS])
    ax.set_xticklabels([m for _, m in MONTH_TICKS], fontsize=8)
    ax.set_xlim(0.5, N_BINS + 0.5)


def plot_irf_single(coef_lag, r2_train, r2_cv, var_label, filename, colour="steelblue"):
    """Bar plot of a single variable's impulse response."""
    bins = np.arange(1, N_BINS + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    colours = [colour if c != 0 else "lightgrey" for c in coef_lag]
    ax.bar(bins, coef_lag, color=colours, edgecolor="none", width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    _add_pheno_bands(ax)
    ax.set_xlabel("4-week period of agronomic year (Dec–Nov)")
    ax.set_ylabel("ElasticNet coefficient (standardised)")
    ax.set_title(f"Olive yield impulse response — {var_label}   "
                 f"(train R² = {r2_train:.3f}, CV R² = {r2_cv:.3f})")
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=200)
    plt.close(fig)
    print(f"  Saved {FIG / filename}")


def plot_irf_overlay(coef_vpd, r2cv_vpd, coef_et0, r2cv_et0, filename):
    """Overlay VPD and ET₀ impulse responses for comparison."""
    bins = np.arange(1, N_BINS + 1)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(bins - 0.2, coef_vpd, width=0.4, color="#c0392b", alpha=0.8,
           label=f"VPD  (CV R² = {r2cv_vpd:.3f})")
    ax.bar(bins + 0.2, coef_et0, width=0.4, color="#2980b9", alpha=0.8,
           label=f"ET₀  (CV R² = {r2cv_et0:.3f})")
    ax.axhline(0, color="black", linewidth=0.5)
    _add_pheno_bands(ax)
    ax.set_xlabel("4-week period of agronomic year (Dec–Nov)")
    ax.set_ylabel("ElasticNet coefficient (standardised)")
    ax.set_title("Olive yield impulse response — VPD vs ET₀ (HYBRID climate)")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=200)
    plt.close(fig)
    print(f"  Saved {FIG / filename}")


def plot_ols_comparison(ols_models, lag_col_sets, labels, colours, filename):
    """3-panel side-by-side OLS impulse response with significance markers."""
    n = len(ols_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=False)
    bins = np.arange(1, N_BINS + 1)

    for ax, model, lag_cols, label, colour in zip(axes, ols_models, lag_col_sets, labels, colours):
        coefs = [model.params[c] for c in lag_cols]
        pvals = [model.pvalues[c] for c in lag_cols]
        bar_colours = []
        for p in pvals:
            if p < 0.001:
                bar_colours.append(colour)
            elif p < 0.05:
                # lighter version
                bar_colours.append(colour + "99")
            else:
                bar_colours.append("#cccccc")

        ax.bar(bins, coefs, color=bar_colours, edgecolor="none", width=0.8)
        ax.axhline(0, color="black", linewidth=0.5)

        # Significance stars
        for b, (c, p) in enumerate(zip(coefs, pvals), start=1):
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                continue
            yoff = 0.005 if c >= 0 else -0.015
            ax.text(b, c + yoff, star, ha="center", va="bottom" if c >= 0 else "top",
                    fontsize=7, fontweight="bold")

        # Pheno bands (no text labels except on first panel)
        for plabel, b0, b1, pcol in PHENO_BANDS:
            ax.axvspan(b0 - 0.5, b1 + 0.5, alpha=0.15, color=pcol, zorder=0)
        if ax == axes[0]:
            for plabel, b0, b1, _ in PHENO_BANDS:
                mid = (b0 + b1) / 2
                ax.text(mid, ax.get_ylim()[1] * 0.95, plabel,
                        ha="center", va="top", fontsize=6, fontstyle="italic", alpha=0.6)

        ax.set_xticks(bins)
        ax.set_xticklabels(MONTH_LABELS, fontsize=6.5, rotation=35, ha="right")
        ax.set_xlim(0.5, N_BINS + 0.5)
        ax.set_title(f"{label}\n(R²={model.rsquared:.3f}, Adj R²={model.rsquared_adj:.3f})",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("OLS coefficient" if ax == axes[0] else "", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Olive yield — OLS impulse response (HYBRID climate, 4-week bins, comarca FE)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG / filename}")


def plot_vmax_with_proxy_overlay(model, lag_cols, proxy_df, proxy_col, filename,
                                 line_label, y2_label, title,
                                 line_colour="#2471a3", fill_colour="#85c1e9",
                                 y2_ylim=None, draw_y2_zero=False,
                                 guide_bands=None, guide_label=None,
                                 bar_label="OLS coefficient"):
    """Plot lagged OLS coefficients with a secondary-axis proxy overlay."""
    bins = np.arange(1, N_BINS + 1)
    coefs = [model.params[c] for c in lag_cols]
    pvals = [model.pvalues[c] for c in lag_cols]

    bar_colours = []
    for p in pvals:
        if p < 0.001:
            bar_colours.append("#c0392b")
        elif p < 0.05:
            bar_colours.append("#c0392b99")
        else:
            bar_colours.append("#cccccc")

    if guide_bands:
        fig, (ax_guide, ax) = plt.subplots(
            2, 1,
            figsize=(12, 5.6),
            sharex=True,
            gridspec_kw={"height_ratios": [0.7, 4.9], "hspace": 0.05},
        )
        ax_guide.set_ylim(0, 1)
        ax_guide.set_yticks([])
        ax_guide.set_xlim(0.5, N_BINS + 0.5)
        ax_guide.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax_guide.tick_params(axis="x", which="both", length=0, labelbottom=False)
        if guide_label:
            ax_guide.set_title(guide_label, fontsize=9, loc="left", pad=2)
        for glabel, b0, b1, gcol in guide_bands:
            ax_guide.axvspan(b0 - 0.5, b1 + 0.5, color=gcol, alpha=0.95)
            ax_guide.text((b0 + b1) / 2, 0.5, glabel,
                          ha="center", va="center", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(12, 4.8))

    ax.bar(bins, coefs, color=bar_colours, edgecolor="none", width=0.8,
           label=bar_label)
    ax.axhline(0, color="black", linewidth=0.5)

    for _, b0, b1, pcol in PHENO_BANDS:
        ax.axvspan(b0 - 0.5, b1 + 0.5, alpha=0.15, color=pcol, zorder=0)

    for b, (c, p) in enumerate(zip(coefs, pvals), start=1):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            continue
        yoff = 0.005 if c >= 0 else -0.015
        ax.text(b, c + yoff, star, ha="center", va="bottom" if c >= 0 else "top",
                fontsize=8, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(
        proxy_df["x"],
        proxy_df[proxy_col],
        color=line_colour,
        linewidth=2.2,
        label=line_label,
    )
    ax2.fill_between(
        proxy_df["x"],
        0,
        proxy_df[proxy_col],
        color=fill_colour,
        alpha=0.18,
    )
    if draw_y2_zero:
        ax2.axhline(0, color=line_colour, linewidth=0.8, alpha=0.6, linestyle="--")

    ax.set_xticks(bins)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8, rotation=35, ha="right")
    ax.set_xlim(0.5, N_BINS + 0.5)
    ax.set_ylabel("OLS coefficient")
    ax.set_xlabel("4-week period of agronomic year (Dec–Nov)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax2.set_ylabel(y2_label, color=line_colour)
    ax2.tick_params(axis="y", colors=line_colour)
    if y2_ylim is not None:
        ax2.set_ylim(*y2_ylim)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", framealpha=0.9)

    if guide_bands:
        fig.subplots_adjust(left=0.07, right=0.93, bottom=0.14, top=0.92, hspace=0.05)
    else:
        fig.tight_layout()
    fig.savefig(FIG / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG / filename}")


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*70)
    print("OLIVE DISTRIBUTED-LAG ANALYSIS - HYBRID CLIMATE")
    print("="*70)
    print(f"Using: {CLIMATE_CSV.name}")
    print("  - Area-weighted climate (23 comarques with CORINE farms)")
    print("  - Centroid climate (13 comarques fallback)")

    df = build_dataset()
    week_cols = [c for c in df.columns if "_w" in c]
    vpd_cols = sorted([c for c in week_cols if c.startswith("vpd_")])
    et0_cols = sorted([c for c in week_cols if c.startswith("et0_")])
    precip_cols = sorted([c for c in week_cols if c.startswith("precip_")])

    print(f"\nDataset: {len(df)} rows, {df['comarca'].nunique()} comarques, "
          f"years {df['agro_year'].min()}–{df['agro_year'].max()}")
    print(f"4-week feature columns: {len(vpd_cols)} VPD, {len(et0_cols)} ET₀, "
          f"{len(precip_cols)} precip")

    # Fit ElasticNet models
    print("\n" + "#"*60)
    print("ELASTICNET MODELS (L1+L2 regularization)")
    print("#"*60)

    m1 = fit_elasticnet(df, precip_cols, "M1: Precipitation → yield")
    m2 = fit_elasticnet(df, vpd_cols, "M2: VPD → yield")
    m3 = fit_elasticnet(df, et0_cols, "M3: ET₀ → yield")
    nvpd_cols = sorted([c for c in week_cols if c.startswith("nvpd_")])
    vmax_cols = sorted([c for c in week_cols if c.startswith("vmax_")])
    m4 = fit_elasticnet(df, nvpd_cols, f"M4: Days VPD > {VPD_THRESHOLD} kPa → yield")
    m5 = fit_elasticnet(df, vmax_cols, "M5: Max VPD → yield")

    # Unpack results
    _, _, _, _, r2t_precip, r2cv_precip, coef_precip = m1
    _, _, _, _, r2t_vpd, r2cv_vpd, coef_vpd = m2
    _, _, _, _, r2t_et0, r2cv_et0, coef_et0 = m3
    _, _, _, _, r2t_nvpd, r2cv_nvpd, coef_nvpd = m4
    _, _, _, _, r2t_vmax, r2cv_vmax, coef_vmax = m5

    # OLS with p-values
    print("\n" + "#"*60)
    print("OLS REGRESSION (for p-values)")
    print("#"*60)
    ols_vmax = fit_ols(df, vmax_cols, "Max VPD")
    emax_cols = sorted([c for c in df.columns if c.startswith("emax_")])
    ols_emax = fit_ols(df, emax_cols, "Max ET\u2080")
    ols_precip = fit_ols(df, precip_cols, "Precipitation")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nDataset used: {CLIMATE_CSV.name}")
    print(f"  - {len(df)} comarca-year observations")
    print(f"  - {df['comarca'].nunique()} comarques")
    print(f"  - Hybrid climate (area-weighted + centroid)")

    print("\nKey results:")
    print(f"  Precipitation CV R²: {r2cv_precip:.3f}")
    print(f"  Max VPD CV R²:       {r2cv_vmax:.3f}")
    print(f"  Max ET₀ CV R²:       {r2cv_et0:.3f}")

    # Generate impulse response plots
    print("\n" + "="*70)
    print("GENERATING FIGURES...")
    print("="*70)

    plot_irf_single(coef_precip, r2t_precip, r2cv_precip, "Precipitation",
                    "olive_irf_precip_hybrid.png", colour="#2e86c1")
    plot_irf_single(coef_vpd, r2t_vpd, r2cv_vpd, "VPD",
                    "olive_irf_vpd_hybrid.png", colour="#c0392b")
    plot_irf_single(coef_nvpd, r2t_nvpd, r2cv_nvpd,
                    f"Days VPD > {VPD_THRESHOLD} kPa",
                    "olive_irf_nvpd_hybrid.png", colour="#8e44ad")
    plot_irf_single(coef_vmax, r2t_vmax, r2cv_vmax,
                    "Max VPD",
                    "olive_irf_vpd_max_hybrid.png", colour="#e74c3c")
    plot_irf_overlay(coef_vpd, r2cv_vpd, coef_et0, r2cv_et0,
                     "olive_irf_vpd_vs_et0_hybrid.png")

    # 3-panel OLS comparison
    plot_ols_comparison(
        [ols_vmax, ols_emax, ols_precip],
        [vmax_cols, emax_cols, precip_cols],
        ["Max VPD", "Max ET\u2080", "Mean Precipitation"],
        ["#c0392b", "#2980b9", "#27ae60"],
        "olive_ols_comparison_hybrid.png"
    )

    # Overlay plots with water proxies
    precip_proxy = compute_precip_proxy(df)
    plot_vmax_with_proxy_overlay(
        ols_vmax,
        vmax_cols,
        precip_proxy,
        "precip_lpf_scaled",
        "olive_ols_vpd_with_precip_overlay_hybrid.png",
        f"LP-filtered precip proxy (τ={int(LP_TAU_DAYS)} d)",
        "Antecedent rain proxy (scaled 0–1)",
        "Olive yield — Max VPD OLS (HYBRID) with antecedent rain proxy",
        y2_ylim=(0, 1.05),
    )

    cwb_proxy = compute_cwb_proxy(df)
    plot_vmax_with_proxy_overlay(
        ols_vmax,
        vmax_cols,
        cwb_proxy,
        "cwb_lpf",
        "olive_ols_vpd_with_cwb_overlay_hybrid.png",
        f"LP-filtered climatic water balance (P - ET₀, τ={int(CWB_TAU_DAYS)} d)",
        "LP-filtered P - ET₀ proxy",
        "Olive yield — Max VPD OLS (HYBRID) with climatic water-balance proxy",
        line_colour="#117864",
        fill_colour="#a9dfbf",
        draw_y2_zero=True,
        guide_bands=RDI_GUIDE_BANDS,
        guide_label="Schematic RDI guide (fruit development / literature)",
        bar_label="Max VPD OLS coefficient",
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nDataset used: {CLIMATE_CSV.name}")
    print(f"  - {len(df)} comarca-year observations")
    print(f"  - {df['comarca'].nunique()} comarques")
    print(f"  - Hybrid climate (area-weighted + centroid)")

    print("\nFigures saved to: figures/")
    print("  - olive_irf_*_hybrid.png (ElasticNet impulse responses)")
    print("  - olive_ols_comparison_hybrid.png (OLS 3-panel)")
    print("  - olive_ols_vpd_with_*_hybrid.png (OLS with water proxies)")

    print("\nNext steps:")
    print("  1. Compare results with centroid-only analysis (PLAN.md)")
    print("  2. Check if stress-sensitive periods align with RDI windows")
    print("  3. Document method in paper")
