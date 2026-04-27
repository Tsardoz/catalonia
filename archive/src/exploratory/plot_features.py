"""
src/plot_features.py — Publication-quality 4-panel scatter via CLI
===================================================================
Usage examples
--------------
  python src/plot_features.py --crop apricot --var vpd_mean --feat lpf_peak --tau 30
  python src/plot_features.py --crop hazelnut --var vpd_max  --feat lpf_doy  --tau 7
  python src/plot_features.py --crop olive    --var vpd_max  --feat raw       --tau 7
  python src/plot_features.py --crop almond   --var precip_mean --feat lpf_width --tau 14 --threshold 1.0
  python src/plot_features.py --list-crops
  python src/plot_features.py --list-vars

--feat choices
--------------
  raw        : mean of the raw daily variable within each phenological window
  lpf_peak   : maximum of the IIR-filtered signal within each window
  lpf_doy    : peak DOY of the filtered signal over the full agronomic year (Oct-Sep)
  lpf_width  : days the filtered signal exceeds --threshold within each window

Panels
------
  One panel per phenological window defined for the crop  +  one panel for peak DOY.
  Points coloured by year (viridis).  Dashed OLS regression line.  R²/p/n annotated.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    PHENO_WINDOWS, VARIABLES, TAUS, VAR_UNITS, EXCL, DEF_THRESH,
    PHENO_KEY_ALIAS, CROP_CATALAN_FILTER,
    DATA_DIR, FIG_DIR, precompute_filters, extract,
)

FEAT_MAP = {
    "raw":       "Raw window mean",
    "lpf_peak":  "LPF peak value",
    "lpf_doy":   "LPF peak value",   # DOY uses the same filtered data; panel logic differs
    "lpf_width": "LPF peak width",
}


# ── axis labels ───────────────────────────────────────────────────────────────
def _xlabel(col, var, feat, tau, threshold):
    unit = VAR_UNITS.get(var, "")
    if col == "peak_doy":
        return f"Peak DOY of {var} LPF{tau}  (agronomic Oct–Sep year)"
    if feat == "raw":
        return f"Mean {var}  –  {col}  ({unit})"
    if feat in ("lpf_peak", "lpf_doy"):
        return f"Peak {var} LPF{tau}  –  {col}  ({unit})"
    return f"Days {var} LPF{tau} > p{threshold:.0f}  –  {col}  (d)"


# ── single panel ──────────────────────────────────────────────────────────────
def _draw_panel(ax, x, y, colors, cmap, norm, col, var, feat, tau, threshold,
                show_ylabel, y_lo, y_hi):
    x = x.astype(float)
    y = y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)

    # Split into inliers (used for regression) and outliers (shown faded)
    in_mask  = mask & (y >= y_lo) & (y <= y_hi)
    out_mask = mask & ~in_mask

    sc = ax.scatter(x[in_mask], y[in_mask], c=np.array(colors)[in_mask],
                    s=36, alpha=0.88, zorder=3, linewidths=0,
                    cmap=cmap, norm=norm)
    if out_mask.any():
        ax.scatter(x[out_mask], y[out_mask], c="0.7",
                   s=28, alpha=0.45, zorder=2, linewidths=0, marker="x")

    r2 = p = sl = delta_y_pct = np.nan
    if in_mask.sum() >= 3 and np.ptp(x[in_mask]) > 0:
        sl, ic, r, p, _ = stats.linregress(x[in_mask], y[in_mask])
        r2      = r ** 2
        x_range = x[in_mask].max() - x[in_mask].min()
        y_mean  = y[in_mask].mean()
        delta_y_pct = sl * x_range / y_mean * 100   # % of clipped-sample mean
        xl = np.linspace(x[in_mask].min(), x[in_mask].max(), 300)
        ax.plot(xl, sl * xl + ic, color="0.2", lw=1.8, ls="--", zorder=4)

    ax.text(0.04, 0.97,
            f"$R^2$ = {r2:.3f}\n$p$ = {p:.2e}\n$n$ = {in_mask.sum()}\n"
            f"slope = {sl:.3f}\n\u0394\u0177 = {delta_y_pct:.1f}% of mean",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))

    ax.set_xlabel(_xlabel(col, var, feat, tau, threshold), fontsize=9, labelpad=4)
    if show_ylabel:
        ax.set_ylabel("Yield  (t ha$^{-1}$)", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    return sc


# ── seasonal time-series panel ────────────────────────────────────────────────
def _draw_timeseries(ax, daily_f, yield_df, pheno_key, var, tau,
                     yr_cmap, yr_norm, years):
    """
    One line per agronomic year (Oct Y-1 → Sep Y) of the LPF-filtered {var}
    averaged across the comarques that have yield data for this crop.
    Lines are coloured by year index (same cmap/norm as scatter panels).
    Phenological windows are shaded; the seasonal peak is marked with a dot.
    """
    lpf_col = f"{var}_lpf{tau}"
    windows = PHENO_WINDOWS[pheno_key]

    csv_key   = PHENO_KEY_ALIAS.get(pheno_key, pheno_key)
    ymask     = yield_df["pheno_key"] == csv_key
    if pheno_key in CROP_CATALAN_FILTER:
        ymask &= yield_df["crop_catalan"] == CROP_CATALAN_FILTER[pheno_key]
    comarques = yield_df[ymask]["comarca"].unique()
    d = daily_f[daily_f["comarca"].isin(comarques)].copy()

    # agronomic-day offset helper (Oct 1 = day 1, non-leap ref year)
    REF = pd.Timestamp(2015, 10, 1)

    def _agr_day(month):
        y = 2015 if month >= 10 else 2016
        return (pd.Timestamp(y, month, 1) - REF).days + 1

    # shade phenological windows
    win_clrs = plt.cm.Set2(np.linspace(0, 0.7, len(windows)))
    for (wname, months), wc in zip(windows.items(), win_clrs):
        day_lo  = min(_agr_day(m) for m in months)
        m_last  = max(months, key=_agr_day)
        y_last  = 2015 if m_last >= 10 else 2016
        next_m  = 1 if m_last == 12 else m_last + 1
        next_y  = y_last + 1 if m_last == 12 else y_last
        day_hi  = (pd.Timestamp(next_y, next_m, 1) - REF).days
        ax.axvspan(day_lo, day_hi, color=wc, alpha=0.18, zorder=0,
                   label=wname.replace("_", " ").title())
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.75, ncol=len(windows))

    # one line per agronomic year
    for yr_idx, year in enumerate(years):
        agr_start = pd.Timestamp(year - 1, 10, 1)
        agr_end   = pd.Timestamp(year, 9, 30)
        sub = d[(d["date"] >= agr_start) & (d["date"] <= agr_end)]
        if sub.empty:
            continue
        dm = (sub.groupby("date")[lpf_col].mean()
                  .reset_index()
                  .sort_values("date"))
        dm["agr_day"] = (dm["date"] - agr_start).dt.days + 1

        color = yr_cmap(yr_norm(yr_idx))
        ax.plot(dm["agr_day"], dm[lpf_col], color=color, lw=1.1, alpha=0.85)

        # peak dot
        pk_i = dm[lpf_col].idxmax()
        ax.scatter([dm.loc[pk_i, "agr_day"]], [dm.loc[pk_i, lpf_col]],
                   color=color, s=40, zorder=5,
                   edgecolors="white", linewidths=0.5)

    # x-axis: month abbreviations in agronomic order
    MONTHS_AGR = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ticks  = [_agr_day(m) for m in MONTHS_AGR]
    labels = [pd.Timestamp(2015 if m >= 10 else 2016, m, 1).strftime("%b")
              for m in MONTHS_AGR]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(1, 366)
    ax.set_ylabel(f"{var} LPF{tau}  ({VAR_UNITS.get(var, '')})", fontsize=9)
    ax.set_xlabel("Agronomic year  (Oct – Sep)", fontsize=9, labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


# ── main plot ─────────────────────────────────────────────────────────────────
def make_plot(pheno_key, var, feat, tau, threshold, output=None, show=True):
    print(f"Loading data…", flush=True)
    daily = pd.read_csv(DATA_DIR / "agera5_daily_catalonia.csv", parse_dates=["date"])
    daily["year"]  = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily_f  = precompute_filters(daily)
    yield_df = pd.read_csv(DATA_DIR / "catalan_woody_yield_climate.csv")

    feat_type = FEAT_MAP[feat]
    print(f"Extracting features  [{pheno_key} | {var} | {feat} | τ={tau}]…", flush=True)
    merged = extract(daily_f, yield_df, pheno_key, var, feat_type, tau, threshold)
    if merged.empty:
        sys.exit("No data returned — check crop/variable combination.")

    windows   = list(PHENO_WINDOWS[pheno_key].keys())
    feat_cols = windows + ["peak_doy"]
    n = len(feat_cols)

    # Clip bounds: mean ± 2.5σ of yield across the full crop sample
    y_all = merged["yield_tha"].values.astype(float)
    y_all = y_all[np.isfinite(y_all)]
    y_lo  = y_all.mean() - 2.5 * y_all.std()
    y_hi  = y_all.mean() + 2.5 * y_all.std()

    years      = sorted(merged["year"].unique())
    n_yr       = len(years)
    yr_to_idx  = {y: i for i, y in enumerate(years)}
    # Map year labels to 0..n-1 so ticks are evenly distributed
    # over the full colorbar height, then relabel with actual years.
    cmap   = plt.get_cmap("viridis", n_yr)
    norm   = plt.Normalize(vmin=-0.5, vmax=n_yr - 0.5)
    colors = np.array([yr_to_idx[y] for y in merged["year"]])

    fig = plt.figure(figsize=(4.6 * n + 1.2, 8.8))
    gs  = gridspec.GridSpec(2, n,
                            height_ratios=[5.2, 3.2],
                            hspace=0.45,
                            left=0.07, right=0.87,
                            wspace=0.08, top=0.91, bottom=0.08)
    axes  = [fig.add_subplot(gs[0, i]) for i in range(n)]
    ts_ax = fig.add_subplot(gs[1, :])

    feat_label = {"raw": "raw mean", "lpf_peak": f"LPF{tau} peak",
                  "lpf_doy": f"LPF{tau} peak DOY", "lpf_width": f"LPF{tau} width"}
    fig.suptitle(
        f"{pheno_key.capitalize()}  ·  {var}  ·  {feat_label[feat]}",
        fontsize=13, fontweight="bold"
    )

    for ax, col in zip(axes, feat_cols):
        if col not in merged.columns:
            ax.axis("off")
            continue
        _draw_panel(ax,
                    merged[col].values, merged["yield_tha"].values,
                    colors, cmap, norm, col, var, feat, tau, threshold,
                    show_ylabel=(ax is axes[0]),
                    y_lo=y_lo, y_hi=y_hi)
        panel_title = "Peak DOY" if col == "peak_doy" else col.replace("_", " ").title()
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=5)

    # seasonal time-series panel (bottom row)
    _draw_timeseries(ts_ax, daily_f, yield_df, pheno_key, var, tau,
                     cmap, norm, years)

    # shared colourbar (spans both rows)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.895, 0.08, 0.018, 0.83])
    cb = fig.colorbar(sm, cax=cbar_ax, label="Year")
    cb.ax.set_yticks(list(range(n_yr)))
    cb.ax.set_yticklabels([str(y) for y in years], fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.canvas.draw()  # lock tick state before save

    # print stats table (clipped OLS)
    print(f"\n{'Window':<30}  {'R²':>6}  {'p':>10}  {'n':>4}")
    print("─" * 56)
    for col in feat_cols:
        if col not in merged.columns:
            continue
        x = merged[col].values.astype(float)
        y = merged["yield_tha"].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y) & (y >= y_lo) & (y <= y_hi)
        if m.sum() >= 3 and np.ptp(x[m]) > 0:
            _, _, r, p, _ = stats.linregress(x[m], y[m])
            print(f"  {col:<28}  {r**2:>6.3f}  {p:>10.3e}  {m.sum():>4d}")

    if output is None:
        output = FIG_DIR / f"{pheno_key}_{var}_{feat}_tau{tau}.png"
    fig.savefig(output, dpi=150)
    print(f"\nSaved → {output}")
    if show:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser():
    p = argparse.ArgumentParser(
        description="4-panel yield–climate scatter (phenological windows + peak DOY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--crop",      metavar="CROP",  help="Pheno key, e.g. apricot")
    p.add_argument("--var",       metavar="VAR",   help="Climate variable",
                   choices=VARIABLES)
    p.add_argument("--feat",      metavar="FEAT",  choices=list(FEAT_MAP),
                   default="lpf_peak",
                   help="Feature type: raw | lpf_peak | lpf_doy | lpf_width  (default: lpf_peak)")
    p.add_argument("--tau",       metavar="TAU",   type=int, choices=TAUS, default=7,
                   help="IIR time-constant in days (default: 7)")
    p.add_argument("--threshold", metavar="PCT",   type=float, default=DEF_THRESH,
                   help="Percentile of filtered signal for lpf_width (default: 75)")
    p.add_argument("--output",    metavar="PATH",  type=Path, default=None,
                   help="Output PNG path (default: figures/<crop>_<var>_<feat>_tau<tau>.png)")
    p.add_argument("--no-show",   action="store_true",
                   help="Don't open the figure window (just save)")
    p.add_argument("--list-crops", action="store_true", help="Print available crops and exit")
    p.add_argument("--list-vars",  action="store_true", help="Print available variables and exit")
    return p


def main():
    p = _build_parser()
    args = p.parse_args()

    if args.list_vars:
        print("Variables:", ", ".join(VARIABLES))
        return

    if args.list_crops:
        yield_df = pd.read_csv(DATA_DIR / "catalan_woody_yield_climate.csv")
        csv_keys = yield_df["pheno_key"].unique()
        crops = sorted(
            k for k in PHENO_WINDOWS if k not in EXCL
            and (k in csv_keys or PHENO_KEY_ALIAS.get(k) in csv_keys)
        )
        print("Crops:", ", ".join(crops))
        return

    if not args.crop or not args.var:
        p.error("--crop and --var are required (unless using --list-crops / --list-vars)")

    make_plot(
        pheno_key=args.crop,
        var=args.var,
        feat=args.feat,
        tau=args.tau,
        threshold=args.threshold,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

