"""
src/explorer.py — Interactive yield–climate scatter explorer (Streamlit)
Run:  streamlit run src/explorer.py

Controls (left sidebar)
-----------------------
  Crop         : pheno_key (all crops with yield data, excl. 'other')
  Variable     : vpd_mean | vpd_max | precip_mean | et0_mean | tmax_mean
  Window feat  : Raw window mean | LPF peak value | LPF peak width
  τ (days)     : IIR filter time-constant
  Threshold    : cutoff for peak-width feature

Plot: 4-panel scatter — one per phenological window + peak DOY — colour = year.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    PHENO_WINDOWS, VARIABLES, TAUS, FEAT_TYPES, EXCL, DEF_THRESH, VAR_UNITS,
    PHENO_KEY_ALIAS, CROP_CATALAN_FILTER, DATA_DIR, FIG_DIR,
    precompute_filters, extract,
)


# ── Streamlit app ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & filtering daily data…")
def load_data():
    daily = pd.read_csv(DATA_DIR / "agera5_daily_catalonia.csv", parse_dates=["date"])
    daily["year"]  = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily_f  = precompute_filters(daily)
    yield_df = pd.read_csv(DATA_DIR / "catalan_woody_yield_climate.csv")
    return daily_f, yield_df


def _draw_timeseries(ax, daily_f, yield_df, pheno_key, var, tau,
                     yr_cmap, yr_norm, years,
                     var2=None, tau2=None, all_years=None):
    """Seasonal LPF-filtered signal (Oct–Sep) averaged across crop comarques.
    One solid line per year for var (left axis). If var2 is given, dashed lines
    on a twin right axis with the same year colours."""
    lpf_col = f"{var}_lpf{tau}"
    windows = PHENO_WINDOWS[pheno_key]

    csv_key   = PHENO_KEY_ALIAS.get(pheno_key, pheno_key)
    ymask     = yield_df["pheno_key"] == csv_key
    if pheno_key in CROP_CATALAN_FILTER:
        ymask &= yield_df["crop_catalan"] == CROP_CATALAN_FILTER[pheno_key]
    comarques = yield_df[ymask]["comarca"].unique()
    d = daily_f[daily_f["comarca"].isin(comarques)].copy()

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
        ax.axvspan(day_lo, day_hi, color=wc, alpha=0.07, zorder=0,
                   label=wname.replace("_", " ").title())
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.75, ncol=len(windows))

    # twin axis for second variable (created here so both share x-ticks)
    ax2 = ax.twinx() if var2 else None
    lpf_col2 = f"{var2}_lpf{tau2}" if var2 else None

    # colour index comes from position in the FULL year list so colours match scatter
    _ref_years = list(all_years) if all_years is not None else list(years)

    # one line per agronomic year
    for year in years:
        yr_idx    = _ref_years.index(year)
        agr_start = pd.Timestamp(year - 1, 10, 1)
        agr_end   = pd.Timestamp(year, 9, 30)
        sub = d[(d["date"] >= agr_start) & (d["date"] <= agr_end)]
        if sub.empty:
            continue
        color = yr_cmap(yr_norm(yr_idx))

        # — primary variable (solid) —
        cols_needed = [lpf_col] + ([lpf_col2] if lpf_col2 else [])
        dm = (sub.groupby("date")[cols_needed].mean()
                  .reset_index()
                  .sort_values("date"))
        dm["agr_day"] = (dm["date"] - agr_start).dt.days + 1

        ax.plot(dm["agr_day"], dm[lpf_col], color=color, lw=1.8, alpha=0.85)
        pk_i = dm[lpf_col].idxmax()
        ax.scatter([dm.loc[pk_i, "agr_day"]], [dm.loc[pk_i, lpf_col]],
                   color=color, s=40, zorder=5,
                   edgecolors="white", linewidths=0.5)

        # — secondary variable (dashed, right axis) —
        if ax2 is not None and lpf_col2 in dm.columns:
            ax2.plot(dm["agr_day"], dm[lpf_col2],
                     color=color, lw=1.8, alpha=0.85, ls="--")

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

    if ax2 is not None:
        unit2 = VAR_UNITS.get(var2, "")
        ax2.set_ylabel(f"{var2} LPF{tau2}  ({unit2})  [dashed]",
                       fontsize=9, color="0.35")
        ax2.tick_params(labelsize=8.5, colors="0.35")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_edgecolor("0.55")


def make_figure(merged, daily_f, yield_df, pheno_key, var, feat_type, tau, threshold,
                var2=None, tau2=None, selected_years=None):
    windows   = list(PHENO_WINDOWS[pheno_key].keys())
    feat_cols = windows + ["peak_doy"]
    unit      = VAR_UNITS.get(var, "")
    n_panels  = len(feat_cols)

    def xlabel(col):
        if col == "peak_doy":
            return f"Peak DOY of {var} LPF{tau} (agronomic year)"
        if feat_type == "Raw window mean":
            return f"Mean {var} — {col} ({unit})"
        if feat_type == "LPF peak value":
            return f"Peak {var} LPF{tau} — {col} ({unit})"
        return f"Days {var} LPF{tau} > p{threshold:.0f} — {col} (d)"

    # Clip bounds: mean ± 2.5σ of yield across the full crop sample
    y_all = merged["yield_tha"].values.astype(float)
    y_all = y_all[np.isfinite(y_all)]
    y_lo  = y_all.mean() - 2.5 * y_all.std()
    y_hi  = y_all.mean() + 2.5 * y_all.std()

    years      = sorted(merged["year"].unique())
    n_yr       = len(years)
    yr_to_idx  = {y: i for i, y in enumerate(years)}
    cmap       = plt.get_cmap("viridis", n_yr)
    norm       = plt.Normalize(vmin=-0.5, vmax=n_yr - 0.5)
    colors     = np.array([yr_to_idx[y] for y in merged["year"]])

    fig = plt.figure(figsize=(5 * n_panels, 9.5))
    gs  = gridspec.GridSpec(2, n_panels,
                            height_ratios=[5, 3.2],
                            hspace=0.45,
                            left=0.07, right=0.87,
                            wspace=0.08, top=0.91, bottom=0.07)
    axes  = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    ts_ax = fig.add_subplot(gs[1, :])

    fig.suptitle(f"{pheno_key}  ·  {var}  ·  {feat_type}  (τ={tau} d)",
                 fontsize=12, fontweight="bold")

    for ax, col in zip(axes, feat_cols):
        if col not in merged.columns:
            ax.axis("off")
            continue
        x = merged[col].values.astype(float)
        y = merged["yield_tha"].values.astype(float)
        in_mask  = np.isfinite(x) & np.isfinite(y) & (y >= y_lo) & (y <= y_hi)
        out_mask = np.isfinite(x) & np.isfinite(y) & ~in_mask
        ax.scatter(x[in_mask], y[in_mask], c=colors[in_mask],
                   s=28, alpha=0.85, zorder=3, edgecolors="none",
                   cmap=cmap, norm=norm)
        if out_mask.any():
            ax.scatter(x[out_mask], y[out_mask], c="0.7",
                       s=22, alpha=0.45, zorder=2, linewidths=0, marker="x")
        if in_mask.sum() >= 2 and np.ptp(x[in_mask]) > 0:
            slope, intercept, r, p, _ = stats.linregress(x[in_mask], y[in_mask])
            r2          = r ** 2
            x_range     = x[in_mask].max() - x[in_mask].min()
            delta_y_pct = slope * x_range / y[in_mask].mean() * 100
            xl = np.linspace(x[in_mask].min(), x[in_mask].max(), 200)
            ax.plot(xl, slope * xl + intercept, "k--", lw=1.6, zorder=4)
            ax.text(0.05, 0.96,
                    f"R² = {r2:.3f}\np = {p:.2e}\nn = {in_mask.sum()}\n"
                    f"slope = {slope:.3f}\nΔŷ = {delta_y_pct:.1f}% of mean",
                    transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        panel_title = "Peak DOY" if col == "peak_doy" else col.replace("_", " ").title()
        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlabel(xlabel(col), fontsize=8.5)
        ax.set_ylabel("Yield (t ha⁻¹)" if ax is axes[0] else "", fontsize=9)

    # seasonal time-series panel (bottom row) — filtered to selected_years only
    sel = set(selected_years) if selected_years is not None else set(years)
    ts_years = [y for y in years if y in sel]
    _draw_timeseries(ts_ax, daily_f, yield_df, pheno_key, var, tau,
                     cmap, norm, ts_years,
                     var2=var2, tau2=tau2, all_years=years)

    # shared colourbar (spans both rows)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.895, 0.07, 0.018, 0.84])
    cb = fig.colorbar(sm, cax=cbar_ax, label="Year")
    cb.set_ticks(list(range(len(years))))
    cb.set_ticklabels([str(y) for y in years])
    cb.ax.tick_params(labelsize=8)
    return fig


# ── page layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Yield–Climate Explorer", layout="wide")
st.title("🌿 Yield–Climate Explorer")

daily_f, yield_df = load_data()
csv_keys = yield_df["pheno_key"].unique()
crops = sorted(
    k for k in PHENO_WINDOWS if k not in EXCL
    and (k in csv_keys or PHENO_KEY_ALIAS.get(k) in csv_keys)
)

with st.sidebar:
    st.header("Controls")
    pheno_key = st.selectbox("Crop",         crops)
    var       = st.selectbox("Variable",     VARIABLES)
    feat_type = st.selectbox("Feature type", FEAT_TYPES, index=1)
    tau       = st.selectbox("τ (days)",     TAUS,       index=1)
    threshold = st.slider("Peak-width percentile",
                          min_value=50, max_value=95, value=DEF_THRESH, step=5,
                          help="Days above this percentile of the filtered signal (per comarca)")
    st.markdown("---")
    st.subheader("Second variable (time series only)")
    _var2_opts = ["None"] + [v for v in VARIABLES if v != var]
    _var2_sel  = st.selectbox("Variable 2", _var2_opts, index=0)
    var2  = None if _var2_sel == "None" else _var2_sel
    tau2  = int(st.selectbox("τ₂ (days)", TAUS, index=1)) if var2 else None
    st.markdown("---")
    save_btn  = st.button("💾 Save figure to figures/")

merged = extract(daily_f, yield_df, pheno_key, var, feat_type, int(tau), threshold)

if merged.empty:
    st.warning("No data found for this selection.")
else:
    all_years = sorted(merged["year"].unique())

    # Year selector — appended to sidebar after data is available
    with st.sidebar:
        st.markdown("---")
        st.subheader("Years")
        selected_years = st.multiselect(
            "Show years",
            options=all_years,
            default=all_years,
            help="Select one or more years to display",
        )

    if not selected_years:
        st.warning("Select at least one year.")
    else:
        fig = make_figure(merged, daily_f, yield_df, pheno_key, var, feat_type,
                          int(tau), threshold, var2=var2, tau2=tau2,
                          selected_years=selected_years)
        st.pyplot(fig, use_container_width=True)

        if save_btn:
            feat_slug = feat_type.replace(" ", "_")
            out = FIG_DIR / f"explorer_{pheno_key}_{var}_{feat_slug}_tau{tau}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            st.success(f"Saved → `{out}`")

        with st.expander("Raw data table"):
            st.dataframe(merged.sort_values(["year", "comarca"]), use_container_width=True)
