"""
src/features.py — Shared IIR-filter helpers used by explorer.py and plot_features.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import filtfilt

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"

sys.path.insert(0, str(Path(__file__).parent))
from aggregate_seasonal import PHENO_WINDOWS

VARIABLES  = ["vpd_mean", "vpd_max", "precip_mean", "et0_mean", "tmax_mean"]
TAUS       = [3, 7, 14, 30]
FEAT_TYPES  = ["Raw window mean", "LPF peak value", "LPF peak width"]
# Retired aggregated keys + miscellaneous bucket — excluded from crop lists
EXCL        = {"other", "stone_fruit", "pome_fruit"}
DEF_THRESH  = 75   # percentile of filtered signal — days above this = stress days
VAR_UNITS  = {"vpd_mean": "kPa", "vpd_max": "kPa", "precip_mean": "mm",
              "et0_mean": "mm/d", "tmax_mean": "°C"}

# peach / plum / apple are split from the aggregated stone_fruit / pome_fruit
# CSV rows by filtering on crop_catalan; their windows mirror the parent key.
PHENO_KEY_ALIAS = {
    "peach": "stone_fruit",
    "plum":  "stone_fruit",
    "apple": "pome_fruit",
}
CROP_CATALAN_FILTER = {
    "peach": "Presseguer",
    "plum":  "Pruner",
    "apple": "Pomera",
}

# ── Olive phenological period definitions (DOY-based) ───────────────────────────
# SP1 (Sensitive Period 1 - Flowering to pit hardening): DOY 95-157 ≈ Apr 5 - Jun 6
# NP (Normal Period - Pit hardening): DOY 153-217 ≈ Jun 2 - Aug 5
# SP2 (Sensitive Period 2 - Oil synthesis to harvest): DOY 216-313 ≈ Aug 4 - Nov 9
OLIVE_PHENO_PERIODS = {
    "sp1": {"doy_min": 95, "doy_max": 157, "label": "SP1: Flowering → pit hardening"},
    "np": {"doy_min": 153, "doy_max": 217, "label": "NP: Pit hardening"},
    "sp2": {"doy_min": 216, "doy_max": 313, "label": "SP2: Oil synthesis → harvest"},
}


def phase_for_doy_olive(doy: int) -> str:
    """Convert day-of-year to olive phenological phase."""
    if OLIVE_PHENO_PERIODS["sp1"]["doy_min"] <= doy <= OLIVE_PHENO_PERIODS["sp1"]["doy_max"]:
        return "sp1"
    if OLIVE_PHENO_PERIODS["np"]["doy_min"] <= doy <= OLIVE_PHENO_PERIODS["np"]["doy_max"]:
        return "np"
    if OLIVE_PHENO_PERIODS["sp2"]["doy_min"] <= doy <= OLIVE_PHENO_PERIODS["sp2"]["doy_max"]:
        return "sp2"
    return "other"


def phase_for_date_olive(date_or_mmdd) -> str:
    """Convert date or mm-dd string to olive phenological phase via DOY."""
    if isinstance(date_or_mmdd, str):
        # Convert mm-dd to DOY (using non-leap year 2001 as reference)
        month, day = int(date_or_mmdd[:2]), int(date_or_mmdd[2:])
        d = pd.Timestamp(2001, month, day)
        doy = d.dayofyear
    else:
        doy = date_or_mmdd.dayofyear
    return phase_for_doy_olive(doy)


def phase_for_agroyear_mmdd(mmdd: str) -> str:
    """
    Convert mm-dd to olive phase for the full agronomic year scan (Dec-Nov).

    The agronomic year starts in December (DOY 335+), then continues into the
    calendar year (DOY 1-334 for Jan-Nov).

    Returns phase name with labels suitable for agronomic year context:
    - "winter": Dec-Feb (pre-growth)
    - "pre_flower": Mar-Apr (spring build-up)
    - "sp1": May-Jun (SP1: Flowering to pit hardening)
    - "np": Jun-Aug (NP: Pit hardening)
    - "sp2": Aug-Nov (SP2: Oil synthesis to harvest)
    """
    month = int(mmdd[:2])
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4}:
        return "pre_flower"
    # Use DOY-based logic for May-Nov
    return phase_for_date_olive(mmdd)


# ── Zero-phase low-pass filter (forward-backward) ─────────────────────────────
def _lpf(x: np.ndarray, tau: float) -> np.ndarray:
    """
    Zero-phase first-order low-pass filter via filtfilt.
    Equivalent IIR: y[t] = α·x[t] + (1-α)·y[t-1], α = 1 - exp(-1/τ).
    filtfilt applies it forward then backward → no phase lag, effectively τ/√2.
    """
    a = 1.0 - np.exp(-1.0 / tau)
    b_coef = [a]
    a_coef = [1.0, -(1.0 - a)]
    return filtfilt(b_coef, a_coef, x.astype(float))


def precompute_filters(daily: pd.DataFrame) -> pd.DataFrame:
    """Return daily DataFrame with extra {var}_lpf{tau} columns for every combination."""
    print("Pre-computing filters (zero-phase filtfilt)…", flush=True)
    frames = []
    for comarca, grp in daily.groupby("comarca"):
        grp = grp.sort_values("date").copy()
        for var in VARIABLES:
            x = grp[var].fillna(0.0).values
            for tau in TAUS:
                grp[f"{var}_lpf{tau}"] = _lpf(x, float(tau))
        frames.append(grp)
    print("  Done.", flush=True)
    return pd.concat(frames, ignore_index=True)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract(daily_f: pd.DataFrame, yield_df: pd.DataFrame,
            pheno_key: str, var: str, feat_type: str,
            tau: int, threshold: float) -> pd.DataFrame:
    """
    For a given crop / variable / feature-type, return a merged DataFrame with
    columns: comarca, year, yield_tha, <one col per pheno window>, peak_doy.

    peach / plum / apple are split aliases: they share windows with stone_fruit /
    pome_fruit but are filtered to a single crop_catalan in the yield CSV.
    """
    # Resolve alias (peach/plum/apple → stone_fruit/pome_fruit in the CSV)
    csv_key = PHENO_KEY_ALIAS.get(pheno_key, pheno_key)
    windows = PHENO_WINDOWS[pheno_key]
    lpf_col = f"{var}_lpf{tau}"

    # Build yield mask: pheno_key match + optional crop_catalan filter
    ymask = yield_df["pheno_key"] == csv_key
    if pheno_key in CROP_CATALAN_FILTER:
        ymask &= yield_df["crop_catalan"] == CROP_CATALAN_FILTER[pheno_key]

    comarques = yield_df[ymask]["comarca"].unique()
    crop_yrs  = sorted(yield_df[ymask]["year"].unique())
    d         = daily_f[daily_f["comarca"].isin(comarques)]

    rows = []
    for comarca, grp in d.groupby("comarca"):
        grp = grp.sort_values("date")
        for year in crop_yrs:
            row = {"comarca": comarca, "year": year}
            for wname, months in windows.items():
                sub = grp[grp["month"].isin(months)]
                if feat_type == "Raw window mean":
                    v = sub[var].dropna().values
                    row[wname] = float(v.mean()) if len(v) else np.nan
                elif feat_type == "LPF peak value":
                    v = sub[lpf_col].values
                    row[wname] = float(v.max()) if len(v) else np.nan
                else:  # LPF peak width: days above Nth percentile of the signal
                    # threshold is treated as a percentile (0–100) of the
                    # full filtered signal for this comarca, making the
                    # feature scale-invariant across variables.
                    all_v = grp[lpf_col].dropna().values
                    thr   = np.nanpercentile(all_v, threshold) if len(all_v) else 0.0
                    v     = sub[lpf_col].values
                    row[wname] = int((v > thr).sum()) if len(v) else np.nan
            # Peak DOY over agronomic year Oct(Y-1)–Sep(Y)
            s = grp[(grp["date"] >= pd.Timestamp(year - 1, 10, 1)) &
                    (grp["date"] <= pd.Timestamp(year, 9, 30))]
            if len(s):
                v    = s[lpf_col].values
                doys = s["date"].dt.dayofyear.values
                row["peak_doy"] = int(doys[np.argmax(v)])
            else:
                row["peak_doy"] = np.nan
            rows.append(row)

    feats      = pd.DataFrame(rows)
    crop_yield = yield_df[ymask][["comarca", "year", "yield_tha"]]
    return crop_yield.merge(feats, on=["comarca", "year"], how="inner")

