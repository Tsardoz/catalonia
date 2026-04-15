"""
aggregate_seasonal.py  —  Step 4
Compute seasonal weather aggregates per comarca × year × pheno_key.

For each pheno_key, phenological periods are defined as calendar month ranges.
Aggregates per period:
    mean_vpd       — mean daily VPD (kPa)
    max_vpd        — mean of top-10% daily VPD values (stress peaks)
    n_days_vpd_gt2 — count of days VPD > 2 kPa
    cum_et0        — cumulative ET0 (mm)
    mean_tmax      — mean daily maximum temperature (°C)
    mean_tmin      — mean daily minimum temperature (°C)
    cum_precip     — cumulative precipitation (mm)

Output CSV columns:
    comarca, year, pheno_key, {period}_{aggregate}, ...
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

VPD_STRESS_THRESHOLD = 2.0  # kPa
TOP_PCT = 0.10               # top 10% for max_vpd

# Phenological windows: pheno_key -> {period_name: [month_numbers]}
# pheno_keys: cherry, stone_fruit, pome_fruit, apricot, almond, hazelnut, walnut, vine, olive, other
# + split aliases: peach, plum (← stone_fruit windows), apple (← pome_fruit windows)
PHENO_WINDOWS = {
    "cherry": {
        "pre_flower":  [1, 2],
        "flower_set":  [3, 4],
        "fruit_dev":   [5, 6],
    },
    "stone_fruit": {
        "pre_flower":  [2, 3],
        "fruit_set":   [4, 5],
        "fruit_fill":  [6, 7, 8],
    },
    # ── split aliases (same windows as stone_fruit / pome_fruit) ──────────────
    "peach": {
        "pre_flower":  [2, 3],
        "fruit_set":   [4, 5],
        "fruit_fill":  [6, 7, 8],
    },
    "plum": {
        "pre_flower":  [2, 3],
        "fruit_set":   [4, 5],
        "fruit_fill":  [6, 7, 8],
    },
    "pome_fruit": {
        "pre_flower":  [3, 4],
        "cell_div":    [5, 6],
        "cell_exp":    [7, 8, 9],
    },
    "apple": {
        "pre_flower":  [3, 4],
        "cell_div":    [5, 6],
        "cell_exp":    [7, 8, 9],
    },
    "apricot": {
        "flower":      [2, 3],       # early flowering, Feb–Mar
        "fruit_set":   [4, 5],
        "harvest":     [6, 7],       # Jun–Jul harvest
    },
    "almond": {
        "flower":       [2, 3],
        "kernel_fill":  [4, 5, 6],
        "maturation":   [7, 8],
    },
    "hazelnut": {
        "flower":       [2, 3],      # early catkins, Feb–Mar
        "fill":         [4, 5, 6, 7],
        "maturation":   [8, 9],
    },
    "walnut": {
        "flower":       [4, 5],      # late flowering, Apr–May
        "fill":         [6, 7, 8],
        "maturation":   [9, 10],
    },
    "vine": {
        "bud_break":          [3, 4],
        "flower_veraison":    [5, 6, 7],
        "veraison_harvest":   [8, 9],
    },
    "olive": {
        "flower":      [5, 6],
        "fruit_set":   [7, 8],
        "maturation":  [9, 10, 11],
    },
    "other": {
        "season":      [4, 5, 6, 7, 8, 9],
    },
}


def _aggregate_period(df: pd.DataFrame) -> dict:
    """Compute aggregates for a subset of daily climate rows."""
    vpd    = df["vpd_mean"].dropna()
    et0    = df["et0_mean"].dropna()
    tmax   = df["tmax_mean"].dropna()
    tmin   = df["tmin_mean"].dropna()
    precip = df["precip_mean"].dropna()

    n = len(vpd)
    top_n = max(1, int(np.ceil(n * TOP_PCT)))

    return {
        "mean_vpd":        vpd.mean() if n > 0 else np.nan,
        "max_vpd":         vpd.nlargest(top_n).mean() if n > 0 else np.nan,
        "n_days_vpd_gt2":  int((vpd > VPD_STRESS_THRESHOLD).sum()),
        "cum_et0":         et0.sum() if len(et0) > 0 else np.nan,
        "mean_tmax":       tmax.mean() if len(tmax) > 0 else np.nan,
        "mean_tmin":       tmin.mean() if len(tmin) > 0 else np.nan,
        "cum_precip":      precip.sum() if len(precip) > 0 else np.nan,
    }


def aggregate_pheno_key(climate: pd.DataFrame, pheno_key: str) -> pd.DataFrame:
    """
    For one pheno_key, compute period aggregates for all comarca × year combos.
    Returns a DataFrame with one row per (comarca, year) and columns
    {period}_{aggregate}.
    """
    windows = PHENO_WINDOWS[pheno_key]
    period_frames = []

    for period, months in windows.items():
        mask = climate["month"].isin(months)
        sub = climate[mask].copy()
        agg = (
            sub.groupby(["comarca", "year"])
            .apply(_aggregate_period)
            .apply(pd.Series)
            .reset_index()
        )
        agg.columns = [
            "comarca", "year",
            f"{period}_mean_vpd",
            f"{period}_max_vpd",
            f"{period}_n_days_vpd_gt2",
            f"{period}_cum_et0",
            f"{period}_mean_tmax",
            f"{period}_mean_tmin",
            f"{period}_cum_precip",
        ]
        period_frames.append(agg)

    if not period_frames:
        return pd.DataFrame()

    # Merge all periods on comarca × year
    merged = period_frames[0]
    for pf in period_frames[1:]:
        merged = merged.merge(pf, on=["comarca", "year"], how="outer")

    merged["pheno_key"] = pheno_key
    return merged


def aggregate_all(climate: pd.DataFrame) -> pd.DataFrame:
    climate = climate.copy()
    climate["date"] = pd.to_datetime(climate["date"])
    climate["year"]  = climate["date"].dt.year
    climate["month"] = climate["date"].dt.month

    frames = []
    for pheno_key in PHENO_WINDOWS:
        print(f"  Aggregating pheno_key='{pheno_key}' ...")
        df = aggregate_pheno_key(climate, pheno_key)
        if not df.empty:
            frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    # Reorder: comarca, year, pheno_key first
    id_cols = ["comarca", "year", "pheno_key"]
    agg_cols = [c for c in result.columns if c not in id_cols]
    return result[id_cols + agg_cols]


if __name__ == "__main__":
    climate_path = DATA_DIR / "agera5_daily_catalonia.csv"
    if not climate_path.exists():
        raise FileNotFoundError("Run extract_climate.py first to generate agera5_daily_catalonia.csv")

    print(f"Loading {climate_path.name} ...")
    climate = pd.read_csv(climate_path)
    print(f"  {len(climate):,} rows, {climate['comarca'].nunique()} comarques")

    df = aggregate_all(climate)

    out = DATA_DIR / "agera5_seasonal_catalonia.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows to {out}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3).to_string(index=False))
