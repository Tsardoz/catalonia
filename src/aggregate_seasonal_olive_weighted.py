"""
Compute seasonal aggregates from area-weighted comarca daily climate.

Reads: agera5_daily_comarca_olive_weighted.csv
Uses: olive phenological windows from aggregate_seasonal.py
Outputs: agera5_seasonal_comarca_olive_weighted.csv

This is the olive-specific version of aggregate_seasonal.py.
Uses elevation-corrected VPD (vpd_cor) instead of raw VPD.

Output columns:
  comarca, year, pheno_key='olive',
  flower_mean_vpd, flower_max_vpd, flower_n_days_vpd_gt2, flower_cum_et0, flower_mean_tmax, flower_mean_tmin, flower_cum_precip,
  fruit_set_mean_vpd, fruit_set_max_vpd, ...,
  maturation_mean_vpd, maturation_max_vpd, ...
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DAILY_PATH = DATA_DIR / "agera5_daily_comarca_olive_weighted.csv"

VPD_STRESS_THRESHOLD = 2.0  # kPa
TOP_PCT = 0.10               # top 10% for max_vpd

# Olive phenological windows (from aggregate_seasonal.py)
OLIVE_WINDOWS = {
    "flower":      [5, 6],      # May–Jun
    "fruit_set":   [7, 8],      # Jul–Aug
    "maturation":  [9, 10, 11], # Sep–Nov
}


def _aggregate_period(df: pd.DataFrame) -> dict:
    """
    Compute aggregates for a single phenological period.

    Input: daily data for one comarca-year filtered to period months
    Returns: dict of {aggregate_name: value}
    """
    if len(df) == 0:
        return {
            "mean_vpd":       np.nan,
            "max_vpd":        np.nan,
            "n_days_vpd_gt2": 0,
            "cum_et0":        np.nan,
            "mean_tmax":      np.nan,
            "mean_tmin":      np.nan,
            "cum_precip":     np.nan,
        }

    # Use elevation-corrected VPD
    vpd = df["vpd_cor"].values

    return {
        "mean_vpd":       np.nanmean(vpd),
        "max_vpd":        np.nanmean(np.sort(vpd)[int(len(vpd) * (1 - TOP_PCT)):]),  # top 10%
        "n_days_vpd_gt2": int(np.sum(vpd > VPD_STRESS_THRESHOLD)),
        "cum_et0":        np.nansum(df["et0"].values),
        "mean_tmax":      np.nanmean(df["tmax_cor"].values),  # corrected
        "mean_tmin":      np.nanmean(df["tmin_cor"].values),  # corrected
        "cum_precip":     np.nansum(df["precip"].values),
    }


def aggregate_comarca_year(df_comarca_year: pd.DataFrame, year: int) -> dict:
    """
    Compute seasonal aggregates for one comarca-year.

    Input: daily data for one comarca, filtered to one agronomic year (Dec Y-1 → Nov Y)
    Returns: dict of {period_aggregate: value} for all olive periods
    """
    row = {"year": year}

    for period_name, months in OLIVE_WINDOWS.items():
        df_period = df_comarca_year[df_comarca_year["date"].dt.month.isin(months)]
        aggs = _aggregate_period(df_period)

        for agg_name, val in aggs.items():
            row[f"{period_name}_{agg_name}"] = val

    return row


def main():
    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DAILY_PATH}\n"
            f"Run: python src/aggregate_farms_to_comarca.py"
        )

    print(f"Loading {DAILY_PATH}...")
    df = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df['comarca'].nunique()} comarques")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    # Check required columns
    required = ['date', 'comarca', 'vpd_cor', 'tmax_cor', 'tmin_cor', 'et0', 'precip']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\nAggregating by agronomic year (Dec Y-1 → Nov Y)...")

    results = []

    for comarca in sorted(df['comarca'].unique()):
        df_comarca = df[df['comarca'] == comarca].copy()
        df_comarca = df_comarca.sort_values('date')

        # Process years 2016–2024 (agronomic year 2015 needs Dec 2014, which we don't have)
        for year in range(2016, 2025):
            # Agronomic year: Dec 1 (year-1) through Nov 30 (year)
            start = pd.Timestamp(f"{year-1}-12-01")
            end = pd.Timestamp(f"{year}-11-30")

            df_year = df_comarca[(df_comarca['date'] >= start) & (df_comarca['date'] <= end)]

            if len(df_year) == 0:
                continue

            row = {"comarca": comarca}
            row.update(aggregate_comarca_year(df_year, year))
            results.append(row)

    df_out = pd.DataFrame(results)
    df_out['pheno_key'] = 'olive'

    # Reorder columns: comarca, year, pheno_key, then all period aggregates
    first_cols = ['comarca', 'year', 'pheno_key']
    other_cols = [c for c in df_out.columns if c not in first_cols]
    df_out = df_out[first_cols + sorted(other_cols)]

    # Save
    out_path = DATA_DIR / "agera5_seasonal_comarca_olive_weighted.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\nSaved {len(df_out)} rows to {out_path}")
    print(f"Comarques: {df_out['comarca'].nunique()}")
    print(f"Years: {sorted(df_out['year'].unique())}")
    print(f"Columns: {list(df_out.columns)}")
    print(f"\nSample output:")
    print(df_out.head(10))

    # Summary stats
    print(f"\nFlowering VPD summary (elevation-corrected):")
    print(df_out['flower_mean_vpd'].describe())


if __name__ == "__main__":
    main()
