"""
Aggregate farm-level daily climate to comarca-level using area weighting.

Reads: agera5_daily_olive_farms.csv (482 farms × 3653 days)
Computes: area-weighted mean per comarca-day
  weighted_mean = sum(value_i × area_i) / sum(area_i)

This gives comarca climate that reflects where olive farms actually are,
rather than assuming uniform conditions across the entire comarca polygon.

Output: agera5_daily_comarca_olive_weighted.csv
  Columns: date, comarca, total_area_ha, n_farms,
           et0, precip, tmax_raw, tmax_cor, tmin_raw, tmin_cor, ea, vpd_raw, vpd_cor
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
FARMS_PATH = DATA_DIR / "agera5_daily_olive_farms.csv"


def main():
    if not FARMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FARMS_PATH}\n"
            f"Run: python src/extract_climate_olive_farms_daily.py"
        )

    print(f"Loading {FARMS_PATH}...")
    df = pd.read_csv(FARMS_PATH, parse_dates=['date'])
    print(f"  {len(df):,} rows loaded")
    print(f"  {df['farm_id'].nunique()} farms, {df['comarca'].nunique()} comarques")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    # Climate variables to aggregate
    climate_vars = ['et0', 'precip', 'tmax_raw', 'tmax_cor', 'tmin_raw', 'tmin_cor', 'ea', 'vpd_raw', 'vpd_cor']

    # Check all variables exist
    missing = [v for v in climate_vars if v not in df.columns]
    if missing:
        raise ValueError(f"Missing climate variables in input: {missing}")

    print("\nComputing area-weighted comarca averages...")

    # Group by comarca and date
    grouped = df.groupby(['comarca', 'date'])

    # Compute weighted means
    def weighted_mean(group, var):
        """Area-weighted mean: sum(value × area) / sum(area)"""
        return (group[var] * group['area_ha']).sum() / group['area_ha'].sum()

    results = []
    for (comarca, date), group in grouped:
        row = {
            'comarca': comarca,
            'date': date,
            'total_area_ha': group['area_ha'].sum(),
            'n_farms': len(group),
        }

        # Weighted average for each climate variable
        for var in climate_vars:
            row[var] = weighted_mean(group, var)

        results.append(row)

    df_comarca = pd.DataFrame(results)
    df_comarca = df_comarca.sort_values(['comarca', 'date']).reset_index(drop=True)

    # Save
    out_path = DATA_DIR / "agera5_daily_comarca_olive_weighted.csv"
    df_comarca.to_csv(out_path, index=False)

    print(f"\nSaved {len(df_comarca):,} rows to {out_path}")
    print(f"Comarques: {df_comarca['comarca'].nunique()}")
    print(f"Date range: {df_comarca['date'].min()} → {df_comarca['date'].max()}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Summary stats
    print(f"\nComarca area distribution (total olive farm area):")
    area_summary = df_comarca.groupby('comarca')['total_area_ha'].first().describe()
    print(area_summary)

    print(f"\nSample output:")
    print(df_comarca.head(20))


if __name__ == "__main__":
    main()
