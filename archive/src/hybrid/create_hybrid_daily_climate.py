"""
Create hybrid daily climate dataset for distributed-lag analysis.

Combines area-weighted and centroid daily climate data for the olive sample:
  - 23 comarques: area-weighted from olive farms (agera5_daily_comarca_olive_weighted.csv)
  - 13 comarques: centroid-based fallback for olive-yield comarques only

Output: agera5_daily_hybrid_olive.csv
  - Same structure as agera5_daily_catalonia.csv
  - 36 comarques × 3653 days = ~131K rows
  - Can be used directly with elasticnet_olive_lag.py
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

DATA_DIR = Path(__file__).parent.parent / "data"

WEIGHTED_DAILY = DATA_DIR / "agera5_daily_comarca_olive_weighted.csv"
CENTROID_DAILY = DATA_DIR / "agera5_daily_catalonia.csv"
FARMS_PATH = DATA_DIR / "olive_groves_catalonia.gpkg"
YIELD_PATH = DATA_DIR / "catalan_woody_yield_raw.csv"


def main():
    print("="*70)
    print("CREATING HYBRID DAILY CLIMATE DATASET")
    print("="*70)

    # Get list of comarques with area-weighted data
    print(f"\nLoading farm data to identify area-weighted comarques...")
    farms = gpd.read_file(FARMS_PATH)
    weighted_comarques = set(farms['comarca'].unique())
    print(f"  {len(weighted_comarques)} comarques with CORINE farms")

    # Restrict fallback daily climate to comarques that actually appear in olive yield data
    print(f"\nLoading olive yield data to identify target comarques...")
    yield_df = pd.read_csv(YIELD_PATH, usecols=['comarca', 'crop_group'])
    olive_comarques = set(yield_df.loc[yield_df['crop_group'] == 'olive', 'comarca'].unique())
    print(f"  {len(olive_comarques)} comarques with olive yield data")

    # Load area-weighted daily data
    print(f"\nLoading area-weighted daily climate from {WEIGHTED_DAILY.name}...")
    df_weighted = pd.read_csv(WEIGHTED_DAILY, parse_dates=['date'])
    print(f"  {len(df_weighted):,} rows")
    print(f"  {df_weighted['comarca'].nunique()} comarques")
    print(f"  Columns: {list(df_weighted.columns)}")

    # Load centroid daily data
    print(f"\nLoading centroid daily climate from {CENTROID_DAILY.name}...")
    df_centroid = pd.read_csv(CENTROID_DAILY, parse_dates=['date'])
    print(f"  {len(df_centroid):,} rows")
    print(f"  {df_centroid['comarca'].nunique()} comarques")

    # Filter centroid data to only missing olive-yield comarques
    fallback_comarques = olive_comarques - weighted_comarques
    df_centroid_filtered = df_centroid[df_centroid['comarca'].isin(fallback_comarques)].copy()
    print(f"\nCentroid data for missing comarques:")
    print(f"  {len(df_centroid_filtered):,} rows")
    print(f"  {df_centroid_filtered['comarca'].nunique()} comarques")
    missing_comarques = sorted(df_centroid_filtered['comarca'].unique())
    for comarca in missing_comarques:
        print(f"    {comarca}")

    # Align column names
    # Area-weighted has: et0, precip, ea, tmax_raw, tmax_cor, tmin_raw, tmin_cor, vpd_raw, vpd_cor
    # Centroid has: et0_mean, et0_min, et0_max, ea_mean, ..., vpd_mean, vpd_min, vpd_max, precip_mean, ...

    # Map area-weighted columns to match centroid naming convention
    # Use elevation-corrected values (tmax_cor, vpd_cor) as the "mean"
    print(f"\nAligning column names...")

    df_weighted_aligned = df_weighted[['date', 'comarca', 'total_area_ha', 'n_farms']].copy()
    df_weighted_aligned['et0_mean'] = df_weighted['et0']
    df_weighted_aligned['precip_mean'] = df_weighted['precip']
    df_weighted_aligned['ea_mean'] = df_weighted['ea']
    df_weighted_aligned['tmax_mean'] = df_weighted['tmax_cor']  # Use elevation-corrected
    df_weighted_aligned['tmin_mean'] = df_weighted['tmin_cor']  # Use elevation-corrected
    df_weighted_aligned['vpd_mean'] = df_weighted['vpd_cor']    # Use elevation-corrected

    # Add climate_method flag
    df_weighted_aligned['climate_method'] = 'area_weighted'

    # Select matching columns from centroid data
    centroid_cols = ['date', 'comarca', 'et0_mean', 'precip_mean', 'ea_mean',
                     'tmax_mean', 'tmin_mean', 'vpd_mean']
    df_centroid_aligned = df_centroid_filtered[centroid_cols].copy()
    df_centroid_aligned['climate_method'] = 'centroid'
    df_centroid_aligned['total_area_ha'] = None
    df_centroid_aligned['n_farms'] = None

    # Combine
    print(f"\nCombining datasets...")
    df_hybrid = pd.concat([df_weighted_aligned, df_centroid_aligned], ignore_index=True)

    # Reorder columns to match standard format
    final_cols = ['date', 'comarca', 'climate_method', 'total_area_ha', 'n_farms',
                  'et0_mean', 'precip_mean', 'ea_mean', 'tmax_mean', 'tmin_mean', 'vpd_mean']
    df_hybrid = df_hybrid[final_cols]
    df_hybrid = df_hybrid.sort_values(['comarca', 'date']).reset_index(drop=True)

    # Save
    out_path = DATA_DIR / "agera5_daily_hybrid_olive.csv"
    df_hybrid.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print("HYBRID DAILY CLIMATE CREATED")
    print('='*70)
    print(f"Saved to: {out_path}")
    print(f"\nTotal rows: {len(df_hybrid):,}")
    print(f"Comarques: {df_hybrid['comarca'].nunique()}")
    print(f"Date range: {df_hybrid['date'].min()} → {df_hybrid['date'].max()}")

    print(f"\nClimate method breakdown:")
    method_counts = df_hybrid.groupby('climate_method')['comarca'].nunique()
    for method, count in method_counts.items():
        print(f"  {method:15s}: {count} comarques")

    print(f"\nRows per method:")
    print(df_hybrid['climate_method'].value_counts())

    print(f"\nSample data:")
    print(df_hybrid.head(10))

    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION")
    print('='*70)

    # Check for duplicates
    dupes = df_hybrid.duplicated(subset=['comarca', 'date'], keep=False)
    if dupes.sum() == 0:
        print("✓ No duplicate comarca-date combinations")
    else:
        print(f"❌ ERROR: {dupes.sum()} duplicates found!")

    # Check date coverage
    date_counts = df_hybrid.groupby('comarca')['date'].nunique()
    print(f"\nDays per comarca:")
    print(f"  Min: {date_counts.min()}")
    print(f"  Max: {date_counts.max()}")
    print(f"  Expected: 3653 (2015-2024, 10 years)")

    # Check for missing values
    missing = df_hybrid[['vpd_mean', 'et0_mean', 'precip_mean', 'tmax_mean']].isna().sum()
    if missing.sum() == 0:
        print(f"\n✓ No missing climate values")
    else:
        print(f"\n⚠ Missing values:")
        print(missing[missing > 0])

    print(f"\n{'='*70}")
    print("READY FOR DISTRIBUTED-LAG ANALYSIS")
    print('='*70)
    print(f"\nTo use with existing regression script:")
    print(f"  1. Update CLIMATE_CSV path in elasticnet_olive_lag.py:")
    print(f"     CLIMATE_CSV = DATA / 'agera5_daily_hybrid_olive.csv'")
    print(f"  2. Run: python src/elasticnet_olive_lag.py")


if __name__ == "__main__":
    main()
