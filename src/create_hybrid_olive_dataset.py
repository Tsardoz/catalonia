"""
Create hybrid olive yield-climate dataset combining area-weighted and centroid approaches.

For 23 comarques with CORINE farms:
  - Use area-weighted climate from olive farm locations (best available)

For 13 comarques without CORINE farms:
  - Use centroid-based climate from standard pipeline (fallback)

Output: catalan_olive_yield_climate_hybrid.csv (320 rows)
  - Same structure as other datasets
  - Added column: climate_method ('area_weighted' or 'centroid')
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

WEIGHTED_PATH = DATA_DIR / "catalan_olive_yield_climate_weighted.csv"
CENTROID_PATH = DATA_DIR / "catalan_woody_yield_climate.csv"


def main():
    # Load area-weighted dataset (23 comarques)
    print(f"Loading area-weighted dataset from {WEIGHTED_PATH.name}...")
    df_weighted = pd.read_csv(WEIGHTED_PATH)
    df_weighted['climate_method'] = 'area_weighted'
    print(f"  {len(df_weighted)} rows, {df_weighted['comarca'].nunique()} comarques")
    weighted_comarques = set(df_weighted['comarca'].unique())

    # Load centroid-based dataset (36 comarques)
    print(f"\nLoading centroid-based dataset from {CENTROID_PATH.name}...")
    df_centroid = pd.read_csv(CENTROID_PATH)
    df_centroid = df_centroid[df_centroid['pheno_key'] == 'olive'].copy()
    print(f"  {len(df_centroid)} olive rows, {df_centroid['comarca'].nunique()} comarques")

    # Filter to missing comarques only
    df_missing = df_centroid[~df_centroid['comarca'].isin(weighted_comarques)].copy()
    df_missing['climate_method'] = 'centroid'
    print(f"\nMissing comarques (using centroid): {df_missing['comarca'].nunique()}")
    for comarca in sorted(df_missing['comarca'].unique()):
        n = len(df_missing[df_missing['comarca'] == comarca])
        print(f"  {comarca:20s} - {n} observations")

    # Ensure column alignment
    # Weighted dataset has elevation-corrected temps (tmax_cor, tmin_cor, vpd_cor)
    # Centroid dataset has raw temps (tmax_mean, tmin_mean, vpd_mean)
    # Need to align column names

    # Get common climate columns (period_aggregate format)
    weighted_cols = set(df_weighted.columns)
    centroid_cols = set(df_centroid.columns)

    # Climate columns that should match
    climate_patterns = [
        'flower_mean_vpd', 'flower_max_vpd', 'flower_n_days_vpd_gt2',
        'flower_cum_et0', 'flower_mean_tmax', 'flower_mean_tmin', 'flower_cum_precip',
        'fruit_set_mean_vpd', 'fruit_set_max_vpd', 'fruit_set_n_days_vpd_gt2',
        'fruit_set_cum_et0', 'fruit_set_mean_tmax', 'fruit_set_mean_tmin', 'fruit_set_cum_precip',
        'maturation_mean_vpd', 'maturation_max_vpd', 'maturation_n_days_vpd_gt2',
        'maturation_cum_et0', 'maturation_mean_tmax', 'maturation_mean_tmin', 'maturation_cum_precip',
    ]

    # Check if all climate columns exist in both
    missing_in_weighted = [c for c in climate_patterns if c not in weighted_cols]
    missing_in_centroid = [c for c in climate_patterns if c not in centroid_cols]

    if missing_in_weighted:
        print(f"\nWARNING: Missing in weighted dataset: {missing_in_weighted}")
    if missing_in_centroid:
        print(f"\nWARNING: Missing in centroid dataset: {missing_in_centroid}")

    # Keep only common columns plus identifiers and climate_method
    keep_cols = [
        'comarca', 'year', 'crop_catalan', 'crop_group', 'pheno_key',
        'seca_ha', 'yield_tha', 'climate_method'
    ] + climate_patterns

    # Filter both datasets to common columns
    df_weighted_clean = df_weighted[[c for c in keep_cols if c in df_weighted.columns]]
    df_missing_clean = df_missing[[c for c in keep_cols if c in df_missing.columns]]

    # Ensure all expected columns exist (fill missing with NaN)
    for col in keep_cols:
        if col not in df_weighted_clean.columns:
            df_weighted_clean[col] = float('nan')
        if col not in df_missing_clean.columns:
            df_missing_clean[col] = float('nan')

    # Reorder columns consistently
    df_weighted_clean = df_weighted_clean[keep_cols]
    df_missing_clean = df_missing_clean[keep_cols]

    # Combine
    print(f"\nCombining datasets...")
    df_hybrid = pd.concat([df_weighted_clean, df_missing_clean], ignore_index=True)

    # Drop 2015 for consistency (area-weighted dataset starts at 2016 due to agronomic year)
    n_before = len(df_hybrid)
    df_hybrid = df_hybrid[df_hybrid['year'] >= 2016].copy()
    n_dropped = n_before - len(df_hybrid)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} observations from 2015 (agronomic year needs Dec 2014)")

    df_hybrid = df_hybrid.sort_values(['comarca', 'year']).reset_index(drop=True)

    # Save
    out_path = DATA_DIR / "catalan_olive_yield_climate_hybrid.csv"
    df_hybrid.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print("HYBRID DATASET CREATED")
    print('='*70)
    print(f"Saved to: {out_path}")
    print(f"\nTotal observations: {len(df_hybrid)}")
    print(f"Comarques: {df_hybrid['comarca'].nunique()}")
    print(f"Years: {sorted(df_hybrid['year'].unique())}")
    print(f"\nClimate method breakdown:")
    print(df_hybrid['climate_method'].value_counts())
    print(f"\nSample:")
    print(df_hybrid.head(10))

    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION")
    print('='*70)
    print(f"Expected: 320 rows (36 comarques × ~9 years)")
    print(f"Actual: {len(df_hybrid)} rows")

    missing_years = df_hybrid.groupby('comarca').size()
    print(f"\nObservations per comarca:")
    print(missing_years.describe())

    if len(df_hybrid) < 310:
        print(f"\nWARNING: Sample size ({len(df_hybrid)}) lower than expected (320)")
        print("Some comarques may have missing years")


if __name__ == "__main__":
    main()
