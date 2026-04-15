"""
Comprehensive validation of the hybrid olive dataset.

Checks for:
  1. Data completeness (missing values, expected rows)
  2. Data quality (outliers, impossible values)
  3. Consistency between datasets
  4. Methodological integrity (area-weighted vs centroid alignment)
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
HYBRID_PATH = DATA_DIR / "catalan_olive_yield_climate_hybrid.csv"
WEIGHTED_PATH = DATA_DIR / "catalan_olive_yield_climate_weighted.csv"
CENTROID_PATH = DATA_DIR / "catalan_woody_yield_climate.csv"


def print_header(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)


def check_completeness(df):
    """Check for missing values and data completeness"""
    print_header("1. DATA COMPLETENESS")

    print(f"Total rows: {len(df)}")
    print(f"Expected: 320 (36 comarques × 9 years, 2016-2024)")

    # Check for missing values
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]

    if len(missing_cols) == 0:
        print("✓ No missing values")
    else:
        print(f"\n⚠ Missing values found:")
        for col, count in missing_cols.items():
            pct = 100 * count / len(df)
            print(f"  {col:30s}: {count:4d} ({pct:.1f}%)")

    # Check year coverage
    print(f"\nYear distribution:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:3d} comarques")

    # Check comarca coverage
    print(f"\nComarca observations:")
    comarca_counts = df.groupby('comarca').size()
    print(f"  Min: {comarca_counts.min()} years")
    print(f"  Max: {comarca_counts.max()} years")
    print(f"  Mean: {comarca_counts.mean():.1f} years")

    # Identify comarques with incomplete data
    incomplete = comarca_counts[comarca_counts < 9]
    if len(incomplete) > 0:
        print(f"\n⚠ Comarques with <9 years of data:")
        for comarca, count in incomplete.items():
            print(f"  {comarca:25s}: {count} years")
    else:
        print("\n✓ All comarques have 9 years of data")


def check_data_quality(df):
    """Check for outliers and impossible values"""
    print_header("2. DATA QUALITY")

    # Yield bounds (olives typically 0.5-5 t/ha, but can vary)
    yield_col = 'yield_tha'
    yield_min, yield_max = df[yield_col].min(), df[yield_col].max()
    print(f"\nYield range: {yield_min:.2f} - {yield_max:.2f} t/ha")

    if yield_min < 0:
        print(f"  ❌ ERROR: Negative yield found!")
    elif yield_min < 0.1:
        print(f"  ⚠ WARNING: Very low yield (<0.1 t/ha)")
    else:
        print(f"  ✓ Yield minimum is reasonable")

    if yield_max > 10:
        print(f"  ⚠ WARNING: Very high yield (>10 t/ha)")
    else:
        print(f"  ✓ Yield maximum is reasonable")

    # VPD bounds (typically 0.5-4 kPa for Mediterranean)
    vpd_cols = [c for c in df.columns if 'vpd' in c and 'n_days' not in c]
    print(f"\nVPD ranges (kPa):")
    for col in vpd_cols:
        vmin, vmax = df[col].min(), df[col].max()
        print(f"  {col:30s}: {vmin:.3f} - {vmax:.3f}")
        if vmin < 0:
            print(f"    ❌ ERROR: Negative VPD!")
        if vmax > 6:
            print(f"    ⚠ WARNING: Very high VPD (>6 kPa)")

    # Temperature bounds
    temp_cols = [c for c in df.columns if 'tmax' in c or 'tmin' in c]
    print(f"\nTemperature ranges (°C):")
    for col in temp_cols:
        tmin, tmax = df[col].min(), df[col].max()
        print(f"  {col:30s}: {tmin:.1f} - {tmax:.1f}")
        if 'tmax' in col and (tmin < 0 or tmax > 50):
            print(f"    ⚠ WARNING: Unusual temperature range")
        if 'tmin' in col and (tmin < -10 or tmax > 30):
            print(f"    ⚠ WARNING: Unusual temperature range")

    # Precipitation bounds
    precip_cols = [c for c in df.columns if 'precip' in c]
    print(f"\nPrecipitation ranges (mm):")
    for col in precip_cols:
        pmin, pmax = df[col].min(), df[col].max()
        print(f"  {col:30s}: {pmin:.1f} - {pmax:.1f}")
        if pmin < 0:
            print(f"    ❌ ERROR: Negative precipitation!")
        if pmax > 1000:
            print(f"    ⚠ WARNING: Very high precipitation (>1000mm)")


def check_consistency(df):
    """Check consistency between area-weighted and centroid data"""
    print_header("3. CONSISTENCY WITH SOURCE DATASETS")

    # Load source datasets
    df_weighted = pd.read_csv(WEIGHTED_PATH)
    df_centroid = pd.read_csv(CENTROID_PATH)
    df_centroid = df_centroid[df_centroid['pheno_key'] == 'olive']

    # Check area-weighted rows match exactly
    hybrid_weighted = df[df['climate_method'] == 'area_weighted'].copy()
    print(f"\nArea-weighted data:")
    print(f"  Hybrid dataset: {len(hybrid_weighted)} rows")
    print(f"  Source dataset: {len(df_weighted)} rows")

    if len(hybrid_weighted) == len(df_weighted):
        print("  ✓ Row counts match")

        # Check if data values match
        merge_check = hybrid_weighted.merge(
            df_weighted[['comarca', 'year', 'yield_tha', 'flower_mean_vpd']],
            on=['comarca', 'year'],
            suffixes=('_hybrid', '_source')
        )

        yield_diff = (merge_check['yield_tha_hybrid'] - merge_check['yield_tha_source']).abs()
        vpd_diff = (merge_check['flower_mean_vpd_hybrid'] - merge_check['flower_mean_vpd_source']).abs()

        if yield_diff.max() < 0.001:
            print("  ✓ Yield values match source")
        else:
            print(f"  ❌ ERROR: Yield values differ (max diff: {yield_diff.max():.6f})")

        if vpd_diff.max() < 0.001:
            print("  ✓ VPD values match source")
        else:
            print(f"  ❌ ERROR: VPD values differ (max diff: {vpd_diff.max():.6f})")
    else:
        print(f"  ❌ ERROR: Row count mismatch!")

    # Check centroid rows
    hybrid_centroid = df[df['climate_method'] == 'centroid'].copy()
    df_centroid_2016plus = df_centroid[df_centroid['year'] >= 2016]

    print(f"\nCentroid data:")
    print(f"  Hybrid dataset: {len(hybrid_centroid)} rows")
    print(f"  Source dataset (2016+): {len(df_centroid_2016plus)} rows (before filtering)")

    # Count expected centroid rows (comarques not in weighted dataset)
    weighted_comarques = set(df_weighted['comarca'].unique())
    expected_centroid = df_centroid_2016plus[
        ~df_centroid_2016plus['comarca'].isin(weighted_comarques)
    ]
    print(f"  Expected centroid rows: {len(expected_centroid)}")

    if len(hybrid_centroid) == len(expected_centroid):
        print("  ✓ Row counts match expected")
    else:
        print(f"  ⚠ Row count mismatch (difference: {len(hybrid_centroid) - len(expected_centroid)})")


def check_method_flag(df):
    """Validate climate_method flag integrity"""
    print_header("4. CLIMATE METHOD FLAG")

    method_counts = df['climate_method'].value_counts()
    print(f"\nMethod distribution:")
    for method, count in method_counts.items():
        pct = 100 * count / len(df)
        print(f"  {method:15s}: {count:3d} ({pct:.1f}%)")

    # Check all rows have a method
    missing_method = df['climate_method'].isna().sum()
    if missing_method == 0:
        print("\n✓ All rows have climate_method assigned")
    else:
        print(f"\n❌ ERROR: {missing_method} rows missing climate_method")

    # Check for invalid values
    valid_methods = {'area_weighted', 'centroid'}
    invalid = df[~df['climate_method'].isin(valid_methods)]
    if len(invalid) == 0:
        print("✓ All climate_method values are valid")
    else:
        print(f"❌ ERROR: {len(invalid)} rows have invalid climate_method")


def check_duplicates(df):
    """Check for duplicate rows"""
    print_header("5. DUPLICATE CHECK")

    dupes = df.duplicated(subset=['comarca', 'year'], keep=False)
    n_dupes = dupes.sum()

    if n_dupes == 0:
        print("✓ No duplicate comarca-year combinations")
    else:
        print(f"❌ ERROR: {n_dupes} duplicate rows found!")
        print("\nDuplicates:")
        print(df[dupes][['comarca', 'year', 'climate_method']].sort_values(['comarca', 'year']))


def check_correlations(df):
    """Check correlations between climate variables"""
    print_header("6. CLIMATE VARIABLE CORRELATIONS")

    # Check VPD vs temperature (should be positive correlation)
    r_flower = df[['flower_mean_vpd', 'flower_mean_tmax']].corr().iloc[0, 1]
    r_fruitset = df[['fruit_set_mean_vpd', 'fruit_set_mean_tmax']].corr().iloc[0, 1]

    print(f"\nVPD vs Tmax correlation:")
    print(f"  Flowering: r = {r_flower:.3f}")
    print(f"  Fruit set: r = {r_fruitset:.3f}")

    if r_flower > 0.5 and r_fruitset > 0.5:
        print("  ✓ Positive correlation as expected (VPD increases with temperature)")
    else:
        print("  ⚠ WARNING: Weak or negative correlation")

    # Check precipitation vs VPD (should be negative/weak correlation)
    r_precip_vpd = df[['flower_cum_precip', 'flower_mean_vpd']].corr().iloc[0, 1]
    print(f"\nPrecipitation vs VPD correlation: r = {r_precip_vpd:.3f}")
    if abs(r_precip_vpd) < 0.5:
        print("  ✓ Weak correlation as expected")


def main():
    print_header("HYBRID OLIVE DATASET VALIDATION")
    print(f"File: {HYBRID_PATH}")

    if not HYBRID_PATH.exists():
        print(f"\n❌ ERROR: File not found!")
        print(f"Run: python src/create_hybrid_olive_dataset.py")
        return

    # Load dataset
    df = pd.read_csv(HYBRID_PATH)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Run all checks
    check_completeness(df)
    check_data_quality(df)
    check_consistency(df)
    check_method_flag(df)
    check_duplicates(df)
    check_correlations(df)

    # Final summary
    print_header("VALIDATION SUMMARY")

    print("\nDataset ready for analysis if:")
    print("  ✓ 320 rows (36 comarques × 9 years)")
    print("  ✓ No missing values in key columns")
    print("  ✓ Reasonable data ranges")
    print("  ✓ Consistency with source datasets")
    print("  ✓ No duplicates")
    print("  ✓ Valid climate_method flags")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
