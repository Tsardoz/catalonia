"""
Join olive yield data with area-weighted comarca seasonal climate.

Reads:
  - catalan_woody_yield_raw.csv (yield data, filter to olive only)
  - agera5_seasonal_comarca_olive_weighted.csv (seasonal climate from farm weighting)

Output: catalan_olive_yield_climate_weighted.csv
  Final dataset for olive yield analysis with area-weighted climate based on
  actual spatial distribution of rainfed olive farms.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
YIELD_PATH = DATA_DIR / "catalan_woody_yield_raw.csv"
CLIMATE_PATH = DATA_DIR / "agera5_seasonal_comarca_olive_weighted.csv"


def main():
    # Load yield data
    if not YIELD_PATH.exists():
        raise FileNotFoundError(f"Missing {YIELD_PATH}")

    print(f"Loading {YIELD_PATH}...")
    df_yield = pd.read_csv(YIELD_PATH)
    print(f"  {len(df_yield):,} total rows")

    # Filter to olive only
    df_olive = df_yield[df_yield['pheno_key'] == 'olive'].copy()
    print(f"  {len(df_olive)} olive rows")
    print(f"  Years: {sorted(df_olive['year'].unique())}")
    print(f"  Comarques: {df_olive['comarca'].nunique()}")

    # Load seasonal climate
    if not CLIMATE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLIMATE_PATH}\n"
            f"Run: python src/aggregate_seasonal_olive_weighted.py"
        )

    print(f"\nLoading {CLIMATE_PATH}...")
    df_climate = pd.read_csv(CLIMATE_PATH)
    print(f"  {len(df_climate)} rows")
    print(f"  Years: {sorted(df_climate['year'].unique())}")
    print(f"  Comarques: {df_climate['comarca'].nunique()}")

    # Join on comarca + year
    print("\nJoining yield + climate...")
    df = df_olive.merge(
        df_climate,
        on=['comarca', 'year', 'pheno_key'],
        how='inner'
    )

    print(f"  {len(df)} rows after join")
    print(f"  Comarques: {df['comarca'].nunique()}")
    print(f"  Years: {sorted(df['year'].unique())}")

    # Drop redundant columns (keep comarca, year, crop info, area, yield, then all climate vars)
    drop_cols = ['crop_en', 'seca_kg_ha']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Reorder: comarca, year, crop info, area/yield, then climate
    key_cols = ['comarca', 'year', 'crop_catalan', 'crop_group', 'pheno_key', 'seca_ha', 'yield_tha']
    climate_cols = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + sorted(climate_cols)]

    # Save
    out_path = DATA_DIR / "catalan_olive_yield_climate_weighted.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e3:.1f} KB")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(10))

    # Sanity checks
    print(f"\nYield range: {df['yield_tha'].min():.2f} – {df['yield_tha'].max():.2f} t/ha")
    print(f"Mean yield: {df['yield_tha'].mean():.2f} t/ha")
    print(f"\nFlowering VPD range: {df['flower_mean_vpd'].min():.3f} – {df['flower_mean_vpd'].max():.3f} kPa")
    print(f"Fruit set VPD range: {df['fruit_set_mean_vpd'].min():.3f} – {df['fruit_set_mean_vpd'].max():.3f} kPa")


if __name__ == "__main__":
    main()
