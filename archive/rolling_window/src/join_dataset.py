"""
join_dataset.py  —  Step 5
Join rainfed yield data with seasonal climate aggregates.

Joins on comarca × year × pheno_key.

If comarca names don't match between yield and climate data, unmatched rows
are reported but not silently dropped — check the name_mismatch output and
fix upstream if needed.

Output: data/catalan_woody_yield_climate.csv
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


# Columns kept in the final output, in this order.
# Extra yield columns (crop_en, seca_kg_ha, grup) are dropped.
YIELD_COLS   = ["comarca", "year", "crop_catalan", "crop_group", "pheno_key",
                "seca_ha", "yield_tha"]


def join(yield_df: pd.DataFrame, seasonal_df: pd.DataFrame) -> pd.DataFrame:
    merged = yield_df.merge(
        seasonal_df,
        on=["comarca", "year", "pheno_key"],
        how="left",
        validate="many_to_one",
    )

    # Report yield rows with no matching climate data
    climate_cols = [c for c in seasonal_df.columns if c not in ("comarca", "year", "pheno_key")]
    # A row has no climate if ALL its pheno_key-specific climate columns are NaN.
    # Use the first climate column that should be populated for every pheno_key (flower/pre_flower).
    sentinel_cols = [c for c in climate_cols if "mean_vpd" in c]
    if sentinel_cols:
        no_climate = merged[sentinel_cols].isna().all(axis=1)
        n_no_climate = no_climate.sum()
        if n_no_climate > 0:
            print(f"Warning: {n_no_climate} yield records have no matching climate data:")
            print(merged.loc[no_climate, ["comarca", "year", "pheno_key"]]
                  .drop_duplicates().to_string(index=False))

    # Reorder: fixed yield columns first, then all climate columns
    keep_yield = [c for c in YIELD_COLS if c in merged.columns]
    keep_climate = [c for c in merged.columns if c not in YIELD_COLS and c not in
                    {"crop_en", "seca_kg_ha", "grup"}]
    return merged[keep_yield + keep_climate]


if __name__ == "__main__":
    yield_path   = DATA_DIR / "catalan_woody_yield_raw.csv"
    seasonal_path = DATA_DIR / "agera5_seasonal_catalonia.csv"

    for p in (yield_path, seasonal_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    yield_df   = pd.read_csv(yield_path)
    seasonal_df = pd.read_csv(seasonal_path)

    print(f"Yield records:   {len(yield_df):,}")
    print(f"Seasonal records: {len(seasonal_df):,}")

    df = join(yield_df, seasonal_df)

    out = DATA_DIR / "catalan_woody_yield_climate.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows to {out}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    # A row has climate data if at least one of its VPD columns is non-NaN
    has_climate = df[[c for c in df.columns if '_vpd' in c]].notna().any(axis=1)
    print(f"\nRecords with climate data: {has_climate.sum():,} / {len(df):,}")
