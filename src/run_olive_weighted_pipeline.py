"""
Master pipeline runner for area-weighted olive climate analysis.

Runs the 3-step pipeline:
  1. Extract daily climate per olive farm (482 farms × 3653 days)
  2. Aggregate to comarca level using area weighting
  3. Bin into phenological windows (flower, fruit_set, maturation)
  4. Join with yield data

This creates an improved dataset where comarca climate represents the actual
spatial distribution of olive farms (weighted by farm area), rather than
assuming uniform conditions across comarca polygons.
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent

STEPS = [
    ("Step 1: Extract farm-level daily climate", "extract_climate_olive_farms_daily.py"),
    ("Step 2: Aggregate farms to comarca (area-weighted)", "aggregate_farms_to_comarca.py"),
    ("Step 3: Bin into phenological windows", "aggregate_seasonal_olive_weighted.py"),
    ("Step 4: Join with yield data", "join_dataset_olive_weighted.py"),
]


def run_step(name: str, script: str):
    """Run a pipeline step and check for errors"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}\n")

    result = subprocess.run(
        [sys.executable, SRC_DIR / script],
        cwd=SRC_DIR.parent,
    )

    if result.returncode != 0:
        print(f"\n❌ ERROR: {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n✓ {name} completed successfully")


def main():
    print("OLIVE AREA-WEIGHTED CLIMATE PIPELINE")
    print("=" * 70)
    print("This will generate:")
    print("  - agera5_daily_olive_farms.csv (~1.8M rows, ~150 MB)")
    print("  - agera5_daily_comarca_olive_weighted.csv")
    print("  - agera5_seasonal_comarca_olive_weighted.csv")
    print("  - catalan_olive_yield_climate_weighted.csv (final dataset)")
    print()

    for name, script in STEPS:
        run_step(name, script)

    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    print("\nFinal dataset: data/catalan_olive_yield_climate_weighted.csv")
    print("\nNext steps:")
    print("  - Adapt elasticnet_olive_lag.py to use the new dataset")
    print("  - Compare results with centroid-based approach")


if __name__ == "__main__":
    main()
