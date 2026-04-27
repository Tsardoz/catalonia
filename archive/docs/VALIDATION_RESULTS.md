# Hybrid Dataset Validation Results

**Date:** 2026-04-15
**File:** `data/catalan_olive_yield_climate_hybrid.csv`
**Status:** ✅ **PASSED ALL CHECKS**

## Summary

**320 observations** validated successfully:
- 207 area-weighted (64.7%)
- 113 centroid (35.3%)
- 36 comarques
- 9 years (2016-2024)

## Validation Results

### ✅ 1. Data Completeness
- **320 rows** (exactly as expected)
- **Zero missing values** in all columns
- Year coverage: 2016-2024 (9 years)
- Minor note: 2 comarques have <9 years due to missing data in source
  - Garrotxa: 8 years
  - Pallars Sobirà: 6 years

### ✅ 2. Data Quality
**Yield (t/ha):**
- Range: 0.20 - 2.58 t/ha ✓ Reasonable for rainfed olives

**VPD (kPa):**
- Flowering mean: 0.900 - 3.462 kPa ✓
- Fruit set mean: 1.415 - 4.034 kPa ✓
- Maturation mean: 0.727 - 1.824 kPa ✓
- All values positive, no outliers

**Temperature (°C):**
- Flowering Tmax: 15.7 - 31.0°C ✓
- Fruit set Tmax: 21.5 - 34.8°C ✓
- Maturation Tmax: 12.8 - 23.6°C ✓
- All values reasonable for Mediterranean climate

**Precipitation (mm):**
- All periods: positive values, range 10-540 mm ✓
- No impossible values

### ✅ 3. Consistency with Source Datasets
**Area-weighted data (207 rows):**
- Row count matches source ✓
- Yield values match exactly ✓
- VPD values match exactly ✓

**Centroid data (113 rows):**
- Row count matches expected ✓
- Correct filtering (excludes 23 comarques with area-weighted data) ✓

### ✅ 4. Climate Method Flag
- All 320 rows have valid `climate_method` value ✓
- No missing flags ✓
- Only valid values: `area_weighted`, `centroid` ✓

### ✅ 5. Duplicate Check
- **Zero duplicates** in comarca-year combinations ✓
- Each observation is unique ✓

### ✅ 6. Climate Variable Correlations
**VPD vs Temperature:**
- Flowering: r = 0.874 ✓ (strong positive, as expected)
- Fruit set: r = 0.876 ✓ (strong positive, as expected)

**Precipitation vs VPD:**
- r = -0.510 (moderate negative, as expected)

All correlations follow expected physical relationships.

## Conclusion

The hybrid dataset is **scientifically valid** and **ready for regression analysis**.

- No data quality issues
- No missing values
- No duplicates
- Consistent with source datasets
- Reasonable value ranges
- Expected correlations between variables

**Recommendation:** Use `catalan_olive_yield_climate_hybrid.csv` for your distributed-lag analysis with confidence.

## How to Re-run Validation

```bash
source .venv/bin/activate
python src/validate_hybrid_dataset.py
```

This validation script checks:
1. Data completeness
2. Data quality (outliers, ranges)
3. Consistency with source datasets
4. Method flag integrity
5. Duplicates
6. Correlations

Run this anytime you regenerate the hybrid dataset to ensure data integrity.
