# Olive Phenological Periods Update

## Summary of Changes

Updated all olive timing scan figures and related scripts to use **DOY-based (day-of-year) sensitive period definitions** instead of month-based windows.

## New Phenological Periods

### SP1: Flowering to Pit Hardening
- **DOY 95–157** ≈ **April 5 – June 6**
- Month-based approximation: May–June (includes April portion)
- **Characteristics**: Water stress during this stage interferes with flowering, fruit set, and fruit drop

### NP: Pit Hardening (Normal Period)
- **DOY 153–217** ≈ **June 2 – August 5**
- Month-based approximation: June–July (extends into August)
- **Characteristics**: Most resistant to water deficit; optimal time for regulated deficit irrigation

### SP2: Oil Synthesis to Harvest
- **DOY 216–313** ≈ **August 4 – November 9**
- Month-based approximation: August–November
- **Characteristics**: Water deficits reduce both production and oil quality; active photosynthesis required

## Files Modified

### Core Implementations
1. **src/features.py** - Added DOY-based period definitions and helper functions:
   - `OLIVE_PHENO_PERIODS` constant with exact DOY ranges
   - `phase_for_doy_olive(doy)` - Convert DOY to phase
   - `phase_for_date_olive(date_or_mmdd)` - Convert date/mmdd string to phase
   - `phase_for_agroyear_mmdd(mmdd)` - Handle full agronomic year context

2. **src/aggregate_seasonal.py** - Updated olive window definitions:
   - Changed from month-based ("flower", "fruit_set", "maturation")
   - To DOY-based ("sp1_flower_hardening", "np_pit_hardening", "sp2_oil_synthesis")

### Timing Scan Scripts (Updated & Regenerated)
3. **src/olive_tmax_timing_scan.py** - Regenerated figure with new phase bands
4. **src/olive_vpd_timing_scan.py** - Regenerated figure with new phase bands
5. **src/olive_cwb_timing_scan.py** - Regenerated figure with new phase bands
6. **src/olive_cwb_deficit_timing_scan.py** - Updated and ready to regenerate
7. **src/olive_tmax_window_compare.py** - Regenerated figure with new phase bands
8. **src/plot_olive_timing_summary.py** - Updated phase labels and colors

## Regenerated Figures

All timing scan figures have been regenerated with the new sensitive period definitions:

- `figures/olive_tmax_timing_scan.png`
- `figures/olive_vpd_timing_scan.png`
- `figures/olive_cwb_timing_scan.png`
- `figures/olive_tmax_window_compare.png`
- `figures/olive_timing_summary.png`

## Data Output Changes

Generated CSV files now include the new phase identifiers:

- `data/olive_tmax_timing_scan.csv` - Phase column: sp1, np, sp2
- `data/olive_vpd_timing_scan.csv` - Phase column: sp1, np, sp2
- `data/olive_cwb_timing_scan.csv` - Phase column: winter, pre_flower, sp1, np, sp2

## Backward Compatibility Notes

- **Season-level aggregates** (aggregate_seasonal.py, aggregate_seasonal_olive_weighted.py) still use month-based windows for compatibility with existing datasets
- **Timing scans** (which generate the publication-ready figures) now exclusively use DOY-based sensitive periods
- Helper functions in features.py provide conversion utilities for both approaches
