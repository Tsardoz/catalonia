# Quick Start: Area-Weighted Climate Pipeline

## Run the Complete Pipeline

```bash
source .venv/bin/activate
python src/run_olive_weighted_pipeline.py
```

This will generate:
- `data/agera5_daily_olive_farms.csv` (368 MB, ~2 min)
- `data/agera5_daily_comarca_olive_weighted.csv` (17 MB, ~30 sec)
- `data/agera5_seasonal_comarca_olive_weighted.csv` (50 KB, ~5 sec)
- `data/catalan_olive_yield_climate_weighted.csv` (85 KB, ~1 sec)

**Total runtime:** ~3 minutes

---

## Compare with Centroid Approach

```bash
source .venv/bin/activate
python src/compare_weighted_vs_centroid.py
```

Outputs:
- Console: Summary statistics, correlations, comarca-level differences
- `data/climate_comparison_weighted_vs_centroid.csv`
- `figures/climate_comparison_weighted_vs_centroid.png`
- `figures/climate_differences_distribution.png`

---

## Key Results

**Correlation:** r = 0.93 (flowering VPD)
**Mean difference:** +0.10 kPa (area-weighted higher)
**Sample size:** 207 rows (23 comarques × 9 years)

**Missing:** 13 comarques with yield but no CORINE farms

---

## Files Created

**New scripts:**
```
src/extract_climate_olive_farms_daily.py
src/aggregate_farms_to_comarca.py
src/aggregate_seasonal_olive_weighted.py
src/join_dataset_olive_weighted.py
src/run_olive_weighted_pipeline.py
src/compare_weighted_vs_centroid.py
```

**Documentation:**
```
OLIVE_WEIGHTED_PIPELINE.md
PIPELINE_SUMMARY.md
AREA_WEIGHTED_RESULTS.md
QUICK_START_WEIGHTED.md (this file)
```

**Updated:**
```
.gitignore (added agera5_daily_olive_farms.csv)
CLAUDE.md (not yet updated - do this next)
```

---

## Next Steps

1. **Review results:** Check `AREA_WEIGHTED_RESULTS.md`
2. **Decide approach:** 23-comarca weighted vs 36-comarca hybrid
3. **Adapt analysis:** Create `elasticnet_olive_lag_weighted.py`
4. **Compare results:** Does area-weighting change conclusions?
5. **Document:** Update PLAN.md with method choice

---

## Questions?

See full documentation:
- **Pipeline details:** `OLIVE_WEIGHTED_PIPELINE.md`
- **Method comparison:** `PIPELINE_SUMMARY.md`
- **Results & interpretation:** `AREA_WEIGHTED_RESULTS.md`
- **Original pipeline:** `WARP.md`
- **Current analysis:** `PLAN.md`
