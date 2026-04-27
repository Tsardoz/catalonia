# Pipeline Comparison Summary

## Area-Weighted vs Centroid-Based Approaches

### Centroid-Based (Original Pipeline)

**Data source:** All AgERA5 grid cells within comarca polygon boundaries

**Coverage:** 36 comarques with olive yield data

**Sample size:** ~320 comarca-year observations (36 comarques × 9 years, 2016-2024)

**Files:**
```
extract_climate.py → agera5_daily_catalonia.csv (42 comarques × 3653 days)
aggregate_seasonal.py → agera5_seasonal_catalonia.csv
join_dataset.py → catalan_woody_yield_climate.csv
```

**Climate calculation:** Spatial mean of all grid cells within comarca polygon (includes non-agricultural areas)

---

### Area-Weighted (New Pipeline)

**Data source:** AgERA5 grid cells matched to actual olive farm locations (CORINE 2018)

**Coverage:** 23 comarques with both olive yield data AND CORINE olive farm polygons

**Sample size:** 207 comarca-year observations (23 comarques × 9 years, 2016-2024)

**Files:**
```
extract_climate_olive_farms_daily.py → agera5_daily_olive_farms.csv (482 farms × 3653 days)
aggregate_farms_to_comarca.py → agera5_daily_comarca_olive_weighted.csv (23 comarques)
aggregate_seasonal_olive_weighted.py → agera5_seasonal_comarca_olive_weighted.csv
join_dataset_olive_weighted.py → catalan_olive_yield_climate_weighted.csv
```

**Climate calculation:**
1. Extract climate at each farm's grid cell location
2. Apply elevation correction to temperature/VPD
3. Area-weighted mean: `weighted_avg = sum(value × farm_area) / sum(farm_area)`

---

## Missing Comarques (13 with yield but no CORINE farms)

The following comarques have olive yield data but are **missing** from the area-weighted dataset because they have no CORINE 2018 olive farm polygons:

1. Alt Urgell (has yield, missing farms)
2. Baix Llobregat (has yield, missing farms)
3. Barcelonès (has yield, missing farms)
4. Conca de Barberà (has yield, missing farms)
5. Garraf (has yield, missing farms)
6. Garrigues (has yield, missing farms)
7. Pallars Jussà (has yield, missing farms)
8. Pallars Sobirà (has yield, missing farms)
9. Pla d'Urgell (has yield, missing farms)
10. Pla de l'Estany (has yield, missing farms)
11. Ribera d'Ebre (has yield, missing farms)
12. Segrià (has yield, missing farms)
13. Urgell (has yield, missing farms)

**Possible reasons:**
- Small olive area below CORINE 25-ha minimum mapping unit
- Olive area reclassified in CORINE 2018 update
- Very dispersed farms (not forming continuous polygons)
- Urban comarques with scattered olive groves (e.g., Barcelonès, Garraf)

**Impact:** Sample size reduced from 320 → 207 rows (35% reduction)

---

## Recommendations

### Option A: Use Area-Weighted for 23 Comarques (Current)

**Pros:**
- Spatially explicit (climate matched to actual farm locations)
- Elevation-corrected temperature and VPD
- Area-weighted (reflects true farm distribution)

**Cons:**
- Smaller sample size (207 vs 320 rows)
- Missing 13 comarques
- Cannot compare results directly with existing analysis

**Best for:** High-resolution spatial analysis, elevation sensitivity studies

---

### Option B: Hybrid Approach (Recommended)

Fill in the 13 missing comarques using centroid-based climate, create a combined dataset:

```python
# 23 comarques: area-weighted from farms
df_weighted = pd.read_csv("data/catalan_olive_yield_climate_weighted.csv")

# 13 comarques: centroid-based (fallback)
df_centroid = pd.read_csv("data/catalan_woody_yield_climate.csv")
df_centroid = df_centroid[df_centroid['pheno_key'] == 'olive']
df_missing = df_centroid[~df_centroid['comarca'].isin(df_weighted['comarca'])]

# Combine
df_hybrid = pd.concat([df_weighted, df_missing], ignore_index=True)
# → 320 rows (23 area-weighted + 13 centroid)
```

Add a flag column `climate_source` to track which method was used.

**Pros:**
- Full sample size (320 rows)
- Best available method for each comarca
- Can test sensitivity to method

**Cons:**
- Mixed methodology (document clearly)

---

### Option C: Use Centroid-Based for All (Existing Pipeline)

Keep using the original pipeline for comparability with existing results.

**Pros:**
- Consistent methodology across all comarques
- Matches PLAN.md results (320 rows)
- Simple, well-documented

**Cons:**
- Ignores spatial distribution of farms
- No elevation correction
- Less accurate in topographically varied comarques

**Best for:** Initial analysis, consistency with published results

---

## Validation: Compare Methods for 23 Comarques

To assess the impact of area weighting vs centroids, compare climate values for the 23 comarques present in both datasets:

```python
import pandas as pd

# Area-weighted
df_w = pd.read_csv("data/catalan_olive_yield_climate_weighted.csv")

# Centroid-based
df_c = pd.read_csv("data/catalan_woody_yield_climate.csv")
df_c = df_c[df_c['pheno_key'] == 'olive']

# Merge on comarca + year
merged = df_w.merge(
    df_c[['comarca', 'year', 'flower_mean_vpd', 'fruit_set_mean_vpd', 'maturation_mean_vpd']],
    on=['comarca', 'year'],
    suffixes=('_weighted', '_centroid')
)

# Differences
merged['flower_vpd_diff'] = merged['flower_mean_vpd_weighted'] - merged['flower_mean_vpd_centroid']
merged['fruitset_vpd_diff'] = merged['fruit_set_mean_vpd_weighted'] - merged['fruit_set_mean_vpd_centroid']

print("VPD differences (weighted - centroid):")
print(merged[['flower_vpd_diff', 'fruitset_vpd_diff']].describe())

# Correlation
print("\nCorrelation:")
print(merged[['flower_mean_vpd_weighted', 'flower_mean_vpd_centroid']].corr())
```

**Expected:** High correlation (r > 0.95) but systematic offsets in mountainous comarques (Alt Urgell, Pallars) due to elevation effects.

---

## File Sizes

| File | Size | Notes |
|------|------|-------|
| `agera5_daily_olive_farms.csv` | 368 MB | Gitignored (regenerate) |
| `agera5_daily_comarca_olive_weighted.csv` | 17 MB | Tracked in git |
| `agera5_seasonal_comarca_olive_weighted.csv` | ~50 KB | Tracked in git |
| `catalan_olive_yield_climate_weighted.csv` | 85 KB | Tracked in git (final dataset) |

---

## Next Steps

1. **Run comparison analysis** (Option B validation) to quantify impact of area weighting
2. **Decide on approach:**
   - Use area-weighted for 23 comarques only (Option A)
   - Create hybrid dataset with fallback to centroids (Option B)
   - Continue with centroid-based for all (Option C)
3. **Update analysis scripts:**
   - Adapt `elasticnet_olive_lag.py` to use new dataset
   - Document which dataset/method was used for final results
4. **Document in PLAN.md:** Method choice and justification
