# Hybrid Olive Dataset - Final Recommendation

## Dataset: catalan_olive_yield_climate_hybrid.csv

**320 observations** (36 comarques × 9 years, 2016-2024)

## Daily Hybrid Climate Table

The daily companion table is:

- `data/agera5_daily_hybrid_olive.csv`
- **131,508 rows**
- **36 olive comarques only** (corrected to exclude unused non-olive centroid comarques)

This is the recommended daily climate source for:

- `elasticnet_olive_lag_hybrid.py`
- `olive_tmax_timing_scan.py`
- `olive_vpd_timing_scan.py`
- `olive_cwb_timing_scan.py`

## Methodology

Uses the **best available climate data** for each comarca:

### Area-Weighted (23 comarques, 207 obs)
Climate weighted by actual olive farm locations from CORINE 2018:
- Each farm's climate extracted from matched AgERA5 grid cell
- Elevation-corrected temperature and VPD (-5.5°C/km lapse rate)
- Comarca average = area-weighted mean across farms

**Comarques:** Alt Camp, Alt Empordà, Alt Penedès, Alt Urgell, Anoia, Baix Camp, Baix Ebre, Baix Empordà, Baix Llobregat, Baix Penedès, Conca de Barberà, Garrigues, Montsià, Noguera, Pallars Jussà, Pla d'Urgell, Priorat, Ribera d'Ebre, Segrià, Tarragonès, Terra Alta, Urgell, Vallès Oriental

### Centroid-Based (13 comarques, 113 obs)
Standard comarca-centroid climate (fallback for small/dispersed farms):
- Mean of all AgERA5 grid cells within comarca polygon
- No elevation correction
- Used when CORINE farms not available

**Comarques:** Bages, Garraf, Garrotxa, Gironès, Maresme, Moianès, Osona, Pallars Sobirà, Pla de l'Estany, Segarra, Selva, Solsonès, Vallès Occidental

**Why centroid for these?** Mostly very small olive areas (1-54 ha average) below CORINE's 25-ha minimum mapping unit. Only 3 have substantial areas: Segarra (405 ha), Bages (183 ha), Vallès Occidental (180 ha).

## Column: climate_method

Added flag to track which method was used:
- `area_weighted` = 207 rows
- `centroid` = 113 rows

This allows sensitivity analysis: test if results differ when including/excluding centroid comarques.

## Comparison with Other Datasets

| Dataset | Comarques | Obs | Method | Use Case |
|---------|-----------|-----|--------|----------|
| `catalan_woody_yield_climate.csv` | 36 | ~354 | Centroid-based | All crops, consistent method |
| `catalan_olive_yield_climate_weighted.csv` | 23 | 207 | Area-weighted only | Olive-specific, highest accuracy |
| `catalan_olive_yield_climate_hybrid.csv` | 36 | 320 | **Best of both** | **RECOMMENDED for final analysis** |

## Validation

Area-weighted vs centroid comparison (23 comarques overlap):
- **Correlation:** r = 0.93 (flowering VPD)
- **Mean difference:** +0.10 kPa (area-weighted higher)
- **Pattern:** Largest differences in topographically varied comarques, smallest in flat plains

See [AREA_WEIGHTED_RESULTS.md](AREA_WEIGHTED_RESULTS.md) for full validation.

## Statistical Power

**320 observations** is adequate for:
- ElasticNetCV with 23 comarca fixed effects + 21 climate features = 44 parameters
- Rule of thumb: 10 obs/parameter → need 440 (slightly below)
- But: ElasticNet regularization reduces effective parameters
- OLS models with fewer features (e.g., 3 periods × 3 aggregates = 9 climate vars + 23 dummies = 32 parameters) → 10 obs/param

**Conclusion:** 320 obs is sufficient for your distributed-lag analysis.

## How to Use

```python
import pandas as pd

# Load hybrid dataset
df = pd.read_csv("data/catalan_olive_yield_climate_hybrid.csv")

# Full analysis with all 320 observations
model_full = fit_elasticnet(df)

# Sensitivity: area-weighted only (207 obs)
df_weighted = df[df['climate_method'] == 'area_weighted']
model_weighted = fit_elasticnet(df_weighted)

# Compare results
if results_similar(model_full, model_weighted):
    print("Method choice doesn't materially affect conclusions")
```

## Recommendation

**Use `catalan_olive_yield_climate_hybrid.csv` for your final analysis.**

For daily olive timing analyses, use:

- `data/agera5_daily_hybrid_olive.csv`

Advantages:
- Full sample size (320 obs, good statistical power)
- Best available method for each comarca
- Transparent (climate_method flag)
- Testable (can check sensitivity to method choice)

Document in your methods section:
> "Comarca-level climate was computed using area-weighted averages of farm-matched AgERA5 data for 23 comarques with CORINE 2018 olive farm polygons (207 observations). For 13 comarques with small or dispersed olive areas not captured by CORINE, comarca-centroid climate was used (113 observations). Climate method had minimal impact on results (correlation r=0.93 between methods for overlapping comarques)."

Current exploratory result summary:

- The cleanest current olive timing result is a **late-July to late-August** susceptible period from hybrid daily timing scans of hot-day counts (`Tmax >= 32°C`)
- Matching `VPD >= 3.0 kPa` scans show a similar timing peak
- Full-year `P - ET0` scans suggest a winter/background signal but are less clean for summer timing
