# ElasticNet Distributed Lag Regression — Olive Impulse Response

## Problem

Characterise *when* during the agronomic year (Dec–Nov) climate variables drive rainfed olive yield in Catalonia. The existing pipeline demonstrated VPD predicts yield (p < 1e-8) using seasonal aggregates; this step recovers the temporal sensitivity structure as a distributed lag impulse response.

## Data

- `data/agera5_daily_catalonia.csv` — 157K rows of daily climate per comarca (VPD, ET₀, precip, tmax, tmin, ea), 42 comarcas, 2015–2024
- `data/catalan_woody_yield_raw.csv` — 1698 rows; 354 are olive (`crop_group == "olive"`), 36 comarcas
- After join and filtering: **320 rows** (36 comarcas × 9 years, 2016–2024). Agronomic year 2015 dropped (no Dec 2014 data).

## Script

`src/elasticnet_olive_lag.py` — self-contained, reads existing CSVs, produces figures and console output. No changes to existing pipeline scripts.

## Methodology

### Agronomic year and binning

Dec 1 of year Y-1 through Nov 30 of year Y → yield attributed to year Y. Daily data binned into **13 four-week periods** (28 days each; period 13 absorbs remainder). This resolution was selected after testing 1-week (52 bins), 2-week (26 bins), and 4-week (13 bins): finer bins caused overfitting and week-to-week sign inversions from multicollinearity; coarser bins smoothed out VPD's narrower sensitivity windows.

### Feature encodings tested

For each 4-week period, the following aggregations were computed:
- **Mean** VPD, ET₀, precipitation
- **Max** VPD, ET₀ (peak stress event per period)
- **Days VPD > 2.0 kPa** (exceedance count)

### Models

All models include **35 comarca fixed effects** (dummy-encoded, `drop_first=True`) to absorb baseline yield differences between comarcas. 320 rows with 13 lag features + 35 dummies = 48 parameters.

**ElasticNetCV** (regularised, for impulse response shape):
- `l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99]`, `alphas=100`, `cv=5`
- Features standardised via `StandardScaler`; comarca dummies excluded from scaling
- Both training R² and CV R² reported

**OLS** (unregularised, for p-values on individual lag coefficients):
- statsmodels `OLS` with same design matrix
- Reports R², Adj R², F-statistic, per-coefficient t-tests

## Results

### ElasticNet CV R² comparison (4-week bins)

- **M1 Precip mean**: CV R² = 0.28
- **M5 Max VPD**: CV R² = 0.22
- **M4 Days VPD > 2 kPa**: CV R² = 0.17
- **M3 ET₀ mean**: CV R² = 0.13
- **M2 VPD mean**: CV R² = 0.06

### OLS model fit (4-week bins, comarca FE)

- **Max ET₀**: R² = 0.63, Adj R² = 0.56, F p = 1.0e-35, 7 significant periods
- **Precip mean**: R² = 0.60, Adj R² = 0.53, F p = 2.2e-32, 5 significant periods
- **Max VPD**: R² = 0.59, Adj R² = 0.51, F p = 1.7e-30, 4 significant periods

### Key significant periods (OLS, p < 0.05)

**Max VPD**: Dec (*** neg), Mar–Apr (*** pos), Jun–Jul (* pos), Aug–Sep (** neg)

**Max ET₀**: Dec (*** neg), Feb–Mar (* pos), Mar–Apr (*** pos), Apr–May (*** neg), May–Jun (*** pos), Aug (*** neg), Oct–Nov (*** pos)

**Precipitation**: Jan (*** pos), Jul–Aug (*** neg), Aug–Sep (** neg), Sep–Oct (* pos), Nov (* neg)

### Interpretation

1. **Aggregation matters more than variable choice.** Mean VPD (CV R² = 0.06) was near-useless, but max VPD (CV R² = 0.22) recovered a strong signal at the same 4-week resolution. Peak stress events, not average conditions, drive yield.

2. **Precipitation is the strongest single predictor** at 4-week resolution (CV R² = 0.28). Winter recharge (Jan, ***) and summer rain damage (Jul–Aug, ***) are the dominant signals.

3. **Max ET₀ has the richest OLS structure** (Adj R² = 0.56, 7 significant periods), suggesting peak evaporative demand is a better integrated stress indicator than VPD alone at this temporal scale.

4. **VPD operates on shorter timescales than precipitation.** VPD signal improved from 0.06 → 0.22 going from mean to max aggregation, and from 0.06 (4-week mean) to 0.22 (2-week mean) going to finer bins. Precipitation was stable across resolutions. This timescale asymmetry is itself a finding.

5. **Interaction between VPD and rainfall** was not detectable at this sample size (320 rows). Combined models (precip + VPD) did not improve over precip alone; same-day and LPF-based rain conditions added to VPD exceedance did not help. The interaction likely requires soil moisture modelling or larger datasets.

## Output files

- `figures/olive_ols_comparison.png` — **primary figure**: 3-panel OLS impulse response (max VPD, max ET₀, precip) with significance markers
- `figures/olive_irf_precip.png` — ElasticNet precip impulse response
- `figures/olive_irf_vpd.png` — ElasticNet mean VPD impulse response
- `figures/olive_irf_vpd_max.png` — ElasticNet max VPD impulse response
- `figures/olive_irf_nvpd.png` — ElasticNet VPD exceedance days impulse response
- `figures/olive_irf_vpd_vs_et0.png` — ElasticNet VPD vs ET₀ overlay

## Design decisions

- **4-week bins (13 periods)**: best trade-off between overfitting (weekly) and signal loss (monthly). Motivated by resolution experiments, not grid search.
- **No held-out test set**: ElasticNetCV handles internal 5-fold CV. OLS R² is in-sample (clearly labelled). The impulse response shape and coefficient significance are the primary results.
- **Comarca fixed effects**: 35 dummies absorb inter-comarca yield differences. They consume degrees of freedom but are necessary — yield varies ~5× across comarcas for structural reasons (altitude, cultivar, soil).
- **Single-variable models only**: with 320 rows and 35 comarca dummies, combined models are underidentified and showed no improvement.

## Dependencies

scikit-learn, matplotlib, pandas, numpy, statsmodels — all installed in `.venv`.
