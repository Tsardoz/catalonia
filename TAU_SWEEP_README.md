# Tau Sensitivity Sweep — Usage Guide

## Purpose

This sweep tests whether the **S_pre winter water balance effect** is robust to the choice of tau (the leaky-integrator memory timescale parameter), or whether it only appears strong at one specific tau value.

For each tau value in the range 15–90 days (step 5):
1. Recomputes CWB_state from scratch with that tau
2. Refits both OLS stage-mean model and Bayesian scalar-on-function model
3. Runs leave-one-year-out (LOYO) cross-validation for the Bayesian model
4. Records S_pre coefficient, uncertainty interval, and LOYO skill

## Quick Start

### Fast OLS-only sweep (testing)
```bash
cd /home/tsardoz/phd/catalonia
source .venv/bin/activate
python src/tau_sensitivity_sweep.py --skip-bayesian --tau-step 15
```
This runs only the OLS model at tau = 15, 30, 45, 60, 75, 90 (6 fits, ~1 minute).

### Full sweep (both models, default tau grid)
```bash
python src/tau_sensitivity_sweep.py --cohort arbequina
```
This runs both OLS and Bayesian models at tau = 15, 20, 25, ..., 90 days (16 tau values).
- OLS: ~1 second per tau
- Bayesian: ~5-10 minutes per tau (MCMC sampling)
- LOYO: ~9 folds × 5-10 minutes = 45-90 minutes per tau
- **Total runtime: ~12-24 hours for full sweep**

### Coarse sweep (faster)
```bash
python src/tau_sensitivity_sweep.py --tau-step 10 --draws 1000 --tune 1000
```
Runs tau = 15, 25, 35, ..., 85 days (8 values) with half the MCMC draws (~6-12 hours).

### Resume interrupted sweep
```bash
python src/tau_sensitivity_sweep.py --reuse-existing
```
Skips refits when output files already exist.

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tau-min` | 15.0 | Minimum tau (days) |
| `--tau-max` | 90.0 | Maximum tau (days) |
| `--tau-step` | 5.0 | Tau grid step (days) |
| `--cohort` | arbequina | Cohort to analyze (arbequina / all_olive) |
| `--draws` | 2000 | MCMC draws per chain |
| `--tune` | 2000 | MCMC warmup draws |
| `--chains` | 4 | MCMC chains |
| `--seed` | 42 | Random seed |
| `--reuse-existing` | False | Skip refits when artifacts exist |
| `--skip-ols` | False | Skip OLS (Bayesian-only) |
| `--skip-bayesian` | False | Skip Bayesian (OLS-only) |
| `--output-stem` | tau_sensitivity_sweep | Output filename prefix |

## Outputs

### 1. Results table
`results/tables/tau_sensitivity_sweep_arbequina.csv`

Columns:
- `tau`: tau value (days)
- `model`: "ols" or "bayesian"
- `cohort`: cohort name
- **OLS row:**
  - `beta_S_pre`: coefficient (standardized scale)
  - `se_S_pre`: HC3 robust standard error
  - `ci_lo_S_pre`, `ci_hi_S_pre`: 95% confidence interval
  - `r2`: R-squared
  - `n_obs`: sample size
- **Bayesian row:**
  - `beta_S_pre_median`: posterior median
  - `beta_S_pre_q_lo`, `beta_S_pre_q_hi`: 90% credible interval
  - `loyo_mae`: LOYO mean absolute error
  - `loyo_rmse`: LOYO root mean squared error
  - `n_folds`: number of LOYO folds completed

### 2. Diagnostic figure
`results/figures/tau_sensitivity_sweep_arbequina.png`

Four panels:
- **Top-left:** OLS β_S_pre with 95% CI band
- **Top-right:** Bayesian β_S_pre median with 90% credible interval
- **Bottom-left:** OLS R² across tau
- **Bottom-right:** LOYO MAE across tau (lower = better out-of-sample skill)

### 3. Summary judgments (logged to console)

The script prints two key judgments:

**1. Is β_S_pre stable in sign and magnitude across tau?**
- Sign stability: do all tau values produce the same sign?
- Coefficient of variation: CV < 50% → STABLE, CV > 50% → FRAGILE
- If stable: "The pre-season water balance effect is robust to the accumulation window" (no selection correction needed)
- If fragile: "The effect only appears at one tau value" (single-tau result not defensible)

**2. Does LOYO-optimal tau match effect-size-optimal tau?**
- LOYO-optimal: tau that minimizes held-out prediction error
- Effect-size-optimal: tau that maximizes β_S_pre
- If they agree: both criteria point to the same tau
- If they disagree: **use LOYO-optimal tau for headline result** (never select tau by maximizing effect size)

## Interpretation

### Expected outcome (robust S_pre)
If the supervisor's May 2026 claim is correct:
- β_S_pre should stay **positive** across all tau values
- Magnitude should vary <50% (e.g., 0.18–0.22 if mean ≈ 0.20)
- LOYO skill should be roughly flat or smoothly varying
- → Reported claim: *"The pre-season water balance effect is robust to the accumulation window (tau = 15–90 days tested)"*

### Failure mode (fragile S_pre)
If the effect is tau-dependent:
- β_S_pre changes sign or collapses outside a narrow range around tau = 45
- LOYO skill peaks sharply at one tau
- → Conclusion: *"The single-tau result (tau = 45) is not defensible; the finding depends on an arbitrary tuning choice"*
- → Action: either report a tau range or acknowledge the finding is fragile

## Technical Notes

### Why refit from scratch?
Each tau changes the **functional form** of the predictor (leaky integrator with different memory). The CWB_state matrix at tau=30 is not a rescaling of tau=60; it's a different time series. Every model parameter must be re-estimated.

### Why LOYO, not full cross-validation?
The effective sample size is ~9 years (spatial replicates share annual climate). Year-level holdout is the natural cross-validation unit for this panel structure.

### Current LOYO limitation
The script currently runs LOYO fits but uses a **placeholder metric** (0.0 for all folds). A full implementation would:
1. Fit model on training years
2. Compute posterior predictive distribution on held-out year
3. Record log-likelihood or MAE for held-out observations
4. Average across folds

For the sensitivity sweep, the proxy is: "does the model converge cleanly at this tau?" If folds fail to converge, LOYO skill is NaN.

### Runtime estimates
- OLS: ~1 second per tau (negligible)
- Bayesian full fit: ~5-10 minutes per tau (4 chains × 2000 warmup + 2000 draws)
- LOYO: ~9 folds × 5-10 minutes = 45-90 minutes per tau
- **16 tau values × ~1 hour each = 16 hours total**

Use `--tau-step 10` (8 values, ~8 hours) or `--tau-step 15` (6 values, ~6 hours) for faster turnaround.

## Example Session

```bash
# Activate environment
cd /home/tsardoz/phd/catalonia
source .venv/bin/activate

# Run coarse sweep (testing)
python src/tau_sensitivity_sweep.py \
  --tau-step 10 \
  --draws 1000 \
  --tune 1000 \
  --cohort arbequina

# Check results
cat results/tables/tau_sensitivity_sweep_arbequina.csv
open results/figures/tau_sensitivity_sweep_arbequina.png

# If stable, run fine sweep for publication
python src/tau_sensitivity_sweep.py \
  --tau-step 5 \
  --reuse-existing \
  --cohort arbequina
```

## Questions for the User

1. **Is the tau grid physically sensible?** (15-90 days covers ~2 weeks to 3 months memory)
2. **Should both cohorts be tested?** (arbequina headline + all_olive breadth check)
3. **Is the LOYO placeholder acceptable for the sweep?** (real implementation requires posterior predictive code)
4. **What CV threshold defines "stable"?** (currently 50%; could be 30% or 70%)

---

**Status (July 2026):** OLS-only sweep complete (τ = 15–90 days, step 5, Arbequina).
Result: β_S_pre positive at all 16 τ values, CV = 11.2%, R² flat at 0.21–0.26. **S_pre is
stable across the plausible τ range at the OLS level** — see `model_complexity_issues.md`
Section 9 for the full write-up. Outputs: `results/tables/tau_sensitivity_sweep_arbequina.csv`,
`results/figures/tau_sensitivity_sweep_arbequina.png`.

Remaining: the Bayesian scalar-on-function refit + LOYO leg (the stricter check against the
actual reported model) has not been run. Use `--skip-ols` (drop `--skip-bayesian`) to launch
it when ready; budget ~12–24 h for the full grid or reduce `--tau-step`/target a subset of τ
for a faster pass.
