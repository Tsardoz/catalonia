#!/usr/bin/env python3
"""
11_fit_fpls_comparison.py — fPLS vs. Bayesian comparison (water balance only).

Compares functional Partial Least Squares (fPLS) regression against the
existing hierarchical Bayesian scalar-on-function model (Step 7) for
predicting olive yield from the climatic water balance state curve
CWB_state(t). Reuses `WD_state_matrix.npy` / `WD_state_reduced.npy` as-is, so
tau=45 (WD_STATE_TAU_DAYS in utils/config.py) and the calendar DOY axis are
automatically matched. Scope is initial: water balance only (no VPD/
temperature, no lag yield). Adding further functional predictors (VPD,
Tmax/Tmin) as extra fPLS blocks is explicit future work, not implemented here.

CAVEATS (also recorded in fpls_vs_bayesian_meta_{cohort}.json — read this
before interpreting the outputs):

1. Smoothers, not deployable models. Both sides of this comparison strip the
   production Step 7 model down to a single-predictor spec
   (`--no-year-re --no-lag-yield`) so fPLS and the Bayesian RW2 smoother can
   be compared on equal footing. Dropping lag_yield and the year RE removes
   real predictive signal, so neither side's LOYO RMSE should be read as
   "the performance of the project's model."
2. Supervised vs. unsupervised. fPLS performs *supervised* dimension
   reduction (components chosen to covary with yield); the Bayesian model
   uses an *unsupervised* RW2 smoothness prior on a fixed 5-basis reduction.
   These are different classes of estimator, not two settings of one knob.
3. Selection-optimism. `fpls_component_selection_{cohort}.csv` sweeps
   n_components via LOYO — that RMSE is optimistic because k would be chosen
   to minimize the same score being reported. It is NOT used to pick the
   headline model. The headline fPLS model always uses a fixed a priori
   `--n-components` (default 1, the most defensible choice at n≈9
   independent years).
4. Flexibility asymmetry. The headline RMSE comparison pits a k=1 fPLS model
   against a 5-basis RW2-penalized Bayesian posterior — these do not have
   comparable effective degrees of freedom, so the RMSE gap (either
   direction) should not be read as "which method is better." This is why
   RMSE is a secondary diagnostic and the beta(t) shape overlay (using a
   small fixed k=1..3 family, not a selection) is the primary result.
5. Underpowered. ~9 LOYO folds means pooled RMSE is a noisy estimator.

Outputs (all prefixed `fpls_` so they never collide with Step 7-10 files):
  - results/tables/fpls_component_selection_{cohort}.csv
  - results/tables/fpls_vs_bayesian_loyo_{cohort}.csv
  - results/tables/fpls_vs_bayesian_beta_t_{cohort}.csv
  - results/tables/fpls_vs_bayesian_stage_means_{cohort}.csv
  - results/figures/fpls_vs_bayesian_beta_t_{cohort}.png
  - results/tables/fpls_vs_bayesian_meta_{cohort}.json

Optional drop-years experiment (--drop-years, e.g. "2018,2024"):
Refits both estimators with the given years fully removed (year-as-unit,
same as the LOYO framework) and attributes any autumn (S8) sign flip to a
specific fPLS component using two diagnostics: (1) a per-component
decomposition of coef_(t) with each component's integrated S8 contribution,
and (2) a check of whether the dropped years sit in the extreme tail of that
component's score distribution. This does not change any full-data output or
the headline k=1 RMSE logic; it only runs when --drop-years is non-empty.
Outputs (tag = "drop" + dropped years, e.g. "drop2018-2024"):
  - results/tables/fpls_vs_bayesian_stage_means_{tag}_{cohort}.csv
  - results/figures/fpls_vs_bayesian_beta_t_{tag}_{cohort}.png
  - results/tables/fpls_component_attribution_{tag}_{cohort}.csv
  - results/tables/fpls_vs_bayesian_meta_{tag}_{cohort}.json
Caveat: dropping years further reduces an already-small n (approx 9 -> 7 for
the default 2018,2024 drop), so this is a qualitative sign-recovery check,
not a precision estimate.
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, TABLES, FIGURES, POSTERIORS, STAGES_RESOLVED, DEFAULT_COHORT,
    N_BASIS, N_CHAINS, N_WARMUP, N_SAMPLES, TARGET_ACCEPT, WD_STATE_TAU_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STEP7_PATH = str(Path(__file__).resolve().parent / "07_fit_numpyro_main.py")

# (full_matrix_file, reduced_matrix_file) per predictor, mirroring 07's
# PREDICTOR_REDUCED_FILES so the two scripts stay in lockstep.
PREDICTOR_FILES = {
    "wd_state": ("WD_state_matrix.npy", "WD_state_reduced.npy"),
    "wd": ("WD_matrix.npy", "WD_reduced.npy"),
    "wd30": ("WD30_matrix.npy", "WD30_reduced.npy"),
    "wd_bucket": ("WD_bucket_matrix.npy", "WD_bucket_reduced.npy"),
    "vpd": ("VPD_matrix.npy", "VPD_reduced.npy"),
    "tmin": ("TMIN_matrix.npy", "TMIN_reduced.npy"),
    "tmax": ("TMAX_matrix.npy", "TMAX_reduced.npy"),
}
PREDICTOR_DISPLAY = {
    "wd_state": "CWB-state", "wd": "CWB", "wd30": "CWB-30d", "wd_bucket": "CWB-bucket",
    "vpd": "VPD", "tmin": "Tmin", "tmax": "Tmax",
}


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fPLS and Bayesian scalar-on-function estimators "
        "of the water-balance sensitivity curve beta(t)."
    )
    parser.add_argument("--cohort", choices=["arbequina", "all_olive"], default="arbequina")
    parser.add_argument(
        "--predictor", choices=list(PREDICTOR_FILES.keys()), default="wd_state",
        help="Functional predictor to compare (default wd_state = CWB-state, "
        "tau=45). Matches 07_fit_numpyro_main.py's --predictor choices, so "
        "the two scripts always fit the same curve.",
    )
    parser.add_argument(
        "--n-components", type=int, default=1,
        help="Headline fPLS complexity for the RMSE table, fixed a priori "
        "and independent of any LOYO score (default 1).",
    )
    parser.add_argument(
        "--shape-components", type=parse_int_list, default="1,2,3",
        help="Comma-separated fixed k family for the beta(t) shape-"
        "triangulation overlay (default 1,2,3). Not a selection.",
    )
    parser.add_argument(
        "--max-components-sweep", type=int, default=5,
        help="Upper bound for the diagnostic-only LOYO k sweep (default 5). "
        "An arbitrary bound, not a claim of complexity parity with the "
        "5-basis RW2 Bayesian model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--draws", type=int, default=N_SAMPLES)
    parser.add_argument("--tune", type=int, default=N_WARMUP)
    parser.add_argument("--chains", type=int, default=N_CHAINS)
    parser.add_argument(
        "--chain-method", choices=["sequential", "parallel", "vectorized"], default="sequential",
    )
    parser.add_argument("--target-accept", type=float, default=TARGET_ACCEPT)
    parser.add_argument("--hdi-prob", type=float, default=0.90)
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Skip Bayesian subprocess fits when expected artifacts already exist.",
    )
    parser.add_argument(
        "--drop-years", type=parse_int_list, default="",
        help="Comma-separated years to drop before any fitting (e.g. "
        "2018,2024), for a storm-year attribution experiment. Default empty "
        "disables the experiment and leaves all other behavior unchanged. "
        "Years are dropped at the year level (all comarca-rows for that "
        "year), consistent with the LOYO framework.",
    )
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────

def load_cohort_data(cohort: str, predictor: str) -> dict:
    full_file, reduced_file = PREDICTOR_FILES[predictor]
    doy_grid = np.load(PROCESSED / "doy_grid.npy")
    X_full = np.load(PROCESSED / full_file)
    X_reduced = np.load(PROCESSED / reduced_file)
    obs_index = pd.read_csv(PROCESSED / "obs_index.csv")
    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")

    if "cohort" not in obs_index.columns:
        obs_index["cohort"] = DEFAULT_COHORT
    if "cohort" not in yield_df.columns:
        yield_df["cohort"] = DEFAULT_COHORT

    mask = (obs_index["cohort"] == cohort).to_numpy()
    if mask.sum() == 0:
        raise ValueError(f"No rows found for cohort='{cohort}' in obs_index.csv.")

    obs_c = obs_index[mask].reset_index(drop=True)
    X_full_c = X_full[mask]
    X_reduced_c = X_reduced[mask]

    merged = obs_c.merge(
        yield_df[["cohort", "comarca", "year", "yield_tha", "lag_yield"]],
        on=["cohort", "comarca", "year"], how="left", validate="one_to_one",
    )
    missing = merged["yield_tha"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} rows could not be matched to olive_yield.csv for cohort={cohort}.")

    return {
        "doy_grid": doy_grid,
        "X_full": X_full_c,
        "X_reduced": X_reduced_c,
        "y": merged["yield_tha"].to_numpy(dtype=float),
        "years": merged["year"].astype(int).to_numpy(),
        "obs": merged,
    }


# ── Bayesian baseline (subprocess reuse of Step 7) ──────────────────────

def bayesian_stem(cohort: str, predictor: str, exclude_year: int | None, run_tag: str) -> str:
    stem = f"numpyro_main_{cohort}_{predictor}_noyearre_nolagyield"
    if exclude_year is not None:
        stem += f"_excl{int(exclude_year)}"
    if run_tag:
        stem += f"_{run_tag}"
    return stem


def run_bayesian_fit(args: argparse.Namespace, exclude_year: int | None) -> str:
    stem = bayesian_stem(args.cohort, args.predictor, exclude_year, args.run_tag)
    beta_t_path = POSTERIORS / f"{stem}_beta_t_draws.npy"
    if args.reuse_existing and beta_t_path.exists():
        log.info(f"Reusing existing Bayesian baseline artifacts: stem={stem}")
        return stem

    cmd = [
        sys.executable, STEP7_PATH,
        "--cohort", args.cohort,
        "--predictor", args.predictor,
        "--no-year-re",
        "--no-lag-yield",
        "--draws", str(args.draws),
        "--tune", str(args.tune),
        "--chains", str(args.chains),
        "--chain-method", args.chain_method,
        "--seed", str(args.seed),
        "--target-accept", str(args.target_accept),
        "--hdi-prob", str(args.hdi_prob),
    ]
    if exclude_year is not None:
        cmd.extend(["--exclude-year", str(int(exclude_year))])
    if args.run_tag:
        cmd.extend(["--stem-suffix", args.run_tag])

    log.info(f"Running Bayesian baseline fit (stem={stem})...")
    subprocess.run(cmd, check=True)
    return stem


def bayesian_stem_dropyears(cohort: str, predictor: str, drop_years: list[int], run_tag: str) -> str:
    """Stem for the drop-years Bayesian baseline. Naming matches what
    07_fit_numpyro_main.py itself produces for --exclude-years (sorted,
    dash-joined), so artifacts line up if run directly with that flag too."""
    stem = f"numpyro_main_{cohort}_{predictor}_noyearre_nolagyield"
    stem += "_excl" + "-".join(str(y) for y in sorted(set(drop_years)))
    if run_tag:
        stem += f"_{run_tag}"
    return stem


def run_bayesian_fit_drop_years(args: argparse.Namespace, drop_years: list[int]) -> str:
    """Run (or reuse) the Bayesian water-balance-only baseline with the given
    years fully removed, using 07_fit_numpyro_main.py's --exclude-years flag.
    This is separate from run_bayesian_fit (single-year LOYO) so the existing
    full-data and LOYO fold paths are untouched."""
    stem = bayesian_stem_dropyears(args.cohort, args.predictor, drop_years, args.run_tag)
    beta_t_path = POSTERIORS / f"{stem}_beta_t_draws.npy"
    if args.reuse_existing and beta_t_path.exists():
        log.info(f"Reusing existing Bayesian drop-years artifacts: stem={stem}")
        return stem

    cmd = [
        sys.executable, STEP7_PATH,
        "--cohort", args.cohort,
        "--predictor", args.predictor,
        "--no-year-re",
        "--no-lag-yield",
        "--exclude-years", ",".join(str(y) for y in sorted(set(drop_years))),
        "--draws", str(args.draws),
        "--tune", str(args.tune),
        "--chains", str(args.chains),
        "--chain-method", args.chain_method,
        "--seed", str(args.seed),
        "--target-accept", str(args.target_accept),
        "--hdi-prob", str(args.hdi_prob),
    ]
    if args.run_tag:
        cmd.extend(["--stem-suffix", args.run_tag])

    log.info(f"Running Bayesian drop-years fit (stem={stem})...")
    subprocess.run(cmd, check=True)
    return stem


def load_bayesian_predict_params(stem: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (alpha_mean, beta_phys_mean, x_mean) for posterior-mean prediction."""
    import arviz as az

    idata = az.from_netcdf(POSTERIORS / f"{stem}.nc")
    alpha_mean = float(idata.posterior["alpha"].values.mean())
    beta_phys = np.load(POSTERIORS / f"{stem}_beta_coefs_phys.npy")
    beta_phys_mean = beta_phys.mean(axis=0)
    x_mean = np.load(POSTERIORS / f"{stem}_x_mean.npy")
    return alpha_mean, beta_phys_mean, x_mean


def predict_bayesian(X_reduced_rows: np.ndarray, alpha_mean: float,
                     beta_phys_mean: np.ndarray, x_mean: np.ndarray) -> np.ndarray:
    """mu = alpha + (X_reduced - x_mean) @ beta_phys; equivalent to the
    training-time standardized computation alpha + Xz @ beta_std, since
    beta_phys = beta_std / x_sd and Xz = (X - x_mean) / x_sd."""
    return alpha_mean + (X_reduced_rows - x_mean[None, :]) @ beta_phys_mean


# ── fPLS ─────────────────────────────────────────────────────────────────

def make_fdata_grid(X: np.ndarray, doy_grid: np.ndarray):
    from skfda.representation import FDataGrid
    return FDataGrid(data_matrix=X, grid_points=doy_grid)


def fit_fpls(X_fd, y_arr: np.ndarray, k: int):
    from skfda.ml.regression import FPLSRegression
    model = FPLSRegression(n_components=k)
    model.fit(X_fd, y_arr)
    return model


def simpson_weights(doy_grid: np.ndarray) -> np.ndarray:
    identity = np.identity(len(doy_grid))
    return scipy.integrate.simpson(identity, x=doy_grid)


def scale_matching_check(X: np.ndarray, y: np.ndarray, X_fd, coef: np.ndarray, model,
                         weights: np.ndarray, bayes_median: np.ndarray, label: str = "") -> dict:
    """Verify fPLS predict-consistency and compute the coef_fpls vs.
    beta_bayesian_median magnitude ratio. Same formula used for the full-data
    check (Step 3) and the drop-years experiment, so both call this one
    function instead of duplicating the logic. label is used only in log
    messages, e.g. "full-data" or "drop-years"."""
    X_bar = X.mean(axis=0)
    manual_pred = y.mean() + (X - X_bar[None, :]) @ (weights * coef)
    model_pred = model.predict(X_fd).ravel()
    max_abs_diff = float(np.max(np.abs(manual_pred - model_pred)))
    consistency_ok = bool(np.allclose(manual_pred, model_pred, atol=1e-6, rtol=1e-4))
    if not consistency_ok:
        log.warning(f"fPLS predict-consistency check FAILED ({label}): max_abs_diff={max_abs_diff:.3g}")
    else:
        log.info(f"fPLS predict-consistency check OK ({label}): max_abs_diff={max_abs_diff:.3g}")

    integral_fpls = float(np.sum(weights * np.abs(coef)))
    integral_bayes = float(np.sum(weights * np.abs(bayes_median)))
    magnitude_ratio = integral_fpls / integral_bayes if integral_bayes > 0 else float("nan")
    scale_suspect = (not consistency_ok) or not (0.1 <= magnitude_ratio <= 10.0)
    log.info(
        f"Scale-matching ({label}): integral|coef_fpls|={integral_fpls:.6g}, "
        f"integral|beta_bayesian_median|={integral_bayes:.6g}, "
        f"ratio={magnitude_ratio:.3g}, scale_suspect={scale_suspect}"
    )
    if scale_suspect:
        log.warning(
            f"SCALE-SUSPECT ({label}): fPLS and Bayesian beta(t) may not be on "
            "comparable physical scales. Overlay artifacts are still written but annotated."
        )
    return {
        "predict_consistency_ok": consistency_ok,
        "predict_consistency_max_abs_diff": max_abs_diff,
        "integral_abs_coef_fpls": integral_fpls,
        "integral_abs_beta_bayesian_median": integral_bayes,
        "magnitude_ratio": magnitude_ratio,
        "scale_suspect": bool(scale_suspect),
    }


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


def fpls_loyo_sweep(X_full: np.ndarray, y: np.ndarray, years: np.ndarray,
                    doy_grid: np.ndarray, k_values: list[int]) -> dict[int, np.ndarray]:
    """Leave-one-year-out pooled predictions for each k. Returns {k: preds (n_obs,)}."""
    unique_years = sorted(set(years.tolist()))
    preds = {k: np.full(len(y), np.nan) for k in k_values}
    for held_out in unique_years:
        train_mask = years != held_out
        test_mask = ~train_mask
        X_train_fd = make_fdata_grid(X_full[train_mask], doy_grid)
        X_test_fd = make_fdata_grid(X_full[test_mask], doy_grid)
        y_train = y[train_mask]
        for k in k_values:
            model = fit_fpls(X_train_fd, y_train, k)
            pred = model.predict(X_test_fd).ravel()
            preds[k][test_mask] = pred
    for k in k_values:
        assert not np.any(np.isnan(preds[k])), f"Missing LOYO predictions for k={k}"
    return preds


def bayesian_loyo(args: argparse.Namespace, X_reduced: np.ndarray, y: np.ndarray,
                  years: np.ndarray) -> np.ndarray:
    unique_years = sorted(set(years.tolist()))
    preds = np.full(len(y), np.nan)
    for held_out in unique_years:
        stem = run_bayesian_fit(args, exclude_year=held_out)
        alpha_mean, beta_phys_mean, x_mean = load_bayesian_predict_params(stem)
        test_mask = years == held_out
        preds[test_mask] = predict_bayesian(X_reduced[test_mask], alpha_mean, beta_phys_mean, x_mean)
    assert not np.any(np.isnan(preds)), "Missing Bayesian LOYO predictions"
    return preds


# ── Stage windows ────────────────────────────────────────────────────────

def window_masks(doy_grid: np.ndarray) -> dict[str, np.ndarray]:
    masks = {}
    for stage, (start, end) in STAGES_RESOLVED.items():
        masks[stage] = (doy_grid >= start) & (doy_grid <= end)
    return masks


def stage_means_point(curve: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    return {stage: float(curve[mask].mean()) for stage, mask in masks.items()}


def stage_means_draws(beta_t_draws: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {stage: beta_t_draws[:, mask].mean(axis=1) for stage, mask in masks.items()}


# ── Drop-years attribution diagnostics ──────────────────────────────────

def component_attribution(fpls_full_models: dict[int, object], shape_components: list[int],
                          weights: np.ndarray, masks: dict[str, np.ndarray],
                          years: np.ndarray, drop_years: list[int]) -> pd.DataFrame:
    """Decompose each shape-component fPLS model's coef_(t) into additive
    per-component contributions and report each component's integrated S8
    (autumn) contribution, plus whether the drop_years rows sit in the
    extreme tail of that component's score distribution.

    skfda 0.10.1's FPLSRegression builds coef_ = x_rotations_matrix_ @
    y_loadings_matrix_.T (see skfda/ml/regression/_fpls_regression.py). Since
    y_loadings_matrix_ has shape (1, k) for a scalar response, each column j
    of x_rotations_matrix_ times y_loadings_matrix_[0, j] is component j's
    additive contribution to coef_(t); summing over j reproduces coef_
    exactly (verified by an assert below). x_scores_ (n_obs, k) gives each
    observation's score on each component (confirmed against skfda source
    and a smoke test: shapes (n_grid, k), (1, k), (n_obs, k) respectively).

    This always runs on the FULL-DATA fits, since the storm-year rows are by
    definition absent from the storm-dropped refit and we need their scores.
    """
    s8_mask = masks["S8"]
    rows = []
    for k in shape_components:
        model = fpls_full_models[k]
        fpls_obj = model.fpls_
        x_rot = fpls_obj.x_rotations_matrix_          # (n_grid, k)
        y_load = fpls_obj.y_loadings_matrix_           # (1, k)
        x_scores = fpls_obj.x_scores_                  # (n_obs, k)
        contributions = x_rot * y_load[0, :][None, :]  # (n_grid, k); columns sum to coef_

        recon = contributions.sum(axis=1)
        assert np.allclose(recon, model.coef_.ravel(), atol=1e-6), (
            f"Per-component contributions do not sum to coef_ for k={k} "
            f"(max abs diff {np.max(np.abs(recon - model.coef_.ravel())):.3g})"
        )

        s8_masses = [float(np.sum(weights[s8_mask] * contributions[s8_mask, j])) for j in range(k)]
        top_component = int(np.argmax(s8_masses))  # component carrying the largest (most positive) S8 mass

        for j in range(k):
            scores_j = x_scores[:, j]
            score_std = scores_j.std(ddof=0)
            z = (scores_j - scores_j.mean()) / score_std if score_std > 0 else np.zeros_like(scores_j)
            abs_score = np.abs(scores_j)
            order = np.argsort(-abs_score)  # rank 1 = most extreme absolute score
            ranks = np.empty(len(scores_j), dtype=int)
            ranks[order] = np.arange(1, len(scores_j) + 1)

            storm_detail_parts = []
            any_extreme = False
            for sy in drop_years:
                year_mask = years == sy
                if not np.any(year_mask):
                    continue
                mean_abs_z = float(np.mean(np.abs(z[year_mask])))
                max_abs_z = float(np.max(np.abs(z[year_mask])))
                best_rank = int(ranks[year_mask].min())
                extreme = (max_abs_z > 2.0) or (best_rank <= 3)
                any_extreme = any_extreme or extreme
                storm_detail_parts.append(
                    f"{sy}: mean|z|={mean_abs_z:.2f} max|z|={max_abs_z:.2f} "
                    f"best_rank={best_rank}/{len(scores_j)}"
                )

            rows.append({
                "n_components": k,
                "component_index": j,
                "s8_integrated_contribution": s8_masses[j],
                "is_top_s8_contributor": (j == top_component),
                "storm_years_extreme": any_extreme,
                "storm_year_detail": "; ".join(storm_detail_parts),
                "note": (
                    "storm_years_extreme=True on the top S8 contributor supports the "
                    "storm-artifact reading; False supports the real-signal reading"
                    if j == top_component else ""
                ),
            })
    return pd.DataFrame(rows)


def build_dropyears_stage_compare(stage_df_full: pd.DataFrame, bayes_stage_draws_drop: dict,
                                  fpls_drop_coefs: dict[int, np.ndarray], masks: dict,
                                  shape_components: list[int], hdi_prob: float) -> pd.DataFrame:
    """Side-by-side S8 mean beta and S7-S6 contrast, full-data vs. storm-dropped,
    for the Bayesian median (plus 90 percent CI) and each fPLS k."""
    lo, hi = 0.5 * (1.0 - hdi_prob), 1.0 - 0.5 * (1.0 - hdi_prob)
    bayes_contrast_draws_drop = bayes_stage_draws_drop["S7"] - bayes_stage_draws_drop["S6"]

    rows = []
    for stage_name in ["S8", "S7_minus_S6"]:
        full_row = stage_df_full[stage_df_full["stage"] == stage_name].iloc[0]
        if stage_name == "S8":
            bmed_d = float(np.median(bayes_stage_draws_drop["S8"]))
            blo_d = float(np.quantile(bayes_stage_draws_drop["S8"], lo))
            bhi_d = float(np.quantile(bayes_stage_draws_drop["S8"], hi))
        else:
            bmed_d = float(np.median(bayes_contrast_draws_drop))
            blo_d = float(np.quantile(bayes_contrast_draws_drop, lo))
            bhi_d = float(np.quantile(bayes_contrast_draws_drop, hi))

        row = {
            "stage": stage_name,
            "beta_bayesian_median_full": full_row["beta_bayesian_median"],
            "beta_bayesian_q_lo_full": full_row["beta_bayesian_q_lo"],
            "beta_bayesian_q_hi_full": full_row["beta_bayesian_q_hi"],
            "beta_bayesian_median_dropyears": bmed_d,
            "beta_bayesian_q_lo_dropyears": blo_d,
            "beta_bayesian_q_hi_dropyears": bhi_d,
        }
        for k in shape_components:
            row[f"beta_fpls_k{k}_full"] = full_row[f"beta_fpls_k{k}"]
            sm_d = stage_means_point(fpls_drop_coefs[k], masks)
            if stage_name == "S8":
                row[f"beta_fpls_k{k}_dropyears"] = sm_d["S8"]
            else:
                row[f"beta_fpls_k{k}_dropyears"] = sm_d["S7"] - sm_d["S6"]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_overlay(doy_grid, bayes_median, bayes_q05, bayes_q95,
                 fpls_curves: dict[int, np.ndarray], cohort: str, predictor: str, scale_suspect: bool,
                 output_tag: str = "", title_suffix: str = ""):
    """output_tag and title_suffix default to empty, so the full-data call
    site (Step 7) is byte-for-byte unchanged for a given predictor. Passing
    output_tag renders fpls_vs_bayesian_beta_t_{output_tag}_{cohort}_{predictor}.png
    instead, used by the drop-years experiment to avoid colliding with the
    full-data figure."""
    disp = PREDICTOR_DISPLAY.get(predictor, predictor)
    plt.figure(figsize=(10, 5.5))
    plt.fill_between(doy_grid, bayes_q05, bayes_q95, alpha=0.2, color="tab:blue",
                     label="Bayesian 90% band")
    plt.plot(doy_grid, bayes_median, lw=2.2, color="tab:blue", label="Bayesian median")
    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for i, (k, curve) in enumerate(sorted(fpls_curves.items())):
        plt.plot(doy_grid, curve, lw=1.6, color=colors[i % len(colors)],
                 label=f"fPLS k={k}")
    for stage, (start, end) in STAGES_RESOLVED.items():
        plt.axvline(start, color="gray", linestyle="--", linewidth=0.6)
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    plt.xlabel("DOY")
    plt.ylabel("β(t)")
    tau_note = f", tau={WD_STATE_TAU_DAYS}d" if predictor == "wd_state" else ""
    title = f"fPLS vs. Bayesian β(t) — {cohort} ({disp}{tau_note})"
    if title_suffix:
        title += f" {title_suffix}"
    if scale_suspect:
        title += " [SCALE-SUSPECT]"
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_name = f"fpls_vs_bayesian_beta_t_{output_tag + '_' if output_tag else ''}{cohort}_{predictor}.png"
    out = FIGURES / out_name
    plt.savefig(out, dpi=180)
    plt.close()
    return out


# ── Unsupervised fPCA storm diagnostic ─────────────────────────────

def fpca_storm_diagnostic(X_fd, weights: np.ndarray, masks: dict[str, np.ndarray],
                          years: np.ndarray, drop_years: list[int],
                          n_components: int = 5, localized_threshold: float = 0.25):
    """Unsupervised functional PCA on the water-balance curves alone (no
    yield anywhere in this function). For each eigenfunction, report the
    fraction of its L2 energy inside the S8 (autumn) window, and whether the
    drop_years rows sit in the extreme tail of that eigenfunction's score
    distribution.

    This is a separate, weaker-assumption line of evidence than the fPLS
    component_attribution diagnostic: fPLS components are fit with
    reference to yield, so "storm years are extreme on this component" could
    in principle be circular (the component was partly shaped by the storm
    years' own yield deviation). An fPCA component has no yield in its
    definition at all, so the same finding here cannot be that kind of
    artifact -- it is direct, yield-independent evidence about the curves.

    S8 is about 8.6% of the 327-day domain, so localized_threshold=0.25
    already means roughly 3x energy concentration relative to a uniformly
    spread eigenfunction. This is a diagnostic, not a model-selection step,
    so no Kaiser-style eigenvalue cutoff or variance-explained target is
    used to choose n_components -- all n_components requested are reported
    and inspected directly.

    Returns (attribution_df, fitted_fpca_model).
    """
    from skfda.preprocessing.dim_reduction import FPCA

    s8_mask = masks["S8"]
    fpca = FPCA(n_components=n_components)
    scores = fpca.fit_transform(X_fd)                      # (n_obs, n_components)
    eigenfunctions = fpca.components_.data_matrix[..., 0]  # (n_components, n_grid)
    explained = fpca.explained_variance_ratio_

    rows = []
    for j in range(n_components):
        eig = eigenfunctions[j]
        total_energy = float(np.sum(weights * eig ** 2))
        s8_energy = float(np.sum(weights[s8_mask] * eig[s8_mask] ** 2))
        s8_fraction = s8_energy / total_energy if total_energy > 0 else float("nan")
        is_localized = s8_fraction > localized_threshold

        scores_j = scores[:, j]
        score_std = scores_j.std(ddof=0)
        z = (scores_j - scores_j.mean()) / score_std if score_std > 0 else np.zeros_like(scores_j)
        abs_score = np.abs(scores_j)
        order = np.argsort(-abs_score)  # rank 1 = most extreme absolute score
        ranks = np.empty(len(scores_j), dtype=int)
        ranks[order] = np.arange(1, len(scores_j) + 1)

        storm_detail_parts = []
        any_extreme = False
        for sy in drop_years:
            year_mask = years == sy
            if not np.any(year_mask):
                continue
            mean_abs_z = float(np.mean(np.abs(z[year_mask])))
            max_abs_z = float(np.max(np.abs(z[year_mask])))
            best_rank = int(ranks[year_mask].min())
            extreme = (max_abs_z > 2.0) or (best_rank <= 3)
            any_extreme = any_extreme or extreme
            storm_detail_parts.append(
                f"{sy}: mean|z|={mean_abs_z:.2f} max|z|={max_abs_z:.2f} "
                f"best_rank={best_rank}/{len(scores_j)}"
            )

        rows.append({
            "component_index": j,
            "explained_variance_ratio": float(explained[j]),
            "s8_energy_fraction": s8_fraction,
            "is_autumn_localized": bool(is_localized),
            "storm_years_extreme": any_extreme,
            "storm_year_detail": "; ".join(storm_detail_parts),
            "note": (
                "unsupervised (no yield used to define this component) -- "
                "storm_years_extreme=True on an autumn-localized component is "
                "yield-independent evidence the storm years are structurally "
                "anomalous in autumn; False on a localized component argues "
                "against the storm-contamination reading from the X side"
                if is_localized else ""
            ),
        })
    return pd.DataFrame(rows), fpca


def plot_fpca_eigenfunctions(doy_grid: np.ndarray, fpca, cohort: str, predictor: str,
                             output_tag: str, n_show: int = 5):
    """Plot the leading unsupervised fPCA eigenfunctions with the S8 window
    shaded, so autumn-localization can be checked visually."""
    disp = PREDICTOR_DISPLAY.get(predictor, predictor)
    eigenfunctions = fpca.components_.data_matrix[..., 0]
    s8_start, s8_end = STAGES_RESOLVED["S8"]
    plt.figure(figsize=(10, 5))
    plt.axvspan(s8_start, s8_end, color="gold", alpha=0.25, label="S8 window")
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for j in range(min(n_show, eigenfunctions.shape[0])):
        plt.plot(doy_grid, eigenfunctions[j], lw=1.6, color=colors[j % len(colors)],
                 label=f"PC{j + 1} ({fpca.explained_variance_ratio_[j] * 100:.1f}% var)")
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    plt.xlabel("DOY")
    plt.ylabel("eigenfunction value")
    plt.title(f"Unsupervised fPCA eigenfunctions -- {cohort} ({disp}, no yield used)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES / f"fpls_fpca_eigenfunctions_{output_tag}_{cohort}_{predictor}.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


# ── Drop-years experiment orchestration ─────────────────────────────────

def run_drop_years_experiment(args: argparse.Namespace, doy_grid: np.ndarray,
                              X_full: np.ndarray, X_full_fd, X_reduced: np.ndarray,
                              y: np.ndarray, years: np.ndarray,
                              fpls_full_models: dict[int, object], weights: np.ndarray,
                              masks: dict[str, np.ndarray], stage_df_full: pd.DataFrame):
    """Storm-year attribution experiment: refit both estimators with
    args.drop_years fully removed, and attribute any S8 (autumn) sign change
    to a specific fPLS component. Does not touch any full-data output or the
    headline k=1 RMSE logic computed earlier in main(); only runs when
    --drop-years is non-empty."""
    drop_years = sorted(set(args.drop_years))
    log.info(f"Drop-years experiment: dropping years {drop_years}")

    all_unique_years = sorted(set(years.tolist()))
    drop_mask = ~np.isin(years, drop_years)
    n_before, n_after = len(y), int(drop_mask.sum())
    if n_after == 0:
        raise ValueError("Drop-years experiment removed all rows; check --drop-years.")

    X_full_d = X_full[drop_mask]
    X_reduced_d = X_reduced[drop_mask]  # noqa: F841 (kept for parity/future use, not used below)
    y_d = y[drop_mask]
    years_d = years[drop_mask]
    unique_years_d = sorted(set(years_d.tolist()))
    log.info(
        f"Drop-years: {n_before} -> {n_after} obs, "
        f"{len(all_unique_years)} -> {len(unique_years_d)} years (dropped: {drop_years})"
    )

    tag = "drop" + "-".join(str(yy) for yy in drop_years)

    # Diagnostic 1 (part a): per-component S8 attribution + storm-year score
    # check, computed on the FULL-DATA fits, since the dropped years' scores
    # only exist there (they are absent from the storm-dropped refit below).
    log.info("Drop-years: component attribution diagnostic (full-data fits)...")
    attribution_df = component_attribution(
        fpls_full_models, args.shape_components, weights, masks, years, drop_years,
    )
    attribution_path = TABLES / f"fpls_component_attribution_{tag}_{args.cohort}_{args.predictor}.csv"
    attribution_df.to_csv(attribution_path, index=False)
    log.info(f"Saved: {attribution_path}")

    # Diagnostic 1b: unsupervised fPCA storm check (yield-independent), also
    # on the FULL-DATA curves -- corroborates diagnostic 1a without using y
    log.info("Drop-years: unsupervised fPCA storm diagnostic (no yield used)...")
    fpca_df, fpca_model = fpca_storm_diagnostic(X_full_fd, weights, masks, years, drop_years)
    fpca_path = TABLES / f"fpls_fpca_storm_diagnostic_{tag}_{args.cohort}_{args.predictor}.csv"
    fpca_df.to_csv(fpca_path, index=False)
    log.info(f"Saved: {fpca_path}")
    fpca_fig_path = plot_fpca_eigenfunctions(doy_grid, fpca_model, args.cohort, args.predictor, tag)
    log.info(f"Saved: {fpca_fig_path}")

    # Refit fPLS on the storm-dropped data (headline k plus shape family, deduped)
    k_needed_d = sorted(set([args.n_components] + args.shape_components))
    X_full_d_fd = make_fdata_grid(X_full_d, doy_grid)
    fpls_d_models = {k: fit_fpls(X_full_d_fd, y_d, k) for k in k_needed_d}
    fpls_d_coefs = {k: m.coef_.ravel() for k, m in fpls_d_models.items()}

    # Bayesian baseline refit on the storm-dropped data
    log.info("Drop-years: Bayesian baseline refit (storm-dropped)...")
    stem_d = run_bayesian_fit_drop_years(args, drop_years)
    bayes_beta_t_d = np.load(POSTERIORS / f"{stem_d}_beta_t_draws.npy")
    if bayes_beta_t_d.shape[1] != len(doy_grid):
        raise ValueError(
            f"Bayesian beta_t grid mismatch (drop-years): {bayes_beta_t_d.shape[1]} vs {len(doy_grid)}."
        )
    bayes_median_d = np.median(bayes_beta_t_d, axis=0)
    bayes_q05_d = np.quantile(bayes_beta_t_d, 0.05, axis=0)
    bayes_q95_d = np.quantile(bayes_beta_t_d, 0.95, axis=0)

    # Scale-matching check, reused unchanged (same function as Step 3),
    # applied to the storm-dropped headline fit
    headline_model_d = fpls_d_models[args.n_components]
    headline_coef_d = fpls_d_coefs[args.n_components]
    scale_check_d = scale_matching_check(
        X_full_d, y_d, X_full_d_fd, headline_coef_d, headline_model_d, weights, bayes_median_d,
        label="drop-years",
    )

    # Diagnostic 1 (part b) / paired comparison: S8 mean beta and the S7-S6
    # contrast, full-data vs. storm-dropped, side by side
    bayes_stage_draws_d = stage_means_draws(bayes_beta_t_d, masks)
    compare_df = build_dropyears_stage_compare(
        stage_df_full, bayes_stage_draws_d, fpls_d_coefs, masks, args.shape_components, args.hdi_prob,
    )
    compare_path = TABLES / f"fpls_vs_bayesian_stage_means_{tag}_{args.cohort}_{args.predictor}.csv"
    compare_df.to_csv(compare_path, index=False)
    log.info(f"Saved: {compare_path}")

    fpls_d_shape_curves = {k: fpls_d_coefs[k] for k in args.shape_components}
    fig_path_d = plot_overlay(
        doy_grid, bayes_median_d, bayes_q05_d, bayes_q95_d, fpls_d_shape_curves,
        args.cohort, args.predictor, scale_check_d["scale_suspect"],
        output_tag=tag, title_suffix=f"(drop-years {','.join(str(yy) for yy in drop_years)})",
    )
    log.info(f"Saved: {fig_path_d}")

    meta_d = {
        "cohort": args.cohort,
        "predictor": args.predictor,
        "tau_days": WD_STATE_TAU_DAYS if args.predictor == "wd_state" else None,
        "dropped_years": drop_years,
        "n_obs_full": n_before,
        "n_obs_dropyears": n_after,
        "n_years_full": len(all_unique_years),
        "n_years_dropyears": len(unique_years_d),
        "headline_n_components": args.n_components,
        "shape_components": args.shape_components,
        "bayesian_stem_dropyears": stem_d,
        "scale_check": scale_check_d,
        "caveat": (
            "Dropping years further reduces an already-small n "
            f"({len(all_unique_years)} -> {len(unique_years_d)} years), so this "
            "is a qualitative sign-recovery test, not a precision estimate."
        ),
        "decision_rule": (
            "Bayesian autumn (S8) recovering positive and fPLS k=1 turning "
            "positive once storms are out indicates storm contamination and a "
            "real autumn window. Autumn positive surviving only in k=2 or k=3, "
            "with the storm years extreme on the top S8-contributing component "
            "(see fpls_component_attribution), indicates storm-fitting rather "
            "than a real signal. With fewer years remaining after the drop, "
            "fPLS k=3 is barely identified, so weight the k=1 and Bayesian "
            "sign-recovery over the k=3 behavior; anything that only moves at "
            "k=3 stays ambiguous."
        ),
        "outputs": {
            "component_attribution_csv": str(attribution_path),
            "fpca_storm_diagnostic_csv": str(fpca_path),
            "fpca_eigenfunctions_png": str(fpca_fig_path),
            "stage_means_compare_csv": str(compare_path),
            "beta_t_overlay_png": str(fig_path_d),
        },
    }
    meta_d_path = TABLES / f"fpls_vs_bayesian_meta_{tag}_{args.cohort}_{args.predictor}.json"
    with open(meta_d_path, "w", encoding="utf-8") as f:
        json.dump(meta_d, f, indent=2)
    log.info(f"Saved: {meta_d_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    POSTERIORS.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading cohort data: cohort={args.cohort}, predictor={args.predictor}")
    data = load_cohort_data(args.cohort, args.predictor)
    doy_grid, X_full, X_reduced, y, years = (
        data["doy_grid"], data["X_full"], data["X_reduced"], data["y"], data["years"]
    )
    n_obs = len(y)
    unique_years = sorted(set(years.tolist()))
    log.info(f"n_obs={n_obs}, n_years={len(unique_years)}: {unique_years}")

    # Step 2: Bayesian full-data baseline (run first — needed for the scale check)
    log.info("Step 2: Bayesian full-data baseline (--no-year-re --no-lag-yield)...")
    full_stem = run_bayesian_fit(args, exclude_year=None)
    bayes_beta_t = np.load(POSTERIORS / f"{full_stem}_beta_t_draws.npy")
    if bayes_beta_t.shape[1] != len(doy_grid):
        raise ValueError(
            f"Bayesian beta_t grid mismatch: {bayes_beta_t.shape[1]} vs doy_grid {len(doy_grid)}."
        )
    bayes_median = np.median(bayes_beta_t, axis=0)
    bayes_q05 = np.quantile(bayes_beta_t, 0.05, axis=0)
    bayes_q95 = np.quantile(bayes_beta_t, 0.95, axis=0)

    # fPLS full-data fits needed for headline k and the shape family (dedup)
    k_needed_full = sorted(set([args.n_components] + args.shape_components))
    log.info(f"Fitting full-data fPLS models for k in {k_needed_full}...")
    X_full_fd = make_fdata_grid(X_full, doy_grid)
    fpls_full_models = {k: fit_fpls(X_full_fd, y, k) for k in k_needed_full}
    fpls_full_coefs = {k: m.coef_.ravel() for k, m in fpls_full_models.items()}

    # Step 3: scale-matching check (headline k)
    log.info("Step 3: scale-matching check...")
    weights = simpson_weights(doy_grid)
    headline_coef = fpls_full_coefs[args.n_components]
    headline_model = fpls_full_models[args.n_components]
    scale_check = scale_matching_check(
        X_full, y, X_full_fd, headline_coef, headline_model, weights, bayes_median, label="full-data",
    )
    consistency_ok = scale_check["predict_consistency_ok"]
    max_abs_diff = scale_check["predict_consistency_max_abs_diff"]
    integral_fpls = scale_check["integral_abs_coef_fpls"]
    integral_bayes = scale_check["integral_abs_beta_bayesian_median"]
    magnitude_ratio = scale_check["magnitude_ratio"]
    scale_suspect = scale_check["scale_suspect"]

    # Step 4: diagnostic-only fPLS LOYO sweep (k=1..max_components_sweep, plus
    # headline/shape k if outside that range, to support later steps)
    sweep_range = list(range(1, args.max_components_sweep + 1))
    k_values_for_loyo = sorted(set(sweep_range) | set(k_needed_full))
    log.info(f"Step 4: fPLS diagnostic LOYO sweep for k in {k_values_for_loyo}...")
    fpls_preds = fpls_loyo_sweep(X_full, y, years, doy_grid, k_values_for_loyo)

    selection_rows = []
    for k in sweep_range:
        rmse, r2 = pooled_metrics(y, fpls_preds[k])
        selection_rows.append({
            "cohort": args.cohort, "n_components": k,
            "pooled_loyo_rmse": rmse, "pooled_loyo_r2": r2,
            "n_obs": n_obs, "n_folds": len(unique_years),
            "note": "selection-optimistic / exploratory — not used to pick the headline model",
        })
    selection_df = pd.DataFrame(selection_rows)
    selection_path = TABLES / f"fpls_component_selection_{args.cohort}_{args.predictor}.csv"
    selection_df.to_csv(selection_path, index=False)
    log.info(f"Saved: {selection_path}")

    # Step 5: headline fPLS metrics (leakage-free, fixed a priori k) + shape family curves
    log.info(f"Step 5: headline fPLS (k={args.n_components}) leakage-free LOYO metrics...")
    fpls_headline_rmse, fpls_headline_r2 = pooled_metrics(y, fpls_preds[args.n_components])

    # Step 6: Bayesian LOYO folds (for RMSE pooling only)
    log.info("Step 6: Bayesian LOYO folds...")
    bayes_preds = bayesian_loyo(args, X_reduced, y, years)
    bayes_rmse, bayes_r2 = pooled_metrics(y, bayes_preds)

    # Step 7: comparison outputs
    log.info("Step 7: writing comparison outputs...")
    loyo_df = pd.DataFrame([
        {
            "cohort": args.cohort, "model": "fpls", "complexity": f"k={args.n_components}",
            "pooled_loyo_rmse": fpls_headline_rmse, "pooled_loyo_r2": fpls_headline_r2,
            "n_obs": n_obs, "n_folds": len(unique_years),
            "note": "secondary, likely-underpowered diagnostic (~9 folds) — not the primary result",
        },
        {
            "cohort": args.cohort, "model": "bayesian_baseline", "complexity": f"n_basis={N_BASIS} (RW2)",
            "pooled_loyo_rmse": bayes_rmse, "pooled_loyo_r2": bayes_r2,
            "n_obs": n_obs, "n_folds": len(unique_years),
            "note": "secondary, likely-underpowered diagnostic (~9 folds) — not the primary result",
        },
    ])
    loyo_path = TABLES / f"fpls_vs_bayesian_loyo_{args.cohort}_{args.predictor}.csv"
    loyo_df.to_csv(loyo_path, index=False)
    log.info(f"Saved: {loyo_path}")

    beta_t_data = {"doy": doy_grid}
    for k in args.shape_components:
        beta_t_data[f"beta_fpls_k{k}"] = fpls_full_coefs[k]
    beta_t_data["beta_bayesian_median"] = bayes_median
    beta_t_data["beta_bayesian_q05"] = bayes_q05
    beta_t_data["beta_bayesian_q95"] = bayes_q95
    beta_t_df = pd.DataFrame(beta_t_data)
    beta_t_path = TABLES / f"fpls_vs_bayesian_beta_t_{args.cohort}_{args.predictor}.csv"
    beta_t_df.to_csv(beta_t_path, index=False)
    log.info(f"Saved: {beta_t_path}")

    masks = window_masks(doy_grid)
    bayes_stage_draws = stage_means_draws(bayes_beta_t, masks)
    lo, hi = 0.5 * (1.0 - args.hdi_prob), 1.0 - 0.5 * (1.0 - args.hdi_prob)
    stage_rows = []
    for stage in STAGES_RESOLVED:
        row = {
            "stage": stage,
            "doy_start": STAGES_RESOLVED[stage][0],
            "doy_end": STAGES_RESOLVED[stage][1],
            "beta_bayesian_median": float(np.median(bayes_stage_draws[stage])),
            "beta_bayesian_q_lo": float(np.quantile(bayes_stage_draws[stage], lo)),
            "beta_bayesian_q_hi": float(np.quantile(bayes_stage_draws[stage], hi)),
        }
        for k in args.shape_components:
            row[f"beta_fpls_k{k}"] = stage_means_point(fpls_full_coefs[k], masks)[stage]
        stage_rows.append(row)
    # S7 - S6 contrast row
    bayes_contrast_draws = bayes_stage_draws["S7"] - bayes_stage_draws["S6"]
    contrast_row = {
        "stage": "S7_minus_S6",
        "doy_start": None, "doy_end": None,
        "beta_bayesian_median": float(np.median(bayes_contrast_draws)),
        "beta_bayesian_q_lo": float(np.quantile(bayes_contrast_draws, lo)),
        "beta_bayesian_q_hi": float(np.quantile(bayes_contrast_draws, hi)),
    }
    for k in args.shape_components:
        sm = stage_means_point(fpls_full_coefs[k], masks)
        contrast_row[f"beta_fpls_k{k}"] = sm["S7"] - sm["S6"]
    stage_rows.append(contrast_row)
    stage_df = pd.DataFrame(stage_rows)
    stage_path = TABLES / f"fpls_vs_bayesian_stage_means_{args.cohort}_{args.predictor}.csv"
    stage_df.to_csv(stage_path, index=False)
    log.info(f"Saved: {stage_path}")

    fpls_shape_curves = {k: fpls_full_coefs[k] for k in args.shape_components}
    fig_path = plot_overlay(doy_grid, bayes_median, bayes_q05, bayes_q95,
                            fpls_shape_curves, args.cohort, args.predictor, scale_suspect)
    log.info(f"Saved: {fig_path}")

    meta = {
        "cohort": args.cohort,
        "predictor": args.predictor,
        "tau_days": WD_STATE_TAU_DAYS if args.predictor == "wd_state" else None,
        "n_obs": n_obs,
        "n_folds": len(unique_years),
        "headline_n_components": args.n_components,
        "shape_components": args.shape_components,
        "max_components_sweep": args.max_components_sweep,
        "bayesian_stem_full": full_stem,
        "scale_check": {
            "predict_consistency_ok": bool(consistency_ok),
            "predict_consistency_max_abs_diff": max_abs_diff,
            "integral_abs_coef_fpls": integral_fpls,
            "integral_abs_beta_bayesian_median": integral_bayes,
            "magnitude_ratio": magnitude_ratio,
            "scale_suspect": bool(scale_suspect),
        },
        "caveats": [
            "Smoothers, not deployable models: both sides drop lag_yield and "
            "the year RE, so neither LOYO RMSE reflects the project's "
            "deployable model performance.",
            "fPLS is supervised dimension reduction; the Bayesian model is an "
            "unsupervised RW2 smoothness prior on a fixed 5-basis reduction — "
            "different estimator classes, not two settings of one knob.",
            "fpls_component_selection is selection-optimistic (k chosen to "
            "minimize the same score reported) and is NOT used to choose the "
            "headline model; the headline k is fixed a priori.",
            "The headline RMSE comparison (k="
            f"{args.n_components} fPLS vs. {N_BASIS}-basis RW2 Bayesian) is "
            "confounded by a flexibility asymmetry, not just fold count — do "
            "not read the RMSE gap as 'which method is better.'",
            "~9 LOYO folds means pooled RMSE is a noisy, underpowered "
            "estimator; the beta(t) shape overlay (primary result) is more "
            "informative than the RMSE table (secondary diagnostic).",
        ],
    }
    meta_path = TABLES / f"fpls_vs_bayesian_meta_{args.cohort}_{args.predictor}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Saved: {meta_path}")

    if args.drop_years:
        run_drop_years_experiment(
            args=args, doy_grid=doy_grid, X_full=X_full, X_full_fd=X_full_fd, X_reduced=X_reduced,
            y=y, years=years, fpls_full_models=fpls_full_models, weights=weights,
            masks=masks, stage_df_full=stage_df,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
