#!/usr/bin/env python3
"""
07_fit_numpyro_main.py — Production hierarchical scalar-on-function model (NumPyro).

Implements Step 7 with:
  - default predictor: WD_state_reduced
  - per-fit predictor standardization (column-wise on fit rows)
  - RW2 (second-order random walk) prior on spline coefficients
  - non-centred comarca/year random effects

Outputs:
  - InferenceData netcdf
  - posterior summary CSV
  - fit design CSV
  - metadata JSON
  - per-fit standardization arrays (mean/sd)
  - posterior beta(t) draws on physical scale for Step 8
"""
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, POSTERIORS, TABLES, DEFAULT_COHORT,
    N_CHAINS, N_WARMUP, N_SAMPLES, TARGET_ACCEPT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PREDICTOR_REDUCED_FILES = {
    "wd_state": "WD_state_reduced.npy",
    "wd": "WD_reduced.npy",
    "wd30": "WD30_reduced.npy",
    "vpd": "VPD_reduced.npy",
    "tmin": "TMIN_reduced.npy",
    "tmax": "TMAX_reduced.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit production NumPyro scalar-on-function model."
    )
    parser.add_argument(
        "--cohort",
        choices=["arbequina", "morruda", "all_olive", "both", "all"],
        default="both",
        help="Cultivar cohort(s) to fit (default: both).",
    )
    parser.add_argument(
        "--predictor",
        choices=list(PREDICTOR_REDUCED_FILES.keys()),
        default="wd_state",
        help="Reduced predictor matrix to use (default: wd_state).",
    )
    parser.add_argument("--draws", type=int, default=N_SAMPLES)
    parser.add_argument("--tune", type=int, default=N_WARMUP)
    parser.add_argument("--chains", type=int, default=N_CHAINS)
    parser.add_argument(
        "--chain-method",
        choices=["sequential", "parallel", "vectorized"],
        default="sequential",
        help="NumPyro chain execution strategy (default: sequential).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=TARGET_ACCEPT)
    parser.add_argument("--hdi-prob", type=float, default=0.90)
    parser.add_argument("--sd-eps", type=float, default=1e-6)
    parser.add_argument(
        "--no-year-re",
        action="store_true",
        help="Disable year random effect (sensitivity spec).",
    )
    parser.add_argument(
        "--exclude-year",
        type=int,
        default=None,
        help="Exclude a single year from fitting (used by LOYO).",
    )
    parser.add_argument(
        "--stem-suffix",
        type=str,
        default="",
        help="Optional suffix appended to output stem (e.g., loyo_y2019).",
    )
    return parser.parse_args()


def resolve_fit_cohorts(requested: str, obs_df: pd.DataFrame) -> list[str]:
    available = set(obs_df["cohort"].astype(str).unique())
    if requested == "both":
        target = ["arbequina", "morruda", "all_olive"]
    elif requested == "all":
        target = ["all"]
    else:
        target = [requested]

    if "all" in target:
        return target

    missing = [c for c in target if c not in available]
    if missing:
        raise ValueError(
            f"Requested cohort(s) missing from processed data: {missing}. "
            "Run Step 1 with --cohort both after creating required whitelist(s)."
        )
    return target


def load_dependencies():
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "arviz is required for Step 7 outputs. Install in venv: pip install arviz"
        ) from exc

    try:
        import jax
        import jax.numpy as jnp
        from jax import lax
    except ImportError as exc:
        raise ImportError(
            "jax is required for Step 7. Install in venv: pip install jax jaxlib"
        ) from exc

    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise ImportError(
            "numpyro is required for Step 7. Install in venv: pip install numpyro"
        ) from exc

    return az, jax, jnp, lax, numpyro, dist, MCMC, NUTS


def load_inputs(predictor: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    predictor_path = PROCESSED / PREDICTOR_REDUCED_FILES[predictor]
    if not predictor_path.exists():
        raise FileNotFoundError(
            f"Reduced predictor missing: {predictor_path}. Run Step 5 first."
        )

    obs_index = pd.read_csv(PROCESSED / "obs_index.csv")
    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")
    X_reduced = np.load(predictor_path)
    B = np.load(PROCESSED / "B_basis.npy")

    if "cohort" not in obs_index.columns:
        obs_index["cohort"] = DEFAULT_COHORT
    if "cohort" not in yield_df.columns:
        yield_df["cohort"] = DEFAULT_COHORT

    key_cols = ["cohort", "comarca", "year"]
    obs_index = obs_index.copy()
    obs_index["row_idx"] = np.arange(len(obs_index))
    merged = obs_index.merge(
        yield_df[key_cols + ["yield_tha", "lag_yield"]],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    missing = merged["yield_tha"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} obs_index rows could not be matched to olive_yield.csv.")
    if X_reduced.shape[0] != len(merged):
        raise ValueError(
            f"Row mismatch: X_reduced has {X_reduced.shape[0]} rows but obs_index has {len(merged)}."
        )
    return merged, X_reduced, B


def standardize_columns(X: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = X.mean(axis=0)
    x_sd = X.std(axis=0, ddof=0)
    x_sd = np.maximum(x_sd, eps)
    Xz = (X - x_mean) / x_sd
    return Xz, x_mean, x_sd


def build_design_arrays(
    fit_df: pd.DataFrame, X_reduced: np.ndarray, sd_eps: float
) -> tuple[pd.DataFrame, dict]:
    fit_df = fit_df.sort_values(["cohort", "comarca", "year"]).reset_index(drop=True)
    fit_df = fit_df.dropna(subset=["yield_tha", "lag_yield"]).reset_index(drop=True)

    row_idx = fit_df["row_idx"].to_numpy(dtype=int)
    X = X_reduced[row_idx]
    Xz, x_mean, x_sd = standardize_columns(X, eps=sd_eps)

    fit_df["year_str"] = fit_df["year"].astype(str)
    fit_df["year_code"] = fit_df["year_str"].astype("category").cat.codes.astype(int)
    fit_df["comarca_code"] = fit_df["comarca"].astype("category").cat.codes.astype(int)
    year_levels = list(fit_df["year_str"].astype("category").cat.categories)
    comarca_levels = list(fit_df["comarca"].astype("category").cat.categories)

    arrays = {
        "Xz": Xz,
        "x_mean": x_mean,
        "x_sd": x_sd,
        "y": fit_df["yield_tha"].to_numpy(dtype=float),
        "lag_yield": fit_df["lag_yield"].to_numpy(dtype=float),
        "year_idx": fit_df["year_code"].to_numpy(dtype=int),
        "comarca_idx": fit_df["comarca_code"].to_numpy(dtype=int),
        "n_years": len(year_levels),
        "n_comarcas": len(comarca_levels),
        "year_levels": year_levels,
        "comarca_levels": comarca_levels,
    }
    return fit_df, arrays


def make_model(numpyro, jnp, lax, dist, include_year_re: bool):
    def model(
        Xz, lag_yield, year_idx, comarca_idx,
        n_years, n_comarcas, y=None,
    ):
        n_basis = Xz.shape[1]
        if n_basis < 3:
            raise ValueError(f"n_basis must be >= 3 for RW2; got {n_basis}")

        beta0 = numpyro.sample("beta0", dist.Normal(0.0, 0.5))
        beta1 = numpyro.sample("beta1", dist.Normal(0.0, 0.5))
        sigma_smooth = numpyro.sample("sigma_smooth", dist.HalfNormal(0.1))
        innovations = numpyro.sample(
            "rw2_innovations",
            dist.Normal(0.0, sigma_smooth).expand([n_basis - 2]),
        )

        def rw2_step(carry, eps):
            b_prev2, b_prev1 = carry
            b_new = 2.0 * b_prev1 - b_prev2 + eps
            return (b_prev1, b_new), b_new

        _, beta_tail = lax.scan(rw2_step, (beta0, beta1), innovations)
        beta_coefs = jnp.concatenate([jnp.array([beta0, beta1]), beta_tail])
        numpyro.deterministic("beta_coefs", beta_coefs)

        functional_effect = jnp.dot(Xz, beta_coefs)

        sigma_comarca = numpyro.sample("sigma_comarca", dist.HalfNormal(0.5))
        comarca_raw = numpyro.sample(
            "comarca_raw", dist.Normal(0.0, 1.0).expand([n_comarcas])
        )
        comarca_re = sigma_comarca * comarca_raw
        numpyro.deterministic("comarca_re", comarca_re)

        year_re = 0.0
        if include_year_re:
            sigma_year = numpyro.sample("sigma_year", dist.HalfNormal(0.5))
            year_raw = numpyro.sample(
                "year_raw", dist.Normal(0.0, 1.0).expand([n_years])
            )
            year_re = sigma_year * year_raw
            numpyro.deterministic("year_re", year_re)

        alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
        lag_coef = numpyro.sample("lag_coef", dist.Normal(0.0, 0.5))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        mu = alpha + functional_effect + lag_coef * lag_yield + comarca_re[comarca_idx]
        if include_year_re:
            mu = mu + year_re[year_idx]

        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

    return model


def fit_one(
    args: argparse.Namespace,
    cohort_name: str,
    fit_df: pd.DataFrame,
    arrays: dict,
    B: np.ndarray,
    az,
    jax,
    numpyro,
    jnp,
    lax,
    dist,
    MCMC,
    NUTS,
):
    include_year_re = not args.no_year_re
    model = make_model(
        numpyro=numpyro, jnp=jnp, lax=lax, dist=dist, include_year_re=include_year_re
    )
    kernel = NUTS(model, target_accept_prob=args.target_accept)
    mcmc = MCMC(
        kernel,
        num_warmup=args.tune,
        num_samples=args.draws,
        num_chains=args.chains,
        chain_method=args.chain_method,
        progress_bar=True,
    )

    rng = jax.random.PRNGKey(args.seed)
    mcmc.run(
        rng,
        Xz=jnp.asarray(arrays["Xz"]),
        lag_yield=jnp.asarray(arrays["lag_yield"]),
        year_idx=jnp.asarray(arrays["year_idx"]),
        comarca_idx=jnp.asarray(arrays["comarca_idx"]),
        n_years=arrays["n_years"],
        n_comarcas=arrays["n_comarcas"],
        y=jnp.asarray(arrays["y"]),
    )

    coords = {
        "basis": np.arange(arrays["Xz"].shape[1]),
        "year": arrays["year_levels"],
        "comarca": arrays["comarca_levels"],
        "obs_id": np.arange(len(arrays["y"])),
    }
    dims = {
        "beta_coefs": ["basis"],
        "rw2_innovations": ["basis_minus2"],
        "year_re": ["year"],
        "comarca_re": ["comarca"],
    }
    idata = az.from_numpyro(mcmc, coords=coords, dims=dims)
    summary = az.summary(idata, hdi_prob=args.hdi_prob)
    bad_rhat = int((summary["r_hat"] > 1.01).sum()) if "r_hat" in summary.columns else 0
    low_ess = int((summary["ess_bulk"] < 400).sum()) if "ess_bulk" in summary.columns else 0

    stem = f"numpyro_main_{cohort_name}_{args.predictor}"
    if not include_year_re:
        stem += "_noyearre"
    if args.exclude_year is not None:
        stem += f"_excl{int(args.exclude_year)}"
    if args.stem_suffix:
        stem += f"_{args.stem_suffix}"

    idata_path = POSTERIORS / f"{stem}.nc"
    summary_path = TABLES / f"{stem}_summary.csv"
    design_path = TABLES / f"{stem}_design.csv"
    meta_path = TABLES / f"{stem}_meta.json"
    xmean_path = POSTERIORS / f"{stem}_x_mean.npy"
    xsd_path = POSTERIORS / f"{stem}_x_sd.npy"
    beta_phys_path = POSTERIORS / f"{stem}_beta_coefs_phys.npy"
    beta_t_path = POSTERIORS / f"{stem}_beta_t_draws.npy"

    idata.to_netcdf(idata_path)
    summary.to_csv(summary_path)
    fit_df.to_csv(design_path, index=False)
    np.save(xmean_path, arrays["x_mean"])
    np.save(xsd_path, arrays["x_sd"])

    beta_std = idata.posterior["beta_coefs"].values.reshape(-1, arrays["Xz"].shape[1])
    beta_phys = beta_std / arrays["x_sd"][None, :]
    beta_t = beta_phys @ B.T
    np.save(beta_phys_path, beta_phys)
    np.save(beta_t_path, beta_t)

    meta = {
        "cohort": cohort_name,
        "predictor": args.predictor,
        "include_year_re": include_year_re,
        "formula": (
            f"yield_tha ~ f({args.predictor}_reduced_standardized) + "
            "lag_yield + (1|comarca) [+ (1|year)]"
        ),
        "n_obs": int(len(fit_df)),
        "n_comarcas": int(arrays["n_comarcas"]),
        "n_years": int(arrays["n_years"]),
        "n_basis": int(arrays["Xz"].shape[1]),
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "chain_method": args.chain_method,
        "target_accept": args.target_accept,
        "sd_eps": args.sd_eps,
        "exclude_year": args.exclude_year,
        "stem_suffix": args.stem_suffix,
        "bad_rhat_count_gt_1.01": bad_rhat,
        "low_ess_bulk_count_lt_400": low_ess,
        "files": {
            "posterior_netcdf": str(idata_path),
            "summary_csv": str(summary_path),
            "design_csv": str(design_path),
            "x_mean_npy": str(xmean_path),
            "x_sd_npy": str(xsd_path),
            "beta_coefs_phys_npy": str(beta_phys_path),
            "beta_t_draws_npy": str(beta_t_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved posterior: {idata_path}")
    log.info(f"Saved summary:   {summary_path}")
    log.info(f"Saved design:    {design_path}")
    log.info(f"Saved metadata:  {meta_path}")
    log.info(f"Saved scaling:   {xmean_path}, {xsd_path}")
    log.info(f"Saved beta(t):   {beta_t_path}")
    if bad_rhat > 0 or low_ess > 0:
        log.warning(
            f"Diagnostic flags ({cohort_name}): r_hat>1.01 count={bad_rhat}, ess_bulk<400 count={low_ess}"
        )

    return {
        "cohort": cohort_name,
        "predictor": args.predictor,
        "include_year_re": include_year_re,
        "n_obs": int(len(fit_df)),
        "n_comarcas": int(arrays["n_comarcas"]),
        "n_years": int(arrays["n_years"]),
        "bad_rhat_count_gt_1.01": bad_rhat,
        "low_ess_bulk_count_lt_400": low_ess,
        "chain_method": args.chain_method,
    }


def main():
    args = parse_args()
    az, jax, jnp, lax, numpyro, dist, MCMC, NUTS = load_dependencies()

    POSTERIORS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    obs_df, X_reduced, B = load_inputs(args.predictor)
    fit_cohorts = resolve_fit_cohorts(args.cohort, obs_df)
    final_rows = []

    for cohort_name in fit_cohorts:
        if cohort_name == "all":
            fit_df = obs_df.copy()
        else:
            fit_df = obs_df[obs_df["cohort"] == cohort_name].copy()
        if fit_df.empty:
            raise ValueError(
                f"No rows available for cohort='{cohort_name}'. Run Step 1 first."
            )
        if args.exclude_year is not None:
            n_before = len(fit_df)
            fit_df = fit_df[fit_df["year"].astype(int) != int(args.exclude_year)].copy()
            n_dropped = n_before - len(fit_df)
            log.info(
                f"Excluded year {int(args.exclude_year)} for cohort={cohort_name}; "
                f"dropped {n_dropped} rows"
            )
            if fit_df.empty:
                raise ValueError(
                    f"No rows left after excluding year {args.exclude_year} "
                    f"for cohort='{cohort_name}'."
                )

        fit_df, arrays = build_design_arrays(fit_df=fit_df, X_reduced=X_reduced, sd_eps=args.sd_eps)
        log.info(
            f"Fitting cohort={cohort_name}: n_obs={len(fit_df)}, n_basis={arrays['Xz'].shape[1]}, "
            f"year_re={'off' if args.no_year_re else 'on'}, chain_method={args.chain_method}"
        )
        row = fit_one(
            args=args,
            cohort_name=cohort_name,
            fit_df=fit_df,
            arrays=arrays,
            B=B,
            az=az,
            jax=jax,
            numpyro=numpyro,
            jnp=jnp,
            lax=lax,
            dist=dist,
            MCMC=MCMC,
            NUTS=NUTS,
        )
        final_rows.append(row)

    final_report = pd.DataFrame(final_rows).sort_values(["cohort", "predictor"]).reset_index(drop=True)
    final_report_path = TABLES / "final_numpyro_model_report.csv"
    final_report.to_csv(final_report_path, index=False)
    log.info(f"Saved final NumPyro model report: {final_report_path}")


if __name__ == "__main__":
    main()
