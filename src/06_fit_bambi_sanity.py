#!/usr/bin/env python3
"""
06_fit_bambi_sanity.py — Quick hierarchical sanity-check model with Bambi.

This script is intentionally lightweight compared to the production NumPyro model.
It aggregates functional predictors into stage means and fits:

    yield_tha ~ stage_flowering + stage_pit + lag_yield + (1|comarca)

Default uses lag_yield WITHOUT year random effects (the plan warns against running
both — they collide and lag_yield collapses to zero). Use --year-re to add (1|year)
for the sensitivity comparison.
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
    PRE_FLOWERING_GDD, FLOWERING_GDD, PIT_HARDENING_GDD, OIL_ACCUMULATION_GDD,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PREDICTOR_FILES = {
    "wd": "WD_matrix.npy",
    "wd30": "WD30_matrix.npy",
    "vpd": "VPD_matrix.npy",
    "tmin": "TMIN_matrix.npy",
    "tmax": "TMAX_matrix.npy",
}

STAGE_ALIASES = {
    "pre_flowering": "preflower",
    "flowering": "flowering",
    "pit_hardening": "pit",
    "oil_accumulation": "oil",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Bambi sanity model using stage-window means."
    )
    parser.add_argument(
        "--cohort",
        choices=["arbequina", "morruda", "all_olive", "both", "all"],
        default="both",
        help="Cultivar cohort(s) to fit (default: both).",
    )
    parser.add_argument(
        "--predictor",
        choices=["wd", "wd30", "vpd", "tmin", "tmax"],
        default="wd",
        help="Primary predictor curve to aggregate (default: wd).",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--hdi-prob", type=float, default=0.90)
    parser.add_argument(
        "--include-oil-stage",
        action="store_true",
        help="Include oil_accumulation stage mean as an extra fixed effect.",
    )
    parser.add_argument(
        "--year-re",
        action="store_true",
        help="Add (1|year) random effect alongside lag_yield (sensitivity check only; "
             "the two collide and lag_yield collapses to zero).",
    )
    return parser.parse_args()


def resolve_fit_cohorts(requested: str, obs_df: pd.DataFrame) -> list[str]:
    """Resolve which cohort fits to run based on CLI selection."""
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
        import bambi as bmb
    except ImportError as exc:
        raise ImportError(
            "bambi is required for Step 6. Install it in your venv (e.g. pip install bambi arviz)."
        ) from exc

    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "arviz is required for Step 6 diagnostics. Install it in your venv (e.g. pip install arviz)."
        ) from exc

    return bmb, az


def load_stage_anchors() -> dict[str, tuple[float, float]]:
    """Load stage anchor windows; fallback to config defaults if needed."""
    defaults = {
        "pre_flowering": PRE_FLOWERING_GDD,
        "flowering": FLOWERING_GDD,
        "pit_hardening": PIT_HARDENING_GDD,
        "oil_accumulation": OIL_ACCUMULATION_GDD,
    }
    path = PROCESSED / "phenology_anchors.csv"
    if not path.exists():
        log.warning(f"{path} not found; using default anchor windows from config.")
        return defaults

    anchors = pd.read_csv(path)
    needed = set(defaults)
    found: dict[str, tuple[float, float]] = {}
    for stage, grp in anchors.groupby("stage"):
        if stage not in needed:
            continue
        row = grp.iloc[0]
        found[stage] = (float(row["gdd_start"]), float(row["gdd_end"]))

    for stage, fallback in defaults.items():
        if stage not in found:
            log.warning(f"Stage '{stage}' missing in phenology_anchors.csv; using default {fallback}.")
            found[stage] = fallback
    return found


def load_inputs(predictor: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load observation index, yield table, and predictor matrix with alignment keys."""
    predictor_path = PROCESSED / PREDICTOR_FILES[predictor]
    if not predictor_path.exists():
        raise FileNotFoundError(
            f"Predictor matrix missing: {predictor_path}. Run Step 3 first."
        )

    obs_index = pd.read_csv(PROCESSED / "obs_index.csv")
    yield_df = pd.read_csv(PROCESSED / "olive_yield.csv")
    gdd_grid = np.load(PROCESSED / "gdd_grid.npy")
    matrix = np.load(predictor_path)

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
    if matrix.shape[0] != len(merged):
        raise ValueError(
            f"Row mismatch: matrix has {matrix.shape[0]} rows but obs_index has {len(merged)}."
        )
    return merged, matrix, gdd_grid


def add_stage_means(
    df: pd.DataFrame,
    matrix: np.ndarray,
    gdd_grid: np.ndarray,
    anchors: dict[str, tuple[float, float]],
    predictor_prefix: str,
    include_oil: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute stage-window means for selected predictor and append to DataFrame."""
    stages = ["pre_flowering", "flowering", "pit_hardening"]
    if include_oil:
        stages.append("oil_accumulation")

    out = df.copy()
    fixed_terms: list[str] = []
    row_idx = out["row_idx"].to_numpy()
    matrix_sub = matrix[row_idx]

    for stage in stages:
        start, end = anchors[stage]
        mask = (gdd_grid >= start) & (gdd_grid <= end)
        if not np.any(mask):
            raise ValueError(f"No grid points found in stage window '{stage}' ({start}-{end} GDD).")

        alias = STAGE_ALIASES[stage]
        col = f"{predictor_prefix}_{alias}_mean"
        out[col] = matrix_sub[:, mask].mean(axis=1)
        fixed_terms.append(col)
        log.info(
            f"Stage {stage}: {start:.0f}-{end:.0f} GDD, points={int(mask.sum())}, column={col}"
        )

    return out, fixed_terms


def fit_model(
    bmb,
    model_df: pd.DataFrame,
    fixed_terms: list[str],
    args: argparse.Namespace,
):
    """Fit Bambi mixed-effects sanity model."""
    rhs = " + ".join(fixed_terms + ["lag_yield"])
    re_terms = "(1|comarca)"
    if args.year_re:
        re_terms = "(1|year) + (1|comarca)"
    formula = f"yield_tha ~ {rhs} + {re_terms}"
    log.info(f"Fitting model: {formula}")

    model = bmb.Model(formula, data=model_df, family="gaussian")
    idata = model.fit(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        random_seed=args.seed,
        target_accept=args.target_accept,
    )
    return model, idata, formula


def main():
    args = parse_args()
    bmb, az = load_dependencies()

    POSTERIORS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    obs_df, matrix, gdd_grid = load_inputs(args.predictor)
    anchors = load_stage_anchors()

    fit_cohorts = resolve_fit_cohorts(args.cohort, obs_df)
    final_rows = []

    for cohort_name in fit_cohorts:
        if cohort_name == "all":
            fit_df = obs_df.copy()
        else:
            fit_df = obs_df[obs_df["cohort"] == cohort_name].copy()
        if fit_df.empty:
            raise ValueError(
                f"No rows available for cohort='{cohort_name}'. "
                "Run Step 1 with that cohort first."
            )

        fit_df = fit_df.sort_values(["cohort", "comarca", "year"]).reset_index(drop=True)
        model_df, fixed_terms = add_stage_means(
            df=fit_df,
            matrix=matrix,
            gdd_grid=gdd_grid,
            anchors=anchors,
            predictor_prefix=args.predictor,
            include_oil=args.include_oil_stage,
        )

        model_df["year"] = model_df["year"].astype(str).astype("category")
        model_df["comarca"] = model_df["comarca"].astype(str).astype("category")
        model_df = model_df.dropna(subset=["yield_tha", "lag_yield"]).reset_index(drop=True)

        _, idata, formula = fit_model(
            bmb=bmb, model_df=model_df, fixed_terms=fixed_terms, args=args
        )

        summary = az.summary(idata, hdi_prob=args.hdi_prob)
        bad_rhat = int((summary["r_hat"] > 1.01).sum()) if "r_hat" in summary.columns else 0
        low_ess = int((summary["ess_bulk"] < 400).sum()) if "ess_bulk" in summary.columns else 0

        stem = f"bambi_sanity_{cohort_name}_{args.predictor}"
        if args.include_oil_stage:
            stem += "_oil"

        idata_path = POSTERIORS / f"{stem}.nc"
        summary_path = TABLES / f"{stem}_summary.csv"
        data_path = TABLES / f"{stem}_design.csv"
        meta_path = TABLES / f"{stem}_meta.json"

        idata.to_netcdf(idata_path)
        summary.to_csv(summary_path)
        model_df.to_csv(data_path, index=False)

        contrast_payload = {}
        preflower_term = f"{args.predictor}_preflower_mean"
        flower_term = f"{args.predictor}_flowering_mean"
        pit_term = f"{args.predictor}_pit_mean"
        contrasts = []
        # pit − flower (original headline contrast)
        if flower_term in idata.posterior and pit_term in idata.posterior:
            delta = (idata.posterior[pit_term] - idata.posterior[flower_term]).values.reshape(-1)
            contrasts.append({
                "delta_name": f"{pit_term} - {flower_term}",
                "delta_median": float(np.median(delta)),
                "delta_q05": float(np.quantile(delta, 0.05)),
                "delta_q95": float(np.quantile(delta, 0.95)),
                "prob_delta_gt_0": float((delta > 0).mean()),
            })
        # flower − preflower (new: does sensitivity differ before vs during flowering?)
        if preflower_term in idata.posterior and flower_term in idata.posterior:
            delta_pf = (idata.posterior[flower_term] - idata.posterior[preflower_term]).values.reshape(-1)
            contrasts.append({
                "delta_name": f"{flower_term} - {preflower_term}",
                "delta_median": float(np.median(delta_pf)),
                "delta_q05": float(np.quantile(delta_pf, 0.05)),
                "delta_q95": float(np.quantile(delta_pf, 0.95)),
                "prob_delta_gt_0": float((delta_pf > 0).mean()),
            })
        if contrasts:
            contrast_payload = {
                **contrasts[0],             # headline: pit − flower
                "all_contrasts": contrasts,  # full list for JSON output
            }

        meta = {
            "formula": formula,
            "cohort": cohort_name,
            "predictor": args.predictor,
            "n_obs": int(len(model_df)),
            "n_comarcas": int(model_df["comarca"].nunique()),
            "n_years": int(model_df["year"].nunique()),
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "target_accept": args.target_accept,
            "bad_rhat_count_gt_1.01": bad_rhat,
            "low_ess_bulk_count_lt_400": low_ess,
            "stage_terms": fixed_terms,
            "contrast": contrast_payload,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log.info(f"Saved posterior: {idata_path}")
        log.info(f"Saved summary:   {summary_path}")
        log.info(f"Saved design:    {data_path}")
        log.info(f"Saved metadata:  {meta_path}")
        if contrast_payload:
            log.info(
                f"Δ={contrast_payload['delta_name']} median={contrast_payload['delta_median']:.4f}, "
                f"90% CI=[{contrast_payload['delta_q05']:.4f}, {contrast_payload['delta_q95']:.4f}], "
                f"P(Δ>0)={contrast_payload['prob_delta_gt_0']:.3f}"
            )
        if bad_rhat > 0 or low_ess > 0:
            log.warning(
                f"Diagnostic flags ({cohort_name}): r_hat>1.01 count={bad_rhat}, ess_bulk<400 count={low_ess}"
            )

        final_rows.append({
            "cohort": cohort_name,
            "predictor": args.predictor,
            "n_obs": int(len(model_df)),
            "n_comarcas": int(model_df["comarca"].nunique()),
            "n_years": int(model_df["year"].nunique()),
            "bad_rhat_count_gt_1.01": bad_rhat,
            "low_ess_bulk_count_lt_400": low_ess,
            "delta_median": contrast_payload.get("delta_median"),
            "delta_q05": contrast_payload.get("delta_q05"),
            "delta_q95": contrast_payload.get("delta_q95"),
            "prob_delta_gt_0": contrast_payload.get("prob_delta_gt_0"),
        })

    final_report = pd.DataFrame(final_rows).sort_values(["cohort", "predictor"]).reset_index(drop=True)
    final_report_path = TABLES / "final_cultivar_model_report.csv"
    final_report.to_csv(final_report_path, index=False)
    log.info(f"Saved final cultivar model report: {final_report_path}")


if __name__ == "__main__":
    main()
