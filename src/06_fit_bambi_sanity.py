#!/usr/bin/env python3
"""
06_fit_bambi_sanity.py — Quick hierarchical sanity-check model with Bambi.

This script is intentionally lightweight compared to the production NumPyro model.
It aggregates functional predictors into five stage-window means and fits a
calendar-DOY sanity check of the form:

    yield_tha ~ wd_S_pre + wd_S5 + wd_S6 + wd_S7 + wd_S8 + lag_yield [+ (1|year)]

Default uses lag_yield WITHOUT year random effects (the plan warns against running
both — they collide and lag_yield collapses to zero). Use --year-re to add (1|year)
for the sensitivity comparison. Comarca RE is omitted because Step 1 centers yield
within comarca.
"""
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    PROCESSED, POSTERIORS, TABLES, DEFAULT_COHORT, PLAN_COHORTS,
    STAGES_RESOLVED,
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
    "S_pre": "S_pre",
    "S5": "S5",
    "S6": "S6",
    "S7": "S7",
    "S8": "S8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Bambi sanity model using five stage-window means."
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
        "--year-re",
        action="store_true",
        help="Add (1|year) random effect (absorbs common-year shocks; "
             "note: lag_yield and year RE partially collide at comarca scale).",
    )
    return parser.parse_args()


def resolve_fit_cohorts(requested: str, obs_df: pd.DataFrame) -> list[str]:
    """Resolve which cohort fits to run based on CLI selection."""
    available = set(obs_df["cohort"].astype(str).unique())
    if requested == "both":
        target = list(PLAN_COHORTS)
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


def load_stage_anchors() -> dict[str, tuple[int, int]]:
    """Load calendar DOY anchor windows from phenology_anchors.csv."""
    path = PROCESSED / "phenology_anchors.csv"
    if not path.exists():
        log.warning(f"{path} not found; using default resolved windows from config.")
        return {name: (int(a), int(b)) for name, (a, b) in STAGES_RESOLVED.items()}

    anchors = pd.read_csv(path)
    needed = set(STAGES_RESOLVED)
    found: dict[str, tuple[int, int]] = {}
    for stage, grp in anchors.groupby("stage"):
        if stage not in needed:
            continue
        row = grp.iloc[0]
        # Prefer resolved (non-overlapping) boundaries
        for col_start, col_end in (("doy_start", "doy_end"), ("doy_start_raw", "doy_end_raw")):
            if col_start in row and col_end in row:
                found[stage] = (int(row[col_start]), int(row[col_end]))
                break

    for stage, fallback in STAGES_RESOLVED.items():
        if stage not in found:
            log.warning(f"Stage '{stage}' missing in phenology_anchors.csv; using default {fallback}.")
            found[stage] = (int(fallback[0]), int(fallback[1]))
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
    doy_grid = np.load(PROCESSED / "doy_grid.npy")
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
    return merged, matrix, doy_grid


def add_stage_means(
    df: pd.DataFrame,
    matrix: np.ndarray,
    doy_grid: np.ndarray,
    anchors: dict[str, tuple[int, int]],
    predictor_prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute stage-window means for all five calendar stages and append to DataFrame."""
    stages = ["S_pre", "S5", "S6", "S7", "S8"]

    out = df.copy()
    fixed_terms: list[str] = []
    row_idx = out["row_idx"].to_numpy()
    matrix_sub = matrix[row_idx]

    for stage in stages:
        start, end = anchors[stage]
        mask = (doy_grid >= start) & (doy_grid <= end)
        if not np.any(mask):
            raise ValueError(f"No grid points found in stage window '{stage}' ({start}-{end} DOY).")

        alias = STAGE_ALIASES[stage]
        col = f"{predictor_prefix}_{alias}_mean"
        out[col] = matrix_sub[:, mask].mean(axis=1)
        fixed_terms.append(col)
        log.info(
            f"Stage {stage}: {start}-{end} DOY, points={int(mask.sum())}, column={col}"
        )

    return out, fixed_terms


def fit_model(
    bmb,
    model_df: pd.DataFrame,
    fixed_terms: list[str],
    args: argparse.Namespace,
):
    """Fit Bambi mixed-effects sanity model.

    Comarca RE is omitted: yield_tha is comarca-mean centered in Step 1,
    so there is no between-comarca variance left to absorb.
    """
    rhs = " + ".join(fixed_terms + ["lag_yield"])
    re_terms = " + (1|year)" if args.year_re else ""
    formula = f"yield_tha ~ {rhs}{re_terms}"
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

    obs_df, matrix, doy_grid = load_inputs(args.predictor)
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
            doy_grid=doy_grid,
            anchors=anchors,
            predictor_prefix=args.predictor,
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

        idata_path = POSTERIORS / f"{stem}.nc"
        summary_path = TABLES / f"{stem}_summary.csv"
        data_path = TABLES / f"{stem}_design.csv"
        meta_path = TABLES / f"{stem}_meta.json"

        idata.to_netcdf(idata_path)
        summary.to_csv(summary_path)
        model_df.to_csv(data_path, index=False)

        # Calendar-axis contrasts
        S_pre_term = f"{args.predictor}_S_pre_mean"
        S5_term = f"{args.predictor}_S5_mean"
        S6_term = f"{args.predictor}_S6_mean"
        S7_term = f"{args.predictor}_S7_mean"
        S8_term = f"{args.predictor}_S8_mean"

        contrasts = []
        # Headline: S7 − S6 (fruit development vs flowering)
        if S7_term in idata.posterior and S6_term in idata.posterior:
            delta = (idata.posterior[S7_term] - idata.posterior[S6_term]).values.reshape(-1)
            contrasts.append({
                "delta_name": f"{S7_term} - {S6_term}",
                "delta_median": float(np.median(delta)),
                "delta_q05": float(np.quantile(delta, 0.05)),
                "delta_q95": float(np.quantile(delta, 0.95)),
                "prob_delta_gt_0": float((delta > 0).mean()),
            })
        # S6 − S5 (flowering vs inflorescence)
        if S6_term in idata.posterior and S5_term in idata.posterior:
            delta_s6_s5 = (idata.posterior[S6_term] - idata.posterior[S5_term]).values.reshape(-1)
            contrasts.append({
                "delta_name": f"{S6_term} - {S5_term}",
                "delta_median": float(np.median(delta_s6_s5)),
                "delta_q05": float(np.quantile(delta_s6_s5, 0.05)),
                "delta_q95": float(np.quantile(delta_s6_s5, 0.95)),
                "prob_delta_gt_0": float((delta_s6_s5 > 0).mean()),
            })
        # S5 − S_pre (inflorescence vs winter rest)
        if S5_term in idata.posterior and S_pre_term in idata.posterior:
            delta_s5_spre = (idata.posterior[S5_term] - idata.posterior[S_pre_term]).values.reshape(-1)
            contrasts.append({
                "delta_name": f"{S5_term} - {S_pre_term}",
                "delta_median": float(np.median(delta_s5_spre)),
                "delta_q05": float(np.quantile(delta_s5_spre, 0.05)),
                "delta_q95": float(np.quantile(delta_s5_spre, 0.95)),
                "prob_delta_gt_0": float((delta_s5_spre > 0).mean()),
            })

        contrast_payload = {}
        if contrasts:
            contrast_payload = {
                **contrasts[0],             # headline: S7 − S6
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
