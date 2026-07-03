#!/usr/bin/env python3
"""
01_load_data.py — Data ingestion for olive water sensitivity analysis.

Loads three sources:
  1. Comarca-level annual olive yield (filtered to cultivar cohorts)
  2. Daily climate per comarca (T_min, T_max, precip, ET0, VPD)
  3. Pre-computed GDD and water balance

Validates coverage, computes lag yield, and writes model-ready files to
data/processed/.
"""
import argparse
import logging
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils.config import (
    YIELD_CSV, DAILY_CLIMATE_CSV, GDD_CSV, WATER_BALANCE_CSV,
    COHORT_WHITELISTS, PROCESSED, MAX_MISSING_FRAC, TABLES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load cohort-filtered yield and climate inputs."
    )
    parser.add_argument(
        "--cohort",
        choices=["arbequina", "morruda", "all_olive", "both"],
        default="both",
        help="Cultivar cohort(s) to include (default: both = all registered).",
    )
    parser.add_argument(
        "--strict-cohorts",
        action="store_true",
        help="Fail when any requested cohort whitelist is missing.",
    )
    parser.add_argument(
        "--irrigated",
        action="store_true",
        help="Use irrigated yield (yield_irrig_tha / regadiu_ha) instead of rainfed.",
    )
    return parser.parse_args()


def resolve_cohort_comarques(cohort_mode: str, strict: bool) -> dict[str, set[str]]:
    """Load cohort whitelist comarques."""
    requested = list(COHORT_WHITELISTS.keys()) if cohort_mode == "both" else [cohort_mode]
    cohort_sets: dict[str, set[str]] = {}

    for cohort in requested:
        path = COHORT_WHITELISTS[cohort]
        if not path.exists():
            msg = (
                f"Whitelist not found for cohort '{cohort}': {path}. "
                "Create it first from DUN cultivar fractions."
            )
            if strict:
                raise FileNotFoundError(msg)
            log.warning(msg + " Skipping this cohort.")
            continue

        whitelist = pd.read_csv(path)
        if "comarca" not in whitelist.columns:
            raise ValueError(f"Whitelist missing required 'comarca' column: {path}")

        comarques = set(whitelist["comarca"].dropna().astype(str).str.strip())
        if not comarques:
            msg = f"Cohort '{cohort}' whitelist has no comarques: {path}"
            if strict:
                raise ValueError(msg)
            log.warning(msg + " Skipping this cohort.")
            continue

        cohort_sets[cohort] = comarques
        log.info(
            f"Cohort '{cohort}': {len(comarques)} comarques from {path.name}"
        )

    if not cohort_sets:
        raise ValueError(
            "No cohort could be loaded. Provide at least one valid whitelist in data/dun/."
        )
    return cohort_sets


def load_yield_base(irrigated: bool = False) -> pd.DataFrame:
    """Load olive yield and compute lagged yield per comarca.

    When *irrigated* is True, uses irrigated columns (regadiu_ha,
    yield_irrig_tha) renamed to seca_ha / yield_tha for downstream
    compatibility.  Rows with zero or missing irrigated area are dropped.
    """
    raw = pd.read_csv(YIELD_CSV)
    olives = raw.query("pheno_key == 'olive'").copy()
    log.info(f"Olive yield rows (all comarques): {len(olives)}")

    if irrigated:
        olives = olives[["year", "comarca", "regadiu_ha", "yield_irrig_tha"]].copy()
        olives = olives.rename(columns={"regadiu_ha": "seca_ha",
                                        "yield_irrig_tha": "yield_tha"})
        olives = olives.dropna(subset=["seca_ha", "yield_tha"])
        olives = olives[(olives["seca_ha"] > 0) & (olives["yield_tha"] > 0)]
        log.info(f"Irrigated olive yield rows after filtering: {len(olives)}")
    else:
        olives = olives[["year", "comarca", "seca_ha", "yield_tha"]].copy()
    olives = olives.sort_values(["comarca", "year"]).reset_index(drop=True)
    olives["lag_yield"] = olives.groupby("comarca")["yield_tha"].shift(1)

    n_before = len(olives)
    olives = olives.dropna(subset=["lag_yield"]).reset_index(drop=True)
    log.info(f"Dropped {n_before - len(olives)} rows without lag yield")
    log.info(
        f"Lag-ready olive yield: {len(olives)} comarca-years, "
        f"years {olives['year'].min()}-{olives['year'].max()}"
    )
    return olives


def apply_cohorts(base_yield: pd.DataFrame, cohort_sets: dict[str, set[str]]) -> pd.DataFrame:
    """Subset lag-ready yield by cohort comarques and append cohort labels."""
    parts = []
    for cohort, comarques in cohort_sets.items():
        cohort_df = base_yield[base_yield["comarca"].isin(comarques)].copy()
        cohort_df["cohort"] = cohort
        parts.append(cohort_df)
        log.info(
            f"Cohort '{cohort}': {len(cohort_df)} comarca-years, "
            f"{cohort_df['comarca'].nunique()} comarques"
        )

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["cohort", "comarca", "year"]).reset_index(drop=True)
    return out


def load_daily_climate(comarcas: set[str]) -> pd.DataFrame:
    """Load daily climate, filter to relevant comarques."""
    clim = pd.read_csv(DAILY_CLIMATE_CSV, parse_dates=["date"])
    clim = clim[clim["comarca"].isin(comarcas)].copy()
    clim["year"] = clim["date"].dt.year
    log.info(f"Daily climate: {len(clim)} rows for {clim['comarca'].nunique()} comarques")
    return clim


def validate_coverage(yield_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that every cohort-comarca-year in yield has near-complete daily climate
    from Jan 1 through Dec 31. Drop rows with >5% missing days.
    """
    coverage = (
        climate_df
        .groupby(["comarca", "year"])["date"]
        .agg(n_days="count")
        .reset_index()
    )
    coverage["expected"] = coverage["year"].apply(
        lambda y: 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    )
    coverage["frac_missing"] = 1.0 - coverage["n_days"] / coverage["expected"]

    merged = yield_df.merge(coverage, on=["comarca", "year"], how="left")
    bad = merged[merged["frac_missing"] > MAX_MISSING_FRAC]
    if len(bad) > 0:
        log.warning(
            f"Dropping {len(bad)} cohort-comarca-years with >{MAX_MISSING_FRAC*100:.0f}% missing climate:"
        )
        for _, r in bad.iterrows():
            log.warning(
                f"  {r['cohort']} | {r['comarca']} {int(r['year'])}: {r['frac_missing']:.1%} missing"
            )
        merged = merged[merged["frac_missing"] <= MAX_MISSING_FRAC]

    good = merged[merged["frac_missing"].notna()]
    log.info(f"Coverage validated: {len(good)} cohort-comarca-years pass")
    return good[yield_df.columns].reset_index(drop=True)


def load_gdd(comarcas: set[str]) -> pd.DataFrame:
    """Load pre-computed GDD (T_b=10)."""
    gdd = pd.read_csv(GDD_CSV, parse_dates=["date"])
    gdd = gdd[gdd["comarca"].isin(comarcas)].copy()
    log.info(f"GDD data: {len(gdd)} rows")
    return gdd


def load_water_balance(comarcas: set[str]) -> pd.DataFrame:
    """Load pre-computed P - ET0."""
    wb = pd.read_csv(WATER_BALANCE_CSV, parse_dates=["date"])
    wb = wb[wb["comarca"].isin(comarcas)].copy()
    log.info(f"Water balance data: {len(wb)} rows")
    return wb


def center_yield_by_comarca(
    yield_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subtract per-(cohort, comarca) mean from yield_tha and lag_yield.

    This within-estimator removes structural between-comarca baseline differences
    at the data level, making a comarca random effect in the model redundant.
    Comarca means are saved separately for back-transformation if needed.

    Note: means are computed on the fitted panel (all available years after lag
    drop). In LOYO refits the held-out year remains in the mean, introducing a
    minor ~1/9-year leakage; this is documented as a known limitation.
    """
    means = (
        yield_df.groupby(["cohort", "comarca"])[["yield_tha", "lag_yield"]]
        .mean()
        .rename(columns={"yield_tha": "yield_tha_mean", "lag_yield": "lag_yield_mean"})
        .reset_index()
    )
    out = yield_df.merge(means, on=["cohort", "comarca"], how="left")
    out["yield_tha"] = out["yield_tha"] - out["yield_tha_mean"]
    out["lag_yield"] = out["lag_yield"] - out["lag_yield_mean"]
    out = out.drop(columns=["yield_tha_mean", "lag_yield_mean"])
    log.info(
        f"Comarca-mean centered yield: grand mean={out['yield_tha'].mean():.4f} "
        f"(should be ~0), within-comarca std={out['yield_tha'].std():.4f}"
    )
    return out, means


def write_cultivar_yield_reports(yield_df: pd.DataFrame) -> None:
    """Write final cultivar yield summary reports for downstream review.
    Note: yield_tha and lag_yield are comarca-mean centered at this point.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    report = (
        yield_df
        .groupby("cohort", as_index=False)
        .agg(
            n_rows=("year", "size"),
            n_comarques=("comarca", "nunique"),
            start_year=("year", "min"),
            end_year=("year", "max"),
            total_seca_ha=("seca_ha", "sum"),
            mean_seca_ha=("seca_ha", "mean"),
            mean_yield_tha=("yield_tha", "mean"),
            median_yield_tha=("yield_tha", "median"),
            std_yield_tha=("yield_tha", "std"),
            mean_lag_yield=("lag_yield", "mean"),
        )
        .sort_values("cohort")
        .reset_index(drop=True)
    )

    overall = pd.DataFrame([{
        "cohort": "all",
        "n_rows": int(len(yield_df)),
        "n_comarques": int(yield_df["comarca"].nunique()),
        "start_year": int(yield_df["year"].min()),
        "end_year": int(yield_df["year"].max()),
        "total_seca_ha": float(yield_df["seca_ha"].sum()),
        "mean_seca_ha": float(yield_df["seca_ha"].mean()),
        "mean_yield_tha": float(yield_df["yield_tha"].mean()),
        "median_yield_tha": float(yield_df["yield_tha"].median()),
        "std_yield_tha": float(yield_df["yield_tha"].std()),
        "mean_lag_yield": float(yield_df["lag_yield"].mean()),
    }])
    report_all = pd.concat([report, overall], ignore_index=True)

    combined_path = TABLES / "final_cultivar_yield_report.csv"
    report_all.to_csv(combined_path, index=False)
    log.info(f"Saved final cultivar yield report: {combined_path}")

    for cohort in sorted(yield_df["cohort"].unique()):
        cohort_df = yield_df[yield_df["cohort"] == cohort].copy()
        detail_path = TABLES / f"final_cultivar_yield_report_{cohort}.csv"
        cohort_df.to_csv(detail_path, index=False)
        log.info(f"Saved cohort detail report: {detail_path} ({len(cohort_df)} rows)")


def main():
    args = parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    cohort_sets = resolve_cohort_comarques(args.cohort, args.strict_cohorts)

    # 1. Yield
    yield_base = load_yield_base(irrigated=args.irrigated)
    yield_df = apply_cohorts(yield_base, cohort_sets)
    comarcas = set(yield_df["comarca"])

    # 2. Daily climate
    climate_df = load_daily_climate(comarcas)

    # 3. Validate coverage
    yield_df = validate_coverage(yield_df, climate_df)
    comarcas = set(yield_df["comarca"])  # update after any drops

    # 4. Center yield by comarca mean (within-estimator; comarca RE not needed in model)
    yield_df, comarca_means = center_yield_by_comarca(yield_df)

    # 5. GDD + water balance
    gdd_df = load_gdd(comarcas)
    wb_df = load_water_balance(comarcas)

    # 6. Save processed datasets
    yield_df.to_csv(PROCESSED / "olive_yield.csv", index=False)
    comarca_means.to_csv(PROCESSED / "comarca_yield_means.csv", index=False)
    climate_df[climate_df["comarca"].isin(comarcas)].to_csv(
        PROCESSED / "daily_climate.csv", index=False
    )
    gdd_df[gdd_df["comarca"].isin(comarcas)].to_csv(
        PROCESSED / "gdd.csv", index=False
    )
    wb_df[wb_df["comarca"].isin(comarcas)].to_csv(
        PROCESSED / "water_balance.csv", index=False
    )
    write_cultivar_yield_reports(yield_df)
    log.info(f"  comarca_yield_means.csv: {len(comarca_means)} rows")

    cohort_counts = yield_df.groupby("cohort").size().to_dict()
    log.info(f"Saved to {PROCESSED}/")
    log.info(f"  olive_yield.csv: {len(yield_df)} rows")
    log.info(
        "  Cohorts: "
        + ", ".join(f"{cohort} ({count})" for cohort, count in cohort_counts.items())
    )
    log.info(f"  Unique years: {sorted(yield_df['year'].unique())}")
    log.info(f"  Unique comarques: {sorted(comarcas)}")
    log.info(f"\nYield summary:\n{yield_df['yield_tha'].describe()}")


if __name__ == "__main__":
    main()
