"""
Shared configuration for the olive water sensitivity pipeline.
All path and parameter defaults live here so individual scripts stay DRY.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = PROJECT / "results"
POSTERIORS = RESULTS / "posteriors"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

# ── Raw data files ───────────────────────────────────────────────────────
YIELD_CSV = DATA / "catalan_woody_yield_raw.csv"
DAILY_CLIMATE_CSV = DATA / "agera5_daily_catalonia.csv"
WATER_BALANCE_CSV = DATA / "agera5_water_balance_catalonia.csv"
ARBEQUINA_WHITELIST = DATA / "dun" / "comarca_arbequina_whitelist.csv"
MORRUDA_WHITELIST = DATA / "dun" / "comarca_morruda_whitelist.csv"
ALL_OLIVE_WHITELIST = DATA / "dun" / "comarca_all_olive_whitelist.csv"
RAINFED_FRACTION = DATA / "dun" / "comarca_olive_rainfed_fraction.csv"

COHORT_WHITELISTS = {
    "arbequina": ARBEQUINA_WHITELIST,
    "morruda": MORRUDA_WHITELIST,
    "all_olive": ALL_OLIVE_WHITELIST,
}
DEFAULT_COHORT = "arbequina"

# Two-cohort plan: arbequina headline + all_olive breadth check
PLAN_COHORTS = ["arbequina", "all_olive"]

# ── Calendar DOY parameters (current methodology) ───────────────────────
DOY_START = 1                  # pre-inflorescence start (Jan 1)
DOY_END = 327                  # S8 end / harvest truncation (≈ 23 November)
DOY_STEP = 1                   # daily grid
WD_ROLLING_WINDOW_DAYS = 30    # rolling cumulative WD comparison predictor
WD_STATE_TAU_DAYS = 45         # leaky-integrator timescale for WD state proxy

# ── Soil-bucket parameters (wd_bucket predictor) ────────────────────────
# Single-layer soil water reservoir as a physical alternative to the leaky
# integrator. Field capacity = root-zone plant-available water; central value
# from Lleida calcareous loam (~150 mm/m AWC × ~1 m effective rooting depth).
SOIL_FIELD_CAPACITY_MM = 150.0   # reservoir size (mm); sweep 100/150/200 in Step 10
SOIL_KC = 0.6                    # crop coefficient for actual ET drawdown
SOIL_INIT_FRAC = 0.5             # initial storage as fraction of field capacity

# Raw stage boundaries (Garrido et al. 2021 for S5–S8; S_pre is pipeline-defined)
S_PRE_DOY = (1, 71)            # pre-inflorescence (winter rest / early vegetative)
S5_DOY = (72, 138)             # inflorescence development (BBCH 51–59)
S6_DOY = (136, 173)            # flowering (BBCH 61–69)
S7_DOY = (168, 299)            # fruit development (BBCH 71–79)
S8_DOY = (299, 327)            # maturation / harvest (BBCH 81–89)

# Resolved non-overlapping boundaries (overlap days assigned to earlier stage)
STAGES_RESOLVED = {
    "S_pre": (1, 71),
    "S5":    (72, 138),
    "S6":    (139, 173),
    "S7":    (174, 299),
    "S8":    (300, 327),
}

# ── Basis parameters ─────────────────────────────────────────────────────
N_BASIS = 5                    # cubic B-splines, start with 5

# ── MCMC defaults ────────────────────────────────────────────────────────
N_CHAINS = 4
N_WARMUP = 2000
N_SAMPLES = 2000
TARGET_ACCEPT = 0.95

# ── Missing data threshold ───────────────────────────────────────────────
MAX_MISSING_FRAC = 0.05        # drop comarca-years with > 5% missing daily
