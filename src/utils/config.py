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
GDD_CSV = DATA / "agera5_gdd_catalonia.csv"
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

# ── GDD parameters ───────────────────────────────────────────────────────
BASE_TEMP = 10.0               # °C, De Melo-Abreu et al. 2004
GDD_UPPER = 1700               # common upper bound (fixed, not per-year)
GDD_STEP = 25                  # GDD grid resolution
GDD_GRID_MIN = 0
WD_ROLLING_WINDOW_GDD = 30     # rolling cumulative WD comparison predictor
WD_STATE_TAU_GDD = 45          # leaky-integrator timescale for WD state proxy

# ── Phenological anchors (GDD at T_b = 10 °C) ───────────────────────────
# Sources: Didevarasl et al. 2023; Sanz-Cortés et al. 2002 (BBCH 50-59)
PRE_FLOWERING_GDD = (0, 350)        # bud break → inflorescence development
FLOWERING_GDD = (350, 450)
PIT_HARDENING_GDD = (900, 1200)
OIL_ACCUMULATION_GDD = (1200, 1700)

# ── Alternate anchors for T_b = 6.5 °C sensitivity (Step 10) ─────────────
# Cultivar-specific estimates for early cultivars (Arbequina-like) in
# inland Catalonia, interpolated from Toledo/Galicia/Croatia bracket.
# NOT used in headline analysis — kept here for the sensitivity sweep.
# Start-date convention: Jan 1 (must be verified against data before use).
ALT_BASE_TEMP = 6.5
ALT_PRE_FLOWERING_GDD = (0, 250)     # bud break / BBCH 51
ALT_FLOWERING_GDD = (400, 600)       # BBCH 61, mid-range Didevarasl/Selak
ALT_PIT_HARDENING_GDD = (1200, 1700) # approximate, needs literature check
ALT_OIL_ACCUMULATION_GDD = (1700, 2500)  # Tunç et al. range, approximate
ALT_GDD_UPPER = 2700                 # harvest truncation at T_b=6.5

# ── Basis parameters ─────────────────────────────────────────────────────
N_BASIS = 5                    # cubic B-splines, start with 5

# ── MCMC defaults ────────────────────────────────────────────────────────
N_CHAINS = 4
N_WARMUP = 2000
N_SAMPLES = 2000
TARGET_ACCEPT = 0.95

# ── Missing data threshold ───────────────────────────────────────────────
MAX_MISSING_FRAC = 0.05        # drop comarca-years with > 5% missing daily
