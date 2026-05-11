#!/usr/bin/env bash
# 10_run_sweeps.sh — Run all Step 10 sensitivity sweeps.
#
# Concurrency safety:
#   - A lockfile (results/.sens_run.lock) prevents two runners from running
#     simultaneously.  Two runners would clobber each other's shared
#     preprocessing files (gdd_grid.npy, WD_state_matrix.npy, ...).
#   - Groups run STRICTLY SEQUENTIALLY within one runner.
#   - Each (param-combo × window-shift) result is written to its own CSV in
#     results/tables/sens_rows/, atomically (tmp + rename).  No shared
#     CSV is appended to during fitting.
#   - sensitivity_summary.csv is rebuilt ONCE at the very end by
#     10_aggregate_sensitivity.py.
#
# Resumability:
#   - Step 7 MCMC fits are skipped if their beta_t_draws.npy artifact
#     exists (--reuse-existing), so completed fits survive interruptions.
#   - Preprocessing (steps 2, 3, 5) always re-runs at the start of each
#     group: it is fast (<60s) and deterministic.
#   - Per-row CSVs survive interruptions, so re-running picks up where
#     the previous run stopped without re-fitting anything.
#
# Usage:
#   bash src/10_run_sweeps.sh            # run all groups
#   bash src/10_run_sweeps.sh --skip-irrigated
#
# Resume after interruption: just re-run the script.

set -uo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate

# Force JAX onto CPU.  Empirically ~10x faster than GPU for this model
# (small data, sequential chains, no batched ops worth offloading).
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu

LOG=results/sens_run.log
LOCK=results/.sens_run.lock
mkdir -p results/tables results/tables/sens_rows results/posteriors

# ── Lockfile guard ──────────────────────────────────────────────────────
if [ -e "$LOCK" ]; then
    other_pid=$(cat "$LOCK" 2>/dev/null || true)
    if [ -n "$other_pid" ] && kill -0 "$other_pid" 2>/dev/null; then
        echo "ERROR: another sensitivity runner is active (PID $other_pid)." >&2
        echo "       Wait for it to finish or remove $LOCK if it crashed." >&2
        exit 1
    else
        echo "WARNING: stale lock $LOCK (PID ${other_pid:-?} not running). Removing."
        rm -f "$LOCK"
    fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

P="python src/10_sensitivity_analysis.py"
# Common flags for every group.  No --append: per-row CSVs are written
# directly by 10_sensitivity_analysis.py.
# Use --window-shifts=... (= syntax) so leading-dash values aren't parsed
# as new argparse flags.
A="--cohorts arbequina --reuse-existing --window-shifts=-50,0,50"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_group() {
    local label="$1"; shift
    log ">>> START $label"
    # Capture Python exit code even through the tee pipe
    $P "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        log "!!! FAILED $label (exit $rc) — check $LOG"
        exit $rc
    fi
    log "<<< DONE  $label"
}

log "=========================================="
log "Step 10 sensitivity sweep"
log "Log: $LOG"
log "=========================================="

# ── Group 1: T_b=10, tau=45, k=5 — both predictors & year-RE modes ───────
# Covers: VPD predictor swap, year-RE toggle.
# Baseline (wd_state + with_year) is reused; 3 new MCMC fits.
run_group "G1: T_b=10 tau=45 k=5 predictors={wd_state,vpd} year-RE={with,no}" \
    $A \
    --base-temps 10 --taus 45 --basis-k 5 \
    --predictors wd_state,vpd \
    --year-re-modes with_year,no_year

# ── Group 2: tau sweep ────────────────────────────────────────────────────
# T_b=10, k=5, wd_state, with_year; tau ∈ {15, 30, 60, 120}
# tau=45 baseline already in summary; 4 new MCMC fits.
for tau in 15 30 60 120; do
    run_group "G2: tau=$tau" \
        $A \
        --base-temps 10 --taus "$tau" --basis-k 5 \
        --predictors wd_state --year-re-modes with_year
done

# ── Group 3: basis-k sweep ───────────────────────────────────────────────
# T_b=10, tau=45, wd_state, with_year; k ∈ {4, 6, 8}
# k=5 baseline already in summary; 3 new MCMC fits.
for k in 4 6 8; do
    run_group "G3: k=$k" \
        $A \
        --base-temps 10 --taus 45 --basis-k "$k" \
        --predictors wd_state --year-re-modes with_year
done

# ── Group 4: T_b sweep ───────────────────────────────────────────────────
# tau=45, k=5, wd_state, with_year; T_b ∈ {7, 8.5, 12.5}
# T_b=10 baseline already in summary; 3 new MCMC fits.
for tb in 7 8.5 12.5; do
    run_group "G4: T_b=$tb" \
        $A \
        --base-temps "$tb" --taus 45 --basis-k 5 \
        --predictors wd_state --year-re-modes with_year
done

# ── Group 5: T_b=6.5 reparameterization ─────────────────────────────────
# Alt anchors from config (ALT_*), GDD_UPPER=2700.  1 new MCMC fit.
run_group "G5: T_b=6.5 (alt anchors, GDD_UPPER=2700)" \
    $A \
    --base-temps 6.5 --taus 45 --basis-k 5 \
    --predictors wd_state --year-re-modes with_year

# ── Group 6: irrigated yield negative control ─────────────────────────────
# Replaces rainfed yield with irrigated yield in olive_yield.csv (backed up
# and restored by 10_sensitivity_analysis.py's try/finally block).
# 1 new MCMC fit.  Preprocessing is restored to T_b=10 tau=45 k=5 afterwards.
if [[ "${1:-}" != "--skip-irrigated" ]]; then
    run_group "G6: irrigated negative control" \
        $A \
        --base-temps 10 --taus 45 --basis-k 5 \
        --predictors wd_state --year-re-modes with_year \
        --irrigated
fi

# ── Aggregate per-row CSVs into the summary file ─────────────────────────────
log ">>> Aggregating per-row CSVs into sensitivity_summary.csv"
python src/10_aggregate_sensitivity.py 2>&1 | tee -a "$LOG"

log "=========================================="
log "All Step 10 sweeps COMPLETE"
log "Per-row CSVs: results/tables/sens_rows/"
log "Summary:      results/tables/sensitivity_summary.csv"
log "=========================================="
