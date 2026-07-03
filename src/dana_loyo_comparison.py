#!/usr/bin/env python3
"""
dana_loyo_comparison.py — Compare β(S8) and S7−S8 contrast for
full model vs LOYO folds excluding each DANA year (2018, 2024),
split by rainfed vs irrigated (Arbequina) and all_olive (rainfed only).

Key question: does the negative autumn (S8) signal survive removing
DANA years, and does it differ between rainfed and irrigated trees?
If storm damage / disease is the mechanism, both panels should show
the same DANA-driven flip; if it is a water-supply effect, irrigated
trees should be less sensitive.

Run: source .venv/bin/activate && python src/dana_loyo_comparison.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import POSTERIORS, PROCESSED, STAGES_RESOLVED

HDI_PROB = 0.90
LO = 0.5 * (1.0 - HDI_PROB)
HI = 1.0 - LO

# (scenario_key, display_label, rainfed_stem_fn, irrigated_stem_fn)
# stem_fn(key) → filename stem inserted into posterior path
SCENARIOS = [
    ("full",     "Full model (all years)"),
    ("excl2018", "excl2018 (no DANA 2018)"),
    ("excl2024", "excl2024 (no DANA 2024)"),
]

# Rainfed: numpyro_main_arbequina_wd_state_{key}_beta_t_draws.npy
# Irrigated full uses a different naming convention: irrigated_full
RAINFED_STEMS = {
    "full":     "numpyro_main_arbequina_wd_state_full_beta_t_draws.npy",
    "excl2018": "numpyro_main_arbequina_wd_state_excl2018_beta_t_draws.npy",
    "excl2024": "numpyro_main_arbequina_wd_state_excl2024_beta_t_draws.npy",
}
IRRIGATED_STEMS = {
    "full":     "numpyro_main_arbequina_wd_state_irrigated_full_beta_t_draws.npy",
    "excl2018": "numpyro_main_arbequina_wd_state_excl2018_irrigated_beta_t_draws.npy",
    "excl2024": "numpyro_main_arbequina_wd_state_excl2024_irrigated_beta_t_draws.npy",
}
ALL_OLIVE_STEMS = {
    "full":     "numpyro_main_all_olive_wd_state_full_beta_t_draws.npy",
    "excl2018": "numpyro_main_all_olive_wd_state_excl2018_beta_t_draws.npy",
    "excl2024": "numpyro_main_all_olive_wd_state_excl2024_beta_t_draws.npy",
}


def stage_mean(beta_t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean β over a stage window per posterior draw. Shape: (n_draws,)."""
    return beta_t[:, mask].mean(axis=1)


def p_diff(delta: np.ndarray) -> float:
    p_gt0 = float((delta > 0).mean())
    return max(p_gt0, 1.0 - p_gt0)


def load_and_summarize(path: Path, s7_mask, s8_mask):
    """Load beta_t_draws and return (b_s8_draws, b_s7_draws) or None if missing."""
    if not path.exists():
        return None, None
    beta_t = np.load(path)
    return stage_mean(beta_t, s8_mask), stage_mean(beta_t, s7_mask)


def print_panel(title, stem_map, scenarios, s7_mask, s8_mask):
    print(f"\n  {title}")
    print(f"  {'─'*76}")
    print(
        f"  {'Scenario':<28}"
        f"  {'β(S8) median':>12}"
        f"  {'90% CI':>24}"
        f"  {'P(β_S8>0)':>10}"
        f"  {'S7−S8 P(diff)':>14}"
    )
    print(f"  {'─'*76}")
    for key, label in scenarios:
        b_s8, b_s7 = load_and_summarize(
            POSTERIORS / stem_map[key], s7_mask, s8_mask
        )
        if b_s8 is None:
            print(f"  {label:<28}  MISSING")
            continue
        delta = b_s7 - b_s8
        p_pos = float((b_s8 > 0).mean())
        pd = p_diff(delta)
        direction = "↑ pos" if p_pos > 0.5 else "↓ neg"
        print(
            f"  {label:<28}"
            f"  {np.median(b_s8):>+12.6f}"
            f"  [{np.quantile(b_s8, LO):+.5f}, {np.quantile(b_s8, HI):+.5f}]"
            f"  {p_pos:>9.3f}  {direction}"
            f"  {pd:>13.3f}"
        )


def main() -> None:
    doy_grid = np.load(PROCESSED / "doy_grid.npy")

    s7_lo, s7_hi = STAGES_RESOLVED["S7"]
    s8_lo, s8_hi = STAGES_RESOLVED["S8"]
    s7_mask = (doy_grid >= s7_lo) & (doy_grid <= s7_hi)
    s8_mask = (doy_grid >= s8_lo) & (doy_grid <= s8_hi)

    print(f"\nS7 mask: DOY {s7_lo}–{s7_hi}  ({s7_mask.sum()} points)")
    print(f"S8 mask: DOY {s8_lo}–{s8_hi}  ({s8_mask.sum()} points)")
    print("\nP(β_S8>0): probability autumn water is beneficial.")
    print("S7−S8 P(diff): probability fruit-dev and maturation stages differ.")
    print("If storm/disease drives the DANA signal, rainfed and irrigated")
    print("should show the same flip when DANA years are removed.")

    print(f"\n{'='*80}")
    print("  ARBEQUINA — rainfed vs irrigated comparison")
    print(f"{'='*80}")
    print_panel("Rainfed",  RAINFED_STEMS,  SCENARIOS, s7_mask, s8_mask)
    print_panel("Irrigated", IRRIGATED_STEMS, SCENARIOS, s7_mask, s8_mask)

    print(f"\n{'='*80}")
    print("  ALL OLIVE ≥500 ha — rainfed only")
    print(f"{'='*80}")
    print_panel("All olive (rainfed)", ALL_OLIVE_STEMS, SCENARIOS, s7_mask, s8_mask)


if __name__ == "__main__":
    main()
