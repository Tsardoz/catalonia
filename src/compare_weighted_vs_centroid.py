"""
Compare area-weighted vs centroid-based climate for olive comarques.

This script validates the impact of area weighting by comparing climate values
for the 23 comarques present in both datasets.

Outputs:
  - Console: Summary statistics and correlation analysis
  - Figure: Scatter plots comparing methods
  - CSV: Full comparison table for inspection
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"

WEIGHTED_PATH = DATA_DIR / "catalan_olive_yield_climate_weighted.csv"
CENTROID_PATH = DATA_DIR / "catalan_woody_yield_climate.csv"


def main():
    # Load datasets
    print("Loading datasets...")
    df_weighted = pd.read_csv(WEIGHTED_PATH)
    print(f"  Area-weighted: {len(df_weighted)} rows, {df_weighted['comarca'].nunique()} comarques")

    df_centroid = pd.read_csv(CENTROID_PATH)
    df_centroid = df_centroid[df_centroid['pheno_key'] == 'olive'].copy()
    print(f"  Centroid-based: {len(df_centroid)} rows, {df_centroid['comarca'].nunique()} comarques")

    # Merge on comarca + year
    print("\nMerging datasets...")

    # Variables to compare
    compare_vars = [
        'flower_mean_vpd', 'flower_max_vpd', 'flower_cum_precip', 'flower_mean_tmax',
        'fruit_set_mean_vpd', 'fruit_set_max_vpd', 'fruit_set_cum_precip', 'fruit_set_mean_tmax',
        'maturation_mean_vpd', 'maturation_max_vpd', 'maturation_cum_precip', 'maturation_mean_tmax',
    ]

    merge_cols = ['comarca', 'year', 'yield_tha'] + compare_vars

    merged = df_weighted.merge(
        df_centroid[merge_cols],
        on=['comarca', 'year'],
        suffixes=('_weighted', '_centroid'),
        how='inner'
    )

    print(f"  {len(merged)} matched rows")
    print(f"  {merged['comarca'].nunique()} comarques with both methods")

    # Compute differences
    print("\nComputing differences (weighted - centroid)...")
    for var in compare_vars:
        merged[f'{var}_diff'] = merged[f'{var}_weighted'] - merged[f'{var}_centroid']

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS: Differences (weighted - centroid)")
    print("="*70)

    diff_cols = [f'{var}_diff' for var in compare_vars]
    summary = merged[diff_cols].describe()
    print(summary.to_string())

    # Correlations
    print("\n" + "="*70)
    print("CORRELATIONS: Weighted vs Centroid")
    print("="*70)

    for var in ['flower_mean_vpd', 'fruit_set_mean_vpd', 'maturation_mean_vpd']:
        r = merged[[f'{var}_weighted', f'{var}_centroid']].corr().iloc[0, 1]
        print(f"{var:30s}: r = {r:.4f}")

    # Per-comarca differences
    print("\n" + "="*70)
    print("COMARCA-LEVEL DIFFERENCES (mean across years)")
    print("="*70)

    comarca_diff = merged.groupby('comarca').agg({
        'flower_mean_vpd_diff': 'mean',
        'fruit_set_mean_vpd_diff': 'mean',
        'maturation_mean_vpd_diff': 'mean',
    }).sort_values('flower_mean_vpd_diff')

    print(comarca_diff.to_string())

    # Save full comparison
    out_csv = DATA_DIR / "climate_comparison_weighted_vs_centroid.csv"
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved full comparison to {out_csv}")

    # Create scatter plots
    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Area-Weighted vs Centroid Climate Comparison\n(23 Olive Comarques, 2016-2024)',
                 fontsize=14, fontweight='bold')

    plot_vars = [
        ('flower_mean_vpd', 'Flowering Mean VPD (kPa)'),
        ('flower_max_vpd', 'Flowering Max VPD (kPa)'),
        ('flower_cum_precip', 'Flowering Cumulative Precip (mm)'),
        ('fruit_set_mean_vpd', 'Fruit Set Mean VPD (kPa)'),
        ('fruit_set_max_vpd', 'Fruit Set Max VPD (kPa)'),
        ('fruit_set_cum_precip', 'Fruit Set Cumulative Precip (mm)'),
    ]

    for ax, (var, label) in zip(axes.flat, plot_vars):
        x = merged[f'{var}_centroid']
        y = merged[f'{var}_weighted']

        ax.scatter(x, y, alpha=0.6, s=30, color='steelblue')

        # 1:1 line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='1:1 line')

        # Correlation
        r = merged[[f'{var}_weighted', f'{var}_centroid']].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(f'Centroid-Based {label}', fontsize=9)
        ax.set_ylabel(f'Area-Weighted {label}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    out_fig = FIG_DIR / "climate_comparison_weighted_vs_centroid.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_fig}")

    plt.close()

    # Difference distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Climate Differences: Area-Weighted minus Centroid-Based',
                 fontsize=14, fontweight='bold')

    diff_vars = [
        ('flower_mean_vpd_diff', 'Flowering Mean VPD Diff (kPa)'),
        ('fruit_set_mean_vpd_diff', 'Fruit Set Mean VPD Diff (kPa)'),
        ('maturation_mean_vpd_diff', 'Maturation Mean VPD Diff (kPa)'),
    ]

    for ax, (var, label) in zip(axes, diff_vars):
        ax.hist(merged[var], bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.axvline(merged[var].mean(), color='red', linestyle='-', linewidth=2,
                   label=f'Mean = {merged[var].mean():.3f}')
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_fig2 = FIG_DIR / "climate_differences_distribution.png"
    plt.savefig(out_fig2, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_fig2}")

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"Matched observations: {len(merged)}")
    print(f"Comarques: {merged['comarca'].nunique()}")
    print(f"\nKey findings:")
    print(f"  - Flowering VPD correlation: r = {merged[['flower_mean_vpd_weighted', 'flower_mean_vpd_centroid']].corr().iloc[0,1]:.4f}")
    print(f"  - Flowering VPD mean difference: {merged['flower_mean_vpd_diff'].mean():.3f} kPa")
    print(f"  - Flowering VPD std of differences: {merged['flower_mean_vpd_diff'].std():.3f} kPa")


if __name__ == "__main__":
    main()
