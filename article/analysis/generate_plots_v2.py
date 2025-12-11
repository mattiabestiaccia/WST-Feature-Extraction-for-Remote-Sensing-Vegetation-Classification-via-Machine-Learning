#!/usr/bin/env python3
"""
Visualization Script for Robustness Analysis v2.0

Generates publication-ready figures from robustness_analysis_v2.py outputs.

Usage:
    python generate_plots_v2.py --data-dir ../output/data --output-dir ../output/graphs

Figures Generated:
    1. Statistical tests summary (bar plot with error bars)
    2. Noise robustness slopes (grouped bar chart)
    3. Method × Noise interaction (heatmap)
    4. Data scarcity retention (line plot)
    5. Pairwise delta distributions (violin plots)
    6. Combined summary figure (multi-panel)

Version: 2.0.0
Author: Mattia Bestiaccia
Date: 2025-10-10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Color palette
COLORS = {
    'advanced_stats': '#E74C3C',  # Red
    'wst': '#3498DB',             # Blue
    'hybrid': '#2ECC71'           # Green
}

METHOD_LABELS = {
    'advanced_stats': 'Advanced Stats',
    'wst': 'WST',
    'hybrid': 'Hybrid'
}


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] [%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations for robustness analysis v2.0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='../output/data',
        help='Directory containing CSV outputs from robustness_analysis_v2.py'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../output/graphs',
        help='Directory to save generated plots'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Output format for figures'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity'
    )

    return parser.parse_args()


def plot_statistical_tests(
    df_stats: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Figure 1: Statistical significance tests with effect sizes.

    Bar plot showing Cohen's d with bootstrap confidence intervals.
    """
    if logger:
        logger.info("Generating Figure 1: Statistical tests summary...")

    # Prepare data
    df_plot = df_stats.copy()
    df_plot['comparison_label'] = df_plot['comparison'].map({
        'wst_vs_advanced': 'WST vs\nAdvanced',
        'hybrid_vs_advanced': 'Hybrid vs\nAdvanced'
    })
    df_plot['metric_label'] = df_plot['metric'].map({
        'delta_macro_f1': 'Macro F1',
        'delta_accuracy': 'Accuracy'
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by comparison
    comparisons = df_plot['comparison_label'].unique()
    metrics = df_plot['metric_label'].unique()
    x = np.arange(len(comparisons))
    width = 0.35

    for i, metric in enumerate(metrics):
        data = df_plot[df_plot['metric_label'] == metric]

        y = data['cohens_d'].values
        yerr_lower = data['cohens_d'].values - data['cohens_d_ci_lower'].values
        yerr_upper = data['cohens_d_ci_upper'].values - data['cohens_d'].values
        yerr = np.array([yerr_lower, yerr_upper])

        # Determine colors based on significance
        colors = []
        for _, row in data.iterrows():
            if row['significant_fdr']:
                colors.append('#2ECC71')  # Green for significant
            elif row['significance_flag'] == 'marginal':
                colors.append('#F39C12')  # Orange for marginal
            else:
                colors.append('#95A5A6')  # Gray for ns

        ax.bar(
            x + i * width,
            y,
            width,
            yerr=yerr,
            label=metric,
            color=colors,
            alpha=0.8,
            capsize=5,
            edgecolor='black',
            linewidth=1.2
        )

    # Horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel('Comparison', fontweight='bold')
    ax.set_ylabel("Cohen's d [95% CI]", fontweight='bold')
    ax.set_title('Statistical Significance Tests: Effect Sizes with Confidence Intervals', fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(comparisons)
    ax.legend(title='Metric', loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Add effect size interpretation lines
    ax.axhspan(-0.2, 0.2, alpha=0.1, color='gray', label='Small effect')
    ax.axhspan(-0.5, -0.2, alpha=0.1, color='orange')
    ax.axhspan(0.2, 0.5, alpha=0.1, color='orange', label='Medium effect')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved: {output_path}")


def plot_noise_robustness(
    df_noise: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Figure 2: Noise robustness slopes by method and noise type.

    Grouped bar chart showing mean degradation slopes.
    """
    if logger:
        logger.info("Generating Figure 2: Noise robustness slopes...")

    # Aggregate by method and noise_type
    df_agg = df_noise.groupby(['noise_type', 'method'])['slope'].mean().reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    noise_types = sorted(df_agg['noise_type'].unique())
    methods = ['advanced_stats', 'wst', 'hybrid']
    x = np.arange(len(noise_types))
    width = 0.25

    for i, method in enumerate(methods):
        data = df_agg[df_agg['method'] == method]
        data = data.set_index('noise_type').reindex(noise_types)

        ax.bar(
            x + i * width,
            data['slope'],
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )

    # Horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel('Noise Type', fontweight='bold')
    ax.set_ylabel('Mean Degradation Slope (less negative = more robust)', fontweight='bold')
    ax.set_title('Noise Robustness: Performance Degradation by Method and Noise Type', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([nt.capitalize() for nt in noise_types])
    ax.legend(title='Method', loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved: {output_path}")


def plot_interaction_heatmap(
    df_interaction: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Figure 3: Method × Noise_Intensity interaction heatmap.

    Shows differential slope changes (Hybrid - Advanced) by noise type.
    """
    if logger:
        logger.info("Generating Figure 3: Interaction heatmap...")

    # Prepare data for heatmap
    df_plot = df_interaction.copy()
    df_plot['noise_type_label'] = df_plot['noise_type'].str.capitalize()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create matrix
    data_matrix = df_plot.pivot_table(
        index='noise_type_label',
        values='slope_diff',
        aggfunc='mean'
    )

    # Plot heatmap
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Δslope (Hybrid - Advanced)'},
        linewidths=1,
        linecolor='black',
        ax=ax
    )

    ax.set_xlabel('')
    ax.set_ylabel('Noise Type', fontweight='bold')
    ax.set_title('Method × Noise Interaction: Differential Robustness\n(Positive = Hybrid more robust)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved: {output_path}")


def plot_data_scarcity(
    df_scarcity: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Figure 4: Data scarcity robustness (retention scores).

    Line plot showing performance retention across dataset sizes.
    """
    if logger:
        logger.info("Generating Figure 4: Data scarcity retention...")

    # Aggregate by method and size
    df_agg = df_scarcity.groupby(['method', 'size'])['retention_pct'].agg(['mean', 'std']).reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    size_order = ['mini', 'small', 'original']
    methods = ['advanced_stats', 'wst', 'hybrid']

    for method in methods:
        data = df_agg[df_agg['method'] == method]
        data = data.set_index('size').reindex(size_order)

        ax.plot(
            size_order,
            data['mean'],
            marker='o',
            markersize=8,
            linewidth=2.5,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.9
        )

        # Add error bars
        ax.fill_between(
            range(len(size_order)),
            data['mean'] - data['std'],
            data['mean'] + data['std'],
            color=COLORS[method],
            alpha=0.2
        )

    # Formatting
    ax.set_xlabel('Dataset Size', fontweight='bold')
    ax.set_ylabel('Performance Retention (%)', fontweight='bold')
    ax.set_title('Data Scarcity Robustness: Performance Retention Across Dataset Sizes', fontweight='bold')
    ax.legend(title='Method', loc='lower right')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim([85, 102])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved: {output_path}")


def plot_delta_distributions(
    df_deltas: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Figure 5: Pairwise delta distributions (violin plots).

    Shows distribution of performance differences for each comparison.
    """
    if logger:
        logger.info("Generating Figure 5: Delta distributions...")

    # Prepare data
    df_plot = df_deltas[['comparison', 'delta_macro_f1', 'delta_accuracy']].copy()
    df_plot = df_plot.melt(
        id_vars='comparison',
        value_vars=['delta_macro_f1', 'delta_accuracy'],
        var_name='metric',
        value_name='delta'
    )

    df_plot['comparison_label'] = df_plot['comparison'].map({
        'wst_vs_advanced': 'WST vs Advanced',
        'hybrid_vs_advanced': 'Hybrid vs Advanced'
    })
    df_plot['metric_label'] = df_plot['metric'].map({
        'delta_macro_f1': 'Macro F1',
        'delta_accuracy': 'Accuracy'
    })

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, metric in enumerate(['Macro F1', 'Accuracy']):
        data = df_plot[df_plot['metric_label'] == metric]

        sns.violinplot(
            data=data,
            x='comparison_label',
            y='delta',
            palette=['#3498DB', '#2ECC71'],
            inner='box',
            ax=axes[i]
        )

        # Add horizontal line at y=0
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        axes[i].set_xlabel('Comparison', fontweight='bold')
        axes[i].set_ylabel(f'Δ {metric}', fontweight='bold')
        axes[i].set_title(f'Distribution of Performance Differences: {metric}', fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved: {output_path}")


def plot_combined_summary(
    df_stats: pd.DataFrame,
    df_noise: pd.DataFrame,
    df_scarcity: pd.DataFrame,
    output_path: Path,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Figure 6: Combined summary figure (4-panel publication figure).

    Multi-panel figure combining key findings:
    - Panel A: Statistical tests
    - Panel B: Noise robustness
    - Panel C: Data scarcity
    - Panel D: Overall performance by method
    """
    if logger:
        logger.info("Generating Figure 6: Combined summary figure...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Statistical tests (bar plot)
    ax1 = fig.add_subplot(gs[0, 0])
    df_f1 = df_stats[df_stats['metric'] == 'delta_macro_f1']
    comparisons = ['wst_vs_advanced', 'hybrid_vs_advanced']
    x = np.arange(len(comparisons))

    for i, comp in enumerate(comparisons):
        row = df_f1[df_f1['comparison'] == comp].iloc[0]
        color = '#F39C12' if row['significance_flag'] == 'marginal' else '#95A5A6'

        ax1.bar(
            i,
            row['cohens_d'],
            yerr=[[row['cohens_d'] - row['cohens_d_ci_lower']],
                  [row['cohens_d_ci_upper'] - row['cohens_d']]],
            color=color,
            alpha=0.8,
            capsize=5,
            edgecolor='black',
            linewidth=1.2
        )

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['WST vs\nAdvanced', 'Hybrid vs\nAdvanced'])
    ax1.set_ylabel("Cohen's d [95% CI]", fontweight='bold')
    ax1.set_title('A. Statistical Significance (Macro F1)', fontweight='bold', loc='left')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel B: Noise robustness (mean slopes)
    ax2 = fig.add_subplot(gs[0, 1])
    df_noise_agg = df_noise.groupby('method')['slope'].mean().reset_index()
    methods = ['advanced_stats', 'wst', 'hybrid']
    x = np.arange(len(methods))

    for i, method in enumerate(methods):
        slope = df_noise_agg[df_noise_agg['method'] == method]['slope'].values[0]
        ax2.bar(
            i,
            slope,
            color=COLORS[method],
            alpha=0.8,
            edgecolor='black',
            linewidth=1.2
        )

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax2.set_ylabel('Mean Degradation Slope', fontweight='bold')
    ax2.set_title('B. Noise Robustness (All Noise Types)', fontweight='bold', loc='left')
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel C: Data scarcity
    ax3 = fig.add_subplot(gs[1, 0])
    df_scarcity_agg = df_scarcity.groupby(['method', 'size'])['retention_pct'].mean().reset_index()
    size_order = ['mini', 'small', 'original']

    for method in methods:
        data = df_scarcity_agg[df_scarcity_agg['method'] == method]
        data = data.set_index('size').reindex(size_order)

        ax3.plot(
            size_order,
            data['retention_pct'],
            marker='o',
            markersize=8,
            linewidth=2.5,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.9
        )

    ax3.set_xlabel('Dataset Size', fontweight='bold')
    ax3.set_ylabel('Performance Retention (%)', fontweight='bold')
    ax3.set_title('C. Data Scarcity Robustness', fontweight='bold', loc='left')
    ax3.legend(title='Method', loc='lower right')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.set_ylim([85, 102])

    # Panel D: Overall method comparison table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Create summary table
    summary_data = []
    for method in methods:
        # Get mean accuracy from aggregated data (would need to load this)
        noise_slope = df_noise_agg[df_noise_agg['method'] == method]['slope'].values[0]
        scarcity = df_scarcity_agg[
            (df_scarcity_agg['method'] == method) &
            (df_scarcity_agg['size'] == 'mini')
        ]['retention_pct'].values[0]

        summary_data.append([
            METHOD_LABELS[method],
            f"{noise_slope:.4f}",
            f"{scarcity:.1f}%"
        ])

    table = ax4.table(
        cellText=summary_data,
        colLabels=['Method', 'Noise Slope', 'Mini Retention'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by method
    for i, method in enumerate(methods):
        table[(i+1, 0)].set_facecolor(COLORS[method])
        table[(i+1, 0)].set_alpha(0.3)

    ax4.set_title('D. Method Performance Summary', fontweight='bold', loc='left', pad=20)

    # Add overall title
    fig.suptitle('Robustness Analysis v2.0: Comprehensive Summary', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved: {output_path}")


def main():
    """Main execution pipeline."""
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    logger.info("=" * 80)
    logger.info("VISUALIZATION GENERATION - ROBUSTNESS ANALYSIS V2.0")
    logger.info("=" * 80)
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Output Format: {args.format}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)

    logger.info("Loading data files...")
    try:
        df_stats = pd.read_csv(data_dir / 'statistical_tests_summary.csv', comment='#')
        df_noise = pd.read_csv(data_dir / 'noise_slope_summary.csv', comment='#')
        df_interaction = pd.read_csv(data_dir / 'method_noise_interaction.csv', comment='#')
        df_scarcity = pd.read_csv(data_dir / 'data_scarcity_retention.csv', comment='#')
        df_deltas = pd.read_csv(data_dir / 'pairwise_deltas.csv', comment='#')
        logger.info("  All data files loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)

    # Generate plots
    logger.info("=" * 80)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 80)

    ext = args.format

    plot_statistical_tests(
        df_stats,
        output_dir / f'fig1_statistical_tests.{ext}',
        logger
    )

    plot_noise_robustness(
        df_noise,
        output_dir / f'fig2_noise_robustness.{ext}',
        logger
    )

    plot_interaction_heatmap(
        df_interaction,
        output_dir / f'fig3_interaction_heatmap.{ext}',
        logger
    )

    plot_data_scarcity(
        df_scarcity,
        output_dir / f'fig4_data_scarcity.{ext}',
        logger
    )

    plot_delta_distributions(
        df_deltas,
        output_dir / f'fig5_delta_distributions.{ext}',
        logger
    )

    plot_combined_summary(
        df_stats,
        df_noise,
        df_scarcity,
        output_dir / f'fig6_combined_summary.{ext}',
        logger
    )

    logger.info("=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"All figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
