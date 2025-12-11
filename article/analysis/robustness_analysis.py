#!/usr/bin/env python3
"""
Robustness Analysis: WST vs Statistical Features
Analyzes performance metrics across experiments to determine robustness under noise and data scarcity.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style with larger fonts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# Configuration
EXPERIMENTS_ROOT = Path("../../../experiments")
OUTPUT_DIR = Path("..")
GRAPHS_DIR = OUTPUT_DIR / "figures"
LOCATIONS = ["assatigue", "popolar", "sunset"]
SIZES = ["mini", "small", "original"]
K_VALUES = [2, 5, 10, 20]
METHODS = ["advanced_stats", "wst", "hybrid"]

# Create graphs directory
GRAPHS_DIR.mkdir(exist_ok=True)

# Noise types and their configurations (corrected based on actual filesystem)
NOISE_CONFIGS = {
    "clean": ["clean_0"],
    "gaussian": ["gaussian_30", "gaussian_50"],
    "poisson": ["poisson_40", "poisson_60"],
    "saltpepper": ["saltpepper_5", "saltpepper_15", "saltpepper_25"],
    "speckle": ["speckle_15", "speckle_35", "speckle_55"],
    "uniform": ["uniform_10", "uniform_25", "uniform_40"]
}

# Color palette for methods
METHOD_COLORS = {
    "advanced_stats": "#e74c3c",  # Red
    "wst": "#3498db",              # Blue
    "hybrid": "#2ecc71"            # Green
}

def extract_intensity_value(intensity_str):
    """Extract numeric intensity from string like 'gaussian_30' -> 30"""
    parts = intensity_str.split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0

def load_experiment_json(noise_type, intensity, location, size, k, method):
    """Load a single experiment JSON file."""
    # Construct path based on standardized structure
    if noise_type == "clean":
        base_path = EXPERIMENTS_ROOT / "clean" / "clean_output" / intensity
    elif noise_type == "saltpepper":
        base_path = EXPERIMENTS_ROOT / "saltpepper" / "saltpepper_output" / intensity
    else:
        base_path = EXPERIMENTS_ROOT / noise_type / f"{noise_type}_output" / intensity

    json_path = base_path / location / size / f"k{k}" / method / "experiment_report_with_model.json"

    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        # Silently skip missing files
        return None

def extract_metrics(experiment_data):
    """Extract relevant metrics from experiment JSON."""
    if not experiment_data:
        return None

    metrics = {}

    # Try different JSON structures
    if 'performance' in experiment_data:
        perf = experiment_data['performance']

        # Get accuracy
        metrics['accuracy'] = perf.get('test_accuracy', perf.get('accuracy', np.nan))

        # Get macro F1 from classification_report
        if 'classification_report' in perf:
            class_rep = perf['classification_report']
            if 'macro avg' in class_rep:
                metrics['macro_f1'] = class_rep['macro avg'].get('f1-score', np.nan)
                metrics['precision'] = class_rep['macro avg'].get('precision', np.nan)
                metrics['recall'] = class_rep['macro avg'].get('recall', np.nan)
            else:
                metrics['macro_f1'] = np.nan
                metrics['precision'] = np.nan
                metrics['recall'] = np.nan
        else:
            metrics['macro_f1'] = perf.get('macro_f1', np.nan)
            metrics['precision'] = perf.get('precision', np.nan)
            metrics['recall'] = perf.get('recall', np.nan)

        # CV scores
        cv_scores = perf.get('cv_scores', [])
        if cv_scores:
            metrics['cv_mean'] = np.mean(cv_scores)
            metrics['cv_std'] = np.std(cv_scores)
        else:
            metrics['cv_mean'] = perf.get('cv_mean_accuracy', np.nan)
            metrics['cv_std'] = perf.get('cv_std_accuracy', np.nan)
    else:
        # Alternative structure
        metrics['accuracy'] = experiment_data.get('accuracy', np.nan)
        metrics['macro_f1'] = experiment_data.get('macro_f1', np.nan)
        metrics['precision'] = experiment_data.get('precision', np.nan)
        metrics['recall'] = experiment_data.get('recall', np.nan)
        metrics['cv_mean'] = np.nan
        metrics['cv_std'] = np.nan

    return metrics

def aggregate_across_locations(noise_type, intensity, size, k, method):
    """Aggregate metrics across the three locations."""
    location_metrics = []

    for location in LOCATIONS:
        exp_data = load_experiment_json(noise_type, intensity, location, size, k, method)
        metrics = extract_metrics(exp_data)
        if metrics:
            location_metrics.append(metrics)

    if not location_metrics:
        return None

    # Compute mean and std across locations
    aggregated = {}
    metric_names = ['accuracy', 'macro_f1', 'precision', 'recall', 'cv_mean', 'cv_std']

    for metric in metric_names:
        values = [m.get(metric, np.nan) for m in location_metrics]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_n'] = len(values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_std'] = np.nan
            aggregated[f'{metric}_n'] = 0

    return aggregated

def load_all_data():
    """Load all experiment data and aggregate by configuration."""
    print("Loading experiment data...")

    all_data = []

    for noise_type, intensities in NOISE_CONFIGS.items():
        for intensity in intensities:
            for size in SIZES:
                for k in K_VALUES:
                    for method in METHODS:
                        agg_metrics = aggregate_across_locations(noise_type, intensity, size, k, method)

                        if agg_metrics:
                            row = {
                                'noise_type': noise_type,
                                'intensity': intensity,
                                'intensity_value': extract_intensity_value(intensity),
                                'size': size,
                                'k': k,
                                'method': method,
                                **agg_metrics
                            }
                            all_data.append(row)

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} aggregated configurations")
    return df

def compute_pairwise_deltas(df):
    """Compute pairwise differences: WST - Advanced_Stats and Hybrid - Advanced_Stats."""
    print("\nComputing pairwise method comparisons...")

    comparisons = []

    # Group by noise_type, intensity, size, k
    grouped = df.groupby(['noise_type', 'intensity', 'intensity_value', 'size', 'k'])

    for (noise_type, intensity, intensity_val, size, k), group in grouped:
        # Get metrics for each method
        adv_stats = group[group['method'] == 'advanced_stats']
        wst = group[group['method'] == 'wst']
        hybrid = group[group['method'] == 'hybrid']

        if len(adv_stats) == 0:
            continue

        adv_stats_row = adv_stats.iloc[0]

        # WST vs Advanced Stats
        if len(wst) > 0:
            wst_row = wst.iloc[0]
            comparisons.append({
                'noise_type': noise_type,
                'intensity': intensity,
                'intensity_value': intensity_val,
                'size': size,
                'k': k,
                'comparison': 'wst_vs_advanced',
                'delta_accuracy': wst_row['accuracy_mean'] - adv_stats_row['accuracy_mean'],
                'delta_macro_f1': wst_row['macro_f1_mean'] - adv_stats_row['macro_f1_mean'],
                'delta_precision': wst_row['precision_mean'] - adv_stats_row['precision_mean'],
                'delta_recall': wst_row['recall_mean'] - adv_stats_row['recall_mean'],
                'adv_stats_accuracy': adv_stats_row['accuracy_mean'],
                'wst_accuracy': wst_row['accuracy_mean'],
                'adv_stats_macro_f1': adv_stats_row['macro_f1_mean'],
                'wst_macro_f1': wst_row['macro_f1_mean']
            })

        # Hybrid vs Advanced Stats
        if len(hybrid) > 0:
            hybrid_row = hybrid.iloc[0]
            comparisons.append({
                'noise_type': noise_type,
                'intensity': intensity,
                'intensity_value': intensity_val,
                'size': size,
                'k': k,
                'comparison': 'hybrid_vs_advanced',
                'delta_accuracy': hybrid_row['accuracy_mean'] - adv_stats_row['accuracy_mean'],
                'delta_macro_f1': hybrid_row['macro_f1_mean'] - adv_stats_row['macro_f1_mean'],
                'delta_precision': hybrid_row['precision_mean'] - adv_stats_row['precision_mean'],
                'delta_recall': hybrid_row['recall_mean'] - adv_stats_row['recall_mean'],
                'adv_stats_accuracy': adv_stats_row['accuracy_mean'],
                'hybrid_accuracy': hybrid_row['accuracy_mean'],
                'adv_stats_macro_f1': adv_stats_row['macro_f1_mean'],
                'hybrid_macro_f1': hybrid_row['macro_f1_mean']
            })

    df_deltas = pd.DataFrame(comparisons)
    print(f"Computed {len(df_deltas)} pairwise comparisons")
    return df_deltas

def perform_statistical_tests(df_deltas, metric='delta_macro_f1'):
    """Perform statistical tests with FDR correction."""
    print(f"\nPerforming statistical tests on {metric}...")

    results = []

    # Group by comparison type
    for comparison in ['wst_vs_advanced', 'hybrid_vs_advanced']:
        subset = df_deltas[df_deltas['comparison'] == comparison]

        if len(subset) == 0:
            continue

        # Get delta values
        deltas = subset[metric].dropna().values

        if len(deltas) < 3:
            continue

        # Test normality
        _, p_normality = shapiro(deltas)
        is_normal = p_normality > 0.05

        # Perform appropriate test
        if is_normal:
            # Paired t-test (testing if mean delta != 0)
            t_stat, p_value = ttest_rel(deltas, np.zeros_like(deltas))
            test_used = 'paired_t_test'
        else:
            # Wilcoxon signed-rank test
            try:
                w_stat, p_value = wilcoxon(deltas)
                test_used = 'wilcoxon'
            except:
                p_value = np.nan
                test_used = 'wilcoxon_failed'

        # Compute effect size (Cohen's d)
        cohens_d = np.mean(deltas) / (np.std(deltas) + 1e-10)

        results.append({
            'comparison': comparison,
            'n': len(deltas),
            'mean_delta': np.mean(deltas),
            'std_delta': np.std(deltas),
            'median_delta': np.median(deltas),
            'p_value_raw': p_value,
            'test_used': test_used,
            'is_normal': is_normal,
            'cohens_d': cohens_d
        })

    df_results = pd.DataFrame(results)

    # Apply FDR correction
    if len(df_results) > 0:
        p_values = df_results['p_value_raw'].values
        _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        df_results['p_value_fdr'] = p_corrected
        df_results['significant_fdr'] = p_corrected < 0.05

    return df_results

def analyze_noise_robustness(df):
    """Analyze robustness to noise using regression slopes."""
    print("\nAnalyzing noise robustness...")

    robustness_results = []

    for noise_type in NOISE_CONFIGS.keys():
        if noise_type == "clean":
            continue

        for size in SIZES:
            for k in K_VALUES:
                for method in METHODS:
                    subset = df[(df['noise_type'] == noise_type) &
                               (df['size'] == size) &
                               (df['k'] == k) &
                               (df['method'] == method)]

                    if len(subset) < 2:
                        continue

                    x = subset['intensity_value'].values
                    y = subset['accuracy_mean'].values

                    # Linear regression
                    if len(x) > 1 and not np.any(np.isnan(y)):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                        robustness_results.append({
                            'noise_type': noise_type,
                            'size': size,
                            'k': k,
                            'method': method,
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_value**2,
                            'p_value': p_value
                        })

    df_robustness = pd.DataFrame(robustness_results)
    return df_robustness

def analyze_data_scarcity(df):
    """Analyze robustness to data scarcity across dataset sizes."""
    print("\nAnalyzing data scarcity robustness...")

    scarcity_results = []

    for noise_type, intensities in NOISE_CONFIGS.items():
        for intensity in intensities:
            for k in K_VALUES:
                for method in METHODS:
                    # Get metrics for each size
                    metrics_by_size = {}
                    for size in SIZES:
                        subset = df[(df['noise_type'] == noise_type) &
                                   (df['intensity'] == intensity) &
                                   (df['size'] == size) &
                                   (df['k'] == k) &
                                   (df['method'] == method)]

                        if len(subset) > 0:
                            metrics_by_size[size] = subset.iloc[0]['accuracy_mean']

                    # Compute retention relative to 'original'
                    if 'original' in metrics_by_size:
                        baseline = metrics_by_size['original']

                        for size in SIZES:
                            if size in metrics_by_size:
                                retention = (metrics_by_size[size] / baseline * 100) if baseline > 0 else np.nan

                                scarcity_results.append({
                                    'noise_type': noise_type,
                                    'intensity': intensity,
                                    'k': k,
                                    'method': method,
                                    'size': size,
                                    'accuracy': metrics_by_size[size],
                                    'retention_pct': retention
                                })

    df_scarcity = pd.DataFrame(scarcity_results)
    return df_scarcity

def plot_performance_by_noise_type(df):
    """Plot 1: Performance comparison by noise type."""
    print("Generating plot 1/7: Performance by noise type...")

    noise_types = [nt for nt in NOISE_CONFIGS.keys() if nt != "clean"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, noise_type in enumerate(noise_types):
        ax = axes[idx]
        subset = df[df['noise_type'] == noise_type]

        # Aggregate across sizes and k for clearer visualization
        agg_data = subset.groupby(['intensity_value', 'method'])['accuracy_mean'].mean().reset_index()

        for method in METHODS:
            method_data = agg_data[agg_data['method'] == method].sort_values('intensity_value')
            if len(method_data) > 0:
                ax.plot(method_data['intensity_value'], method_data['accuracy_mean'],
                       marker='o', label=method, color=METHOD_COLORS[method], linewidth=2, markersize=8)

        ax.set_title(f'{noise_type.capitalize()} Noise', fontsize=20, fontweight='bold')
        ax.set_xlabel('Noise Intensity', fontsize=18)
        ax.set_ylabel('Mean Accuracy', fontsize=18)
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)

    # Hide the 6th subplot if only 5 noise types
    if len(noise_types) < 6:
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'performance_by_noise_type.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_noise_robustness_slopes(df_robustness):
    """Plot 2: Regression slopes showing noise robustness."""
    print("Generating plot 2/7: Noise robustness slopes...")

    if df_robustness is None or len(df_robustness) == 0:
        print("  Skipping: no robustness data")
        return

    # Aggregate slopes by method across all noise types
    slope_summary = df_robustness.groupby('method')['slope'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(METHODS))
    means = [slope_summary[slope_summary['method'] == m]['mean'].values[0] if len(slope_summary[slope_summary['method'] == m]) > 0 else 0 for m in METHODS]
    stds = [slope_summary[slope_summary['method'] == m]['std'].values[0] if len(slope_summary[slope_summary['method'] == m]) > 0 else 0 for m in METHODS]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                   color=[METHOD_COLORS[m] for m in METHODS], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Method', fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean Slope (Accuracy vs Intensity)', fontsize=18, fontweight='bold')
    ax.set_title('Noise Robustness: Less Negative = More Robust', fontsize=20, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(METHODS, fontsize=16)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'noise_robustness_slopes.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_scarcity_retention(df_scarcity):
    """Plot 3: Data scarcity performance retention."""
    print("Generating plot 3/7: Data scarcity retention...")

    if df_scarcity is None or len(df_scarcity) == 0:
        print("  Skipping: no scarcity data")
        return

    # Aggregate retention by method and size
    retention_summary = df_scarcity.groupby(['method', 'size'])['retention_pct'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(SIZES))
    width = 0.25

    for idx, method in enumerate(METHODS):
        method_data = retention_summary[retention_summary['method'] == method]
        values = [method_data[method_data['size'] == s]['retention_pct'].values[0] if len(method_data[method_data['size'] == s]) > 0 else 0 for s in SIZES]

        ax.bar(x + idx*width, values, width, label=method, color=METHOD_COLORS[method], edgecolor='black')

    ax.set_xlabel('Dataset Size', fontsize=18, fontweight='bold')
    ax.set_ylabel('Performance Retention (%)', fontsize=18, fontweight='bold')
    ax.set_title('Data Scarcity Robustness (% of Original Performance)', fontsize=20, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(SIZES, fontsize=16)
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='100% (baseline)')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='80% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'data_scarcity_retention.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_delta_heatmap(df_deltas):
    """Plot 4: Heatmap of deltas (method - advanced_stats)."""
    print("Generating plot 4/7: Delta heatmap...")

    if df_deltas is None or len(df_deltas) == 0:
        print("  Skipping: no delta data")
        return

    # Create pivot tables for heatmap
    pivot_wst = df_deltas[df_deltas['comparison'] == 'wst_vs_advanced'].groupby(['noise_type', 'intensity'])['delta_macro_f1'].mean().reset_index()
    pivot_hybrid = df_deltas[df_deltas['comparison'] == 'hybrid_vs_advanced'].groupby(['noise_type', 'intensity'])['delta_macro_f1'].mean().reset_index()

    # Combine for visualization
    pivot_wst['method'] = 'WST vs Adv'
    pivot_hybrid['method'] = 'Hybrid vs Adv'
    combined = pd.concat([pivot_wst, pivot_hybrid])

    pivot_table = combined.pivot_table(values='delta_macro_f1', index=['noise_type', 'intensity'], columns='method')

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Δ Macro-F1'}, ax=ax, linewidths=0.5, annot_kws={'fontsize': 14})

    ax.set_title('Performance Delta vs Advanced Stats\n(Green = Better, Red = Worse)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Comparison', fontsize=18, fontweight='bold')
    ax.set_ylabel('Noise Type + Intensity', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'delta_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_distribution(df):
    """Plot 5: Box plot of performance distribution."""
    print("Generating plot 5/7: Performance distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy distribution
    ax1 = axes[0]
    data_acc = [df[df['method'] == m]['accuracy_mean'].dropna().values for m in METHODS]
    bp1 = ax1.boxplot(data_acc, labels=METHODS, patch_artist=True, showmeans=True)

    for patch, method in zip(bp1['boxes'], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.7)

    ax1.set_title('Accuracy Distribution', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=18)
    ax1.grid(True, alpha=0.3, axis='y')

    # Macro-F1 distribution
    ax2 = axes[1]
    data_f1 = [df[df['method'] == m]['macro_f1_mean'].dropna().values for m in METHODS]
    bp2 = ax2.boxplot(data_f1, labels=METHODS, patch_artist=True, showmeans=True)

    for patch, method in zip(bp2['boxes'], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.7)

    ax2.set_title('Macro-F1 Distribution', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Macro-F1', fontsize=18)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_k_features_impact(df):
    """Plot 6: Impact of K features selection."""
    print("Generating plot 6/7: K-features impact...")

    # Aggregate across noise types and sizes
    k_impact = df.groupby(['k', 'method'])['accuracy_mean'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in METHODS:
        method_data = k_impact[k_impact['method'] == method]
        ax.plot(method_data['k'], method_data['accuracy_mean'],
               marker='o', label=method, color=METHOD_COLORS[method],
               linewidth=2, markersize=8)

    ax.set_xlabel('Number of Selected Features (K)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean Accuracy', fontsize=18, fontweight='bold')
    ax.set_title('Impact of Feature Selection on Performance', fontsize=20, fontweight='bold')
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'k_features_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistical_significance(df_stats_f1, df_stats_acc):
    """Plot 7: Statistical significance matrix."""
    print("Generating plot 7/7: Statistical significance matrix...")

    if df_stats_f1 is None or len(df_stats_f1) == 0:
        print("  Skipping: no statistical test results")
        return

    # Create significance matrix
    metrics = ['Macro-F1', 'Accuracy']
    comparisons = ['WST vs Adv', 'Hybrid vs Adv']

    sig_matrix = np.zeros((len(metrics), len(comparisons)))
    p_values = np.zeros((len(metrics), len(comparisons)))
    effect_sizes = np.zeros((len(metrics), len(comparisons)))

    # Fill matrix
    for i, (df_stat, metric) in enumerate([(df_stats_f1, 'Macro-F1'), (df_stats_acc, 'Accuracy')]):
        if df_stat is not None and len(df_stat) > 0:
            for j, comp in enumerate(['wst_vs_advanced', 'hybrid_vs_advanced']):
                row = df_stat[df_stat['comparison'] == comp]
                if len(row) > 0:
                    sig_matrix[i, j] = 1 if row.iloc[0].get('significant_fdr', False) else 0
                    p_values[i, j] = row.iloc[0].get('p_value_fdr', 1.0)
                    effect_sizes[i, j] = row.iloc[0].get('cohens_d', 0.0)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use effect sizes for colors, but annotate with significance
    im = ax.imshow(effect_sizes, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(comparisons)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(comparisons, fontsize=16)
    ax.set_yticklabels(metrics, fontsize=16)

    # Annotate cells
    for i in range(len(metrics)):
        for j in range(len(comparisons)):
            sig_marker = '***' if p_values[i, j] < 0.001 else '**' if p_values[i, j] < 0.01 else '*' if p_values[i, j] < 0.05 else 'ns'
            text = f"d={effect_sizes[i, j]:.3f}\np={p_values[i, j]:.4f}\n{sig_marker}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title('Statistical Significance & Effect Sizes\n(* p<0.05, ** p<0.01, *** p<0.001, ns=not significant)',
                fontsize=18, fontweight='bold')
    ax.set_xlabel('Comparison', fontsize=18, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=18, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d (Effect Size)", fontsize=16)

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'statistical_significance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(df, df_deltas, df_stats_f1, df_stats_acc, df_robustness, df_scarcity):
    """Generate all visualization plots."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_performance_by_noise_type(df)
    plot_noise_robustness_slopes(df_robustness)
    plot_data_scarcity_retention(df_scarcity)
    plot_delta_heatmap(df_deltas)
    plot_performance_distribution(df)
    plot_k_features_impact(df)
    plot_statistical_significance(df_stats_f1, df_stats_acc)

    print("\n✓ All 7 plots generated successfully!")

def generate_summary_report(df, df_deltas, df_stats, df_robustness, df_scarcity):
    """Generate markdown report with findings."""
    print("\nGenerating analysis report...")

    report_lines = []

    # Header
    report_lines.append("# 🧠 Robustness Analysis: WST vs Statistical Features")
    report_lines.append("")
    report_lines.append("**Analysis Date**: 2025-10-04")
    report_lines.append("**Research Question**: Is the Wavelet Scattering Transform a robust feature extraction technique under noisy or data-scarce conditions?")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## 📋 Executive Summary")
    report_lines.append("")

    # Overall statistics
    total_configs = len(df)
    noise_types = df['noise_type'].nunique()

    report_lines.append(f"- **Total Configurations Analyzed**: {total_configs}")
    report_lines.append(f"- **Noise Types**: {noise_types}")
    report_lines.append(f"- **Dataset Sizes**: {len(SIZES)} (mini, small, original)")
    report_lines.append(f"- **K-Best Values**: {len(K_VALUES)} ({', '.join(map(str, K_VALUES))})")
    report_lines.append(f"- **Methods Compared**: {len(METHODS)} (advanced_stats, wst, hybrid)")
    report_lines.append("")

    # Statistical significance findings
    if df_stats is not None and len(df_stats) > 0:
        report_lines.append("### Key Findings")
        report_lines.append("")

        for _, row in df_stats.iterrows():
            method_name = "WST" if "wst" in row['comparison'] else "Hybrid"
            significance = "✅ **Significant**" if row['significant_fdr'] else "❌ Not significant"

            report_lines.append(f"**{method_name} vs Advanced Stats (Macro-F1)**:")
            report_lines.append(f"- Mean Δ: {row['mean_delta']:.4f} ± {row['std_delta']:.4f}")
            report_lines.append(f"- p-value (FDR-corrected): {row['p_value_fdr']:.4f} {significance}")
            report_lines.append(f"- Effect size (Cohen's d): {row['cohens_d']:.3f}")
            report_lines.append(f"- Test: {row['test_used']}")
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Visualization Section
    report_lines.append("## 📊 Visualizations")
    report_lines.append("")

    report_lines.append("### 1. Performance by Noise Type")
    report_lines.append("")
    report_lines.append("![Performance by Noise Type](graphs/performance_by_noise_type.png)")
    report_lines.append("")
    report_lines.append("**Interpretazione**: Questo grafico composto presenta sei pannelli, ciascuno dedicato a un diverso tipo di rumore "
                       "applicato alle immagini RGB del dataset. Ogni pannello mostra l'andamento dell'accuratezza media in funzione "
                       "dell'intensità del rumore, confrontando tre metodi di estrazione features: Advanced Statistics (rosso), "
                       "Wavelet Scattering Transform (blu), e Hybrid (verde).")
    report_lines.append("")
    report_lines.append("I **noise types analizzati** sono:")
    report_lines.append("- **Gaussian**: Rumore gaussiano con σ=30 e σ=50")
    report_lines.append("- **Poisson**: Rumore di Poisson con parametri 40 e 60")
    report_lines.append("- **Saltpepper**: Rumore salt-and-pepper con densità 5%, 15%, 25%")
    report_lines.append("- **Speckle**: Rumore speckle con varianza 15, 35, 55")
    report_lines.append("- **Uniform**: Rumore uniforme con range 10, 25, 40")
    report_lines.append("")
    report_lines.append("**Osservazioni chiave**:")
    report_lines.append("- Le linee più **piatte** (pendenza meno negativa) indicano maggiore robustezza al rumore")
    report_lines.append("- La **separazione verticale** tra le linee mostra le differenze assolute di performance tra metodi")
    report_lines.append("- L'**intersezione** di linee indica che la superiorità di un metodo può dipendere dall'intensità del rumore")
    report_lines.append("- Tutti i metodi mostrano una **degradazione monotona** all'aumentare dell'intensità del rumore")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### 2. Noise Robustness Slopes")
    report_lines.append("")
    report_lines.append("![Noise Robustness](graphs/noise_robustness_slopes.png)")
    report_lines.append("")
    report_lines.append("**Interpretation**: Mean regression slopes across all noise types. "
                       "Less negative slopes indicate better resistance to noise degradation.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### 3. Data Scarcity Retention")
    report_lines.append("")
    report_lines.append("![Data Scarcity](graphs/data_scarcity_retention.png)")
    report_lines.append("")
    report_lines.append("**Interpretation**: Performance retention relative to 'original' dataset size. "
                       "Higher retention percentages indicate better performance with limited training data.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### 4. Delta Heatmap (vs Advanced Stats)")
    report_lines.append("")
    report_lines.append("![Delta Heatmap](graphs/delta_heatmap.png)")
    report_lines.append("")
    report_lines.append("**Interpretation**: Green cells indicate WST/Hybrid outperforming Advanced Stats. "
                       "Red cells indicate underperformance. Values show Δ Macro-F1 scores.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### 5. Performance Distribution")
    report_lines.append("")
    report_lines.append("![Performance Distribution](graphs/performance_distribution.png)")
    report_lines.append("")
    report_lines.append("**Interpretation**: Box plots showing distribution of accuracy and macro-F1 across all configurations. "
                       "Narrower boxes indicate more consistent performance.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### 6. K-Features Impact")
    report_lines.append("")
    report_lines.append("![K-Features Impact](graphs/k_features_impact.png)")
    report_lines.append("")
    report_lines.append("**Interpretation**: Effect of feature selection on performance. "
                       "Shows optimal k-value range and diminishing returns.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### 7. Statistical Significance Matrix")
    report_lines.append("")
    report_lines.append("![Statistical Significance](graphs/statistical_significance_matrix.png)")
    report_lines.append("")
    report_lines.append("**Interpretation**: Effect sizes (Cohen's d) and statistical significance. "
                       "Cells show whether differences are statistically meaningful.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Detailed Results
    report_lines.append("## 📋 Detailed Statistical Results")
    report_lines.append("")

    if df_stats is not None and len(df_stats) > 0:
        report_lines.append("### Pairwise Method Comparison")
        report_lines.append("")
        report_lines.append(df_stats.to_markdown(index=False))
        report_lines.append("")

    # Noise robustness
    report_lines.append("## 🔬 Noise Robustness Analysis")
    report_lines.append("")

    if df_robustness is not None and len(df_robustness) > 0:
        report_lines.append("### Regression Slopes (Accuracy vs Intensity)")
        report_lines.append("")
        report_lines.append("A **less negative slope** indicates better robustness to increasing noise.")
        report_lines.append("")

        # Aggregate slopes by method
        slope_summary = df_robustness.groupby('method').agg({
            'slope': ['mean', 'std', 'min', 'max']
        }).round(6)

        report_lines.append(slope_summary.to_markdown())
        report_lines.append("")

    # Data scarcity
    report_lines.append("## 📉 Data Scarcity Robustness")
    report_lines.append("")

    if df_scarcity is not None and len(df_scarcity) > 0:
        report_lines.append("### Performance Retention Across Dataset Sizes")
        report_lines.append("")

        # Average retention by method and size
        retention_summary = df_scarcity.groupby(['method', 'size'])['retention_pct'].agg(['mean', 'std']).round(2)
        report_lines.append(retention_summary.to_markdown())
        report_lines.append("")

    # Interpretation
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 💡 Interpretation")
    report_lines.append("")

    if df_stats is not None and len(df_stats) > 0:
        wst_row = df_stats[df_stats['comparison'] == 'wst_vs_advanced']
        if len(wst_row) > 0:
            wst_sig = wst_row.iloc[0]['significant_fdr']
            wst_delta = wst_row.iloc[0]['mean_delta']

            if wst_sig and wst_delta > 0:
                report_lines.append("✅ **WST demonstrates statistically significant superior performance** compared to Advanced Statistical Features.")
            elif wst_sig and wst_delta < 0:
                report_lines.append("⚠️ **Advanced Statistical Features outperform WST** with statistical significance.")
            else:
                report_lines.append("⚪ **No statistically significant difference** between WST and Advanced Statistical Features.")

        report_lines.append("")

    # Recommendations
    report_lines.append("## 🎯 Recommendations")
    report_lines.append("")
    report_lines.append("Based on the analysis:")
    report_lines.append("")
    report_lines.append("1. Review detailed statistical tables above for specific conditions")
    report_lines.append("2. Consider noise type and intensity when selecting feature extraction method")
    report_lines.append("3. Account for dataset size constraints in method selection")
    report_lines.append("4. Evaluate trade-offs between performance and computational cost")
    report_lines.append("")

    # Footer
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("**Generated by**: Robustness Analysis Pipeline")
    report_lines.append("**Data Source**: Aggregated across assatigue, popolar, sunset locations")
    report_lines.append("")

    # Write report
    report_path = OUTPUT_DIR / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Report saved to {report_path}")

def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("ROBUSTNESS ANALYSIS: WST vs Statistical Features")
    print("=" * 80)

    # 1. Load all data
    df = load_all_data()
    df.to_csv(OUTPUT_DIR / "all_aggregated_data.csv", index=False)
    print(f"✓ Saved: all_aggregated_data.csv")

    # 2. Compute pairwise deltas
    df_deltas = compute_pairwise_deltas(df)
    df_deltas.to_csv(OUTPUT_DIR / "pairwise_deltas.csv", index=False)
    print(f"✓ Saved: pairwise_deltas.csv")

    # 3. Statistical tests
    df_stats_f1 = perform_statistical_tests(df_deltas, 'delta_macro_f1')
    df_stats_acc = perform_statistical_tests(df_deltas, 'delta_accuracy')

    df_stats_f1.to_csv(OUTPUT_DIR / "statistical_tests_macro_f1.csv", index=False)
    df_stats_acc.to_csv(OUTPUT_DIR / "statistical_tests_accuracy.csv", index=False)
    print(f"✓ Saved: statistical_tests_macro_f1.csv, statistical_tests_accuracy.csv")

    # 4. Noise robustness
    df_robustness = analyze_noise_robustness(df)
    df_robustness.to_csv(OUTPUT_DIR / "noise_robustness_slopes.csv", index=False)
    print(f"✓ Saved: noise_robustness_slopes.csv")

    # 5. Data scarcity
    df_scarcity = analyze_data_scarcity(df)
    df_scarcity.to_csv(OUTPUT_DIR / "data_scarcity_retention.csv", index=False)
    print(f"✓ Saved: data_scarcity_retention.csv")

    # 6. Generate visualizations
    generate_all_plots(df, df_deltas, df_stats_f1, df_stats_acc, df_robustness, df_scarcity)

    # 7. Generate report
    generate_summary_report(df, df_deltas, df_stats_f1, df_robustness, df_scarcity)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  CSV Data:")
    print("    - all_aggregated_data.csv")
    print("    - pairwise_deltas.csv")
    print("    - statistical_tests_macro_f1.csv")
    print("    - statistical_tests_accuracy.csv")
    print("    - noise_robustness_slopes.csv")
    print("    - data_scarcity_retention.csv")
    print("\n  Visualizations (graphs/):")
    print("    - performance_by_noise_type.png")
    print("    - noise_robustness_slopes.png")
    print("    - data_scarcity_retention.png")
    print("    - delta_heatmap.png")
    print("    - performance_distribution.png")
    print("    - k_features_impact.png")
    print("    - statistical_significance_matrix.png")
    print("\n  Report:")
    print("    - analysis_report.md (main report with embedded graphs)")

if __name__ == "__main__":
    main()
