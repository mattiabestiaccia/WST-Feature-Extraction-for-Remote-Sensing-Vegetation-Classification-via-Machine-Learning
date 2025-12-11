#!/usr/bin/env python3
"""
Comprehensive analysis of Random Forest experiments across all parameters.

This script analyzes trends in model performance metrics as parameters vary:
- Feature extraction method (advanced_stats, wst, hybrid)
- K-value (2, 5, 10, 20)
- Dataset size (mini, small, original)
- Noise type and intensity (clean, gaussian, poisson, s&p, speckle, uniform)

Averages across geographic areas (assatigue, popolar, sunset) to focus on
parameter effects rather than location-specific variations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

class ComprehensiveExperimentAnalyzer:
    """Analyzes all experiments and extracts parameter-wise trends."""

    def __init__(self, experiments_dir: str = "/home/brusc/Projects/random_forest/experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.results = []
        self.df = None

        # Define parameter spaces
        self.noise_types = ['rgb_clean', 'gaussian', 'poisson', 's&p', 'speckle', 'uniform']
        self.areas = ['assatigue', 'popolar', 'sunset']
        self.sizes = ['mini', 'small', 'original']
        self.k_values = [2, 5, 10, 20]
        self.methods = ['advanced_stats', 'wst', 'hybrid']

        # Noise intensity mapping
        self.noise_labels = {
            'rgb_clean': 'Clean',
            'gaussian': 'Gaussian',
            'poisson': 'Poisson',
            's&p': 'Salt & Pepper',
            'speckle': 'Speckle',
            'uniform': 'Uniform'
        }

    def load_all_experiments(self) -> pd.DataFrame:
        """Load all experiment JSON files into a structured DataFrame."""
        print("🔍 Scanning for experiment results...")

        # Find all JSON files recursively
        json_files = list(self.experiments_dir.rglob("experiment_report*.json"))
        print(f"  Found {len(json_files)} JSON files total")

        for json_file in json_files:
            try:
                # Parse path to extract parameters
                parts = json_file.parts
                exp_idx = parts.index('experiments')

                # Skip if not enough path components
                if len(parts) < exp_idx + 6:
                    continue

                # Extract noise type from various possible locations
                noise_type = None
                for part in parts[exp_idx+1:]:
                    if 'rgb_clean' in part:
                        noise_type = 'rgb_clean'
                        break
                    elif 'gaussian' in part:
                        noise_type = 'gaussian'
                        break
                    elif 'poisson' in part:
                        noise_type = 'poisson'
                        break
                    elif 's&p' in part or 'salt' in part:
                        noise_type = 's&p'
                        break
                    elif 'speckle' in part:
                        noise_type = 'speckle'
                        break
                    elif 'uniform' in part:
                        noise_type = 'uniform'
                        break

                if not noise_type:
                    continue

                # Find area, size, k_value, method in path
                area = None
                size = None
                k_value = None
                method = None

                for part in parts:
                    if part in self.areas:
                        area = part
                    elif part in self.sizes:
                        size = part
                    elif part.startswith('k') and part[1:].isdigit():
                        k_value = int(part[1:])
                    elif part in self.methods:
                        method = part

                # Skip if missing required parameters
                if not all([area, size, k_value, method]):
                    continue

                # Load JSON data
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extract metrics
                result = {
                    'noise_type': noise_type,
                    'area': area,
                    'size': size,
                    'k_value': k_value,
                    'method': method,
                    'test_accuracy': data['performance']['test_accuracy'],
                    'cv_mean_accuracy': data['performance']['cv_mean_accuracy'],
                    'cv_std_accuracy': data['performance']['cv_std_accuracy'],
                    'n_samples': data['dataset_info']['total_images'],
                    'n_features': data['dataset_info']['total_features_available'],
                    'file_path': str(json_file)
                }

                # Extract per-class metrics if available
                if 'classification_report' in data['performance']:
                    for class_name in ['low_veg', 'trees', 'water']:
                        if class_name in data['performance']['classification_report']:
                            result[f'{class_name}_precision'] = data['performance']['classification_report'][class_name]['precision']
                            result[f'{class_name}_recall'] = data['performance']['classification_report'][class_name]['recall']
                            result[f'{class_name}_f1'] = data['performance']['classification_report'][class_name]['f1-score']

                self.results.append(result)

            except Exception as e:
                print(f"    ⚠️  Error loading {json_file.name}: {e}")
                continue

        self.df = pd.DataFrame(self.results)
        print(f"\n✅ Loaded {len(self.df)} experiments")
        print(f"   Noise types: {self.df['noise_type'].unique()}")
        print(f"   Methods: {self.df['method'].unique()}")
        print(f"   Sizes: {self.df['size'].unique()}")
        print(f"   K-values: {sorted(self.df['k_value'].unique())}")

        return self.df

    def aggregate_by_parameters(self) -> pd.DataFrame:
        """
        Aggregate results across geographic areas to focus on parameter effects.
        Returns DataFrame with mean and std across areas for each parameter combination.
        """
        print("\n📊 Aggregating across geographic areas...")

        groupby_cols = ['noise_type', 'size', 'k_value', 'method']

        agg_funcs = {
            'test_accuracy': ['mean', 'std', 'count'],
            'cv_mean_accuracy': ['mean', 'std'],
            'cv_std_accuracy': 'mean'
        }

        aggregated = self.df.groupby(groupby_cols).agg(agg_funcs).reset_index()

        # Flatten column names
        aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                             for col in aggregated.columns.values]

        print(f"✅ Aggregated to {len(aggregated)} unique parameter combinations")

        return aggregated

    def analyze_method_trends(self, df_agg: pd.DataFrame):
        """Analyze performance trends by feature extraction method."""
        print("\n🔬 Analyzing Feature Extraction Method Trends...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Extraction Method Performance Comparison', fontsize=14, fontweight='bold')

        # 1. Overall accuracy by method
        ax = axes[0, 0]
        method_perf = df_agg.groupby('method')['test_accuracy_mean'].agg(['mean', 'std'])
        method_perf = method_perf.sort_values('mean', ascending=False)

        x = np.arange(len(method_perf))
        ax.bar(x, method_perf['mean'], yerr=method_perf['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(method_perf.index, rotation=0)
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Overall Performance by Method')
        ax.set_ylim([0.5, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (mean, std) in enumerate(zip(method_perf['mean'], method_perf['std'])):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)

        # 2. Method performance across noise types
        ax = axes[0, 1]
        for method in self.methods:
            method_data = df_agg[df_agg['method'] == method].groupby('noise_type')['test_accuracy_mean'].mean()
            noise_order = ['rgb_clean', 'gaussian', 'poisson', 's&p', 'speckle', 'uniform']
            noise_order = [n for n in noise_order if n in method_data.index]
            ax.plot(range(len(noise_order)), [method_data.get(n, np.nan) for n in noise_order],
                   marker='o', label=method, linewidth=2)

        ax.set_xticks(range(len(noise_order)))
        ax.set_xticklabels([self.noise_labels.get(n, n) for n in noise_order], rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Method Robustness Across Noise Types')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Method performance across dataset sizes
        ax = axes[1, 0]
        for method in self.methods:
            size_data = df_agg[df_agg['method'] == method].groupby('size')['test_accuracy_mean'].mean()
            size_order = ['mini', 'small', 'original']
            ax.plot(range(len(size_order)), [size_data.get(s, np.nan) for s in size_order],
                   marker='s', label=method, linewidth=2, markersize=8)

        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order)
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Method Performance vs. Dataset Size')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Method performance across k-values
        ax = axes[1, 1]
        for method in self.methods:
            k_data = df_agg[df_agg['method'] == method].groupby('k_value')['test_accuracy_mean'].mean()
            k_order = sorted(k_data.index)
            ax.plot(k_order, [k_data[k] for k in k_order],
                   marker='D', label=method, linewidth=2, markersize=8)

        ax.set_xlabel('Number of Features (k)')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Method Performance vs. Feature Count')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_method_trends.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: analysis_method_trends.png")

        # Print summary statistics
        print("\n   📈 Method Performance Summary:")
        for method in self.methods:
            method_df = df_agg[df_agg['method'] == method]
            print(f"      {method:15s}: μ={method_df['test_accuracy_mean'].mean():.3f}, "
                  f"σ={method_df['test_accuracy_mean'].std():.3f}, "
                  f"n={len(method_df)}")

    def analyze_k_value_trends(self, df_agg: pd.DataFrame):
        """Analyze performance trends by number of selected features (k)."""
        print("\n🔢 Analyzing K-Value (Feature Count) Trends...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Selection (K-Value) Analysis', fontsize=14, fontweight='bold')

        # 1. Overall accuracy by k-value
        ax = axes[0, 0]
        k_perf = df_agg.groupby('k_value')['test_accuracy_mean'].agg(['mean', 'std'])
        k_order = sorted(k_perf.index)

        ax.errorbar(k_order, [k_perf.loc[k, 'mean'] for k in k_order],
                   yerr=[k_perf.loc[k, 'std'] for k in k_order],
                   marker='o', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('Number of Features (k)')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Overall Performance vs. K-Value')
        ax.grid(alpha=0.3)

        # Add value labels
        for k in k_order:
            mean_val = k_perf.loc[k, 'mean']
            ax.text(k, mean_val + 0.02, f'{mean_val:.3f}', ha='center', fontsize=9)

        # 2. K-value performance across dataset sizes
        ax = axes[0, 1]
        for size in ['mini', 'small', 'original']:
            size_data = df_agg[df_agg['size'] == size].groupby('k_value')['test_accuracy_mean'].mean()
            k_order = sorted(size_data.index)
            ax.plot(k_order, [size_data.get(k, np.nan) for k in k_order],
                   marker='o', label=size, linewidth=2, markersize=8)

        ax.set_xlabel('Number of Features (k)')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('K-Value Effect Across Dataset Sizes')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. K-value performance across noise types (heatmap)
        ax = axes[1, 0]
        pivot_data = df_agg.groupby(['noise_type', 'k_value'])['test_accuracy_mean'].mean().unstack()
        noise_order = ['rgb_clean', 'gaussian', 'poisson', 's&p', 'speckle', 'uniform']
        noise_order = [n for n in noise_order if n in pivot_data.index]
        pivot_data = pivot_data.loc[noise_order]

        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                   ax=ax, cbar_kws={'label': 'Accuracy'})
        ax.set_ylabel('Noise Type')
        ax.set_xlabel('K-Value')
        ax.set_title('Accuracy Heatmap: K-Value vs. Noise Type')
        ax.set_yticklabels([self.noise_labels.get(n, n) for n in noise_order], rotation=0)

        # 4. K-value CV std (stability analysis)
        ax = axes[1, 1]
        k_stability = df_agg.groupby('k_value')['cv_std_accuracy_mean'].mean()
        k_order = sorted(k_stability.index)

        ax.bar(k_order, [k_stability[k] for k in k_order], alpha=0.7, color='coral')
        ax.set_xlabel('Number of Features (k)')
        ax.set_ylabel('Mean CV Std Deviation')
        ax.set_title('Model Stability vs. K-Value (Lower is Better)')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_k_value_trends.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: analysis_k_value_trends.png")

        # Print summary
        print("\n   📈 K-Value Performance Summary:")
        for k in sorted(df_agg['k_value'].unique()):
            k_df = df_agg[df_agg['k_value'] == k]
            print(f"      k={k:2d}: μ={k_df['test_accuracy_mean'].mean():.3f}, "
                  f"σ={k_df['test_accuracy_mean'].std():.3f}, "
                  f"CV_std={k_df['cv_std_accuracy_mean'].mean():.3f}")

    def analyze_dataset_size_trends(self, df_agg: pd.DataFrame):
        """Analyze performance trends by dataset size."""
        print("\n📏 Analyzing Dataset Size Trends...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dataset Size Impact Analysis', fontsize=14, fontweight='bold')

        size_order = ['mini', 'small', 'original']

        # 1. Overall accuracy by dataset size
        ax = axes[0, 0]
        size_perf = df_agg.groupby('size')['test_accuracy_mean'].agg(['mean', 'std'])
        size_perf = size_perf.loc[size_order]

        x = np.arange(len(size_order))
        ax.bar(x, size_perf['mean'], yerr=size_perf['std'], capsize=5, alpha=0.7, color=['#ff9999', '#ffcc99', '#99cc99'])
        ax.set_xticks(x)
        ax.set_xticklabels(size_order)
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Overall Performance vs. Dataset Size')
        ax.set_ylim([0.5, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels and improvement percentages
        for i, (size, row) in enumerate(size_perf.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.02, f'{row["mean"]:.3f}', ha='center', fontsize=9)
            if i > 0:
                prev_mean = size_perf.iloc[i-1]['mean']
                improvement = ((row['mean'] - prev_mean) / prev_mean) * 100
                ax.text(i, 0.52, f'+{improvement:.1f}%', ha='center', fontsize=8, color='green')

        # 2. Dataset size effect across methods
        ax = axes[0, 1]
        for method in self.methods:
            method_data = df_agg[df_agg['method'] == method].groupby('size')['test_accuracy_mean'].mean()
            ax.plot(range(len(size_order)), [method_data.get(s, np.nan) for s in size_order],
                   marker='o', label=method, linewidth=2, markersize=8)

        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order)
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Dataset Size Effect by Method')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Dataset size effect across noise types
        ax = axes[1, 0]
        for noise in ['rgb_clean', 'gaussian', 'poisson']:
            noise_data = df_agg[df_agg['noise_type'] == noise].groupby('size')['test_accuracy_mean'].mean()
            ax.plot(range(len(size_order)), [noise_data.get(s, np.nan) for s in size_order],
                   marker='s', label=self.noise_labels.get(noise, noise), linewidth=2, markersize=8)

        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order)
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Dataset Size Effect Across Noise Types')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Sample count vs accuracy scatter
        ax = axes[1, 1]
        sample_counts = {'mini': 30, 'small': 150, 'original': 600}

        for size in size_order:
            size_df = df_agg[df_agg['size'] == size]
            n_samples = sample_counts[size]
            accuracies = size_df['test_accuracy_mean'].values

            # Add jitter for visibility
            x_vals = np.random.normal(n_samples, n_samples * 0.05, len(accuracies))
            ax.scatter(x_vals, accuracies, alpha=0.6, s=50, label=size)

        ax.set_xlabel('Approximate Number of Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs. Sample Count')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_dataset_size_trends.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: analysis_dataset_size_trends.png")

        # Print summary
        print("\n   📈 Dataset Size Performance Summary:")
        for size in size_order:
            size_df = df_agg[df_agg['size'] == size]
            print(f"      {size:8s}: μ={size_df['test_accuracy_mean'].mean():.3f}, "
                  f"σ={size_df['test_accuracy_mean'].std():.3f}, "
                  f"samples~{sample_counts[size]}")

    def analyze_noise_trends(self, df_agg: pd.DataFrame):
        """Analyze performance trends across noise types and intensities."""
        print("\n🔊 Analyzing Noise Robustness Trends...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Noise Robustness Analysis', fontsize=14, fontweight='bold')

        noise_order = ['rgb_clean', 'gaussian', 'poisson', 's&p', 'speckle', 'uniform']
        noise_order = [n for n in noise_order if n in df_agg['noise_type'].unique()]

        # 1. Overall accuracy by noise type
        ax = axes[0, 0]
        noise_perf = df_agg.groupby('noise_type')['test_accuracy_mean'].agg(['mean', 'std'])
        noise_perf = noise_perf.loc[noise_order]

        x = np.arange(len(noise_order))
        colors = ['green' if n == 'rgb_clean' else 'orange' for n in noise_order]
        ax.bar(x, noise_perf['mean'], yerr=noise_perf['std'], capsize=5, alpha=0.7, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([self.noise_labels.get(n, n) for n in noise_order], rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Performance Degradation by Noise Type')
        ax.set_ylim([0.5, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add degradation percentages
        clean_acc = noise_perf.loc['rgb_clean', 'mean']
        for i, (noise, row) in enumerate(noise_perf.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.02, f'{row["mean"]:.3f}', ha='center', fontsize=8)
            if noise != 'rgb_clean':
                degradation = ((clean_acc - row['mean']) / clean_acc) * 100
                ax.text(i, 0.52, f'-{degradation:.1f}%', ha='center', fontsize=8, color='red')

        # 2. Noise robustness heatmap (method x noise)
        ax = axes[0, 1]
        pivot_data = df_agg.groupby(['method', 'noise_type'])['test_accuracy_mean'].mean().unstack()
        pivot_data = pivot_data[noise_order]

        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                   ax=ax, cbar_kws={'label': 'Accuracy'})
        ax.set_ylabel('Feature Method')
        ax.set_xlabel('Noise Type')
        ax.set_title('Method Robustness Heatmap')
        ax.set_xticklabels([self.noise_labels.get(n, n) for n in noise_order], rotation=45, ha='right')

        # 3. Noise impact on different dataset sizes
        ax = axes[1, 0]
        for size in ['mini', 'small', 'original']:
            size_data = df_agg[df_agg['size'] == size].groupby('noise_type')['test_accuracy_mean'].mean()
            ax.plot(range(len(noise_order)), [size_data.get(n, np.nan) for n in noise_order],
                   marker='o', label=size, linewidth=2, markersize=8)

        ax.set_xticks(range(len(noise_order)))
        ax.set_xticklabels([self.noise_labels.get(n, n) for n in noise_order], rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Noise Impact Across Dataset Sizes')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Accuracy degradation bar chart
        ax = axes[1, 1]
        degradations = []
        for noise in noise_order:
            if noise == 'rgb_clean':
                continue
            clean_acc = df_agg[df_agg['noise_type'] == 'rgb_clean']['test_accuracy_mean'].mean()
            noise_acc = df_agg[df_agg['noise_type'] == noise]['test_accuracy_mean'].mean()
            degradation = ((clean_acc - noise_acc) / clean_acc) * 100
            degradations.append((self.noise_labels.get(noise, noise), degradation))

        degradations.sort(key=lambda x: x[1])
        labels, values = zip(*degradations)

        x = np.arange(len(labels))
        ax.barh(x, values, alpha=0.7, color='coral')
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Accuracy Degradation (%)')
        ax.set_title('Noise-Induced Performance Loss')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('analysis_noise_robustness.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: analysis_noise_robustness.png")

        # Print summary
        print("\n   📈 Noise Impact Summary:")
        clean_acc = df_agg[df_agg['noise_type'] == 'rgb_clean']['test_accuracy_mean'].mean()
        print(f"      Clean baseline: {clean_acc:.3f}")
        for noise in noise_order:
            if noise == 'rgb_clean':
                continue
            noise_df = df_agg[df_agg['noise_type'] == noise]
            noise_acc = noise_df['test_accuracy_mean'].mean()
            degradation = ((clean_acc - noise_acc) / clean_acc) * 100
            print(f"      {self.noise_labels.get(noise, noise):15s}: μ={noise_acc:.3f}, "
                  f"degradation={degradation:.1f}%")

    def create_comprehensive_report(self, df_agg: pd.DataFrame):
        """Generate comprehensive analysis report."""
        print("\n📝 Generating Comprehensive Analysis Report...")

        report = []
        report.append("# Comprehensive Experiment Analysis Report")
        report.append("=" * 80)
        report.append(f"\nTotal Experiments Analyzed: {len(self.df)}")
        report.append(f"Unique Parameter Combinations: {len(df_agg)}\n")

        # Overall statistics
        report.append("\n## Overall Performance Statistics")
        report.append("-" * 80)
        report.append(f"Mean Accuracy (all experiments): {self.df['test_accuracy'].mean():.4f} ± {self.df['test_accuracy'].std():.4f}")
        report.append(f"Mean CV Accuracy: {self.df['cv_mean_accuracy'].mean():.4f} ± {self.df['cv_mean_accuracy'].std():.4f}")
        report.append(f"Best Single Experiment: {self.df['test_accuracy'].max():.4f}")
        report.append(f"Worst Single Experiment: {self.df['test_accuracy'].min():.4f}")

        # Method comparison
        report.append("\n## Feature Extraction Method Comparison")
        report.append("-" * 80)
        method_stats = df_agg.groupby('method')['test_accuracy_mean'].agg(['mean', 'std', 'min', 'max'])
        method_stats = method_stats.sort_values('mean', ascending=False)
        report.append(method_stats.to_string())

        best_method = method_stats.index[0]
        report.append(f"\n🏆 Best Method: {best_method} (μ={method_stats.loc[best_method, 'mean']:.4f})")

        # K-value analysis
        report.append("\n## K-Value (Feature Count) Analysis")
        report.append("-" * 80)
        k_stats = df_agg.groupby('k_value')['test_accuracy_mean'].agg(['mean', 'std', 'min', 'max'])
        k_stats = k_stats.sort_values('mean', ascending=False)
        report.append(k_stats.to_string())

        best_k = k_stats.index[0]
        report.append(f"\n🏆 Best K-Value: {best_k} (μ={k_stats.loc[best_k, 'mean']:.4f})")

        # Dataset size analysis
        report.append("\n## Dataset Size Analysis")
        report.append("-" * 80)
        size_stats = df_agg.groupby('size')['test_accuracy_mean'].agg(['mean', 'std', 'min', 'max'])
        size_order = ['mini', 'small', 'original']
        size_stats = size_stats.loc[size_order]
        report.append(size_stats.to_string())

        # Calculate improvements
        mini_acc = size_stats.loc['mini', 'mean']
        small_acc = size_stats.loc['small', 'mean']
        orig_acc = size_stats.loc['original', 'mean']

        report.append(f"\n📈 Improvement Trajectory:")
        report.append(f"   Mini → Small: +{((small_acc - mini_acc) / mini_acc) * 100:.2f}%")
        report.append(f"   Small → Original: +{((orig_acc - small_acc) / small_acc) * 100:.2f}%")
        report.append(f"   Mini → Original: +{((orig_acc - mini_acc) / mini_acc) * 100:.2f}% (total)")

        # Noise robustness
        report.append("\n## Noise Robustness Analysis")
        report.append("-" * 80)
        noise_stats = df_agg.groupby('noise_type')['test_accuracy_mean'].agg(['mean', 'std', 'min', 'max'])
        noise_stats = noise_stats.sort_values('mean', ascending=False)
        report.append(noise_stats.to_string())

        clean_acc = noise_stats.loc['rgb_clean', 'mean']
        report.append(f"\n📊 Noise-Induced Degradation (from clean baseline {clean_acc:.4f}):")
        for noise in noise_stats.index:
            if noise != 'rgb_clean':
                noise_acc = noise_stats.loc[noise, 'mean']
                degradation = ((clean_acc - noise_acc) / clean_acc) * 100
                report.append(f"   {self.noise_labels.get(noise, noise):15s}: -{degradation:5.2f}% (acc={noise_acc:.4f})")

        # Best configuration
        report.append("\n## Best Configuration")
        report.append("-" * 80)
        best_config = df_agg.loc[df_agg['test_accuracy_mean'].idxmax()]
        report.append(f"Method: {best_config['method']}")
        report.append(f"K-Value: {best_config['k_value']}")
        report.append(f"Dataset Size: {best_config['size']}")
        report.append(f"Noise Type: {self.noise_labels.get(best_config['noise_type'], best_config['noise_type'])}")
        report.append(f"Accuracy: {best_config['test_accuracy_mean']:.4f}")

        # Save report
        report_text = '\n'.join(report)
        with open('comprehensive_analysis_report.txt', 'w') as f:
            f.write(report_text)

        print("   ✅ Saved: comprehensive_analysis_report.txt")

        # Also save aggregated data to CSV
        df_agg.to_csv('aggregated_results.csv', index=False)
        print("   ✅ Saved: aggregated_results.csv")

        # Print summary to console
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"🏆 Best Method: {best_method}")
        print(f"🏆 Best K-Value: {best_k}")
        print(f"🏆 Best Size: {size_stats.index[0]}")
        print(f"📈 Dataset Size Impact: Mini→Original = +{((orig_acc - mini_acc) / mini_acc) * 100:.1f}%")
        print(f"🔊 Worst Noise Degradation: {max(degradation for noise in noise_stats.index if noise != 'rgb_clean' for degradation in [((clean_acc - noise_stats.loc[noise, 'mean']) / clean_acc) * 100]):.1f}%")
        print("=" * 80)

def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RANDOM FOREST EXPERIMENT ANALYSIS")
    print("=" * 80)

    analyzer = ComprehensiveExperimentAnalyzer()

    # Load all experiments
    df = analyzer.load_all_experiments()

    if len(df) == 0:
        print("❌ No experiments found!")
        return

    # Aggregate across geographic areas
    df_agg = analyzer.aggregate_by_parameters()

    # Run analyses
    analyzer.analyze_method_trends(df_agg)
    analyzer.analyze_k_value_trends(df_agg)
    analyzer.analyze_dataset_size_trends(df_agg)
    analyzer.analyze_noise_trends(df_agg)
    analyzer.create_comprehensive_report(df_agg)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 analysis_method_trends.png")
    print("  📊 analysis_k_value_trends.png")
    print("  📊 analysis_dataset_size_trends.png")
    print("  📊 analysis_noise_robustness.png")
    print("  📝 comprehensive_analysis_report.txt")
    print("  📋 aggregated_results.csv")

if __name__ == "__main__":
    main()
