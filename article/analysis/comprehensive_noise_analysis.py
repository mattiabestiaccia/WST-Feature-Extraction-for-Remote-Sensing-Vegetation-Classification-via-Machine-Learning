#!/usr/bin/env python3
"""
Analisi Completa dei Risultati Sperimentali
============================================

Questo script analizza tutti i file CSV averaged per studiare l'impatto di:
1. Dimensione dataset (mini, small, original)
2. Tipo di rumore (clean, gaussian, poisson, salt&pepper, speckle, uniform)
3. Intensità del rumore (3 livelli per ogni tipo)
4. Numero di features (k=2, 5, 10, 20)

Dati aggregati per area geografica (già mediati nei file _averaged.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Configurazione grafica
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_LARGE = (16, 10)
FIGSIZE_MEDIUM = (12, 8)
DPI = 300

class NoiseAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data = None
        self.noise_files = {
            'gaussian': 'experiments/gaussian/gaussian_analysis/experiments_summary_averaged.csv',
            'poisson': 'experiments/poisson/poisson_analysis/poisson_experiments_summary_averaged.csv',
            'saltpepper': 'experiments/saltpepper/saltpepper_analysis/saltpepper_experiments_summary_averaged.csv',
            'speckle': 'experiments/speckle/speckle_analysis/speckle_experiments_summary_averaged.csv',
            'uniform': 'experiments/uniform/uniform_analysis/uniform_experiments_summary_averaged.csv'
        }

    def load_all_data(self):
        """Carica tutti i CSV e li combina in un unico DataFrame"""
        print("📊 Caricamento dati da tutti i file CSV...")

        all_dfs = []
        for noise_type, rel_path in self.noise_files.items():
            file_path = self.base_dir / rel_path
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['noise_family'] = noise_type
                all_dfs.append(df)
                print(f"  ✓ {noise_type}: {len(df)} righe")
            else:
                print(f"  ✗ {noise_type}: FILE NON TROVATO - {file_path}")

        self.data = pd.concat(all_dfs, ignore_index=True)

        # Parsing intensità del rumore dalla colonna noise_condition
        self.data['noise_intensity'] = self.data['noise_condition'].apply(self._extract_intensity)
        self.data['noise_type_full'] = self.data['noise_condition']

        # Crea una colonna per distinguere clean vs noisy
        self.data['is_clean'] = self.data['noise_condition'] == 'clean'

        print(f"\n✅ Dataset completo: {len(self.data)} righe totali")
        print(f"   - Noise conditions uniche: {self.data['noise_condition'].nunique()}")
        print(f"   - Dataset types: {sorted(self.data['dataset_type'].unique())}")
        print(f"   - K values: {sorted(self.data['k_features'].unique())}")
        print(f"   - Methods: {sorted(self.data['feature_method'].unique())}")

        return self.data

    def _extract_intensity(self, noise_condition):
        """Estrae l'intensità numerica dal nome della noise condition"""
        if noise_condition == 'clean':
            return 0

        # Estrai il numero alla fine (es: gaussian30 -> 30)
        match = re.search(r'(\d+)$', noise_condition)
        if match:
            return int(match.group(1))
        return 0

    def get_noise_type_from_condition(self, condition):
        """Estrae il tipo di rumore dalla condizione"""
        if condition == 'clean':
            return 'clean'

        for noise in ['gaussian', 'poisson', 'saltpepper', 'speckle', 'uniform']:
            if noise in condition.lower():
                return noise
        return 'unknown'

    def analyze_dataset_size_impact(self):
        """Analisi 1: Impatto della dimensione del dataset sull'accuratezza"""
        print("\n" + "="*70)
        print("ANALISI 1: IMPATTO DIMENSIONE DATASET")
        print("="*70)

        # Raggruppa per dataset_type e calcola media/std
        size_stats = self.data.groupby('dataset_type')['mean_accuracy'].agg(['mean', 'std', 'count'])
        size_stats = size_stats.reindex(['mini', 'small', 'original'])

        print("\n📈 Accuratezza media per dimensione dataset:")
        print(size_stats)

        # Calcola miglioramento percentuale
        mini_acc = size_stats.loc['mini', 'mean']
        small_acc = size_stats.loc['small', 'mean']
        original_acc = size_stats.loc['original', 'mean']

        print(f"\n🔍 Miglioramenti:")
        print(f"   Mini → Small:    {(small_acc - mini_acc)*100:+.2f}% (+{(small_acc/mini_acc - 1)*100:.1f}%)")
        print(f"   Small → Original: {(original_acc - small_acc)*100:+.2f}% (+{(original_acc/small_acc - 1)*100:.1f}%)")
        print(f"   Mini → Original:  {(original_acc - mini_acc)*100:+.2f}% (+{(original_acc/mini_acc - 1)*100:.1f}%)")

        # Crea grafico
        self._plot_dataset_size_impact()

        return size_stats

    def _plot_dataset_size_impact(self):
        """Genera grafico per l'analisi dimensione dataset"""
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

        # 1. Bar plot generale
        ax = axes[0, 0]
        size_order = ['mini', 'small', 'original']
        size_data = self.data.groupby('dataset_type')['mean_accuracy'].mean().reindex(size_order)
        size_std = self.data.groupby('dataset_type')['mean_accuracy'].std().reindex(size_order)

        bars = ax.bar(size_order, size_data, yerr=size_std, capsize=10,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance vs. Dataset Size', fontsize=14, fontweight='bold')
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Aggiungi valori sulle barre
        for bar, val in zip(bars, size_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 2. Effetto per metodo
        ax = axes[0, 1]
        for method in ['advanced_stats', 'wst', 'hybrid']:
            method_data = self.data[self.data['feature_method'] == method]
            method_by_size = method_data.groupby('dataset_type')['mean_accuracy'].mean().reindex(size_order)
            ax.plot(size_order, method_by_size, marker='o', linewidth=2,
                   label=method, markersize=8)

        ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_title('Dataset Size Effect by Method', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.85, 1.0)

        # 3. Distribuzione per dataset size
        ax = axes[1, 0]
        violin_data = []
        labels = []
        for size in size_order:
            violin_data.append(self.data[self.data['dataset_type'] == size]['mean_accuracy'])
            labels.append(size)

        parts = ax.violinplot(violin_data, positions=range(len(size_order)),
                             showmeans=True, showmedians=True)
        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order)
        ax.set_ylabel('Accuracy Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Distribution by Dataset Size', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 4. Heatmap: Dataset Size vs K-value
        ax = axes[1, 1]
        heatmap_data = self.data.pivot_table(
            values='mean_accuracy',
            index='dataset_type',
            columns='k_features',
            aggfunc='mean'
        ).reindex(size_order)

        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.85, vmax=1.0, ax=ax, cbar_kws={'label': 'Mean Accuracy'})
        ax.set_title('Dataset Size vs. K-Value Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Features (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dataset Size', fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_file = self.base_dir / 'analysis_dataset_size_comprehensive.png'
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"\n💾 Grafico salvato: {output_file}")
        plt.close()

    def analyze_noise_type_impact(self):
        """Analisi 2: Impatto del tipo di rumore sull'accuratezza"""
        print("\n" + "="*70)
        print("ANALISI 2: IMPATTO TIPO DI RUMORE")
        print("="*70)

        # Estrai tipo di rumore
        self.data['noise_type'] = self.data['noise_condition'].apply(self.get_noise_type_from_condition)

        # Raggruppa per noise_type
        noise_stats = self.data.groupby('noise_type')['mean_accuracy'].agg(['mean', 'std', 'count'])
        noise_stats = noise_stats.sort_values('mean', ascending=False)

        print("\n📈 Accuratezza media per tipo di rumore:")
        print(noise_stats)

        # Calcola degradazione rispetto al clean
        clean_acc = noise_stats.loc['clean', 'mean']
        print(f"\n🔍 Degradazione rispetto a Clean (baseline: {clean_acc:.4f}):")
        for noise in noise_stats.index:
            if noise != 'clean':
                acc = noise_stats.loc[noise, 'mean']
                degradation = (acc - clean_acc) * 100
                print(f"   {noise:12s}: {degradation:+.2f}% (acc: {acc:.4f})")

        # Crea grafico
        self._plot_noise_type_impact()

        return noise_stats

    def _plot_noise_type_impact(self):
        """Genera grafico per l'analisi tipo di rumore"""
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

        # 1. Bar plot con degradazione
        ax = axes[0, 0]
        self.data['noise_type'] = self.data['noise_condition'].apply(self.get_noise_type_from_condition)
        noise_order = ['clean', 'uniform', 'poisson', 'saltpepper', 'speckle', 'gaussian']
        noise_data = self.data.groupby('noise_type')['mean_accuracy'].mean().reindex(noise_order)
        noise_std = self.data.groupby('noise_type')['mean_accuracy'].std().reindex(noise_order)

        colors = ['#2ECC71' if i == 0 else '#E74C3C' if i == 5 else '#F39C12'
                 for i in range(len(noise_order))]

        bars = ax.bar(range(len(noise_order)), noise_data, yerr=noise_std,
                      capsize=8, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(noise_order)))
        ax.set_xticklabels(noise_order, rotation=45, ha='right')
        ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Noise Type', fontsize=14, fontweight='bold')
        ax.set_ylim(0.80, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Aggiungi valori
        for bar, val in zip(bars, noise_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 2. Heatmap: Noise Type vs. Method
        ax = axes[0, 1]
        heatmap_data = self.data.pivot_table(
            values='mean_accuracy',
            index='noise_type',
            columns='feature_method',
            aggfunc='mean'
        ).reindex(noise_order)

        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.85, vmax=1.0, ax=ax, cbar_kws={'label': 'Mean Accuracy'})
        ax.set_title('Noise Type vs. Method Heatmap', fontsize=14, fontweight='bold')
        ax.set_ylabel('Noise Type', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Method', fontsize=12, fontweight='bold')

        # 3. Line plot: Noise type per dataset size
        ax = axes[1, 0]
        size_order = ['mini', 'small', 'original']
        for noise in noise_order:
            noise_subset = self.data[self.data['noise_type'] == noise]
            noise_by_size = noise_subset.groupby('dataset_type')['mean_accuracy'].mean().reindex(size_order)
            ax.plot(size_order, noise_by_size, marker='o', linewidth=2,
                   label=noise, markersize=7)

        ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_title('Noise Type Effect Across Dataset Sizes', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

        # 4. Degradazione relativa a clean
        ax = axes[1, 1]
        clean_acc = noise_data['clean']
        degradations = [(noise_data[noise] - clean_acc) * 100 for noise in noise_order if noise != 'clean']
        noise_labels = [n for n in noise_order if n != 'clean']

        bars = ax.barh(range(len(noise_labels)), degradations,
                       color=['#2ECC71' if d >= 0 else '#E74C3C' for d in degradations],
                       alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(noise_labels)))
        ax.set_yticklabels(noise_labels)
        ax.set_xlabel('Accuracy Degradation (%)', fontsize=12, fontweight='bold')
        ax.set_title('Noise-Induced Performance Loss vs. Clean', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax.grid(axis='x', alpha=0.3)

        # Aggiungi valori
        for i, (bar, val) in enumerate(zip(bars, degradations)):
            ax.text(val + (0.2 if val > 0 else -0.2), i, f'{val:.1f}%',
                   va='center', ha='left' if val > 0 else 'right', fontweight='bold', fontsize=9)

        plt.tight_layout()
        output_file = self.base_dir / 'analysis_noise_type_comprehensive.png'
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"\n💾 Grafico salvato: {output_file}")
        plt.close()

    def analyze_noise_intensity_impact(self):
        """Analisi 3: Impatto dell'intensità del rumore sull'accuratezza"""
        print("\n" + "="*70)
        print("ANALISI 3: IMPATTO INTENSITÀ RUMORE")
        print("="*70)

        # Filtra solo i dati con rumore (esclude clean)
        noisy_data = self.data[self.data['noise_intensity'] > 0].copy()

        # Raggruppa per noise_family e intensità
        intensity_stats = noisy_data.groupby(['noise_family', 'noise_intensity'])['mean_accuracy'].agg(['mean', 'std', 'count'])

        print("\n📈 Accuratezza media per tipo di rumore e intensità:")
        print(intensity_stats)

        # Crea grafico
        self._plot_noise_intensity_impact(noisy_data)

        return intensity_stats

    def _plot_noise_intensity_impact(self, noisy_data):
        """Genera grafico per l'analisi intensità rumore"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        noise_families = sorted(noisy_data['noise_family'].unique())

        for idx, noise_type in enumerate(noise_families):
            ax = axes[idx]
            noise_subset = noisy_data[noisy_data['noise_family'] == noise_type]

            # Ordina per intensità
            intensity_data = noise_subset.groupby('noise_intensity')['mean_accuracy'].agg(['mean', 'std'])
            intensity_data = intensity_data.sort_index()

            # Plot con error bars
            intensities = intensity_data.index
            means = intensity_data['mean']
            stds = intensity_data['std']

            ax.errorbar(intensities, means, yerr=stds, marker='o', linewidth=2,
                       markersize=8, capsize=5, capthick=2, label=f'{noise_type}')

            ax.set_xlabel('Noise Intensity', fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Accuracy', fontsize=11, fontweight='bold')
            ax.set_title(f'{noise_type.upper()} - Accuracy vs. Intensity',
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_ylim(0.75, 1.0)

            # Aggiungi valori
            for x, y in zip(intensities, means):
                ax.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=9, fontweight='bold')

        # Rimuovi subplot extra
        if len(noise_families) < len(axes):
            for idx in range(len(noise_families), len(axes)):
                fig.delaxes(axes[idx])

        plt.tight_layout()
        output_file = self.base_dir / 'analysis_noise_intensity_comprehensive.png'
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"\n💾 Grafico salvato: {output_file}")
        plt.close()

    def analyze_k_features_impact(self):
        """Analisi 4: Impatto del numero di features (k) sull'accuratezza"""
        print("\n" + "="*70)
        print("ANALISI 4: IMPATTO NUMERO FEATURES (k)")
        print("="*70)

        # Raggruppa per k_features
        k_stats = self.data.groupby('k_features')['mean_accuracy'].agg(['mean', 'std', 'count'])
        k_stats = k_stats.sort_index()

        print("\n📈 Accuratezza media per numero di features:")
        print(k_stats)

        # Trova il k ottimale
        best_k = k_stats['mean'].idxmax()
        best_acc = k_stats.loc[best_k, 'mean']
        print(f"\n🏆 K ottimale: {best_k} (accuratezza: {best_acc:.4f})")

        # Crea grafico
        self._plot_k_features_impact()

        return k_stats

    def _plot_k_features_impact(self):
        """Genera grafico per l'analisi numero features"""
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

        # 1. Performance generale per k
        ax = axes[0, 0]
        k_order = sorted(self.data['k_features'].unique())
        k_data = self.data.groupby('k_features')['mean_accuracy'].mean()
        k_std = self.data.groupby('k_features')['mean_accuracy'].std()

        ax.errorbar(k_order, k_data, yerr=k_std, marker='o', linewidth=3,
                   markersize=10, capsize=8, capthick=2, color='#3498db')
        ax.set_xlabel('Number of Features (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance vs. K-Value', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(0.88, 0.96)

        # Aggiungi valori
        for x, y in zip(k_order, k_data):
            ax.text(x, y + 0.003, f'{y:.3f}', ha='center', fontsize=10, fontweight='bold')

        # 2. K effect per metodo
        ax = axes[0, 1]
        for method in ['advanced_stats', 'wst', 'hybrid']:
            method_data = self.data[self.data['feature_method'] == method]
            method_by_k = method_data.groupby('k_features')['mean_accuracy'].mean()
            ax.plot(k_order, method_by_k, marker='o', linewidth=2, label=method, markersize=8)

        ax.set_xlabel('Number of Features (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('K-Value Effect by Method', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # 3. Heatmap: K vs Dataset Size
        ax = axes[1, 0]
        size_order = ['mini', 'small', 'original']
        heatmap_data = self.data.pivot_table(
            values='mean_accuracy',
            index='dataset_type',
            columns='k_features',
            aggfunc='mean'
        ).reindex(size_order)

        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.85, vmax=1.0, ax=ax, cbar_kws={'label': 'Mean Accuracy'})
        ax.set_title('Dataset Size vs. K-Value', fontsize=14, fontweight='bold')
        ax.set_ylabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Features (k)', fontsize=12, fontweight='bold')

        # 4. K effect per noise type
        ax = axes[1, 1]
        self.data['noise_type'] = self.data['noise_condition'].apply(self.get_noise_type_from_condition)

        for noise in ['clean', 'gaussian', 'poisson', 'speckle']:
            noise_data = self.data[self.data['noise_type'] == noise]
            noise_by_k = noise_data.groupby('k_features')['mean_accuracy'].mean()
            ax.plot(k_order, noise_by_k, marker='o', linewidth=2, label=noise, markersize=7)

        ax.set_xlabel('Number of Features (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('K-Value Effect Across Noise Types', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_file = self.base_dir / 'analysis_k_features_comprehensive.png'
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        print(f"\n💾 Grafico salvato: {output_file}")
        plt.close()

    def generate_summary_report(self):
        """Genera un report riassuntivo con tutte le analisi"""
        print("\n" + "="*70)
        print("GENERAZIONE REPORT RIASSUNTIVO")
        print("="*70)

        report = []
        report.append("# Analisi Comparativa Completa dei Risultati Sperimentali\n")
        report.append("*Dati aggregati per area geografica*\n\n")

        # Dataset globale
        report.append("## Dataset Globale\n")
        report.append(f"- **Numero totale osservazioni**: {len(self.data)}\n")
        report.append(f"- **Noise conditions**: {self.data['noise_condition'].nunique()}\n")
        report.append(f"- **Dataset sizes**: {', '.join(sorted(self.data['dataset_type'].unique()))}\n")
        report.append(f"- **K values**: {', '.join(map(str, sorted(self.data['k_features'].unique())))}\n")
        report.append(f"- **Methods**: {', '.join(sorted(self.data['feature_method'].unique()))}\n\n")

        # Statistiche globali
        report.append("## Statistiche Globali\n")
        report.append(f"- **Accuratezza media**: {self.data['mean_accuracy'].mean():.4f}\n")
        report.append(f"- **Deviazione standard**: {self.data['mean_accuracy'].std():.4f}\n")
        report.append(f"- **Accuratezza massima**: {self.data['mean_accuracy'].max():.4f}\n")
        report.append(f"- **Accuratezza minima**: {self.data['mean_accuracy'].min():.4f}\n\n")

        # Configurazione ottimale
        best_row = self.data.loc[self.data['mean_accuracy'].idxmax()]
        report.append("## Configurazione Ottimale (Massima Accuratezza)\n")
        report.append(f"- **Accuratezza**: {best_row['mean_accuracy']:.4f}\n")
        report.append(f"- **Noise condition**: {best_row['noise_condition']}\n")
        report.append(f"- **Dataset size**: {best_row['dataset_type']}\n")
        report.append(f"- **K features**: {best_row['k_features']}\n")
        report.append(f"- **Method**: {best_row['feature_method']}\n\n")

        # Salva report
        report_path = self.base_dir / 'paper' / 'analysis_summary_report.md'
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.writelines(report)

        print(f"💾 Report salvato: {report_path}")
        print("\n✅ Analisi completata!")


def main():
    base_dir = Path('/home/brusc/Projects/random_forest')

    print("="*70)
    print("ANALISI COMPLETA DEI RISULTATI SPERIMENTALI")
    print("="*70)
    print(f"Base directory: {base_dir}\n")

    analyzer = NoiseAnalyzer(base_dir)

    # Carica tutti i dati
    analyzer.load_all_data()

    # Esegui tutte le analisi
    analyzer.analyze_dataset_size_impact()
    analyzer.analyze_noise_type_impact()
    analyzer.analyze_noise_intensity_impact()
    analyzer.analyze_k_features_impact()

    # Genera report finale
    analyzer.generate_summary_report()


if __name__ == '__main__':
    main()
