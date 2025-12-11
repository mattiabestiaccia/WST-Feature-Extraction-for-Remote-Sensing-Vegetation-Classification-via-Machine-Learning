# Code Context - Random Forest Vegetation Classification

## Panoramica del Progetto
Questo progetto implementa un sistema di classificazione di immagini di vegetazione utilizzando **Random Forest** con diverse tecniche di estrazione delle feature, incluso il **Wavelet Scattering Transform (WST)**. Il progetto include pipeline complete per training, inferenza, analisi di robustezza al rumore e generazione di report per pubblicazioni scientifiche.

---

## Struttura Completa del Progetto

```
random_forest/
├── scripts/                         # Script principali (produzione)
│   ├── experiments/
│   │   └── train_and_save_model.py  # Training modelli RF
│   ├── analysis/                    # Analisi per tipo di rumore
│   │   ├── analyze_gaussian_experiments.py
│   │   ├── analyze_poisson_experiments.py
│   │   ├── analyze_speckle_experiments.py
│   │   ├── analyze_salt_and_pepper_experiments.py
│   │   └── analyze_uniform_experiments.py
│   ├── inference.py                 # Inferenza universale
│   ├── add_noise.py                 # Generazione rumore
│   └── compare_wst_coefficients.py  # Confronto WST clean vs noisy
│
├── article/                         # Script per paper scientifico
│   ├── scripts/                     # Mirror di scripts/ per articolo
│   │   ├── experiments/
│   │   │   └── train_and_save_model.py
│   │   ├── analysis/
│   │   │   ├── analyze_gaussian_experiments.py
│   │   │   ├── analyze_poisson_experiments.py
│   │   │   ├── analyze_speckle_experiments.py
│   │   │   ├── analyze_salt_and_pepper_experiments.py
│   │   │   └── analyze_uniform_experiments.py
│   │   ├── inference.py
│   │   ├── add_noise.py
│   │   └── compare_wst_coefficients.py
│   │
│   ├── data_analysis/               # Analisi dati per paper
│   │   ├── analysis_1/scripts/
│   │   │   └── robustness_analysis.py      # Analisi robustezza WST vs Stats
│   │   ├── analysis_1_b/scripts/
│   │   │   ├── robustness_analysis_v2.py   # Versione 2 analisi robustezza
│   │   │   └── generate_plots_v2.py        # Generazione grafici v2
│   │   ├── analysis_2/
│   │   │   └── comprehensive_analysis.py   # Analisi comprensiva
│   │   └── analysis_3/
│   │       └── comprehensive_noise_analysis.py  # Analisi noise aggregata
│   │
│   └── features_visual/             # Visualizzazione features
│       ├── visualize_features.py    # Visualizzazione Advanced Stats vs WST
│       └── verify_output.py         # Verifica output generati
│
├── analisi/                         # Analisi aggregata generale
│   └── comprehensive_noise_analysis.py
│
├── datasets/                        # Dataset immagini RGB
├── experiments/                     # Output esperimenti
└── paper/                           # Output per pubblicazione
```

---

## Moduli Principali

### 1. Training (`scripts/experiments/train_and_save_model.py`)

**Scopo**: Addestra un classificatore Random Forest con feature extraction configurabile.

**Utilizzo CLI**:
```bash
python train_and_save_model.py <dataset_path> <area_name> <feature_method> <k_features> <output_dir> [options]
```

**Parametri**:
| Parametro | Valori | Descrizione |
|-----------|--------|-------------|
| `area_name` | `assatigue`, `popolar`, `sunset` | Area geografica |
| `feature_method` | `advanced_stats`, `wst`, `hybrid` | Metodo estrazione features |
| `k_features` | `2`, `5`, `10`, `20` | Numero features da selezionare |
| `--n_estimators` | default=50 | Numero alberi RF (auto: 3 per mini, 10 per small) |
| `--test_size` | default=0.2 | Frazione test set |
| `--cv_folds` | default=5 | Fold cross-validation |

**Funzioni Chiave**:
| Funzione | Linea | Descrizione |
|----------|-------|-------------|
| `load_rgb_image()` | ~51 | Carica PNG RGB come array (C,H,W) normalizzato [0,1] |
| `extract_advanced_features()` | ~58 | 18 features statistiche per canale RGB |
| `extract_wst_features()` | ~346 | Coefficienti WST (J=2, L=8) con mean/std |
| `extract_hybrid_features()` | ~380 | Combinazione advanced_stats + WST |
| `select_features_kbest()` | ~147 | Selezione con Mutual Information |
| `train_final_model()` | ~167 | Training RF con stratified split + CV |
| `save_model_and_artifacts()` | ~200 | Salva model, scaler, selector, report |

**Output Generati**:
```
output_dir/
├── trained_model.joblib           # Modello RF serializzato
├── scaler.joblib                  # StandardScaler fitted
├── feature_selector.joblib        # SelectKBest fitted
├── feature_names.json             # Features selezionate + scores
├── experiment_report_with_model.json  # Report completo
└── model_usage_instructions.md    # Istruzioni utilizzo
```

---

### 2. Inference (`scripts/inference.py`)

**Scopo**: Esegue inferenza su dataset usando modelli addestrati, con auto-detection della configurazione.

**Utilizzo CLI**:
```bash
python inference.py --model-dir /path/to/model [--dataset-type TYPE] [--sample N]
```

**Classe Principale**: `ModelInference`

**Metodi Chiave**:
| Metodo | Descrizione |
|--------|-------------|
| `parse_model_directory()` | Auto-detect area, feature_method, dataset_type dal path |
| `load_model_components()` | Carica model, scaler, selector, feature_info |
| `extract_features()` | Estrae features secondo il metodo rilevato |
| `predict_single_image()` | Predizione singola con probabilità |
| `predict_dataset()` | Inferenza batch con sampling opzionale |
| `evaluate_predictions()` | Calcola accuracy, F1, confusion matrix |
| `save_results()` | Esporta CSV, JSON, confusion matrix plot |

**Dataset Supportati** (definiti in `DATASET_DIRS`):
- `original`, `salt_pepper25`, `gaussian30`, `gaussian50`, `poisson60`

---

### 3. Aggiunta Rumore (`scripts/add_noise.py`)

**Scopo**: Genera versioni rumorose dei dataset per test di robustezza.

**Utilizzo CLI**:
```bash
python add_noise.py --noise-type <TYPE> --intensity <0-100> [--input-dir PATH] [--seed N]
```

**Funzioni di Rumore**:
| Funzione | Tipo | Formula/Descrizione |
|----------|------|---------------------|
| `add_gaussian_noise()` | Additivo | `noisy = image + N(0, σ)` dove `σ = intensity * 255 / 100` |
| `add_salt_and_pepper_noise()` | Impulsivo | Pixel random → 0 o 255 con prob `intensity/100` |
| `add_speckle_noise()` | Moltiplicativo | `noisy = image + image * N(0,1) * factor` |
| `add_poisson_noise()` | Shot noise | Basato su distribuzione Poisson |
| `add_uniform_noise()` | Additivo uniforme | `noisy = image + U(-range/2, range/2)` |

**Output**: Crea directory `datasets_<noise_type>_<intensity>/dataset_rgb_<noise_type>_<intensity>/`

---

### 4. Confronto WST (`scripts/compare_wst_coefficients.py`)

**Scopo**: Visualizza differenze nei coefficienti WST tra immagini clean e noisy.

**Funzioni Chiave**:
| Funzione | Descrizione |
|----------|-------------|
| `load_and_preprocess_image()` | Carica come grayscale 32x32, normalizza |
| `compute_scattering_coefficients()` | Scattering2D con J=3, L=6, max_order=2 |
| `plot_scattering_disk()` | Visualizzazione "disk plot" dei coefficienti |
| `compare_images()` | Confronta coppie clean/noisy con statistiche |

**Output**: PNG `wst_comparison_sigma50_pair_*.png` con:
- Immagini clean vs noisy
- WST disk plots per entrambe
- Statistiche (mean, std differenze)

---

### 5. Analisi Esperimenti (`scripts/analysis/analyze_*_experiments.py`)

**Pattern comune** per tutti i tipi di rumore (gaussian, poisson, speckle, salt_and_pepper, uniform).

**Classe Base**: `GaussianExperimentAnalyzer` (e analoghi)

**Attributi di Configurazione**:
```python
noise_conditions = ['clean', 'gaussian30', 'gaussian50']  # varia per tipo
areas = ['assatigue', 'popolar', 'sunset']
datasets = ['mini', 'small', 'original']
k_values = ['k2', 'k5', 'k10', 'k20']
feature_methods = ['advanced_stats', 'wst', 'hybrid']
```

**Metodi Principali**:
| Metodo | Output |
|--------|--------|
| `load_all_experiments()` | Lista di dict con tutti i risultati JSON |
| `generate_comprehensive_report()` | Report markdown statistico |
| `generate_qualitative_analysis()` | Analisi qualitativa robustezza |
| `create_comparison_plots()` | Grafici accuracy vs noise/dataset/k |
| `create_detailed_plots()` | Grafici dettagliati per combinazione |
| `export_to_csv()` | Export dati tabulari |
| `create_analysis_summary()` | Indice file generati |

**Output Generati**:
```
analysis_output/
├── comprehensive_report.md
├── qualitative_analysis.md
├── experiments_summary.csv
├── comparisons/
│   ├── accuracy_vs_noise_overall.png
│   ├── accuracy_vs_dataset_size_overall.png
│   ├── accuracy_vs_method_boxplot.png
│   └── accuracy_heatmap_summary.png
└── detailed/
    ├── accuracy_vs_noise_<dataset>_k<N>.png
    └── accuracy_vs_k_<noise>_<dataset>.png
```

---

### 6. Analisi Aggregata (`analisi/comprehensive_noise_analysis.py`)

**Scopo**: Analisi comparativa su TUTTI i tipi di rumore combinati.

**Classe**: `NoiseAnalyzer`

**File CSV di Input**:
```python
noise_files = {
    'gaussian': 'experiments/gaussian/gaussian_analysis/experiments_summary_averaged.csv',
    'poisson': 'experiments/poisson/poisson_analysis/poisson_experiments_summary_averaged.csv',
    'saltpepper': 'experiments/saltpepper/saltpepper_analysis/saltpepper_experiments_summary_averaged.csv',
    'speckle': 'experiments/speckle/speckle_analysis/speckle_experiments_summary_averaged.csv',
    'uniform': 'experiments/uniform/uniform_analysis/uniform_experiments_summary_averaged.csv'
}
```

**Analisi Eseguite**:
| Metodo | Analisi |
|--------|---------|
| `analyze_dataset_size_impact()` | Impatto mini → small → original |
| `analyze_noise_type_impact()` | Degradazione per tipo di rumore vs clean |
| `analyze_noise_intensity_impact()` | Curva accuracy vs intensità per famiglia |
| `analyze_k_features_impact()` | Effetto numero features selezionate |
| `generate_summary_report()` | Report markdown finale |

**Output Grafici** (4 plot multi-pannello):
- `analysis_dataset_size_comprehensive.png`
- `analysis_noise_type_comprehensive.png`
- `analysis_noise_intensity_comprehensive.png`
- `analysis_k_features_comprehensive.png`

---

### 7. Robustness Analysis (`article/data_analysis/analysis_1/scripts/robustness_analysis.py`)

**Scopo**: Analisi statistica rigorosa per paper scientifico (WST vs Advanced Stats).

**Configurazione Rumore**:
```python
NOISE_CONFIGS = {
    "clean": ["clean_0"],
    "gaussian": ["gaussian_30", "gaussian_50"],
    "poisson": ["poisson_40", "poisson_60"],
    "saltpepper": ["saltpepper_5", "saltpepper_15", "saltpepper_25"],
    "speckle": ["speckle_15", "speckle_35", "speckle_55"],
    "uniform": ["uniform_10", "uniform_25", "uniform_40"]
}
```

**Funzioni Statistiche**:
| Funzione | Descrizione |
|----------|-------------|
| `load_all_data()` | Carica e aggrega esperimenti per location |
| `compute_pairwise_deltas()` | Δ = WST - AdvStats, Δ = Hybrid - AdvStats |
| `perform_statistical_tests()` | Shapiro-Wilk + t-test/Wilcoxon + FDR correction |
| `analyze_noise_robustness()` | Regressione lineare accuracy vs intensità (slope) |
| `analyze_data_scarcity()` | Retention % relativa a dataset "original" |

**Visualizzazioni Generate** (7 plot):
1. `performance_by_noise_type.png` - 6 pannelli per tipo rumore
2. `noise_robustness_slopes.png` - Bar chart pendenze regressione
3. `data_scarcity_retention.png` - Retention % per dataset size
4. `delta_heatmap.png` - Heatmap Δ Macro-F1 vs Advanced Stats
5. `performance_distribution.png` - Boxplot accuracy e F1
6. `k_features_impact.png` - Accuracy vs K per metodo
7. `statistical_significance_matrix.png` - Cohen's d + p-values FDR

**Output Principale**: `analysis_report.md` con grafici embedded

---

### 8. Visualizzazione Features (`article/features_visual/visualize_features.py`)

**Scopo**: Genera visualizzazioni didattiche per spiegare Advanced Stats vs WST.

**Pattern di Test Generati**:
| Pattern | Funzione | Scopo |
|---------|----------|-------|
| `gradient_horizontal` | `generate_gradient_horizontal()` | Test sensibilità direzionale |
| `gradient_vertical` | `generate_gradient_vertical()` | Test sensibilità direzionale |
| `checkerboard` | `generate_checkerboard()` | Pattern ad alta frequenza |
| `circles` | `generate_circles()` | Pattern concentrico |
| `texture` | `generate_texture()` | Rumore random |
| `vertical_texture` | `generate_vertical_texture()` | Texture direzionale |
| `edge` | `generate_edge()` | Bordi netti |

**Visualizzazioni per Pattern**:
- `*_original.png` - Immagine test
- `*_advanced_stats.png` - Features statistiche + gradienti + istogramma
- `*_wst.png` - Coefficienti WST a diverse scale/orientazioni
- `*_comparison.png` - Dashboard comparativo

**Output Finale**: `overall_comparison.png` con:
- Dimensionalità (18 vs ~162 features)
- Costo computazionale
- Radar chart caratteristiche qualitative
- Tabella riassuntiva

---

## Pipeline di Estrazione Features

### Advanced Statistics (54 features per immagine RGB)

Per ogni canale RGB (3 × 18 = 54 features):

| Categoria | Features | Indici |
|-----------|----------|--------|
| **Base** | mean, std, var, min, max, range | 0-5 |
| **Shape** | skewness, kurtosis, coefficient of variation | 6-8 |
| **Percentili** | P10, P25, P50, P75, P90, IQR | 9-14 |
| **Dispersione** | MAD (Mean Absolute Deviation) | 15 |
| **Texture** | gradient_mean (Sobel), edge_density (Laplace) | 16-17 |

### WST Features (~486 features per immagine RGB)

**Configurazione**: `Scattering2D(J=2, L=8, shape=(H,W))`

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| J | 2 | Numero di scale |
| L | 8 | Numero di orientazioni (0°, 22.5°, 45°, ..., 157.5°) |
| Coefficienti | ~81 | Per canale (1 low-pass + 8×J first-order + second-order) |
| Statistiche | mean, std | Per ogni coefficiente |

**Totale**: 3 canali × 81 coeff × 2 stats ≈ **486 features**

### Hybrid

Concatenazione: `[Advanced Stats (54)] + [WST (~486)]` ≈ **540 features**

---

## Dipendenze Principali

```python
# Core ML
sklearn (RandomForestClassifier, SelectKBest, StandardScaler, StratifiedKFold)
numpy, pandas

# WST
kymatio (Scattering2D)
torch

# Image Processing
PIL (Image)
scipy.ndimage (sobel, laplace)
scipy.stats (skew, kurtosis, shapiro, ttest_rel, wilcoxon, linregress)

# Visualization
matplotlib, seaborn

# Statistical Analysis
statsmodels.stats.multitest (multipletests - FDR correction)

# Serialization
joblib
```

---

## Configurazione Modello RF

```python
RandomForestClassifier(
    n_estimators=n_est,       # 3 (mini), 10 (small), 50 (original)
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

---

## Workflow Sperimentale Completo

```
1. Dataset Clean (RGB images)
       ↓
2. add_noise.py → Dataset con rumore (varie intensità/tipi)
       ↓
3. train_and_save_model.py → Modelli per ogni combinazione:
   (area × dataset_size × feature_method × k_features × noise_type × noise_intensity)
       ↓
4. inference.py → Valutazione su test set
       ↓
5. analyze_*_experiments.py → Report per tipo rumore
       ↓
6. comprehensive_noise_analysis.py → Analisi comparativa globale
       ↓
7. robustness_analysis.py → Analisi statistica per paper
       ↓
8. visualize_features.py → Figure esplicative per paper
```

---

## Aree Geografiche

| Area | Descrizione | Caratteristiche |
|------|-------------|-----------------|
| `assatigue` | Isola di Assateague | Vegetazione costiera, dune |
| `popolar` | Foresta di pioppi | Canopy densa, ombre |
| `sunset` | Area critica | Condizioni di illuminazione complesse |

---

## Key Findings (dal codice di analisi)

### Robustezza al Rumore
- **WST** mostra maggiore robustezza, specialmente con rumore gaussiano ad alta intensità
- Pendenza regressione accuracy vs intensità meno negativa per WST
- **Hybrid** bilancia robustezza WST con diversità di features
- **Advanced Stats** più vulnerabile a noise additivo

### Effetto Dataset Size
- Dataset **original** performa ~5-10% meglio di **mini**
- Retention % più alta per WST su dataset ridotti

### Selezione Features
- k=10-20 features tipicamente ottimali
- Diminishing returns oltre k=10 per Advanced Stats
- WST beneficia di più features (k=20)

### Significatività Statistica
- Test Shapiro-Wilk per normalità
- Paired t-test o Wilcoxon signed-rank
- FDR correction (Benjamini-Hochberg)
- Cohen's d per effect size

---

## Comandi Rapidi

```bash
# Training completo
python scripts/experiments/train_and_save_model.py \
    datasets/dataset_rgb assatigue wst 10 output/model

# Aggiunta rumore
python scripts/add_noise.py --noise-type gaussian --intensity 30

# Inferenza
python scripts/inference.py --model-dir output/model --dataset-type original

# Analisi gaussiano
python scripts/analysis/analyze_gaussian_experiments.py

# Analisi completa
python analisi/comprehensive_noise_analysis.py

# Visualizzazione features
python article/features_visual/visualize_features.py
```
