# RGB Random Forest Experiments

This directory contains scripts for running comprehensive RGB Random Forest experiments with different feature extraction methods and dataset configurations.

## Scripts Overview

### 1. `rgb_rf_generalized.py` - Main Experiment Script

A unified Python script that replaces all individual experiment scripts.

**Usage:**
```bash
python rgb_rf_generalized.py <dataset_type> <area_name> <feature_method> <k_features> [base_dataset_dir]
```

**Parameters:**
- `dataset_type`: mini, small, original (augmented available but not used in main experiments)
- `area_name`: assatigue, popolar, sunset
- `feature_method`: wst, advanced_stats, hybrid
- `k_features`: 2, 5, 10, 20 (30, 50 supported but not used in main experiments)
- `base_dataset_dir`: (optional) custom path to dataset_rgb

**Feature Methods:**
- `wst`: Wavelet Scattering Transform + basic features
- `advanced_stats`: Advanced statistical features (54 features)
- `hybrid`: Combination of WST + advanced stats

**Examples:**
```bash
# WST with 20 features
python rgb_rf_generalized.py mini assatigue wst 20

# Advanced stats with 10 features
python rgb_rf_generalized.py small popolar advanced_stats 10

# Hybrid with 5 features
python rgb_rf_generalized.py original sunset hybrid 5

# Custom dataset path
python rgb_rf_generalized.py augmented assatigue wst 20 /custom/path/to/dataset_rgb
```

### 2. `run_all_experiments.sh` - Complete Experiment Runner

Runs all possible experiment combinations systematically.

**Usage:**
```bash
./run_all_experiments.sh [base_dataset_dir]
```

**Features:**
- Runs all combinations: 3 datasets × 3 areas × 3 features × 4 k-values = 108 experiments per noise condition
- Colored logging with timestamps
- Individual log files for each experiment
- Automatic summary generation
- Progress tracking
- Error handling and recovery

**Note**: Original experiments used mini, small, and original datasets only. The augmented dataset was excluded after preliminary tests showed minimal accuracy improvement beyond the original dataset size.

**Output:**
- `experiment_logs/`: Individual experiment logs
- `all_results/experiment_summary.txt`: Human-readable summary
- `all_results/experiment_results.csv`: CSV results for analysis

### 3. `run_subset_experiments.sh` - Selective Experiment Runner

Runs specific subsets of experiments based on filters.

**Usage:**
```bash
./run_subset_experiments.sh [OPTIONS] [base_dataset_dir]
```

**Options:**
- `-d, --datasets`: Comma-separated datasets (mini,small,original,augmented)
- `-a, --areas`: Comma-separated areas (assatigue,popolar,sunset)
- `-f, --features`: Comma-separated features (wst,advanced_stats,hybrid)
- `-k, --k-values`: Comma-separated k values (2,5,10,20,30,50)
- `-h, --help`: Show help message

**Examples:**
```bash
# Run only mini and small datasets with WST and hybrid features
./run_subset_experiments.sh -d mini,small -f wst,hybrid

# Run only assatigue area with all features and k=10,20
./run_subset_experiments.sh -a assatigue -k 10,20

# Run original dataset with advanced_stats on sunset area
./run_subset_experiments.sh -d original -a sunset -f advanced_stats

# Multiple selections
./run_subset_experiments.sh -d mini,small -a assatigue,popolar -f wst,hybrid -k 10,20
```

**Output:**
- `subset_experiment_logs/`: Individual experiment logs
- `subset_results/subset_experiment_summary.txt`: Human-readable summary
- `subset_results/subset_experiment_results.csv`: CSV results for analysis

## Experiment Matrix

The experiment matrix used in the main study:

| Parameter | Values | Notes |
|-----------|---------|-------|
| Dataset Types | mini, small, original | augmented excluded (plateau effect) |
| Areas | assatigue, popolar, sunset | All three geographic locations |
| Feature Methods | wst, advanced_stats, hybrid | All three extraction methods |
| K Values | 2, 5, 10, 20 | 30, 50 available but not in main suite |

**Total Combinations:** 3 × 3 × 3 × 4 = 108 experiments per noise condition

### Why Augmented Was Excluded

Preliminary experiments revealed:
1. **Accuracy improvement trajectory**: mini (baseline) → small (+improvement) → original (+improvement) → augmented (+minimal/none)
2. **Plateau effect**: The original dataset (~200 images/class) already provides sufficient training data
3. **Research focus**: Study emphasizes robustness under **data scarcity** and **noise corruption**, not maximum performance with abundant data
4. **Computational efficiency**: Excluding augmented reduces experiments by ~25% without loss of research insights

This finding is scientifically valuable as it identifies the point of diminishing returns in the dataset size vs. accuracy curve for this classification task.

## Feature Extraction Methods

### WST (Wavelet Scattering Transform)
- Basic features: mean, std for each RGB channel (6 features)
- WST features: Wavelet scattering coefficients (mean/std) for each channel
- Total features: ~60-80 depending on image size

### Advanced Statistics
- 18 statistical features per RGB channel (54 total):
  - Basic: mean, std, var, min, max, range
  - Shape: skewness, kurtosis, coefficient of variation
  - Percentiles: 10th, 25th, 50th, 75th, 90th, IQR
  - Robust: MAD (Median Absolute Deviation)
  - Spatial: gradient mean, edge density

### Hybrid
- Combination of advanced statistics + WST features
- Most comprehensive feature set (~120-140 features)

## Output Structure

Each experiment generates:

1. **JSON Report**: Detailed results with selected features and scores
2. **Log Files**: Console output and error messages
3. **Summary Reports**: Aggregated results across experiments
4. **CSV Files**: Machine-readable results for further analysis

## Directory Structure

```
experiments/
├── rgb_rf_generalized.py          # Main experiment script
├── run_all_experiments.sh         # Complete experiment runner
├── run_subset_experiments.sh      # Selective experiment runner
├── README.md                      # This file
├── experiment_logs/               # Full experiment logs
├── all_results/                   # Complete experiment results
├── subset_experiment_logs/        # Subset experiment logs
├── subset_results/               # Subset experiment results
└── experiments/                  # Individual experiment outputs
    ├── wst_mini_k20_assatigue/
    ├── advanced_stats_small_k10_popolar/
    └── hybrid_original_k5_sunset/
```

## Tips for Usage

1. **Start Small**: Test with subset experiments before running all combinations
2. **Monitor Resources**: 216 experiments can take considerable time
3. **Check Logs**: Individual experiment logs help debug issues
4. **Incremental Runs**: Use subset script to fill gaps in failed experiments
5. **Analysis**: Use generated CSV files for comparative analysis

## Error Handling

- Individual experiment failures don't stop the batch
- Error logs are saved separately for each experiment
- Summary reports indicate success/failure status
- Scripts can be interrupted and resumed

## Requirements

- Python 3.x with required packages:
  - numpy, pandas, torch, PIL, tqdm
  - kymatio, sklearn, scipy
- Sufficient disk space for logs and results
- Access to the dataset directory

## Example Workflow

1. **Test a single experiment:**
   ```bash
   python rgb_rf_generalized.py mini assatigue wst 10
   ```

2. **Run a small subset:**
   ```bash
   ./run_subset_experiments.sh -d mini -a assatigue -f wst -k 10,20
   ```

3. **Run all experiments:**
   ```bash
   ./run_all_experiments.sh
   ```

4. **Analyze results:**
   ```bash
   # View summary
   cat all_results/experiment_summary.txt
   
   # Analyze CSV data
   python -c "import pandas as pd; df = pd.read_csv('all_results/experiment_results.csv'); print(df.groupby('FeatureMethod')['Accuracy'].mean())"
   ```