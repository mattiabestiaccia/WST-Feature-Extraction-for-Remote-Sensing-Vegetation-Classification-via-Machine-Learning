# Random Forest Classification for Land Cover Analysis

A comprehensive machine learning project for land cover classification using Random Forest algorithms with advanced feature engineering, including RGB statistics, Wavelet Scattering Transform (WST), and hybrid approaches.

## 🎯 Project Overview

This project focuses on classifying land cover types from aerial imagery using multiple feature extraction techniques:

- **RGB Statistical Features**: Mean and standard deviation of color channels
- **Wavelet Scattering Transform (WST)**: Advanced texture and pattern analysis
- **Hybrid Approaches**: Combination of RGB and WST features
- **Noise Robustness**: Evaluation across different noise conditions

### Land Cover Classes
- **Assatigue**: Agricultural/farmland areas
- **Popolar**: Forest/woodland areas  
- **Sunset**: Urban/built-up areas

## 🏗️ Project Structure

```
random_forest/
├── 📁 datasets/                     # Image datasets (NOT in git - large files)
│   ├── dataset_rgb_clean/           # Clean images (mini, small, original, augmented)
│   ├── dataset_rgb_gaussian_30/     # Gaussian noise σ=30
│   ├── dataset_rgb_gaussian_50/     # Gaussian noise σ=50
│   ├── dataset_rgb_poisson_60/      # Poisson noise scale=60
│   └── [other noise variants]/      # Additional noise conditions
│
├── 📁 experiments/                  # Experiment results organized by noise type
│   ├── rgb_clean/                   # Clean dataset experiments (mini, small, original)
│   ├── gaussian/                    # Gaussian noise experiments
│   ├── poisson/                     # Poisson noise experiments
│   ├── s&p/                         # Salt & pepper noise experiments
│   ├── speckle/                     # Speckle noise experiments
│   ├── uniform/                     # Uniform noise experiments
│   └── README.md                    # Experiment infrastructure documentation
│
├── 📁 scripts/                      # Python scripts and shell utilities
│   ├── experiments/                 # Training and batch execution scripts
│   │   ├── train_and_save_model.py  # Main training script
│   │   └── run_*_experiments.sh     # Batch runners per noise type
│   ├── analysis/                    # Results analysis scripts
│   │   └── analyze_*_experiments.py # Analysis per noise type
│   ├── add_noise.py                 # Noise injection utility
│   ├── inference.py                 # Model inference script
│   └── generate_noise_datasets.sh   # Batch noise generation
│
├── 📁 paper/                        # Research paper materials
│   ├── random forest matherials and methods chapter.md
│   └── images/                      # Figures for publication
│
├── 📁 inference_k5_popolar_results/ # Sample inference validation results
│
├── 📓 Poplar.ipynb                  # Main analysis notebook
├── 📋 requirements.txt              # Python dependencies
├── 📋 claude.md                     # AI assistant context file
└── 📋 README.md                     # This file
```


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster WST computation)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/mattiabestiaccia/random_forest.git
cd random_forest
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Train a Model
```bash
python scripts/train_and_save_model.py dataset_path area_name feature_method k_features output_dir [options]
```

**Required Arguments:**
- `dataset_path`: Path to the dataset directory
- `area_name`: Land cover area (`assatigue`, `popolar`, `sunset`)
- `feature_method`: Feature extraction method (`advanced_stats`, `wst`, `hybrid`)
- `k_features`: Number of features to select (`2`, `5`, `10`, `20`)
- `output_dir`: Output directory for model and results

**Optional Arguments:**
- `--n_estimators`: Number of trees in the forest (default: 50)
- `--test_size`: Test set size fraction (default: 0.2)
- `--random_state`: Random seed for reproducibility (default: 42)
- `--cv_folds`: Number of cross-validation folds (default: 5)

**Examples:**
```bash
# Basic usage with advanced statistics
python scripts/train_and_save_model.py /path/to/dataset popolar advanced_stats 5 ./output

# Using WST features with custom parameters
python scripts/train_and_save_model.py /path/to/dataset assatigue wst 10 ./output --n_estimators 100 --test_size 0.3

# Hybrid features with all parameters
python scripts/train_and_save_model.py /path/to/dataset sunset hybrid 20 ./output --n_estimators 200 --test_size 0.25 --random_state 123 --cv_folds 10
```

This script will:
- Load the dataset from the specified path
- Extract features using the chosen method
- Train a Random Forest classifier
- Save the trained model and all artifacts

#### 2. Run Inference
```bash
python scripts/inference_on_dataset.py
```
Performs classification on new datasets and generates:
- Confusion matrices
- Classification reports
- Performance metrics

#### 3. Analyze Results
```bash
python scripts/analyze_experiments.py
```
Comprehensive analysis of experimental results across different conditions.

#### 4. Generate Visualizations
```bash
python scripts/create_visualizations.py
```
Creates plots and charts for result interpretation.

#### 5. Run Batch Experiments
```bash
cd experiments_organized
./run_all_experiments.sh [base_dataset_dir]
```

**Arguments:**
- `base_dataset_dir`: Base directory containing different dataset versions (optional, defaults to `/home/brusc/Projects/random_forest/dataset_rgb`)

**What it does:**
Executes all experiment combinations systematically:
- **Dataset sizes**: mini, small, original (augmented not used - see note below)
- **Noise conditions**: clean, gaussian30, gaussian50, poisson60, salt_pepper25, speckle55, uniform40
- **Land cover areas**: assatigue, popolar, sunset
- **Feature methods**: advanced_stats, wst, hybrid
- **K-best values**: 2, 5, 10, 20

**Note on Augmented Dataset**: The augmented dataset (~400 images/class) exists but was excluded from systematic experiments after preliminary tests showed minimal accuracy improvement beyond the original dataset (~200 images/class). This finding indicates the original dataset size represents a practical plateau in the accuracy vs. dataset size curve for this task.

**Example:**
```bash
# Use default dataset directory
cd experiments_organized
./run_all_experiments.sh

# Use custom dataset directory
cd experiments_organized
./run_all_experiments.sh /custom/path/to/datasets
```

The script generates:
- Organized JSON reports in structured directories
- Detailed experiment logs
- Summary reports with success/failure statistics
- CSV files with all results for analysis

**Note:** This script requires the `rgb_rf_generalized.py` script to be present in the experiments_organized directory.

## 📊 Features

### Experimental Infrastructure

The project includes a comprehensive experimental infrastructure located in `experiments_organized/`:

- **`run_all_experiments.sh`**: Automated batch script that executes all possible experiment combinations
- **`structure.md`**: Defines the hierarchical organization of experiments
- **758 JSON result files**: Organized across multiple noise conditions and parameters

The experiments were generated using the `run_all_experiments.sh` script, which systematically runs all combinations of:
- Dataset sizes (mini, small, original) - augmented excluded after showing plateau effect
- Land cover areas (assatigue, popolar, sunset)
- Feature methods (advanced_stats, WST, hybrid)
- K-best feature selection values (2, 5, 10, 20)
- Noise conditions (clean + 6 noise variants)

**Total**: 3 sizes × 3 areas × 3 methods × 4 k-values × 7 noise types = **756 experiments**

### Feature Extraction Methods

1. **RGB Statistical Features**
   - Mean and standard deviation per color channel
   - Simple but effective for color-based classification

2. **Wavelet Scattering Transform (WST)**
   - Multi-scale texture analysis
   - Translation-invariant features
   - Robust to deformations

3. **Hybrid Features**
   - Combination of RGB and WST
   - Leverages both color and texture information
   - Best performance across most scenarios

### Experimental Design

- **K-Best Feature Selection**: Tests with k=2, 5, 10, 20 features
- **Dataset Sizes**: Mini (~10 img/class), Small (~50), Original (~200)
  - *Note: Augmented (~400) excluded - negligible improvement beyond original*
- **Noise Conditions**: Clean, Gaussian, Poisson, Salt & Pepper, Speckle, Uniform
- **Cross-validation**: 5-fold stratified validation
- **Total Experiments**: 756 systematic combinations

## 📈 Results & Analysis

### Complete Analysis Suite

The project now includes a comprehensive analysis of 324 experiments with professional reports and visualizations. Use the complete analysis script:

```bash
# Activate virtual environment
source venv_rf/bin/activate

# Run complete analysis
python create_complete_analysis.py
```

This generates:
- **Statistical reports**: Comprehensive analysis with performance metrics
- **Qualitative analysis**: In-depth interpretation of robustness findings  
- **Executive summary**: Key findings and recommendations
- **Data export**: Complete CSV dataset (186KB)
- **Visualizations**: 37 high-quality plots (6.6MB)

### Key Findings from Analysis

- **🏆 Best Method**: WST achieves highest accuracy (0.913 average)
- **🔊 Noise Impact**: 11.4% performance loss from clean to Gaussian σ=50
- **📊 Dataset Size Effect**: 7.4% improvement from mini to original
- **📈 Dataset Size Plateau**: Minimal improvement from original to augmented (plateau reached at ~200 img/class)
- **🛡️ Robustness**: WST shows lowest degradation under noise
- **⚖️ Feature Selection**: k=10-20 provides optimal performance

### Experimental Results Structure

```
experiments/
└── {noise_type}/                    # e.g., rgb_clean, gaussian, poisson
    └── {land_cover_area}/           # assatigue, popolar, sunset
        └── {dataset_size}/          # mini, small, original (augmented not used)
            └── {k_features}/        # k2, k5, k10, k20
                └── {method}/        # advanced_stats, wst, hybrid
                    └── {method}_k{k}_{size}_report.json
```

### Analysis Outputs

```
analysis/
├── comprehensive_report.md         # Complete statistical analysis
├── qualitative_analysis.md         # In-depth qualitative interpretation
├── analysis_summary.md            # Executive summary with key findings
├── experiments_summary.csv # Complete dataset export
├── comparisons/                    # High-level comparison plots (4 plots)
└── detailed/                      # Detailed analysis plots (33+ plots)
```

### Professional Visualizations

The analysis includes publication-ready plots:
- **Comparison plots**: Accuracy vs method/noise/dataset (averaged across dimensions)
- **Detailed plots**: Individual condition analysis (geographic-only averaging)
- **Statistical plots**: Performance distributions and confidence intervals
- **Clean formatting**: Professional styling suitable for research publications

## 🔬 Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Image normalization
   - Noise injection (if applicable)
   - Patch extraction and augmentation

2. **Feature Engineering**
   - RGB statistical moments
   - WST coefficient extraction
   - Feature standardization

3. **Model Training**
   - Random Forest with 100 estimators
   - Stratified train/test split (80/20)
   - Cross-validation for hyperparameter tuning

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrices
   - Feature importance analysis

### Dependencies

Key libraries used:
- `scikit-learn`: Machine learning algorithms
- `kymatio`: Wavelet Scattering Transform
- `opencv-python`: Image processing
- `numpy`, `pandas`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `torch`: Deep learning framework (for WST)

## 📝 Experiments

The project includes extensive experiments across different conditions, with results stored in `experiments_organized/`:

### Noise Types
- **Clean**: Original images without noise 
- **Gaussian**
- **Poisson**
- **Salt & Pepper**
- **Speckle**
- **Uniform**

### Feature Selection
- **k=2**: Minimal feature set
- **k=5**: Small feature set
- **k=10**: Medium feature set
- **k=20**: Large feature set

### Dataset Sizes Used in Experiments
- **Mini**: ~10 images/class - Rapid prototyping, extreme data scarcity testing
- **Small**: ~50 images/class - Moderate data scarcity scenarios
- **Original**: ~200 images/class - Full dataset, sufficient for effective training
- **Augmented**: ~400 images/class - Available but **not used** (showed negligible improvement over original)

**Research Finding**: The accuracy vs. dataset size curve plateaus at the original dataset size (~200 images/class), indicating this is a practical threshold for data collection in similar land cover classification tasks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Kymatio**: For the excellent Wavelet Scattering Transform implementation
- **scikit-learn**: For robust machine learning tools
- **OpenCV**: For comprehensive image processing capabilities

## 📧 Contact

For questions or collaborations, please contact:
- **Author**: Mattia Bestiaccia
- **Email**: mattiabestiaccia@gmail.com
- **GitHub**: [@mattiabestiaccia](https://github.com/mattiabestiaccia)

---

*This project demonstrates advanced machine learning techniques for remote sensing applications, with emphasis on robustness and feature engineering.*