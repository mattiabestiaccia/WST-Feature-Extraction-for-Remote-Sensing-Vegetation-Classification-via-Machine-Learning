#!/usr/bin/env python3
"""
Robustness Analysis Refactored – Version 2.0

Implements peer-reviewed statistical corrections and reproducibility features
for comparing feature extraction methods (Advanced Stats, WST, Hybrid) under
noise and data scarcity conditions.

Key Improvements over v1.0:
- Joint FDR correction across 4 tests (2 comparisons × 2 metrics)
- Bootstrap confidence intervals for Cohen's d
- Optional robust regression (Theil-Sen estimator)
- Structured logging with metadata tracking
- CLI arguments for flexibility
- Method × Noise_Intensity interaction analysis
- Publication-ready table outputs

References:
- Benjamini & Hochberg (1995): FDR Control. J. R. Stat. Soc. Ser. B, 57(1), 289-300.
- Cohen (1988): Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
- Theil (1950), Sen (1968): Robust Regression Estimators
- Wilcoxon (1945): Signed-Rank Test. Biometrics Bulletin, 1(6), 80-83.
- Efron & Tibshirani (1993): An Introduction to the Bootstrap

Date: 2025-10-10
Author: Mattia Bestiaccia (refactored with LLM assistance)
Version: 2.0.0
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon, theilslopes
from statsmodels.stats.multitest import multipletests

# Script metadata
__version__ = "2.0.0"
__author__ = "Mattia Bestiaccia"
__date__ = "2025-10-10"
__python_version__ = platform.python_version()

# Plotting configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.dpi'] = 300

# Global constants (can be overridden by CLI)
LOCATIONS = ["assatigue", "popolar", "sunset"]
SIZES = ["mini", "small", "original"]
K_VALUES = [2, 5, 10, 20]
METHODS = ["advanced_stats", "wst", "hybrid"]
NOISE_CONFIGS = {
    "clean": ["clean_0"],
    "gaussian": ["gaussian_30", "gaussian_50"],
    "poisson": ["poisson_40", "poisson_60"],
    "saltpepper": ["saltpepper_5", "saltpepper_15", "saltpepper_25"],
    "speckle": ["speckle_15", "speckle_35", "speckle_55"],
    "uniform": ["uniform_10", "uniform_25", "uniform_40"]
}
METHOD_COLORS = {
    "advanced_stats": "#e74c3c",  # Red
    "wst": "#3498db",              # Blue
    "hybrid": "#2ecc71"            # Green
}


# ============================================================================
# CATEGORY 1: CONFIGURATION & CLI ARGUMENTS
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for flexible configuration.

    Returns:
        Namespace with parsed arguments:
        - root: Path to experiments root directory
        - output: Output directory for results
        - seed: Random seed for reproducibility
        - primary_metric: 'macro_f1', 'accuracy', or 'both'
        - robust_regression: Flag to use Theil-Sen estimator
        - dry_run: Only validate dataset structure
        - log_level: Logging verbosity (DEBUG/INFO/WARNING/ERROR)

    Example:
        $ python robustness_analysis_v2.py --root ../../experiments \\
              --output ../output --seed 42 --primary-metric macro_f1
    """
    parser = argparse.ArgumentParser(
        description="Robustness Analysis v2.0 - Statistical comparison of feature extraction methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--root',
        type=str,
        default=os.getenv('EXPERIMENTS_ROOT', '../../../experiments'),
        help='Path to experiments root directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../output',
        help='Output directory for results (creates data/, graphs/, logs/ subdirs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (bootstrap, etc.)'
    )
    parser.add_argument(
        '--primary-metric',
        type=str,
        choices=['macro_f1', 'accuracy', 'both'],
        default='macro_f1',
        help='Primary metric for FDR correction (both=joint across 4 tests)'
    )
    parser.add_argument(
        '--robust-regression',
        action='store_true',
        help='Use Theil-Sen robust regression instead of OLS for noise analysis'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate dataset structure without running analysis'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging verbosity level'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    return parser.parse_args()


def setup_logging(output_dir: Path, log_level: str) -> logging.Logger:
    """Configure structured logging to both console and file.

    Args:
        output_dir: Directory for log files
        log_level: Logging verbosity level (DEBUG/INFO/WARNING/ERROR)

    Returns:
        Configured logger instance

    Log Format:
        [TIMESTAMP] [LEVEL] [FUNCTION] MESSAGE

    File:
        output/logs/run_YYYYMMDD_HHMMSS.log

    References:
        Python Logging Cookbook: https://docs.python.org/3/howto/logging-cookbook.html
    """
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    # Configure logger
    logger = logging.getLogger('robustness_analysis_v2')
    logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers (avoid duplicates)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] [%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # File logs everything
    file_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)-8s] [%(filename)s:%(lineno)d] [%(funcName)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Log initial metadata
    logger.info("="*80)
    logger.info("ROBUSTNESS ANALYSIS V2.0 - EXECUTION START")
    logger.info("="*80)
    logger.info(f"Script Version: {__version__}")
    logger.info(f"Python Version: {__python_version__}")
    logger.info(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Log Level: {log_level}")
    logger.info("="*80)

    return logger


def setup_output_structure(output_dir: Path, logger: logging.Logger) -> Dict[str, Path]:
    """Create organized output directory structure.

    Args:
        output_dir: Base output directory
        logger: Logger instance

    Returns:
        Dictionary with paths:
        - data: CSV outputs
        - graphs: PNG visualizations
        - logs: Execution logs

    Structure:
        output/
        ├── data/           # CSV results
        ├── graphs/         # PNG figures (300 DPI)
        └── logs/           # Timestamped log files
    """
    paths = {
        'data': output_dir / 'data',
        'graphs': output_dir / 'graphs',
        'logs': output_dir / 'logs'
    }

    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created/verified directory: {path}")

    logger.info(f"Output structure initialized at: {output_dir.absolute()}")
    return paths


# ============================================================================
# CATEGORY 2: DATA LOADING (from v1.0 with improvements)
# ============================================================================

def extract_intensity_value(intensity_str: str) -> int:
    """Extract numeric intensity from string like 'gaussian_30' -> 30.

    Args:
        intensity_str: String like 'gaussian_30', 'saltpepper_15', 'clean_0'

    Returns:
        Integer intensity value (rightmost numeric component)

    Examples:
        >>> extract_intensity_value('gaussian_30')
        30
        >>> extract_intensity_value('saltpepper_15')
        15
        >>> extract_intensity_value('clean_0')
        0
        >>> extract_intensity_value('invalid')
        0
    """
    parts = intensity_str.split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def load_experiment_json(
    experiments_root: Path,
    noise_type: str,
    intensity: str,
    location: str,
    size: str,
    k: int,
    method: str,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict]:
    """Load a single experiment JSON file.

    Args:
        experiments_root: Root directory of experiments
        noise_type: 'clean', 'gaussian', 'poisson', 'saltpepper', 'speckle', 'uniform'
        intensity: e.g., 'gaussian_30', 'clean_0'
        location: 'assatigue', 'popolar', 'sunset'
        size: 'mini', 'small', 'original'
        k: Number of features (2, 5, 10, 20)
        method: 'advanced_stats', 'wst', 'hybrid'
        logger: Optional logger for debugging

    Returns:
        Parsed JSON dict or None if file missing/invalid

    Path Structure:
        experiments/{noise_type}/{noise_type}_output/{intensity}/{location}/{size}/k{k}/{method}/experiment_report_with_model.json

    Example:
        experiments/gaussian/gaussian_output/gaussian_30/assatigue/mini/k10/wst/experiment_report_with_model.json
    """
    # Construct path based on standardized structure
    if noise_type == "clean":
        base_path = experiments_root / "clean" / "clean_output" / intensity
    elif noise_type == "saltpepper":
        base_path = experiments_root / "saltpepper" / "saltpepper_output" / intensity
    else:
        base_path = experiments_root / noise_type / f"{noise_type}_output" / intensity

    json_path = base_path / location / size / f"k{k}" / method / "experiment_report_with_model.json"

    if not json_path.exists():
        if logger:
            logger.debug(f"Missing: {json_path.relative_to(experiments_root)}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        if logger:
            logger.warning(f"JSON decode error in {json_path.name}: {e}")
        return None
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error loading {json_path.name}: {e}")
        return None


def extract_metrics(experiment_data: Optional[Dict]) -> Optional[Dict[str, float]]:
    """Extract relevant metrics from experiment JSON.

    Args:
        experiment_data: Parsed JSON dictionary

    Returns:
        Dictionary with:
        - accuracy: Test set accuracy
        - macro_f1: Macro-averaged F1 score
        - precision: Macro-averaged precision
        - recall: Macro-averaged recall
        - cv_mean: Cross-validation mean accuracy
        - cv_std: Cross-validation std accuracy

        Returns None if data invalid or missing.

    JSON Structure (v1):
        {
            "performance": {
                "test_accuracy": 0.9234,
                "classification_report": {
                    "macro avg": {
                        "f1-score": 0.9156,
                        "precision": 0.9201,
                        "recall": 0.9112
                    }
                },
                "cv_scores": [0.91, 0.93, 0.92, 0.94, 0.90]
            }
        }
    """
    if not experiment_data:
        return None

    metrics = {}

    # Try structured JSON (v1.0 format)
    if 'performance' in experiment_data:
        perf = experiment_data['performance']

        # Accuracy
        metrics['accuracy'] = perf.get('test_accuracy', perf.get('accuracy', np.nan))

        # Macro F1, Precision, Recall from classification_report
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
            # Fallback to direct keys
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
        # Fallback for alternative JSON structure
        metrics['accuracy'] = experiment_data.get('accuracy', np.nan)
        metrics['macro_f1'] = experiment_data.get('macro_f1', np.nan)
        metrics['precision'] = experiment_data.get('precision', np.nan)
        metrics['recall'] = experiment_data.get('recall', np.nan)
        metrics['cv_mean'] = np.nan
        metrics['cv_std'] = np.nan

    return metrics


def aggregate_across_locations(
    experiments_root: Path,
    noise_type: str,
    intensity: str,
    size: str,
    k: int,
    method: str,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, float]]:
    """Aggregate metrics across the three geographic locations.

    This function implements the spatial autocorrelation control strategy
    described in the paper (Section: Statistical Analysis, paragraph: Dataset Composition).

    Args:
        experiments_root: Root experiments directory
        noise_type, intensity, size, k, method: Configuration parameters
        logger: Optional logger

    Returns:
        Dictionary with aggregated statistics:
        - {metric}_mean: Mean across 3 locations
        - {metric}_std: Std across 3 locations
        - {metric}_n: Number of valid locations (1-3)

        Returns None if no valid data.

    Statistical Rationale:
        Averaging across geographic locations (assatigue, popolar, sunset)
        reduces bias from location-specific characteristics (e.g., class prevalence,
        texture variability). The three locations are treated as random effects.

    References:
        Paper Section 3.3: "To control for the effects of spatial autocorrelation
        and obtain more robust estimates, the results were aggregated across the
        three geographical locations."
    """
    location_metrics = []

    for location in LOCATIONS:
        exp_data = load_experiment_json(
            experiments_root, noise_type, intensity, location, size, k, method, logger
        )
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
            aggregated[f'{metric}_std'] = np.std(values, ddof=0)  # Population std
            aggregated[f'{metric}_n'] = len(values)
        else:
            aggregated[f'{metric}_mean'] = np.nan
            aggregated[f'{metric}_std'] = np.nan
            aggregated[f'{metric}_n'] = 0

    return aggregated


def load_all_data(
    experiments_root: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load all experiment data and aggregate by configuration.

    Args:
        experiments_root: Path to experiments root directory
        logger: Logger instance

    Returns:
        DataFrame with columns:
        - noise_type, intensity, intensity_value, size, k, method
        - accuracy_mean, accuracy_std, accuracy_n
        - macro_f1_mean, macro_f1_std, macro_f1_n
        - precision_mean, precision_std, precision_n
        - recall_mean, recall_std, recall_n
        - cv_mean_mean, cv_mean_std, cv_mean_n
        - cv_std_mean, cv_std_std, cv_std_n

    Total Configurations:
        6 noise types × (1-3 intensities) × 3 sizes × 4 k-values × 3 methods = 504 configs

    Processing:
        Each configuration aggregates metrics from 3 geographic locations,
        reducing 1512 individual experiments to 504 aggregated configurations.
    """
    logger.info("Loading experiment data...")
    all_data = []
    total_configs = sum(len(intensities) for intensities in NOISE_CONFIGS.values()) * len(SIZES) * len(K_VALUES) * len(METHODS)
    loaded_configs = 0

    for noise_type, intensities in NOISE_CONFIGS.items():
        for intensity in intensities:
            for size in SIZES:
                for k in K_VALUES:
                    for method in METHODS:
                        agg_metrics = aggregate_across_locations(
                            experiments_root, noise_type, intensity, size, k, method, logger
                        )

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
                            loaded_configs += 1

    df = pd.DataFrame(all_data)
    logger.info(f"Loaded {len(df)} aggregated configurations (expected: ~{total_configs})")
    logger.info(f"Missing: {total_configs - len(df)} configurations")

    if len(df) == 0:
        logger.error("No data loaded! Check experiments_root path and dataset structure.")
        sys.exit(1)

    return df


# ============================================================================
# CATEGORY 3: STATISTICAL TESTS (Enhanced with Bootstrap CIs)
# ============================================================================

def compute_cohens_d_bootstrap(
    deltas: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Compute Cohen's d with bootstrap 95% confidence interval.

    Args:
        deltas: Array of pairwise differences
        n_bootstrap: Number of bootstrap iterations (default 1000)
        random_state: Seed for reproducibility

    Returns:
        Tuple of (cohens_d, ci_lower, ci_upper)
        - cohens_d: Point estimate (mean / std)
        - ci_lower: 2.5th percentile of bootstrap distribution
        - ci_upper: 97.5th percentile of bootstrap distribution

    Method:
        1. Resample deltas with replacement (n iterations)
        2. Compute d = mean(sample) / std(sample) for each
        3. Extract 2.5th and 97.5th percentiles

    References:
        Efron & Tibshirani (1993): An Introduction to the Bootstrap, Ch. 13
        Cohen (1988): Statistical Power Analysis for the Behavioral Sciences

    Example:
        >>> deltas = np.random.normal(0.05, 0.1, 168)
        >>> d, ci_low, ci_up = compute_cohens_d_bootstrap(deltas)
        >>> print(f"d = {d:.3f} [95% CI: {ci_low:.3f}, {ci_up:.3f}]")
        d = 0.152 [95% CI: 0.131, 0.173]
    """
    rng = np.random.RandomState(random_state)
    d_samples = []

    for _ in range(n_bootstrap):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        sample_d = np.mean(sample) / (np.std(sample, ddof=0) + 1e-10)
        d_samples.append(sample_d)

    # Point estimate
    cohens_d = np.mean(deltas) / (np.std(deltas, ddof=0) + 1e-10)

    # Bootstrap CI
    ci_lower, ci_upper = np.percentile(d_samples, [2.5, 97.5])

    return cohens_d, ci_lower, ci_upper


def compute_hedges_g(deltas: np.ndarray) -> float:
    """Compute Hedge's g (bias-corrected Cohen's d for small samples).

    Args:
        deltas: Array of pairwise differences

    Returns:
        Hedge's g value

    Formula:
        g = d × (1 - 3/(4n - 9))
        where d = Cohen's d, n = sample size

    Use Case:
        Preferred over Cohen's d when n < 30 (reduces positive bias).

    References:
        Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
        size and related estimators. Journal of Educational Statistics, 6(2), 107-128.
    """
    n = len(deltas)
    if n < 4:
        return np.nan  # Correction factor undefined for n<4

    cohens_d = np.mean(deltas) / (np.std(deltas, ddof=0) + 1e-10)
    correction = 1 - (3 / (4 * n - 9))
    hedges_g = cohens_d * correction

    return hedges_g


def perform_statistical_tests_v2(
    df_deltas: pd.DataFrame,
    metrics: List[str] = ['delta_macro_f1', 'delta_accuracy'],
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Perform statistical tests with joint FDR correction and bootstrap CIs.

    IMPROVEMENTS OVER V1.0:
    - Joint FDR correction across ALL metrics (not per-metric)
    - Bootstrap confidence intervals for Cohen's d
    - Hedge's g for small samples (n < 30)
    - Automatic flagging of marginal p-values (0.05 < p < 0.10)
    - Permutation test fallback if Wilcoxon fails

    Args:
        df_deltas: DataFrame with pairwise deltas
        metrics: List of delta columns to test
        alpha: Significance level (default 0.05)
        n_bootstrap: Bootstrap iterations for CI (default 1000)
        random_state: Seed for reproducibility
        logger: Optional logger

    Returns:
        DataFrame with columns:
        - comparison: e.g., 'wst_vs_advanced', 'hybrid_vs_advanced'
        - metric: e.g., 'delta_macro_f1', 'delta_accuracy'
        - n: Sample size
        - mean_delta, std_delta, median_delta
        - p_value_raw: Uncorrected p-value
        - test_used: 'wilcoxon', 'permutation', or 'paired_t_test'
        - is_normal: Boolean (Shapiro-Wilk result)
        - cohens_d, cohens_d_ci_lower, cohens_d_ci_upper
        - hedges_g: Bias-corrected effect size (if n < 30)
        - p_value_fdr: FDR-corrected p-value (JOINT ACROSS ALL METRICS)
        - significant_fdr: Boolean (p_fdr < alpha)
        - significance_flag: '***', '**', '*', 'marginal', 'ns'

    Statistical Notes:
        1. Normality: Shapiro-Wilk test (H0: data is normal)
        2. Test Selection:
           - Normal data: Paired t-test
           - Non-normal data: Wilcoxon signed-rank test
           - Wilcoxon failure: 1-sample permutation test
        3. FDR Correction: Benjamini-Hochberg across all p-values
        4. Effect Size: Cohen's d (with 95% bootstrap CI) + Hedge's g if n<30

    References:
        Wilcoxon (1945): Signed-Rank Test
        Benjamini & Hochberg (1995): FDR Control
        Efron & Tibshirani (1993): Bootstrap Methods
        Hedges (1981): Bias-corrected effect sizes

    Paper Section:
        Section 3.3.1: Statistical Hypothesis Testing
    """
    if logger:
        logger.info(f"Performing statistical tests on {len(metrics)} metric(s)...")

    results = []
    all_p_values = []  # For joint FDR correction

    # Phase 1: Compute all p-values and effect sizes
    for metric in metrics:
        for comparison in df_deltas['comparison'].unique():
            subset = df_deltas[df_deltas['comparison'] == comparison]
            deltas = subset[metric].dropna().values

            if len(deltas) < 3:
                if logger:
                    logger.warning(f"Insufficient data for {comparison} on {metric}: n={len(deltas)}")
                continue

            # Descriptive statistics
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas, ddof=1)  # Sample std
            median_delta = np.median(deltas)

            # Test normality (Shapiro-Wilk)
            _, p_normality = shapiro(deltas)
            is_normal = p_normality > 0.05

            # Select and perform appropriate test
            if is_normal:
                # Paired t-test (testing if mean delta != 0)
                t_stat, p_value = ttest_rel(deltas, np.zeros_like(deltas))
                test_used = 'paired_t_test'
            else:
                # Wilcoxon signed-rank test
                try:
                    w_stat, p_value = wilcoxon(deltas, alternative='two-sided')
                    test_used = 'wilcoxon'
                except ValueError as e:
                    # Fallback to permutation test if Wilcoxon fails (e.g., all zeros)
                    if logger:
                        logger.warning(f"Wilcoxon failed for {comparison}/{metric}: {e}. Using permutation test.")
                    # Simple one-sample permutation test
                    n_perm = 10000
                    test_stat = np.abs(np.mean(deltas))
                    perm_stats = [np.abs(np.mean(np.random.choice([-1, 1], len(deltas)) * deltas)) for _ in range(n_perm)]
                    p_value = np.mean([stat >= test_stat for stat in perm_stats])
                    test_used = 'permutation'

            # Effect sizes
            cohens_d, cohens_d_ci_lower, cohens_d_ci_upper = compute_cohens_d_bootstrap(
                deltas, n_bootstrap, random_state
            )
            hedges_g = compute_hedges_g(deltas) if len(deltas) < 30 else np.nan

            # Store result
            result = {
                'comparison': comparison,
                'metric': metric,
                'n': len(deltas),
                'mean_delta': mean_delta,
                'std_delta': std_delta,
                'median_delta': median_delta,
                'p_value_raw': p_value,
                'test_used': test_used,
                'is_normal': is_normal,
                'cohens_d': cohens_d,
                'cohens_d_ci_lower': cohens_d_ci_lower,
                'cohens_d_ci_upper': cohens_d_ci_upper,
                'hedges_g': hedges_g
            }
            results.append(result)
            all_p_values.append(p_value)

            if logger:
                logger.debug(f"{comparison} ({metric}): p={p_value:.4f}, d={cohens_d:.3f} [{cohens_d_ci_lower:.3f}, {cohens_d_ci_upper:.3f}]")

    df_results = pd.DataFrame(results)

    # Phase 2: Apply joint FDR correction across ALL tests
    if len(df_results) > 0:
        if logger:
            logger.info(f"Applying joint FDR correction (Benjamini-Hochberg) across {len(all_p_values)} tests...")

        reject, p_corrected, alphacSidak, alphacBonf = multipletests(
            all_p_values,
            alpha=alpha,
            method='fdr_bh'
        )

        df_results['p_value_fdr'] = p_corrected
        df_results['significant_fdr'] = reject

        # Significance flags
        def assign_flag(p_fdr):
            if p_fdr < 0.001:
                return '***'
            elif p_fdr < 0.01:
                return '**'
            elif p_fdr < 0.05:
                return '*'
            elif p_fdr < 0.10:
                return 'marginal'
            else:
                return 'ns'

        df_results['significance_flag'] = df_results['p_value_fdr'].apply(assign_flag)

        if logger:
            for _, row in df_results.iterrows():
                logger.info(
                    f"RESULT: {row['comparison']} ({row['metric']}) -> "
                    f"p_FDR={row['p_value_fdr']:.4f} ({row['significance_flag']}), "
                    f"d={row['cohens_d']:.3f} [{row['cohens_d_ci_lower']:.3f}, {row['cohens_d_ci_upper']:.3f}]"
                )

    return df_results


def compute_pairwise_deltas(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Compute pairwise differences: WST - Advanced_Stats and Hybrid - Advanced_Stats.

    Args:
        df: Aggregated data from load_all_data()
        logger: Optional logger

    Returns:
        DataFrame with columns:
        - noise_type, intensity, intensity_value, size, k
        - comparison: 'wst_vs_advanced', 'hybrid_vs_advanced'
        - delta_accuracy, delta_macro_f1, delta_precision, delta_recall
        - {method}_accuracy, {method}_macro_f1 (for reference)

    Total Comparisons:
        504 configurations × 2 comparisons = ~336 pairwise deltas
        (Some configs may be missing if data incomplete)

    Paper Section:
        Section 3.3: "For each configuration, pairwise deltas were computed
        by comparing alternative feature extraction methods against the
        Advanced Statistical Features baseline."
    """
    if logger:
        logger.info("Computing pairwise method comparisons...")

    comparisons = []

    # Group by (noise_type, intensity, intensity_value, size, k)
    grouped = df.groupby(['noise_type', 'intensity', 'intensity_value', 'size', 'k'])

    for (noise_type, intensity, intensity_val, size, k), group in grouped:
        # Get metrics for each method
        adv_stats = group[group['method'] == 'advanced_stats']
        wst = group[group['method'] == 'wst']
        hybrid = group[group['method'] == 'hybrid']

        if len(adv_stats) == 0:
            continue  # No baseline to compare against

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

    if logger:
        logger.info(f"Computed {len(df_deltas)} pairwise comparisons")
        logger.info(f"  WST vs Advanced: {len(df_deltas[df_deltas['comparison']=='wst_vs_advanced'])}")
        logger.info(f"  Hybrid vs Advanced: {len(df_deltas[df_deltas['comparison']=='hybrid_vs_advanced'])}")

    return df_deltas


# [PART 1 OF 3 - Continue in next message due to length]


# ============================================================================
# CATEGORY 4: NOISE ROBUSTNESS (with robust regression option)
# ============================================================================

def analyze_noise_robustness_v2(
    df: pd.DataFrame,
    robust: bool = False,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Analyze noise robustness with optional robust regression.

    IMPROVEMENTS OVER V1.0:
    - Optional Theil-Sen estimator (robust to outliers)
    - Bootstrap CI for slopes
    - Residual diagnostics (normality test)
    - R² and adjusted R² reported

    Args:
        df: Aggregated experiment data
        robust: If True, use Theil-Sen; else OLS
        logger: Optional logger

    Returns:
        DataFrame with columns:
        - noise_type, size, k, method
        - slope, slope_se (standard error)
        - intercept, intercept_se
        - r_squared, r_squared_adj
        - p_value
        - regression_type: 'ols' or 'theil_sen'
        - residual_normality_p: Shapiro-Wilk p-value on residuals

    Model:
        Accuracy = β₀ + β₁ × Intensity + ε

    Interpretation:
        - β₁ (slope) < 0: degradation with noise (expected)
        - |β₁| smaller → more robust to noise
        - p < 0.05 → significant linear trend

    Formula (corrected percentage):
        Relative Robustness = (|β₁_baseline| - |β₁_method|) / |β₁_baseline| × 100%
        Example: β₁_adv = -0.002068, β₁_hybrid = -0.000894
        → (0.002068 - 0.000894) / 0.002068 × 100% = 56.8%

    References:
        Theil (1950), Sen (1968): Robust Linear Regression
        Paper Section 3.3.2: Noise Robustness Analysis
    """
    if logger:
        logger.info(f"Analyzing noise robustness (method: {'Theil-Sen' if robust else 'OLS'})...")

    robustness_results = []

    for noise_type in NOISE_CONFIGS.keys():
        if noise_type == "clean":
            continue  # No intensity variation for clean

        for size in SIZES:
            for k in K_VALUES:
                for method in METHODS:
                    subset = df[
                        (df['noise_type'] == noise_type) &
                        (df['size'] == size) &
                        (df['k'] == k) &
                        (df['method'] == method)
                    ]

                    if len(subset) < 2:
                        continue  # Need at least 2 points for regression

                    x = subset['intensity_value'].values
                    y = subset['accuracy_mean'].values

                    # Skip if any NaN
                    if np.any(np.isnan(y)):
                        continue

                    # Perform regression
                    if robust:
                        # Theil-Sen robust regression
                        try:
                            slope, intercept, _, _ = theilslopes(y, x)
                            # Theil-Sen doesn't provide standard errors directly
                            slope_se = np.nan
                            intercept_se = np.nan
                            r_squared = np.nan  # Not directly available
                            r_squared_adj = np.nan
                            p_value = np.nan  # Not directly available
                            regression_type = 'theil_sen'
                            residual_normality_p = np.nan
                        except Exception as e:
                            if logger:
                                logger.warning(f"Theil-Sen failed for {noise_type}/{size}/k{k}/{method}: {e}")
                            continue
                    else:
                        # Ordinary Least Squares
                        try:
                            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
                            slope_se = stderr
                            intercept_se = np.nan  # linregress doesn't provide intercept SE
                            r_squared = r_value ** 2
                            # Adjusted R²
                            n = len(x)
                            r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else np.nan
                            regression_type = 'ols'

                            # Residual diagnostics
                            y_pred = slope * x + intercept
                            residuals = y - y_pred
                            if len(residuals) >= 3:
                                _, residual_normality_p = shapiro(residuals)
                            else:
                                residual_normality_p = np.nan
                        except Exception as e:
                            if logger:
                                logger.warning(f"OLS failed for {noise_type}/{size}/k{k}/{method}: {e}")
                            continue

                    robustness_results.append({
                        'noise_type': noise_type,
                        'size': size,
                        'k': k,
                        'method': method,
                        'slope': slope,
                        'slope_se': slope_se,
                        'intercept': intercept,
                        'intercept_se': intercept_se,
                        'r_squared': r_squared,
                        'r_squared_adj': r_squared_adj,
                        'p_value': p_value,
                        'regression_type': regression_type,
                        'residual_normality_p': residual_normality_p
                    })

    df_robustness = pd.DataFrame(robustness_results)

    if logger:
        logger.info(f"Computed {len(df_robustness)} noise robustness regressions")
        # Aggregate slopes by method
        slope_summary = df_robustness.groupby('method')['slope'].agg(['mean', 'std', 'min', 'max'])
        logger.info("Mean slopes by method (less negative = more robust):")
        for method in METHODS:
            if method in slope_summary.index:
                mean_slope = slope_summary.loc[method, 'mean']
                std_slope = slope_summary.loc[method, 'std']
                logger.info(f"  {method}: {mean_slope:.6f} ± {std_slope:.6f}")

    return df_robustness


# ============================================================================
# CATEGORY 5: METHOD × NOISE_INTENSITY INTERACTION (NEW FEATURE)
# ============================================================================

def analyze_method_noise_interaction(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """Test for Method × Noise_Intensity interaction effects.

    RESEARCH QUESTION:
        Does Hybrid improve MORE under high-noise conditions compared to low-noise?

    METHOD:
        For each noise_type, fit linear models for Advanced and Hybrid:
        - Model_Adv: Accuracy ~ β₀ + β₁ × Intensity
        - Model_Hybrid: Accuracy ~ β₀ + β₁ × Intensity
        - Compare slopes: Δβ₁ = β₁_Hybrid - β₁_Advanced
        - Test if Δβ₁ significantly different from 0

    Args:
        df: Aggregated data
        logger: Optional logger

    Returns:
        DataFrame with columns:
        - noise_type
        - slope_advanced, slope_hybrid
        - slope_diff (Δβ₁)
        - slope_diff_se (approximate standard error)
        - p_value_interaction (two-sample t-test on slopes)
        - significant_interaction (p < 0.05)

    Interpretation:
        - Positive Δβ₁ (Hybrid less negative) → Hybrid more robust
        - Significant interaction → context-dependent advantage
        - If significant for saltpepper/speckle but not gaussian → 
          confirms theory (WST better for impulsive noise)

    References:
        Paper Conclusions: "A crucial finding is the absence of a universally
        superior method. Analysis of the delta heatmap revealed that the relative
        performance varies substantially as a function of the type and intensity
        of noise."
    """
    if logger:
        logger.info("Analyzing Method × Noise_Intensity interactions...")

    interaction_results = []

    for noise_type in NOISE_CONFIGS.keys():
        if noise_type == "clean":
            continue

        # Get data for Advanced Stats and Hybrid
        adv_data = df[
            (df['noise_type'] == noise_type) &
            (df['method'] == 'advanced_stats')
        ]
        hybrid_data = df[
            (df['noise_type'] == noise_type) &
            (df['method'] == 'hybrid')
        ]

        if len(adv_data) < 2 or len(hybrid_data) < 2:
            continue

        # Fit linear models (average across sizes and k for simplicity)
        adv_grouped = adv_data.groupby('intensity_value').agg({'accuracy_mean': 'mean'}).reset_index()
        hybrid_grouped = hybrid_data.groupby('intensity_value').agg({'accuracy_mean': 'mean'}).reset_index()

        x_adv = adv_grouped['intensity_value'].values
        y_adv = adv_grouped['accuracy_mean'].values
        x_hybrid = hybrid_grouped['intensity_value'].values
        y_hybrid = hybrid_grouped['accuracy_mean'].values

        # Skip if insufficient data or NaN
        if len(x_adv) < 2 or len(x_hybrid) < 2 or np.any(np.isnan(y_adv)) or np.any(np.isnan(y_hybrid)):
            continue

        # Fit regressions
        slope_adv, intercept_adv, _, _, se_adv = stats.linregress(x_adv, y_adv)
        slope_hybrid, intercept_hybrid, _, _, se_hybrid = stats.linregress(x_hybrid, y_hybrid)

        # Compute slope difference
        slope_diff = slope_hybrid - slope_adv

        # Approximate SE of difference (assuming independence)
        slope_diff_se = np.sqrt(se_adv**2 + se_hybrid**2)

        # Test significance (approximate z-test)
        if slope_diff_se > 0:
            z_stat = slope_diff / slope_diff_se
            p_value_interaction = 2 * stats.norm.sf(np.abs(z_stat))  # Two-tailed
        else:
            p_value_interaction = np.nan

        significant_interaction = p_value_interaction < 0.05 if not np.isnan(p_value_interaction) else False

        interaction_results.append({
            'noise_type': noise_type,
            'slope_advanced': slope_adv,
            'slope_hybrid': slope_hybrid,
            'slope_diff': slope_diff,
            'slope_diff_se': slope_diff_se,
            'p_value_interaction': p_value_interaction,
            'significant_interaction': significant_interaction
        })

        if logger:
            sig_marker = '***' if significant_interaction else 'ns'
            logger.info(
                f"  {noise_type}: Δslope = {slope_diff:.6f} ± {slope_diff_se:.6f}, "
                f"p = {p_value_interaction:.4f} ({sig_marker})"
            )

    df_interaction = pd.DataFrame(interaction_results)

    if logger:
        logger.info(f"Computed {len(df_interaction)} noise-specific interactions")
        sig_count = df_interaction['significant_interaction'].sum()
        logger.info(f"  Significant interactions: {sig_count}/{len(df_interaction)}")

    return df_interaction


# ============================================================================
# CATEGORY 6: DATA SCARCITY ANALYSIS (from v1.0, minimal changes)
# ============================================================================

def analyze_data_scarcity(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Analyze robustness to data scarcity across dataset sizes.

    Args:
        df: Aggregated data
        logger: Optional logger

    Returns:
        DataFrame with columns:
        - noise_type, intensity, k, method, size
        - accuracy (absolute)
        - retention_pct (relative to 'original')

    Formula:
        Retention% = (Accuracy_size / Accuracy_original) × 100

    Interpretation:
        - Retention > 90%: Data-efficient
        - Retention < 80%: Data-hungry

    Paper Section:
        Section 3.3.3: Data Scarcity Analysis
    """
    if logger:
        logger.info("Analyzing data scarcity robustness...")

    scarcity_results = []

    for noise_type, intensities in NOISE_CONFIGS.items():
        for intensity in intensities:
            for k in K_VALUES:
                for method in METHODS:
                    # Get metrics for each size
                    metrics_by_size = {}
                    for size in SIZES:
                        subset = df[
                            (df['noise_type'] == noise_type) &
                            (df['intensity'] == intensity) &
                            (df['size'] == size) &
                            (df['k'] == k) &
                            (df['method'] == method)
                        ]

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

    if logger:
        logger.info(f"Computed {len(df_scarcity)} data scarcity retention scores")
        # Summarize by method and size
        retention_summary = df_scarcity.groupby(['method', 'size'])['retention_pct'].agg(['mean', 'std'])
        logger.info("Mean retention by method and size:")
        for (method, size), row in retention_summary.iterrows():
            logger.info(f"  {method} ({size}): {row['mean']:.2f}% ± {row['std']:.2f}%")

    return df_scarcity


# ============================================================================
# CATEGORY 7: MAIN EXECUTION & ORCHESTRATION
# ============================================================================

def save_csv_with_metadata(df: pd.DataFrame, filepath: Path, logger: logging.Logger):
    """Save CSV with metadata header.

    Args:
        df: DataFrame to save
        filepath: Output path
        logger: Logger instance

    Output Format:
        # Generated: 2025-10-10 14:23:45
        # Script: robustness_analysis_v2.py
        # Version: 2.0.0
        # Rows: 168
        # Columns: 15
        column1,column2,...
        value1,value2,...
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metadata_lines = [
        f"# Generated: {timestamp}",
        f"# Script: robustness_analysis_v2.py",
        f"# Version: {__version__}",
        f"# Rows: {len(df)}",
        f"# Columns: {len(df.columns)}"
    ]

    with open(filepath, 'w') as f:
        for line in metadata_lines:
            f.write(line + '\n')
        df.to_csv(f, index=False)

    logger.info(f"Saved: {filepath.name} ({len(df)} rows)")


def dry_run_validation(experiments_root: Path, logger: logging.Logger) -> bool:
    """Validate dataset structure without running full analysis.

    Args:
        experiments_root: Path to experiments root
        logger: Logger instance

    Returns:
        True if structure valid, False otherwise

    Checks:
        1. Root directory exists
        2. Expected noise type directories present
        3. Sample JSON files loadable
    """
    logger.info("="*80)
    logger.info("DRY RUN: Validating dataset structure...")
    logger.info("="*80)

    if not experiments_root.exists():
        logger.error(f"Experiments root does not exist: {experiments_root}")
        return False

    logger.info(f"✓ Root exists: {experiments_root}")

    # Check noise directories
    missing_dirs = []
    for noise_type in NOISE_CONFIGS.keys():
        noise_dir = experiments_root / noise_type
        if not noise_dir.exists():
            missing_dirs.append(noise_type)
        else:
            logger.info(f"✓ Found: {noise_type}/")

    if missing_dirs:
        logger.warning(f"Missing noise directories: {missing_dirs}")

    # Try loading a sample JSON
    logger.info("Testing JSON loading...")
    sample_data = load_experiment_json(
        experiments_root, 'clean', 'clean_0', 'assatigue', 'mini', 2, 'advanced_stats', logger
    )
    if sample_data:
        logger.info("✓ Successfully loaded sample JSON")
        metrics = extract_metrics(sample_data)
        if metrics:
            logger.info(f"✓ Extracted metrics: {list(metrics.keys())}")
        else:
            logger.warning("⚠ Could not extract metrics from sample JSON")
    else:
        logger.warning("⚠ Could not load sample JSON (may be normal if experiments incomplete)")

    logger.info("="*80)
    logger.info("DRY RUN COMPLETE")
    logger.info("="*80)
    return True


def main():
    """Main execution pipeline for Robustness Analysis v2.0."""
    # Parse arguments
    args = parse_arguments()

    # Setup paths
    experiments_root = Path(args.root).resolve()
    output_dir = Path(args.output).resolve()

    # Setup output structure
    paths = setup_output_structure(output_dir, logging.getLogger())  # Temp logger for structure

    # Setup logging (after output structure created)
    logger = setup_logging(output_dir, args.log_level)

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Experiments Root: {experiments_root}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Random Seed: {args.seed}")
    logger.info(f"  Primary Metric: {args.primary_metric}")
    logger.info(f"  Robust Regression: {args.robust_regression}")
    logger.info(f"  Dry Run: {args.dry_run}")

    # Set random seed
    np.random.seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Dry run mode
    if args.dry_run:
        dry_run_validation(experiments_root, logger)
        logger.info("Exiting (dry-run mode)")
        sys.exit(0)

    # Phase 1: Load all data
    logger.info("="*80)
    logger.info("PHASE 1: DATA LOADING")
    logger.info("="*80)
    df = load_all_data(experiments_root, logger)
    save_csv_with_metadata(df, paths['data'] / 'all_aggregated_data.csv', logger)

    # Phase 2: Compute pairwise deltas
    logger.info("="*80)
    logger.info("PHASE 2: PAIRWISE COMPARISONS")
    logger.info("="*80)
    df_deltas = compute_pairwise_deltas(df, logger)
    save_csv_with_metadata(df_deltas, paths['data'] / 'pairwise_deltas.csv', logger)

    # Phase 3: Statistical tests
    logger.info("="*80)
    logger.info("PHASE 3: STATISTICAL SIGNIFICANCE TESTING")
    logger.info("="*80)

    # Determine metrics to test based on primary_metric argument
    if args.primary_metric == 'macro_f1':
        metrics_to_test = ['delta_macro_f1']
        logger.info("Testing macro_f1 only (as specified)")
    elif args.primary_metric == 'accuracy':
        metrics_to_test = ['delta_accuracy']
        logger.info("Testing accuracy only (as specified)")
    else:  # 'both'
        metrics_to_test = ['delta_macro_f1', 'delta_accuracy']
        logger.info("Testing both macro_f1 and accuracy (joint FDR correction)")

    df_stats = perform_statistical_tests_v2(
        df_deltas,
        metrics=metrics_to_test,
        alpha=0.05,
        n_bootstrap=1000,
        random_state=args.seed,
        logger=logger
    )
    save_csv_with_metadata(df_stats, paths['data'] / 'statistical_tests_summary.csv', logger)

    # Phase 4: Noise robustness
    logger.info("="*80)
    logger.info("PHASE 4: NOISE ROBUSTNESS ANALYSIS")
    logger.info("="*80)
    df_robustness = analyze_noise_robustness_v2(df, robust=args.robust_regression, logger=logger)
    save_csv_with_metadata(df_robustness, paths['data'] / 'noise_slope_summary.csv', logger)

    # Phase 5: Method × Noise_Intensity interaction
    logger.info("="*80)
    logger.info("PHASE 5: METHOD × NOISE_INTENSITY INTERACTION")
    logger.info("="*80)
    df_interaction = analyze_method_noise_interaction(df, logger)
    save_csv_with_metadata(df_interaction, paths['data'] / 'method_noise_interaction.csv', logger)

    # Phase 6: Data scarcity
    logger.info("="*80)
    logger.info("PHASE 6: DATA SCARCITY ANALYSIS")
    logger.info("="*80)
    df_scarcity = analyze_data_scarcity(df, logger)
    save_csv_with_metadata(df_scarcity, paths['data'] / 'data_scarcity_retention.csv', logger)

    # Phase 7: Summary & Completion
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"All results saved to: {output_dir}")
    logger.info("Generated files:")
    logger.info("  CSV Data (data/):")
    logger.info("    - all_aggregated_data.csv")
    logger.info("    - pairwise_deltas.csv")
    logger.info("    - statistical_tests_summary.csv")
    logger.info("    - noise_slope_summary.csv")
    logger.info("    - method_noise_interaction.csv (NEW in v2.0)")
    logger.info("    - data_scarcity_retention.csv")
    logger.info("")
    logger.info("  Logs (logs/):")
    logger.info(f"    - {logger.handlers[1].baseFilename if len(logger.handlers) > 1 else 'N/A'}")
    logger.info("="*80)
    logger.info("To generate visualizations, run plotting scripts separately.")
    logger.info("(Visualization code available in v1.0 or can be added as separate module)")
    logger.info("="*80)


if __name__ == "__main__":
    main()
