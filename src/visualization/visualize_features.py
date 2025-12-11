#!/usr/bin/env python3
"""
Feature Visualization Script for PowerPoint Presentation

This script generates visualizations comparing Advanced Statistics and
Wavelet Scattering Transform (WST) feature extraction methods on
simple black-and-white test patterns.

Author: Mattia Bestiaccia
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.ndimage import sobel, laplace
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import from train_and_save_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'experiments'))

# Import feature extraction functions
try:
    from kymatio.numpy import Scattering2D
    KYMATIO_AVAILABLE = True
except ImportError:
    print("Warning: kymatio not available. WST features will not work.")
    KYMATIO_AVAILABLE = False

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Constants
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
DPI = 300
IMAGE_SIZE = 128  # Size of test images


# ============================================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================================

def generate_gradient_horizontal(size=128):
    """Generate horizontal gradient from black (left) to white (right)"""
    img = np.linspace(0, 1, size).reshape(1, -1).repeat(size, axis=0)
    return img

def generate_gradient_vertical(size=128):
    """Generate vertical gradient from black (top) to white (bottom)"""
    img = np.linspace(0, 1, size).reshape(-1, 1).repeat(size, axis=1)
    return img

def generate_checkerboard(size=128, squares=8):
    """Generate checkerboard pattern"""
    square_size = size // squares
    img = np.zeros((size, size))
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                img[i*square_size:(i+1)*square_size,
                    j*square_size:(j+1)*square_size] = 1.0
    return img

def generate_circles(size=128, num_circles=5):
    """Generate concentric circles pattern"""
    img = np.zeros((size, size))
    center = size / 2
    max_radius = size / 2

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            normalized_dist = dist / max_radius
            img[i, j] = np.sin(normalized_dist * num_circles * np.pi) * 0.5 + 0.5

    return img

def generate_texture(size=128, seed=42):
    """Generate random texture pattern"""
    np.random.seed(seed)
    img = np.random.rand(size, size)
    return img

def generate_vertical_texture(size=128, seed=42, frequency=8):
    """
    Generate vertical striped texture with random variations

    Creates vertical stripes with noise to demonstrate directional texture.
    This pattern has strong vertical directionality while maintaining texture complexity.
    """
    np.random.seed(seed)

    # Create base vertical stripes
    x = np.linspace(0, frequency * 2 * np.pi, size)
    vertical_pattern = np.sin(x).reshape(1, -1).repeat(size, axis=0)

    # Normalize to [0, 1]
    vertical_pattern = (vertical_pattern + 1) / 2

    # Add random noise to create texture (but keep vertical structure)
    noise = np.random.rand(size, size) * 0.3  # 30% noise
    img = vertical_pattern * 0.7 + noise

    # Ensure [0, 1] range
    img = np.clip(img, 0, 1)

    return img

def generate_edge(size=128, border_width=20):
    """Generate sharp edge (white square on black background)"""
    img = np.zeros((size, size))
    img[border_width:size-border_width, border_width:size-border_width] = 1.0
    return img


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS (adapted from train_and_save_model.py)
# ============================================================================

def extract_advanced_features(grayscale_image):
    """
    Extract advanced statistical features from a grayscale image.
    For grayscale, we extract 18 features (1 channel).

    Returns: array of 18 features
    """
    features_per_channel = 18
    features = np.zeros(features_per_channel)

    channel = grayscale_image
    ch_flat = channel.ravel()
    ch_clean = ch_flat[np.isfinite(ch_flat)]

    if len(ch_clean) == 0:
        return features

    # Basic statistics
    features[0] = np.mean(ch_clean)
    features[1] = np.std(ch_clean)
    features[2] = np.var(ch_clean)
    features[3] = np.min(ch_clean)
    features[4] = np.max(ch_clean)
    features[5] = np.ptp(ch_clean)  # range

    # Shape statistics
    features[6] = stats.skew(ch_clean)
    features[7] = stats.kurtosis(ch_clean)
    mean_val = features[0]
    features[8] = features[1] / max(mean_val, 1e-8)  # coefficient of variation

    # Percentiles
    features[9] = np.percentile(ch_clean, 10)
    features[10] = np.percentile(ch_clean, 25)
    features[11] = np.percentile(ch_clean, 50)
    features[12] = np.percentile(ch_clean, 75)
    features[13] = np.percentile(ch_clean, 90)
    features[14] = features[12] - features[10]  # IQR

    # MAD
    features[15] = np.mean(np.abs(ch_clean - mean_val))

    # Gradient and edge density
    try:
        grad_x = sobel(channel, axis=0)
        grad_y = sobel(channel, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        features[16] = np.mean(grad_mag.ravel())

        edges = np.abs(laplace(channel))
        edge_thr = np.percentile(edges.ravel(), 90)
        features[17] = np.mean(edges.ravel() > edge_thr)
    except:
        features[16] = 0
        features[17] = 0

    return features

def get_feature_names_advanced():
    """Get names for advanced statistics features"""
    return [
        'mean', 'std', 'var', 'min', 'max', 'range',
        'skew', 'kurt', 'cv',
        'p10', 'p25', 'p50', 'p75', 'p90', 'iqr',
        'mad', 'grad_mean', 'edge_density'
    ]

def extract_wst_features(grayscale_image):
    """
    Extract Wavelet Scattering Transform features from grayscale image.

    Returns: array of WST features (mean and std of scattering coefficients)
    """
    if not KYMATIO_AVAILABLE:
        raise ImportError("kymatio not available. Cannot extract WST features.")

    # Initialize scattering transform
    J = 2  # Number of scales
    L = 8  # Number of angles

    H, W = grayscale_image.shape

    # Initialize scattering for this image size
    scattering = Scattering2D(J=J, L=L, shape=(H, W))

    # Compute scattering coefficients
    scattering_coeffs = scattering(grayscale_image)

    # Calculate mean and std across spatial dimensions for each coefficient
    coeffs_mean = np.mean(scattering_coeffs, axis=(-2, -1))
    coeffs_std = np.std(scattering_coeffs, axis=(-2, -1))

    # Combine mean and std features
    wst_features = np.concatenate([coeffs_mean, coeffs_std])

    return wst_features, scattering_coeffs


# ============================================================================
# VISUALIZATION FUNCTIONS - ADVANCED STATISTICS
# ============================================================================

def visualize_advanced_stats(image, pattern_name, output_subdir):
    """Create comprehensive visualization of advanced statistics features"""
    features = extract_advanced_features(image)
    feature_names = get_feature_names_advanced()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Input: {pattern_name}', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Feature values bar chart
    ax2 = fig.add_subplot(gs[0, 1:])
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax2.barh(feature_names, features, color=colors)
    ax2.set_xlabel('Feature Value', fontsize=12)
    ax2.set_title('Advanced Statistics Features (18 total)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Gradient visualization
    ax3 = fig.add_subplot(gs[1, 0])
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    im3 = ax3.imshow(grad_mag, cmap='hot')
    ax3.set_title('Gradient Magnitude (Sobel)', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # 4. Edge detection
    ax4 = fig.add_subplot(gs[1, 1])
    edges = np.abs(laplace(image))
    im4 = ax4.imshow(edges, cmap='hot')
    ax4.set_title('Edge Detection (Laplace)', fontsize=12)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # 5. Histogram with percentiles
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(image.ravel(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    percentiles = [10, 25, 50, 75, 90]
    colors_p = ['red', 'orange', 'green', 'orange', 'red']
    for p, c in zip(percentiles, colors_p):
        val = np.percentile(image.ravel(), p)
        ax5.axvline(val, color=c, linestyle='--', linewidth=2, label=f'P{p}')
    ax5.set_xlabel('Pixel Intensity', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Histogram with Percentiles', fontsize=12)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Key statistics table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')

    stats_data = [
        ['Statistic', 'Value', 'Description'],
        ['Mean', f'{features[0]:.4f}', 'Average pixel intensity'],
        ['Std Dev', f'{features[1]:.4f}', 'Standard deviation'],
        ['Skewness', f'{features[6]:.4f}', 'Distribution asymmetry'],
        ['Kurtosis', f'{features[7]:.4f}', 'Distribution tail heaviness'],
        ['IQR', f'{features[14]:.4f}', 'Interquartile range (P75-P25)'],
        ['Gradient Mean', f'{features[16]:.4f}', 'Average edge strength'],
        ['Edge Density', f'{features[17]:.4f}', 'Proportion of edge pixels'],
    ]

    table = ax6.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.2, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.suptitle(f'Advanced Statistics Feature Extraction: {pattern_name}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_path = os.path.join(output_subdir, f'{pattern_name}_advanced_stats.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return features


# ============================================================================
# VISUALIZATION FUNCTIONS - WST
# ============================================================================

def visualize_wst(image, pattern_name, output_subdir):
    """Create comprehensive visualization of WST features"""
    if not KYMATIO_AVAILABLE:
        print("  ✗ Skipping WST visualization (kymatio not available)")
        return None

    wst_features, scattering_coeffs = extract_wst_features(image)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Input: {pattern_name}', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. Scattering coefficients at different orientations
    num_coeffs = scattering_coeffs.shape[0]

    # Show specific orientations to demonstrate directional sensitivity
    # WST with L=8 orientations: angles = [0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°]
    # Index mapping (approximate): J=0 is index 0, J=1 starts at index 1

    coefficients_to_show = [
        (0, 'J=0 (Low-pass)', 'Overall energy'),
        (1, 'Horizontal (0°)', 'Detects horizontal variations'),
        (5, 'Vertical (90°)', 'Detects vertical variations'),
        (3, 'Diagonal (45°)', 'Detects diagonal variations'),
        (7, 'Diagonal (135°)', 'Detects opposite diagonal'),
        (2, 'Orientation 22.5°', 'Intermediate angle'),
        (9, 'J=2 Example', 'Second-order coefficient'),
    ]

    plot_idx = 0
    for coeff_idx, title, description in coefficients_to_show:
        if coeff_idx >= num_coeffs:
            continue

        # Position calculation: skip first column in row 0
        if plot_idx < 3:  # First 3 go in row 0, columns 1-3
            row = 0
            col = plot_idx + 1
        else:  # Remaining go in row 1, columns 0-3
            row = 1
            col = plot_idx - 3

        if col >= 4:  # Safety check
            break

        ax = fig.add_subplot(gs[row, col])
        plot_idx += 1

        # Display scattering coefficient
        coeff = scattering_coeffs[coeff_idx]
        im = ax.imshow(coeff, cmap='viridis')

        # Title with orientation
        ax.set_title(f'{title}\n({description})', fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # Add coefficient index in corner
        ax.text(0.02, 0.98, f'Coeff {coeff_idx}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

    # 3. Feature statistics visualization
    ax_stats = fig.add_subplot(gs[2, :2])

    num_coeffs_actual = len(scattering_coeffs)
    coeffs_mean = np.mean(scattering_coeffs, axis=(-2, -1))
    coeffs_std = np.std(scattering_coeffs, axis=(-2, -1))

    x = np.arange(min(50, num_coeffs_actual))  # Show first 50 coefficients
    ax_stats.plot(x, coeffs_mean[:len(x)], 'o-', label='Mean', linewidth=2, markersize=4)
    ax_stats.plot(x, coeffs_std[:len(x)], 's-', label='Std Dev', linewidth=2, markersize=4)
    ax_stats.set_xlabel('Coefficient Index', fontsize=12)
    ax_stats.set_ylabel('Value', fontsize=12)
    ax_stats.set_title('WST Coefficient Statistics (Mean & Std)', fontsize=12, fontweight='bold')
    ax_stats.legend(fontsize=10)
    ax_stats.grid(True, alpha=0.3)

    # 4. Energy distribution across scales
    ax_energy = fig.add_subplot(gs[2, 2:])

    # Approximate energy per scale (simplified)
    scale_energies = []
    scale_labels = []

    # J=0 (low-pass)
    scale_energies.append(np.sum(coeffs_mean[0]**2))
    scale_labels.append('J=0\n(Low-pass)')

    # J=1 (first scale)
    if num_coeffs_actual > 1:
        scale_energies.append(np.sum(coeffs_mean[1:min(9, num_coeffs_actual)]**2))
        scale_labels.append('J=1\n(8 orientations)')

    # J=2 (second scale)
    if num_coeffs_actual > 9:
        scale_energies.append(np.sum(coeffs_mean[9:]**2))
        scale_labels.append('J=2+\n(Higher scales)')

    colors_energy = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax_energy.bar(scale_labels, scale_energies, color=colors_energy[:len(scale_energies)],
                         alpha=0.7, edgecolor='black', linewidth=2)
    ax_energy.set_ylabel('Energy (sum of squared means)', fontsize=12)
    ax_energy.set_title('Energy Distribution Across Scales', fontsize=12, fontweight='bold')
    ax_energy.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, energy in zip(bars, scale_energies):
        height = bar.get_height()
        ax_energy.text(bar.get_x() + bar.get_width()/2., height,
                      f'{energy:.2e}',
                      ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle(f'Wavelet Scattering Transform (WST) Feature Extraction: {pattern_name}\n'
                 f'J=2 scales, L=8 orientations (0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°), {len(wst_features)} features total',
                 fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    output_path = os.path.join(output_subdir, f'{pattern_name}_wst.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")
    return wst_features


# ============================================================================
# COMPARATIVE DASHBOARD
# ============================================================================

def create_comparison_dashboard(image, pattern_name, output_subdir):
    """Create side-by-side comparison of both methods"""

    # Extract features
    adv_features = extract_advanced_features(image)
    adv_feature_names = get_feature_names_advanced()

    wst_available = KYMATIO_AVAILABLE
    if wst_available:
        wst_features, _ = extract_wst_features(image)

    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Original image (larger, spanning 2 rows)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Input Image\n{pattern_name}', fontsize=16, fontweight='bold')
    ax1.axis('off')

    # Add image properties text
    textstr = f'Size: {image.shape[0]}×{image.shape[1]}\n'
    textstr += f'Mean: {np.mean(image):.3f}\n'
    textstr += f'Std: {np.std(image):.3f}\n'
    textstr += f'Range: [{np.min(image):.3f}, {np.max(image):.3f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # 2. Advanced Stats - Top features
    ax2 = fig.add_subplot(gs[0, 1])
    top_n = 10
    top_indices = np.argsort(np.abs(adv_features))[-top_n:][::-1]
    top_features = adv_features[top_indices]
    top_names = [adv_feature_names[i] for i in top_indices]

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax2.barh(top_names, top_features, color=colors)
    ax2.set_xlabel('Feature Value', fontsize=12)
    ax2.set_title(f'Advanced Statistics\nTop {top_n} Features (of {len(adv_features)} total)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Advanced Stats - Dimensionality info
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    adv_info = f"""
    ADVANCED STATISTICS METHOD

    Total Features: {len(adv_features)}

    Feature Categories:
    • Basic Statistics (6): mean, std, var, min, max, range
    • Shape Statistics (3): skewness, kurtosis, CV
    • Percentiles (6): P10, P25, P50, P75, P90, IQR
    • Dispersion (1): MAD
    • Texture/Edge (2): gradient mean, edge density

    Computation: Fast (milliseconds)
    Interpretability: High
    Robustness: Moderate
    """

    ax3.text(0.1, 0.5, adv_info, transform=ax3.transAxes, fontsize=11,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 4. WST visualization
    if wst_available:
        ax4 = fig.add_subplot(gs[0, 2])

        # Show top N WST coefficients (similar to Advanced Stats visualization)
        top_n = 20  # Show top 20 coefficients

        # Get indices sorted by absolute value
        top_indices = np.argsort(np.abs(wst_features))[-top_n:][::-1]
        top_values = wst_features[top_indices]

        # Create labels for coefficients
        # First half are means, second half are stds
        num_coeffs = len(wst_features) // 2
        coeff_labels = []
        for idx in top_indices:
            if idx < num_coeffs:
                # Mean coefficient
                coeff_labels.append(f'μ{idx}')
            else:
                # Std coefficient
                coeff_labels.append(f'σ{idx - num_coeffs}')

        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        bars = ax4.barh(range(top_n), top_values, color=colors)
        ax4.set_yticks(range(top_n))
        ax4.set_yticklabels(coeff_labels, fontsize=9)
        ax4.set_xlabel('Coefficient Value', fontsize=12)
        ax4.set_title(f'WST Top {top_n} Coefficients\n({len(wst_features)} total: {num_coeffs} means + {num_coeffs} stds)',
                      fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.invert_yaxis()  # Highest at top

        # 5. WST - Dimensionality info
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        wst_info = f"""
        WAVELET SCATTERING TRANSFORM METHOD

        Total Features: {len(wst_features)}

        Configuration:
        • Scales (J): 2
        • Orientations (L): 8
        • Statistics per coeff: mean, std

        Feature Structure:
        • J=0: 1 low-pass coefficient
        • J=1: 8 oriented wavelets
        • J=2: 64 second-order coefficients

        Computation: Slower (seconds, GPU accelerated)
        Interpretability: Low (complex multiscale)
        Robustness: High (deformation invariant)
        """

        ax5.text(0.1, 0.5, wst_info, transform=ax5.transAxes, fontsize=11,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    else:
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.text(0.5, 0.5, 'WST Not Available\n(kymatio not installed)',
                ha='center', va='center', fontsize=14, color='red')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

    plt.suptitle(f'Feature Extraction Methods Comparison: {pattern_name}',
                 fontsize=18, fontweight='bold', y=0.98)

    # Save figure
    output_path = os.path.join(output_subdir, f'{pattern_name}_comparison.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# OVERALL COMPARISON CHART
# ============================================================================

def create_overall_comparison(all_patterns, output_dir):
    """Create overall comparison chart across all patterns"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Extraction Methods: Overall Comparison',
                 fontsize=18, fontweight='bold')

    # 1. Dimensionality comparison
    ax1 = axes[0, 0]
    methods = ['Advanced\nStatistics', 'WST', 'Hybrid']

    if KYMATIO_AVAILABLE:
        # Get actual dimensions from a sample image
        sample_img = generate_gradient_horizontal(IMAGE_SIZE)
        adv_feats = extract_advanced_features(sample_img)
        wst_feats, _ = extract_wst_features(sample_img)
        dimensions = [len(adv_feats), len(wst_feats), len(adv_feats) + len(wst_feats)]
    else:
        dimensions = [18, 486, 504]  # Approximate values

    colors = ['#3498db', '#e74c3c', '#9b59b6']
    bars = ax1.bar(methods, dimensions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Features', fontsize=12)
    ax1.set_title('Feature Space Dimensionality', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, dim in zip(bars, dimensions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{dim}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 2. Computational complexity (approximate)
    ax2 = axes[0, 1]
    comp_times = [0.01, 0.5, 0.51]  # Approximate seconds per image
    bars2 = ax2.bar(methods, comp_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Computational Cost (Approximate)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, time in zip(bars2, comp_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Characteristics comparison (radar chart style)
    ax3 = axes[1, 0]
    categories = ['Interpretability', 'Robustness', 'Speed', 'Simplicity']

    # Scores (0-10)
    adv_stats_scores = [9, 6, 10, 10]
    wst_scores = [3, 10, 3, 2]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    adv_stats_scores += adv_stats_scores[:1]
    wst_scores += wst_scores[:1]
    angles += angles[:1]

    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(angles, adv_stats_scores, 'o-', linewidth=2, label='Advanced Stats', color=colors[0])
    ax3.fill(angles, adv_stats_scores, alpha=0.25, color=colors[0])
    ax3.plot(angles, wst_scores, 's-', linewidth=2, label='WST', color=colors[1])
    ax3.fill(angles, wst_scores, alpha=0.25, color=colors[1])
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 10)
    ax3.set_title('Qualitative Characteristics\n(Higher = Better)',
                  fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.grid(True)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    summary_data = [
        ['Characteristic', 'Advanced Stats', 'WST', 'Hybrid'],
        ['Features', f'{dimensions[0]}', f'{dimensions[1]}', f'{dimensions[2]}'],
        ['Computation', 'Very Fast', 'Slow', 'Slow'],
        ['Interpretability', 'High', 'Low', 'Medium'],
        ['Robustness', 'Moderate', 'High', 'High'],
        ['Use Case', 'Baseline', 'Texture-rich', 'Best overall'],
        ['GPU Benefit', 'No', 'Yes (10×)', 'Yes (10×)'],
        ['Memory', 'Low', 'High', 'High'],
    ]

    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')

    ax4.set_title('Method Comparison Summary', fontsize=14, fontweight='bold', pad=20)

    # Save figure
    output_path = os.path.join(output_dir, 'overall_comparison.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved overall comparison: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("="*80)
    print("FEATURE VISUALIZATION SCRIPT")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"DPI: {DPI}")
    print(f"Kymatio available: {KYMATIO_AVAILABLE}")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define patterns to generate
    patterns = {
        'gradient_horizontal': generate_gradient_horizontal,
        'gradient_vertical': generate_gradient_vertical,
        'checkerboard': generate_checkerboard,
        'circles': generate_circles,
        'texture': generate_texture,
        'vertical_texture': generate_vertical_texture,  # NEW: Directional texture
        'edge': generate_edge,
    }

    # Process each pattern
    for pattern_name, pattern_func in patterns.items():
        print(f"\nProcessing: {pattern_name}")
        print("-" * 40)

        # Create subdirectory for this pattern
        pattern_dir = os.path.join(OUTPUT_DIR, pattern_name)
        os.makedirs(pattern_dir, exist_ok=True)

        # Generate image
        image = pattern_func(IMAGE_SIZE)

        # Save original image
        img_path = os.path.join(pattern_dir, f'{pattern_name}_original.png')
        plt.imsave(img_path, image, cmap='gray')
        print(f"  ✓ Generated image: {img_path}")

        # Visualize advanced statistics
        visualize_advanced_stats(image, pattern_name, pattern_dir)

        # Visualize WST
        if KYMATIO_AVAILABLE:
            visualize_wst(image, pattern_name, pattern_dir)

        # Create comparison dashboard
        create_comparison_dashboard(image, pattern_name, pattern_dir)

    # Create overall comparison
    print("\n" + "="*80)
    print("Creating overall comparison chart...")
    create_overall_comparison(list(patterns.keys()), OUTPUT_DIR)

    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY")
    print(f"✓ Output saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
