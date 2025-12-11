"""
Compare WST coefficients between clean and noisy images
======================================================
This script compares Wavelet Scattering Transform coefficients between clean images 
and their noisy counterparts with Gaussian noise (sigma=30).

Based on the original plot_scattering_disk.py script.
"""

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from kymatio import Scattering2D
from PIL import Image
import os
import glob

# Dataset paths
clean_dataset_path = "/home/brusc/Projects/random_forest/datasets/dataset_rgb_clean"
noisy_dataset_path = "/home/brusc/Projects/random_forest/datasets/dataset_rgb_gaussian_50"

def get_sample_images(dataset_path, n_samples=2):
    """Get sample original images from the dataset"""
    pattern = os.path.join(dataset_path, "**/*_original.png")
    image_files = glob.glob(pattern, recursive=True)
    return image_files[:n_samples]

def load_and_preprocess_image(img_path, target_size=(32, 32)):
    """Load image and convert to grayscale"""
    img = Image.open(img_path).convert('L').resize(target_size)
    return np.array(img).astype(np.float32) / 255.0

def compute_scattering_coefficients(img_tensor, L=6, J=3):
    """Compute scattering coefficients for an image"""
    scattering = Scattering2D(J=J, shape=img_tensor.shape, L=L, max_order=2, frontend='numpy')
    scat_coeffs = scattering(img_tensor)
    return -scat_coeffs  # Invert colors as in original

def plot_scattering_disk(scat_coeffs, title, ax_main, gs_spec, L=6, J=3):
    """Plot scattering coefficients in disk format"""
    
    len_order_1 = J * L
    scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]
    norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
    mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap="gray")
    
    len_order_2 = (J*(J-1)//2)*(L**2)
    scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]
    norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
    mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap="gray")
    
    window_rows, window_columns = scat_coeffs.shape[1:]
    l_offset = int(L - L / 2 - 1)
    
    # Plot first-order coefficients
    for row in range(window_rows):
        for column in range(window_columns):
            ax = plt.subplot(gs_spec[row, column], projection='polar')
            ax.axis('off')
            coefficients = scat_coeffs_order_1[:, row, column]
            for j in range(J):
                for l in range(L):
                    coeff = coefficients[l + j * L]
                    color = mapper_order_1.to_rgba(coeff)
                    angle = (l_offset - l) * np.pi / L
                    radius = 2 ** (-j - 1)
                    ax.bar(x=angle, height=radius, width=np.pi / L,
                           bottom=radius, color=color)
                    ax.bar(x=angle + np.pi, height=radius, width=np.pi / L,
                           bottom=radius, color=color)
    
    ax_main.set_title(title, fontsize=14, pad=20)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.axis('off')

def compare_images():
    """Main function to compare WST coefficients"""
    
    # Get sample images from clean dataset
    clean_images = get_sample_images(clean_dataset_path, n_samples=2)
    
    if len(clean_images) < 2:
        print("Error: Not enough images found in clean dataset")
        return
    
    # Find corresponding noisy images
    noisy_images = []
    for clean_path in clean_images:
        # Extract relative path from clean dataset
        rel_path = os.path.relpath(clean_path, clean_dataset_path)
        noisy_path = os.path.join(noisy_dataset_path, rel_path)
        
        if os.path.exists(noisy_path):
            noisy_images.append(noisy_path)
        else:
            print(f"Warning: Corresponding noisy image not found for {clean_path}")
    
    if len(noisy_images) < 2:
        print("Error: Not enough corresponding noisy images found")
        return
    
    # Process each pair of images separately
    for i, (clean_path, noisy_path) in enumerate(zip(clean_images[:2], noisy_images[:2])):
        
        print(f"Processing image pair {i+1}/2:")
        print(f"  Clean: {os.path.basename(clean_path)}")
        print(f"  Noisy: {os.path.basename(noisy_path)}")
        
        # Load and preprocess images
        clean_img = load_and_preprocess_image(clean_path)
        noisy_img = load_and_preprocess_image(noisy_path)
        
        # Compute scattering coefficients
        clean_coeffs = compute_scattering_coefficients(clean_img)
        noisy_coeffs = compute_scattering_coefficients(noisy_img)
        
        window_rows, window_columns = clean_coeffs.shape[1:]
        
        # Create figure for this pair
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.3)
        
        # Original images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(clean_img, cmap='gray', interpolation='nearest', aspect='auto')
        ax1.set_title(f'Clean Image {i+1}', fontsize=14)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(noisy_img, cmap='gray', interpolation='nearest', aspect='auto')
        ax2.set_title(f'Noisy Image {i+1}', fontsize=14)
        ax2.axis('off')
        
        # WST coefficients for clean image
        gs_clean = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[0, 1:])
        ax3 = fig.add_subplot(gs[0, 1])
        plot_scattering_disk(clean_coeffs, f'WST Clean {i+1}', ax3, gs_clean)
        
        # WST coefficients for noisy image
        gs_noisy = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1, 1:])
        ax4 = fig.add_subplot(gs[1, 1])
        plot_scattering_disk(noisy_coeffs, f'WST Noisy {i+1}', ax4, gs_noisy)
        
        # Compute and print statistics
        clean_mean = np.mean(clean_coeffs)
        noisy_mean = np.mean(noisy_coeffs)
        clean_std = np.std(clean_coeffs)
        noisy_std = np.std(noisy_coeffs)
        
        print(f"  Clean coeffs - Mean: {clean_mean:.4f}, Std: {clean_std:.4f}")
        print(f"  Noisy coeffs - Mean: {noisy_mean:.4f}, Std: {noisy_std:.4f}")
        print(f"  Difference - Mean: {abs(clean_mean - noisy_mean):.4f}, Std: {abs(clean_std - noisy_std):.4f}")
        print()
        
        plt.suptitle(f'WST Coefficients Comparison: Image Pair {i+1} (σ=50)', fontsize=16)
        
        # Save the figure
        output_path = f"/home/brusc/Projects/random_forest/wst_comparison_sigma50_pair_{i+1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {output_path}")
        
        plt.show()

if __name__ == "__main__":
    compare_images()