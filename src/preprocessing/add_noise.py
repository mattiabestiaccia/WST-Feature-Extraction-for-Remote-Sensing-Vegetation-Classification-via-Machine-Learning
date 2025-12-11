#!/usr/bin/env python3
"""
Script to add different types of noise to images in the dataset directory.
Creates new folders with noisy versions of the images.
"""

import os
import argparse
import numpy as np
from PIL import Image
import random
from pathlib import Path

def add_gaussian_noise(image_array, intensity):
    """Add Gaussian noise to image."""
    row, col, ch = image_array.shape
    mean = 0
    sigma = intensity * 255 / 100  # Convert percentage to pixel value range
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image_array + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image_array, intensity):
    """Add salt and pepper noise to image."""
    row, col, ch = image_array.shape
    s_vs_p = 0.5  # Salt vs pepper ratio
    amount = intensity / 100  # Convert percentage to probability
    
    # Copy the image
    noisy = np.copy(image_array)
    
    # Salt noise
    num_salt = np.ceil(amount * image_array.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    noisy[coords[0], coords[1], :] = 255
    
    # Pepper noise
    num_pepper = np.ceil(amount * image_array.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy

def add_speckle_noise(image_array, intensity):
    """Add speckle noise to image."""
    row, col, ch = image_array.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    
    # Scale intensity
    noise_factor = intensity / 100
    noisy = image_array + image_array * gauss * noise_factor
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_poisson_noise(image_array, intensity):
    """Add Poisson noise to image."""
    # Scale image based on intensity (higher intensity = more noise)
    # Use a scaling factor that preserves image brightness
    scale_factor = 10 + (intensity / 100) * 90  # Range from 10 to 100
    scaled = image_array * scale_factor / 255.0
    
    # Add Poisson noise
    noisy = np.random.poisson(scaled) * 255.0 / scale_factor
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_uniform_noise(image_array, intensity):
    """Add uniform noise to image."""
    row, col, ch = image_array.shape
    noise_range = intensity * 255 / 100  # Convert percentage to pixel value range
    noise = np.random.uniform(-noise_range/2, noise_range/2, (row, col, ch))
    noisy = image_array + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def process_image(input_path, output_path, noise_type, intensity):
    """Process a single image by adding noise."""
    try:
        # Load image
        image = Image.open(input_path)
        image_array = np.array(image)
        
        # Add noise based on type
        if noise_type == 'gaussian':
            noisy_array = add_gaussian_noise(image_array, intensity)
        elif noise_type == 'salt_and_pepper':
            noisy_array = add_salt_and_pepper_noise(image_array, intensity)
        elif noise_type == 'speckle':
            noisy_array = add_speckle_noise(image_array, intensity)
        elif noise_type == 'poisson':
            noisy_array = add_poisson_noise(image_array, intensity)
        elif noise_type == 'uniform':
            noisy_array = add_uniform_noise(image_array, intensity)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Save noisy image
        noisy_image = Image.fromarray(noisy_array)
        noisy_image.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def create_output_directory(base_path, noise_type, intensity):
    """Create output directory with naming convention."""
    parent_dir = base_path.parent / f"datasets_{noise_type}_{intensity}"
    output_dir = parent_dir / f"dataset_rgb_{noise_type}_{intensity}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def copy_directory_structure(source_dir, dest_dir):
    """Copy directory structure without files."""
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, source_dir)
        
        # Create corresponding directory in destination
        if rel_path != '.':
            dest_path = os.path.join(dest_dir, rel_path)
            Path(dest_path).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Add noise to images in datasets directory')
    parser.add_argument('--noise-type', '-t', 
                      choices=['gaussian', 'salt_and_pepper', 'speckle', 'poisson', 'uniform'],
                      required=True,
                      help='Type of noise to add')
    parser.add_argument('--intensity', '-i', 
                      type=int, 
                      required=True,
                      help='Noise intensity (0-100)')
    parser.add_argument('--input-dir', '-d',
                      default='/home/brusc/Projects/random_forest/datasets',
                      help='Input directory containing images')
    parser.add_argument('--seed', '-s',
                      type=int,
                      default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate intensity
    if not 0 <= args.intensity <= 100:
        print("Error: Intensity must be between 0 and 100")
        return
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Check if input directory exists
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Create output directory
    output_dir = create_output_directory(input_dir, args.noise_type, args.intensity)
    print(f"Output directory: {output_dir}")
    
    # Copy directory structure
    copy_directory_structure(input_dir, output_dir)
    
    # Process all images
    processed_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                # Calculate relative path and create output path
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process image
                if process_image(input_path, output_path, args.noise_type, args.intensity):
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} images...")
                else:
                    error_count += 1
    
    print(f"\nCompleted!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count} images")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()