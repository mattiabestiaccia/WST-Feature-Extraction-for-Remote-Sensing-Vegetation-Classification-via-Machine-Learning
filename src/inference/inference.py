#!/usr/bin/env python3
"""
Universal Inference Script for Random Forest Models
===================================================

This script performs inference on datasets using any trained Random Forest model 
from the experiments. It automatically detects the area, feature method, and dataset type
from the model directory structure.

Usage:
    python inference_on_dataset.py --model-dir /path/to/model [OPTIONS]

Examples:
    # Full inference auto-detecting area from model directory
    python inference_on_dataset.py --model-dir /path/to/model_dir

    # Sample 10 images per class
    python inference_on_dataset.py --model-dir /path/to/model_dir --sample 10

    # Test on different dataset type
    python inference_on_dataset.py --model-dir /path/to/model_dir --dataset-type mini
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from scipy.ndimage import sobel, laplace
import torch
from kymatio.torch import Scattering2D
import warnings
warnings.filterwarnings('ignore')

# Base dataset directories
DATASET_DIRS = {
    'original': '/home/brusc/Projects/random_forest/dataset_rgb',
    'salt_and_pepper_25': '/home/brusc/Projects/random_forest/dataset_rgb_salt_and_pepper_25',
    'salt_pepper25': '/home/brusc/Projects/random_forest/dataset_rgb_salt_and_pepper_25',
    'gaussian30': '/home/brusc/Projects/random_forest/dataset_rgb_gaussian_30',
    'gaussian50': '/home/brusc/Projects/random_forest/dataset_rgb_gaussian_50',
    'poisson60': '/home/brusc/Projects/random_forest/dataset_rgb_poisson_60',
    'poisson60_new': '/home/brusc/Projects/random_forest/dataset_rgb_poisson_60_new'
}

class ModelInference:
    def __init__(self, model_dir):
        """Initialize the inference system by loading the trained model and components"""
        self.model_dir = model_dir
        self.parse_model_directory()
        self.load_model_components()
        
    def parse_model_directory(self):
        """Parse model directory to extract area, feature method, and dataset information"""
        # Extract area from the last part of the path
        dir_parts = self.model_dir.rstrip('/').split('/')
        last_part = dir_parts[-1]
        
        # Extract area (last component after splitting by underscore)
        parts = last_part.split('_')
        self.area_name = parts[-1]
        
        # Extract feature method and other info from directory structure
        if len(dir_parts) >= 2:
            parent_dir = dir_parts[-2]  # e.g., "experiments"
            if len(dir_parts) >= 3:
                grandparent_dir = dir_parts[-3]  # e.g., "rgb_salt_pepper25_kbest"
                
                # Parse grandparent directory for dataset info
                if 'rgb_' in grandparent_dir:
                    # Extract dataset type from grandparent directory
                    # e.g., "rgb_salt_pepper25_kbest" -> "salt_pepper25"
                    dataset_parts = grandparent_dir.split('_')
                    if len(dataset_parts) >= 2:
                        dataset_type_parts = dataset_parts[1:-1]  # Remove 'rgb' and 'kbest'
                        self.dataset_type_from_dir = '_'.join(dataset_type_parts)
                    else:
                        self.dataset_type_from_dir = 'original'
                else:
                    self.dataset_type_from_dir = 'original'
            else:
                self.dataset_type_from_dir = 'original'
        else:
            self.dataset_type_from_dir = 'original'
        
        # Extract feature method from last_part
        # e.g., "advanced_stats_original_k5_popolar" -> "advanced_stats"
        if 'advanced_stats' in last_part:
            self.feature_method = 'advanced_stats'
        elif 'wst' in last_part:
            self.feature_method = 'wst'
        elif 'hybrid' in last_part:
            self.feature_method = 'hybrid'
        else:
            self.feature_method = 'advanced_stats'  # default
        
        # Extract dataset type from last_part
        # e.g., "advanced_stats_original_k5_popolar" -> "original"
        if 'original' in last_part:
            self.dataset_type_from_name = 'original'
        elif 'mini' in last_part:
            self.dataset_type_from_name = 'mini'
        elif 'small' in last_part:
            self.dataset_type_from_name = 'small'
        else:
            self.dataset_type_from_name = 'original'  # default
        
        print(f"Auto-detected configuration:")
        print(f"  Area: {self.area_name}")
        print(f"  Feature method: {self.feature_method}")
        print(f"  Dataset type (from directory): {self.dataset_type_from_dir}")
        print(f"  Dataset type (from name): {self.dataset_type_from_name}")
        
        # Validate area
        if self.area_name not in ['assatigue', 'popolar', 'sunset']:
            raise ValueError(f"Invalid area detected: {self.area_name}. Must be one of: assatigue, popolar, sunset")
        
    def load_model_components(self):
        """Load all model components"""
        print(f"Loading model components from: {self.model_dir}")
        
        # Load model
        model_path = os.path.join(self.model_dir, 'trained_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Load feature selector
        selector_path = os.path.join(self.model_dir, 'feature_selector.joblib')
        if not os.path.exists(selector_path):
            raise FileNotFoundError(f"Feature selector not found: {selector_path}")
        self.feature_selector = joblib.load(selector_path)
        
        # Load feature information
        features_path = os.path.join(self.model_dir, 'feature_names.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_info = json.load(f)
                self.selected_features = self.feature_info['selected_features']
        else:
            self.feature_info = None
            self.selected_features = None
        
        print(f"Model loaded successfully!")
        print(f"Classes: {list(self.model.classes_)}")
        if self.selected_features:
            print(f"Selected features: {self.selected_features}")
    
    def load_rgb_image(self, file_path):
        """Load RGB PNG image using PIL"""
        image = Image.open(file_path).convert('RGB')
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        return image_array
    
    def extract_basic_features(self, rgb_image):
        """Extract basic statistical features (mean, std) from each RGB channel"""
        features = np.zeros(6)  # 2 features per channel * 3 channels
        
        for i in range(3):  # RGB channels
            channel = rgb_image[i]
            features[2*i] = np.mean(channel)
            features[2*i + 1] = np.std(channel)
        
        return features

    def extract_advanced_features(self, rgb_image):
        """Extract advanced statistical features from each RGB channel"""
        features_per_channel = 18
        features = np.zeros(3 * features_per_channel)
        
        for i in range(3):
            channel = rgb_image[i]
            ch_flat = channel.ravel()
            ch_clean = ch_flat[np.isfinite(ch_flat)]
            
            if len(ch_clean) == 0:
                continue
                
            base = i * features_per_channel
            
            # Basic statistics
            features[base + 0] = np.mean(ch_clean)
            features[base + 1] = np.std(ch_clean)
            features[base + 2] = np.var(ch_clean)
            features[base + 3] = np.min(ch_clean)
            features[base + 4] = np.max(ch_clean)
            features[base + 5] = np.ptp(ch_clean)
            
            # Shape statistics
            features[base + 6] = stats.skew(ch_clean)
            features[base + 7] = stats.kurtosis(ch_clean)
            mean_val = features[base + 0]
            features[base + 8] = features[base + 1] / max(mean_val, 1e-8)
            
            # Percentiles
            features[base + 9] = np.percentile(ch_clean, 10)
            features[base + 10] = np.percentile(ch_clean, 25)
            features[base + 11] = np.percentile(ch_clean, 50)
            features[base + 12] = np.percentile(ch_clean, 75)
            features[base + 13] = np.percentile(ch_clean, 90)
            features[base + 14] = features[base + 12] - features[base + 10]
            
            # MAD
            features[base + 15] = np.mean(np.abs(ch_clean - mean_val))
            
            # Gradient and edge density
            try:
                grad_x = sobel(channel, axis=0)
                grad_y = sobel(channel, axis=1)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                features[base + 16] = np.mean(grad_mag.ravel())
                
                edges = np.abs(laplace(channel))
                edge_thr = np.percentile(edges.ravel(), 90)
                features[base + 17] = np.mean(edges.ravel() > edge_thr)
            except:
                features[base + 16] = 0
                features[base + 17] = 0
        
        return features
    
    def extract_wst_features(self, rgb_image, J=2, L=8):
        """Extract Wavelet Scattering Transform features from each RGB channel"""
        num_channels, height, width = rgb_image.shape
        
        # Initialize scattering transform
        S = Scattering2D(J=J, shape=(height, width), L=L)
        
        all_features = []
        
        for channel_idx in range(num_channels):
            channel = rgb_image[channel_idx]
            
            # Convert to tensor and add batch dimension
            channel_tensor = torch.from_numpy(channel).unsqueeze(0).unsqueeze(0).contiguous()
            
            # Apply scattering transform
            with torch.no_grad():
                coeffs = S(channel_tensor)
            
            # Remove batch and order dimensions: (1, 1, C, H', W') -> (C, H', W')
            coeffs = coeffs.squeeze(0).squeeze(0).numpy()
            
            # Extract statistics (mean, std) for each scattering coefficient
            num_coeffs = coeffs.shape[0]
            channel_features = np.zeros(2 * num_coeffs)
            
            for i in range(num_coeffs):
                coeff = coeffs[i].ravel()
                channel_features[2*i] = np.mean(coeff)
                channel_features[2*i + 1] = np.std(coeff)
            
            all_features.append(channel_features)
        
        return np.concatenate(all_features)
    
    def extract_features(self, rgb_image):
        """Extract features based on the detected method"""
        if self.feature_method == 'advanced_stats':
            return self.extract_advanced_features(rgb_image)
        elif self.feature_method == 'wst':
            # WST + basic features
            basic_features = self.extract_basic_features(rgb_image)
            wst_features = self.extract_wst_features(rgb_image, J=2, L=8)
            return np.hstack([basic_features, wst_features])
        elif self.feature_method == 'hybrid':
            # Hybrid: Advanced stats + WST
            advanced_features = self.extract_advanced_features(rgb_image)
            wst_features = self.extract_wst_features(rgb_image, J=2, L=8)
            return np.hstack([advanced_features, wst_features])
        else:
            raise ValueError(f"Unknown feature method: {self.feature_method}")
    
    def predict_single_image(self, image_path):
        """Predict class for a single image"""
        try:
            # Load and preprocess image
            image = self.load_rgb_image(image_path)
            
            # Extract features using the appropriate method
            features = self.extract_features(image)
            X = np.array([features])
            
            # Apply preprocessing pipeline
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Make prediction
            prediction = self.model.predict(X_selected)[0]
            probabilities = self.model.predict_proba(X_selected)[0]
            
            return {
                'predicted_class': prediction,
                'probabilities': dict(zip(self.model.classes_, probabilities)),
                'confidence': np.max(probabilities),
                'success': True
            }
        except Exception as e:
            return {
                'predicted_class': None,
                'probabilities': None,
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def load_dataset_images(self, dataset_path, area_name, sample_per_class=None):
        """Load images from dataset with optional sampling"""
        area_path = os.path.join(dataset_path, area_name)
        
        if not os.path.exists(area_path):
            raise ValueError(f"Area directory not found: {area_path}")
        
        class_dirs = [d for d in os.listdir(area_path) if os.path.isdir(os.path.join(area_path, d))]
        class_dirs.sort()
        
        print(f"Found classes in {area_name}: {class_dirs}")
        
        all_image_paths = []
        all_labels = []
        
        for class_dir in class_dirs:
            class_path = os.path.join(area_path, class_dir)
            png_files = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
            
            # Sample if requested
            if sample_per_class and len(png_files) > sample_per_class:
                png_files = np.random.choice(png_files, sample_per_class, replace=False)
            
            print(f"Processing {len(png_files)} images from {area_name}/{class_dir}")
            
            for png_file in png_files:
                file_path = os.path.join(class_path, png_file)
                all_image_paths.append(file_path)
                all_labels.append(class_dir)
        
        return all_image_paths, np.array(all_labels)
    
    def get_dataset_path(self, dataset_type_override=None):
        """Get the appropriate dataset path based on detected or override dataset type"""
        if dataset_type_override:
            dataset_type = dataset_type_override
        else:
            dataset_type = self.dataset_type_from_name
        
        # First try to find exact match
        if dataset_type in DATASET_DIRS:
            return DATASET_DIRS[dataset_type]
        
        # Try to find by dataset type from directory parsing
        if self.dataset_type_from_dir in DATASET_DIRS:
            return DATASET_DIRS[self.dataset_type_from_dir]
        
        # Default fallback
        return DATASET_DIRS['original']
    
    def predict_dataset(self, dataset_type_override=None, sample_per_class=None):
        """Perform inference on entire dataset or samples"""
        dataset_path = self.get_dataset_path(dataset_type_override)
        
        print(f"{'='*80}")
        print(f"INFERENCE ON DATASET: {self.area_name.upper()}")
        print(f"Dataset path: {dataset_path}")
        print(f"Dataset type: {dataset_type_override or self.dataset_type_from_name}")
        print(f"Feature method: {self.feature_method}")
        if sample_per_class:
            print(f"Sampling {sample_per_class} images per class")
        print(f"{'='*80}")
        
        # Load dataset
        image_paths, true_labels = self.load_dataset_images(
            dataset_path, self.area_name, sample_per_class
        )
        
        print(f"Total images to process: {len(image_paths)}")
        
        # Perform inference
        predictions = []
        confidences = []
        probabilities_list = []
        successful_predictions = 0
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = self.predict_single_image(image_path)
            
            if result['success']:
                predictions.append(result['predicted_class'])
                confidences.append(result['confidence'])
                probabilities_list.append(result['probabilities'])
                successful_predictions += 1
            else:
                predictions.append('ERROR')
                confidences.append(0.0)
                probabilities_list.append(None)
                print(f"Error processing {image_path}: {result['error']}")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'true_label': true_labels,
            'predicted_label': predictions,
            'confidence': confidences
        })
        
        # Add probability columns
        if probabilities_list and probabilities_list[0] is not None:
            for class_name in self.model.classes_:
                results_df[f'prob_{class_name}'] = [
                    prob[class_name] if prob is not None else 0.0 
                    for prob in probabilities_list
                ]
        
        return results_df, successful_predictions
    
    def evaluate_predictions(self, results_df):
        """Evaluate prediction performance"""
        # Filter out errors
        valid_results = results_df[results_df['predicted_label'] != 'ERROR'].copy()
        
        if len(valid_results) == 0:
            print("No valid predictions to evaluate!")
            return None
        
        true_labels = valid_results['true_label'].values
        predicted_labels = valid_results['predicted_label'].values
        confidences = valid_results['confidence'].values
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Classification report
        class_report = classification_report(
            true_labels, predicted_labels, 
            target_names=self.model.classes_, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=self.model.classes_)
        
        # Per-class accuracy
        per_class_accuracy = {}
        for i, class_name in enumerate(self.model.classes_):
            class_mask = (true_labels == class_name)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(
                    true_labels[class_mask], 
                    predicted_labels[class_mask]
                )
                per_class_accuracy[class_name] = class_acc
        
        evaluation_results = {
            'overall_accuracy': accuracy,
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'classification_report': class_report,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_accuracy,
            'total_predictions': len(valid_results),
            'successful_predictions': len(valid_results)
        }
        
        return evaluation_results
    
    def print_evaluation_results(self, evaluation_results):
        """Print detailed evaluation results"""
        if evaluation_results is None:
            return
        
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        
        print(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.4f}")
        print(f"Mean Confidence: {evaluation_results['mean_confidence']:.4f} ± {evaluation_results['std_confidence']:.4f}")
        print(f"Total Predictions: {evaluation_results['total_predictions']}")
        
        print(f"\nPer-Class Accuracy:")
        for class_name, acc in evaluation_results['per_class_accuracy'].items():
            print(f"  {class_name}: {acc:.4f}")
        
        print(f"\nClassification Report:")
        report = evaluation_results['classification_report']
        for class_name in self.model.classes_:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name:10} - Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
        
        print(f"\nConfusion Matrix:")
        cm = evaluation_results['confusion_matrix']
        print("True\\Predicted", end="")
        for class_name in self.model.classes_:
            print(f"{class_name:>10}", end="")
        print()
        
        for i, true_class in enumerate(self.model.classes_):
            print(f"{true_class:>12}", end="")
            for j in range(len(self.model.classes_)):
                print(f"{cm[i,j]:>10}", end="")
            print()
    
    def save_results(self, results_df, evaluation_results, output_dir, dataset_type_override=None):
        """Save inference results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename suffix
        dataset_type = dataset_type_override or self.dataset_type_from_name
        filename_suffix = f'{self.area_name}_{dataset_type}_{self.feature_method}'
        
        # Save detailed results
        results_path = os.path.join(output_dir, f'inference_results_{filename_suffix}.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Detailed results saved: {results_path}")
        
        # Save evaluation summary
        if evaluation_results:
            summary_path = os.path.join(output_dir, f'evaluation_summary_{filename_suffix}.json')
            
            # Convert numpy arrays to lists for JSON serialization
            eval_json = evaluation_results.copy()
            eval_json['confusion_matrix'] = eval_json['confusion_matrix'].tolist()
            
            # Add configuration information
            eval_json['configuration'] = {
                'area_name': self.area_name,
                'feature_method': self.feature_method,
                'dataset_type': dataset_type,
                'model_directory': self.model_dir
            }
            
            with open(summary_path, 'w') as f:
                json.dump(eval_json, f, indent=2)
            print(f"Evaluation summary saved: {summary_path}")
        
        # Create confusion matrix plot
        if evaluation_results:
            self.plot_confusion_matrix(
                evaluation_results['confusion_matrix'], 
                self.model.classes_,
                os.path.join(output_dir, f'confusion_matrix_{filename_suffix}.png')
            )
    
    def plot_confusion_matrix(self, cm, class_names, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Universal inference script for Random Forest models from experiments'
    )
    parser.add_argument(
        '--model-dir', '-m',
        required=True,
        help='Directory containing trained model (auto-detects area and feature method)'
    )
    parser.add_argument(
        '--dataset-type', '-t',
        default=None,
        choices=['original', 'mini', 'small', 'salt_pepper25', 'gaussian30', 'gaussian50', 'poisson60', 'poisson60_new'],
        help='Override dataset type (default: auto-detect from model directory)'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help='Number of images to sample per class (default: all images)'
    )
    parser.add_argument(
        '--output', '-o',
        default='inference_results',
        help='Output directory for results (default: inference_results)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    try:
        # Initialize inference system (auto-detects area and feature method)
        inference = ModelInference(args.model_dir)
        
        # Perform inference
        results_df, successful_predictions = inference.predict_dataset(
            dataset_type_override=args.dataset_type,
            sample_per_class=args.sample
        )
        
        # Evaluate results
        evaluation_results = inference.evaluate_predictions(results_df)
        
        # Print results
        inference.print_evaluation_results(evaluation_results)
        
        # Save results
        inference.save_results(results_df, evaluation_results, args.output, args.dataset_type)
        
        print(f"\n{'='*80}")
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print(f"Area: {inference.area_name}")
        print(f"Feature method: {inference.feature_method}")
        print(f"Dataset type: {args.dataset_type or inference.dataset_type_from_name}")
        print(f"Processed {len(results_df)} images")
        print(f"Successful predictions: {successful_predictions}")
        if evaluation_results:
            print(f"Overall accuracy: {evaluation_results['overall_accuracy']:.4f}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()