#!/usr/bin/env python3
"""
Flexible Random Forest training script with command line parameters.

This script trains a Random Forest classifier with configurable parameters and saves the trained model.

Usage:
    python train_and_save_model.py [dataset_path] [area_name] [feature_method] [k_features] [output_dir] [options]

Arguments:
    dataset_path    Path to the dataset directory
    area_name       Area name (assatigue, popolar, sunset)
    feature_method  Feature extraction method (advanced_stats, wst, hybrid)
    k_features      Number of features to select (2, 5, 10, 20)
    output_dir      Output directory for model and results
    
Optional arguments:
    --n_estimators  Number of trees in the forest (default: 50)
    --test_size     Test set size fraction (default: 0.2)
    --random_state  Random seed (default: 42)
    --cv_folds      Cross-validation folds (default: 5)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from scipy.ndimage import sobel, laplace
import warnings
warnings.filterwarnings('ignore')

# Add kymatio import for WST (will be used if feature_method includes WST)
try:
    from kymatio.numpy import Scattering2D
except ImportError:
    print("Warning: kymatio not available. WST and hybrid features will not work.")
    Scattering2D = None

def load_rgb_image(file_path):
    """Load RGB PNG image using PIL"""
    image = Image.open(file_path).convert('RGB')
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return image_array

def extract_advanced_features(rgb_image):
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
        features[base + 5] = np.ptp(ch_clean)  # range
        
        # Shape statistics
        features[base + 6] = stats.skew(ch_clean)
        features[base + 7] = stats.kurtosis(ch_clean)
        mean_val = features[base + 0]
        features[base + 8] = features[base + 1] / max(mean_val, 1e-8)  # coefficient of variation
        
        # Percentiles
        features[base + 9] = np.percentile(ch_clean, 10)
        features[base + 10] = np.percentile(ch_clean, 25)
        features[base + 11] = np.percentile(ch_clean, 50)
        features[base + 12] = np.percentile(ch_clean, 75)
        features[base + 13] = np.percentile(ch_clean, 90)
        features[base + 14] = features[base + 12] - features[base + 10]  # IQR
        
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

def load_area_images_and_labels(root_dir, area_name):
    """Load RGB images and their labels from a specific area"""
    images = []
    labels = []
    
    area_path = os.path.join(root_dir, area_name)
    
    if not os.path.exists(area_path):
        raise ValueError(f"Area directory not found: {area_path}")
    
    class_dirs = [d for d in os.listdir(area_path) if os.path.isdir(os.path.join(area_path, d))]
    class_dirs.sort()
    
    print(f"Found classes in {area_name}: {class_dirs}")
    
    for class_dir in class_dirs:
        class_path = os.path.join(area_path, class_dir)
        png_files = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
        
        print(f"Loading {len(png_files)} images from {area_name}/{class_dir}")
        
        for png_file in tqdm(png_files, desc=f"Loading {area_name}/{class_dir}"):
            file_path = os.path.join(class_path, png_file)
            try:
                image = load_rgb_image(file_path)
                images.append(image)
                labels.append(class_dir)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    return images, np.array(labels)

def select_features_kbest(X, y, feature_names, k=5):
    """Apply SelectKBest with Mutual Information"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)
    
    selected_indices = selector.get_support(indices=True)
    
    # Ensure we have enough feature names
    if len(feature_names) < X.shape[1]:
        # Pad with generic names if needed
        feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), X.shape[1])]
    
    selected_features = [feature_names[i] for i in selected_indices]
    feature_scores = selector.scores_[selected_indices]
    
    return X_selected, selected_features, feature_scores, scaler, selector

def train_final_model(X, y, test_size=0.2, random_state=42, n_estimators=50, cv_folds=5):
    """Train the final model on full dataset with train/test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation on full dataset
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    
    return rf, {
        'test_accuracy': test_accuracy,
        'cv_mean_accuracy': float(np.mean(cv_scores)),
        'cv_std_accuracy': float(np.std(cv_scores)),
        'cv_scores': cv_scores.tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

def save_model_and_artifacts(model, scaler, selector, selected_features, feature_scores, 
                           performance_results, dataset_info, config):
    """Save the trained model and all artifacts"""
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the Random Forest model
    model_path = os.path.join(output_dir, 'trained_model.joblib')
    joblib.dump(model, model_path)
    print(f"Saved trained model: {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler: {scaler_path}")
    
    # Save the feature selector
    selector_path = os.path.join(output_dir, 'feature_selector.joblib')
    joblib.dump(selector, selector_path)
    print(f"Saved feature selector: {selector_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, 'feature_names.json')
    with open(features_path, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'feature_scores': feature_scores.tolist() if isinstance(feature_scores, np.ndarray) else feature_scores,
            'all_feature_names': get_feature_names(config['feature_method'])
        }, f, indent=2)
    print(f"Saved feature information: {features_path}")
    
    # Save comprehensive experiment report
    experiment_name = f"{config['feature_method']}_{config['area_name']}_k{config['k_features']}_WithModel"
    report = {
        "experiment_name": experiment_name,
        "config": config,
        "dataset_info": dataset_info,
        "feature_selection": {
            "method": f"SelectKBest_k{config['k_features']}",
            "num_features": len(selected_features),
            "selected_features": selected_features,
            "feature_scores": feature_scores.tolist() if isinstance(feature_scores, np.ndarray) else feature_scores
        },
        "performance": performance_results,
        "model_files": {
            "trained_model": "trained_model.joblib",
            "scaler": "scaler.joblib",
            "feature_selector": "feature_selector.joblib",
            "feature_names": "feature_names.json"
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    report_path = os.path.join(output_dir, 'experiment_report_with_model.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved experiment report: {report_path}")
    
    # Save model usage instructions
    feature_method_desc = {
        'advanced_stats': 'Advanced Statistics',
        'wst': 'Wavelet Scattering Transform',
        'hybrid': 'Hybrid (RGB Stats + WST)'
    }
    
    usage_instructions = f'''# Model Usage Instructions

## Loading the Model
```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# Load the trained model
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.joblib')
feature_selector = joblib.load('feature_selector.joblib')

# Load feature information
with open('feature_names.json', 'r') as f:
    feature_info = json.load(f)
    selected_features = feature_info['selected_features']
```

## Making Predictions
```python
# Assuming you have new_images (list of RGB images) and extract_features function
feature_data = []
for img in new_images:
    features = extract_features(img, "{config['feature_method']}")  # You need this function
    feature_data.append(features)

X_new = np.array(feature_data)

# Apply the same preprocessing pipeline
X_new_scaled = scaler.transform(X_new)
X_new_selected = feature_selector.transform(X_new_scaled)

# Make predictions
predictions = model.predict(X_new_selected)
prediction_probabilities = model.predict_proba(X_new_selected)
```

## Model Details
- **Dataset**: {dataset_info['dataset_type']} - {config['area_name']} area
- **Feature Method**: {feature_method_desc.get(config['feature_method'], config['feature_method'])}
- **Selected Features**: {selected_features}
- **Model**: Random Forest with {config['n_estimators']} estimators
- **Classes**: {list(dataset_info['classes'].keys())}
- **Test Accuracy**: {performance_results['test_accuracy']:.4f}
- **CV Accuracy**: {performance_results['cv_mean_accuracy']:.4f} ± {performance_results['cv_std_accuracy']:.4f}
'''
    
    usage_path = os.path.join(output_dir, 'model_usage_instructions.md')
    with open(usage_path, 'w') as f:
        f.write(usage_instructions)
    print(f"Saved usage instructions: {usage_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Random Forest classifier with configurable parameters')
    
    # Required arguments
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('area_name', choices=['assatigue', 'popolar', 'sunset'], 
                       help='Area name (assatigue, popolar, sunset)')
    parser.add_argument('feature_method', choices=['advanced_stats', 'wst', 'hybrid'],
                       help='Feature extraction method (advanced_stats, wst, hybrid)')
    parser.add_argument('k_features', type=int, choices=[2, 5, 10, 20],
                       help='Number of features to select (2, 5, 10, 20)')
    parser.add_argument('output_dir', help='Output directory for model and results')
    
    # Optional arguments
    parser.add_argument('--n_estimators', type=int, default=50,
                       help='Number of trees in the forest (default: 50)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size fraction (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Cross-validation folds (default: 5)')
    
    return parser.parse_args()

def extract_wst_features(rgb_image):
    """Extract Wavelet Scattering Transform features from RGB image using mean and std of coefficients"""
    if Scattering2D is None:
        raise ImportError("kymatio not available. Cannot extract WST features.")
    
    # Initialize scattering transform
    J = 2  # Number of scales
    L = 8  # Number of angles
    
    # Get image dimensions
    C, H, W = rgb_image.shape
    
    # Initialize scattering for this image size
    scattering = Scattering2D(J=J, L=L, shape=(H, W))
    
    all_features = []
    
    # Process each channel
    for c in range(C):
        channel = rgb_image[c]
        
        # Compute scattering coefficients
        scattering_coeffs = scattering(channel)
        
        # Calculate mean and std across spatial dimensions for each coefficient
        coeffs_mean = np.mean(scattering_coeffs, axis=(-2, -1))
        coeffs_std = np.std(scattering_coeffs, axis=(-2, -1))
        
        # Combine mean and std features for this channel
        channel_features = np.concatenate([coeffs_mean, coeffs_std])
        all_features.extend(channel_features)
    
    return np.array(all_features)

def extract_hybrid_features(rgb_image):
    """Extract hybrid features (combination of advanced stats and WST)"""
    advanced_features = extract_advanced_features(rgb_image)
    wst_features = extract_wst_features(rgb_image)
    
    # Combine features
    hybrid_features = np.concatenate([advanced_features, wst_features])
    return hybrid_features

def extract_features(rgb_image, feature_method):
    """Extract features based on the specified method"""
    if feature_method == 'advanced_stats':
        return extract_advanced_features(rgb_image)
    elif feature_method == 'wst':
        return extract_wst_features(rgb_image)
    elif feature_method == 'hybrid':
        return extract_hybrid_features(rgb_image)
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")

def get_feature_names(feature_method):
    """Get feature names based on the extraction method"""
    if feature_method == 'advanced_stats':
        stat_names = [
            'mean', 'std', 'var', 'min', 'max', 'range', 'skew', 'kurt', 'cv',
            'p10', 'p25', 'p50', 'p75', 'p90', 'iqr', 'mad', 'grad_mean', 'edge_density'
        ]
        return [f"{c}_{stat}" for c in ['R', 'G', 'B'] for stat in stat_names]
    elif feature_method == 'wst':
        # WST feature names - now includes mean and std for each coefficient per channel
        # With J=2, L=8, we get approximately 81 coefficients per channel
        # For 3 channels (RGB) and 2 statistics (mean, std): 3 * 81 * 2 = 486 features
        wst_names = []
        channels = ['R', 'G', 'B']
        stats = ['mean', 'std']
        
        for channel in channels:
            for stat in stats:
                for i in range(81):  # Approximate number of scattering coefficients
                    wst_names.append(f"{channel}_wst_{stat}_{i}")
        
        return wst_names
    elif feature_method == 'hybrid':
        advanced_names = get_feature_names('advanced_stats')
        wst_names = get_feature_names('wst')
        return advanced_names + wst_names
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine n_estimators based on dataset type if not specified
    dataset_type = os.path.basename(args.dataset_path)
    if args.n_estimators == 50:  # Default value, user didn't specify
        if 'mini' in dataset_type:
            n_estimators = 3
        elif 'small' in dataset_type:
            n_estimators = 10
        elif 'original' in dataset_type:
            n_estimators = 50
        else:
            n_estimators = 50  # Default fallback
    else:
        n_estimators = args.n_estimators
    
    # Create config from arguments
    config = {
        'dataset_path': args.dataset_path,
        'area_name': args.area_name,
        'feature_method': args.feature_method,
        'k_features': args.k_features,
        'output_dir': args.output_dir,
        'n_estimators': n_estimators,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'cv_folds': args.cv_folds
    }
    
    print("="*80)
    print("CONFIGURABLE RANDOM FOREST TRAINING SCRIPT")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Area: {config['area_name']}")
    print(f"Feature Method: {config['feature_method']}")
    print(f"K Features: {config['k_features']}")
    print(f"Output: {config['output_dir']}")
    print(f"N Estimators: {config['n_estimators']}")
    print("="*80)
    
    # Load images and labels
    print(f"Loading images from {config['area_name']} area...")
    images, labels = load_area_images_and_labels(config['dataset_path'], config['area_name'])
    
    if len(images) == 0:
        raise ValueError(f"No images found for area: {config['area_name']}")
    
    print(f"Loaded {len(images)} images with {len(np.unique(labels))} classes")
    print(f"Classes: {list(np.unique(labels))}")
    print(f"Image shape: {images[0].shape}")
    
    # Extract features
    print(f"Extracting {config['feature_method']} features...")
    feature_data = []
    feature_names = get_feature_names(config['feature_method'])
    
    for img in tqdm(images, desc="Extracting features"):
        features = extract_features(img, config['feature_method'])
        feature_data.append(features)
    
    X = np.array(feature_data)
    print(f"Feature matrix shape: {X.shape}")
    
    # Dataset info
    class_counts = pd.Series(labels).value_counts()
    dataset_info = {
        "data_directory": config['dataset_path'],
        "area_name": config['area_name'],
        "total_images": len(images),
        "classes": class_counts.to_dict(),
        "image_shape": list(images[0].shape),
        "total_features_available": len(feature_names),
        "feature_method": config['feature_method'],
        "dataset_type": os.path.basename(config['dataset_path']),
        "k_features": config['k_features']
    }
    
    # Feature selection
    print(f"Applying SelectKBest feature selection (k={config['k_features']})...")
    X_selected, selected_features, feature_scores, scaler, selector = select_features_kbest(
        X, labels, feature_names, k=config['k_features']
    )
    
    print(f"Selected features: {selected_features}")
    print(f"Feature scores: {feature_scores}")
    
    # Train final model
    print("Training final Random Forest model...")
    model, performance_results = train_final_model(
        X_selected, labels, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        n_estimators=config['n_estimators'],
        cv_folds=config['cv_folds']
    )
    
    # Save everything
    print("Saving model and artifacts...")
    save_model_and_artifacts(
        model, scaler, selector, selected_features, feature_scores,
        performance_results, dataset_info, config
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Area: {config['area_name']}")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Feature method: {config['feature_method']}")
    print(f"Total images: {dataset_info['total_images']}")
    print(f"Classes: {list(dataset_info['classes'].keys())}")
    print(f"Original features: {dataset_info['total_features_available']}")
    print(f"Selected features: {config['k_features']}")
    print(f"N estimators: {config['n_estimators']}")
    print(f"\nPERFORMANCE RESULTS:")
    print(f"Test Accuracy: {performance_results['test_accuracy']:.4f}")
    print(f"CV Accuracy: {performance_results['cv_mean_accuracy']:.4f} ± {performance_results['cv_std_accuracy']:.4f}")
    print(f"\nTop {len(selected_features)} selected features:")
    for i, (feat, score) in enumerate(zip(selected_features, feature_scores)):
        print(f"  {i+1}. {feat}: {score:.4f}")
    print(f"\nModel and artifacts saved to: {config['output_dir']}")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)