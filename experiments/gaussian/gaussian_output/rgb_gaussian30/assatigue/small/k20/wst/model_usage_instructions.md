# Model Usage Instructions

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
    features = extract_features(img, "wst")  # You need this function
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
- **Dataset**: small - assatigue area
- **Feature Method**: Wavelet Scattering Transform
- **Selected Features**: ['R_wst_mean_0', 'R_wst_mean_54', 'R_wst_std_0', 'G_wst_mean_0', 'G_wst_std_0', 'B_wst_mean_0', 'B_wst_mean_7', 'B_wst_mean_8', 'B_wst_mean_12', 'B_wst_mean_17', 'B_wst_mean_23', 'B_wst_mean_24', 'B_wst_mean_29', 'B_wst_mean_36', 'B_wst_mean_45', 'B_wst_mean_46', 'B_wst_mean_49', 'B_wst_mean_58', 'B_wst_mean_65', 'B_wst_mean_79']
- **Model**: Random Forest with 10 estimators
- **Classes**: ['low_veg', 'trees', 'water']
- **Test Accuracy**: 1.0000
- **CV Accuracy**: 0.8889 ± 0.0703
