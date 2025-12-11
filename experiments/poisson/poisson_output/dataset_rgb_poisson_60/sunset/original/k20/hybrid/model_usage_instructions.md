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
    features = extract_features(img, "hybrid")  # You need this function
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
- **Dataset**: original - sunset area
- **Feature Method**: Hybrid (RGB Stats + WST)
- **Selected Features**: ['R_std', 'R_var', 'R_iqr', 'R_mad', 'G_std', 'G_var', 'B_skew', 'R_wst_std_0', 'R_wst_std_9', 'R_wst_std_11', 'R_wst_std_12', 'R_wst_std_13', 'G_wst_mean_10', 'G_wst_mean_12', 'G_wst_mean_14', 'G_wst_std_12', 'B_wst_mean_12', 'B_wst_mean_13', 'B_wst_mean_15', 'B_wst_std_10']
- **Model**: Random Forest with 50 estimators
- **Classes**: ['garden', 'low_veg', 'trees']
- **Test Accuracy**: 0.9583
- **CV Accuracy**: 0.9917 ± 0.0167
