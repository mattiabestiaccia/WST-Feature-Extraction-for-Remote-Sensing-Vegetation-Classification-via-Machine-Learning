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
- **Selected Features**: ['B_mean', 'B_cv', 'B_p10', 'B_p25', 'B_p75', 'R_wst_std_0', 'B_wst_mean_0', 'B_wst_mean_2', 'B_wst_mean_3', 'B_wst_mean_4', 'B_wst_mean_5', 'B_wst_mean_9', 'B_wst_mean_10', 'B_wst_mean_13', 'B_wst_mean_42', 'B_wst_mean_47', 'B_wst_mean_73', 'B_wst_std_16', 'B_wst_std_25', 'B_wst_std_50']
- **Model**: Random Forest with 50 estimators
- **Classes**: ['garden', 'low_veg', 'trees']
- **Test Accuracy**: 1.0000
- **CV Accuracy**: 0.9833 ± 0.0333
