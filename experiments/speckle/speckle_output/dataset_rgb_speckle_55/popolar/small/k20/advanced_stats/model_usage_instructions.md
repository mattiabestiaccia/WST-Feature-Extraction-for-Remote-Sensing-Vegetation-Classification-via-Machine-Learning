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
    features = extract_features(img, "advanced_stats")  # You need this function
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
- **Dataset**: small - popolar area
- **Feature Method**: Advanced Statistics
- **Selected Features**: ['R_mean', 'R_skew', 'R_cv', 'R_p10', 'R_p25', 'R_p50', 'R_p75', 'R_p90', 'R_grad_mean', 'G_p10', 'B_mean', 'B_skew', 'B_kurt', 'B_cv', 'B_p10', 'B_p25', 'B_p50', 'B_p75', 'B_iqr', 'B_grad_mean']
- **Model**: Random Forest with 10 estimators
- **Classes**: ['low_veg', 'trees', 'water']
- **Test Accuracy**: 0.7778
- **CV Accuracy**: 0.8667 ± 0.0831
