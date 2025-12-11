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
- **Dataset**: original - assatigue area
- **Feature Method**: Hybrid (RGB Stats + WST)
- **Selected Features**: ['R_std', 'R_var', 'R_p10', 'R_iqr', 'R_mad', 'R_grad_mean', 'G_std', 'G_var', 'G_kurt', 'G_cv', 'G_p10', 'G_p25', 'G_iqr', 'G_mad', 'B_cv', 'B_p10', 'B_mad', 'B_grad_mean', 'R_wst_std_0', 'G_wst_std_0']
- **Model**: Random Forest with 50 estimators
- **Classes**: ['low_veg', 'water', 'trees']
- **Test Accuracy**: 1.0000
- **CV Accuracy**: 0.9826 ± 0.0213
