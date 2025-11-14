import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from pmlb_fetcher import PmlbFetcher
from adapter import NICEExperimentsDataAdapter
from preprocessing import OHE_minmax  

# Load raw data
fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
nice_data = fetcher.dataset

# Create data adapter
data_adapted = NICEExperimentsDataAdapter(nice_data)

# Create preprocessor (matching NICE_experiments)
preprocessor = OHE_minmax(  
    cat_feat=nice_data['cat_feat'],
    con_feat=nice_data['con_feat']
)

# Get raw training data
X_train_raw = data_adapted.df_train.drop(columns=['y']).values
y_train = data_adapted.df_train['y'].values

# Fit preprocessor and transform
preprocessor.fit(X_train_raw)  
X_train = preprocessor.transform(X_train_raw)  

# Data info (after preprocessing)
print("="*80)
print("DATA INFORMATION (After Preprocessing)")
print("="*80)
print(f"\nTrain data:")
print(f"  Shape: {X_train.shape}")
print(f"  Rows: {X_train.shape[0]}")
print(f"  Columns (after OHE): {X_train.shape[1]}")  

print(f"\nOriginal features: {len(data_adapted.feature_names)}")
print(f"After One-Hot Encoding: {X_train.shape[1]} features")

print(f"\nCategorical features ({len(data_adapted.categorical)}):")
for feat in data_adapted.categorical:
    print(f"    - {feat}")

print(f"\nContinuous features ({len(data_adapted.continuous)}):")
for feat in data_adapted.continuous:
    print(f"    - {feat}")

# Model
model = MLPClassifier(random_state=42, max_iter=300)  

# Hidden layer sizes based on preprocessed input size
k = X_train.shape[1] 
print(f"\nInput size (k) after preprocessing: {k}")
print(f"Grid search range: 2 to {int(1.5*k)} (step ~{int(0.15*k)})")

param_grid = {
    'hidden_layer_sizes': [
        (int(2 + i * 0.15 * k),)  
        for i in range(11)  
    ]
}

print(f"Hidden layer sizes to test: {param_grid['hidden_layer_sizes']}")

print("\nStarting grid search...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Evaluate on test set
model_best = grid_search.best_estimator_

X_test_raw = data_adapted.df_test.drop(columns=['y']).values
X_test = preprocessor.transform(X_test_raw)  
y_test = data_adapted.df_test['y'].values

test_acc = model_best.score(X_test, y_test)
y_pred_proba = model_best.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save best model
import pickle
with open('best_mlp_model.pkl', 'wb') as f:
    pickle.dump(model_best, f)
print("\nBest model saved to best_mlp_model.pkl")

# Also save preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("Preprocessor saved to preprocessor.pkl")