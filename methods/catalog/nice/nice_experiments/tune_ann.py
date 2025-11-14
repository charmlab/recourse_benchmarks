import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from pmlb_fetcher import PmlbFetcher
from adapter import NICEExperimentsDataAdapter, NICEExperimentsModelAdapter

fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
nice_data = fetcher.dataset

data_adapted = NICEExperimentsDataAdapter(nice_data)

# Train data
print(f"\nTrain data:")
print(f"  Shape: {data_adapted.df_train.shape}")
print(f"  Rows: {data_adapted.df_train.shape[0]}")
print(f"  Columns: {data_adapted.df_train.shape[1]}")

# Test data
print(f"\nTest data:")
print(f"  Shape: {data_adapted.df_test.shape}")
print(f"  Rows: {data_adapted.df_test.shape[0]}")
print(f"  Columns: {data_adapted.df_test.shape[1]}")

# Features
print(f"\nFeatures:")
print(f"  Total features: {len(data_adapted.feature_names)}")
print(f"  Feature names: {data_adapted.feature_names}")

print(f"\nCategorical features ({len(data_adapted.categorical)}):")
for feat in data_adapted.categorical:
    print(f"    - {feat}")

print(f"\nContinuous features ({len(data_adapted.continuous)}):")
for feat in data_adapted.continuous:
    print(f"    - {feat}")

# Train model using NICE_experiments data
X_train = data_adapted.df_train.drop(columns=['y']).values
y_train = data_adapted.df_train['y'].values

model = MLPClassifier(random_state=42)

# Hidden layer sizes: 2 to 1.5k, step size=0.15k 
# where k is the input size, i.e., number of features
param_grid = {
    'hidden_layer_sizes': [
        (2,),           
        (4,),
        (6,),
        (8,),
        (10,),
        (13,),
        (15,),  
        (17,),
        (19,),
        (21,),          
    ]
}

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

model_best = grid_search.best_estimator_
X_test = data_adapted.df_test.drop(columns=['y']).values
y_test = data_adapted.df_test['y'].values
test_acc = model_best.score(X_test, y_test)
y_pred_proba = model_best.predict_proba(X_test)[:, 1]  # Get probability for class 1
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")