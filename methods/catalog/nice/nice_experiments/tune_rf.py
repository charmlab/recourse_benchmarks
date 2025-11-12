import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from pmlb_fetcher import PmlbFetcher
from adapter import NICEExperimentsDataAdapter, NICEExperimentsModelAdapter

fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
nice_data = fetcher.dataset

data_adapted = NICEExperimentsDataAdapter(nice_data)

# Train model using NICE_experiments data
X_train = data_adapted.df_train.drop(columns=['y']).values
y_train = data_adapted.df_train['y'].values

model = RandomForestClassifier(random_state=42)

param_grid = {
    # 'n_estimators': [50, 100, 250, 500, 1000],
    'n_estimators': [50, 100, 250, 500],
    'max_depth': [1, 2, 5, 10, 25, None],
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

# note: Test AUC: 0.9168