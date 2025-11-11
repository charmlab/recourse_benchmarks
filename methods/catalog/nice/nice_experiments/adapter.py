"""
Adapters to make NICE_experiments data API-compatible with recourse benchmarks
"""
import pandas as pd
import numpy as np
from typing import List, Dict


class NICEExperimentsDataAdapter:
    """
    Wraps NICE_experiments data to be API-compatible with recourse benchmarks
    
    The data content is different (NICE_experiments preprocessing),
    but the interface "looks like" what the repo expects.
    """
    
    def __init__(self, nice_dataset: dict):
        """
        Parameters
        ----------
        nice_dataset : dict
            Dictionary from PmlbFetcher with keys:
            - X_train, X_test: numpy arrays (label encoded)
            - y_train, y_test: numpy arrays
            - feature_names: list of str
            - cat_feat: list of int (indices)
            - con_feat: list of int (indices)
            - feature_map: dict (categorical value mappings)
        """
        self.nice_dataset = nice_dataset
        
        # Extract data
        X_train = nice_dataset['X_train']
        y_train = nice_dataset['y_train']
        X_test = nice_dataset['X_test']
        y_test = nice_dataset['y_test']
        feature_names = nice_dataset['feature_names']
        cat_indices = nice_dataset['cat_feat']
        num_indices = nice_dataset['con_feat']
        
        # Convert to DataFrames (repo format expects DataFrames)
        self.df_train = pd.DataFrame(X_train, columns=feature_names)
        self.df_train['y'] = y_train
        
        self.df_test = pd.DataFrame(X_test, columns=feature_names)
        self.df_test['y'] = y_test
        
        # Feature lists as names (repo expects feature names, not indices)
        self.categorical = [feature_names[i] for i in cat_indices]
        self.continuous = [feature_names[i] for i in num_indices]
        
        # Store for reference
        self.feature_names = feature_names
        self.cat_indices = cat_indices
        self.num_indices = num_indices
        self.feature_map = nice_dataset['feature_map']
        
        print(f"Data adapter created:")
        print(f"  Train: {self.df_train.shape}")
        print(f"  Test: {self.df_test.shape}")
        print(f"  Categorical features ({len(self.categorical)}): {self.categorical}")
        print(f"  Continuous features ({len(self.continuous)}): {self.continuous}")
        

class NICEExperimentsModelAdapter:
    """
    Wraps sklearn model to be API-compatible with repo's MLModel interface
    """
    
    def __init__(self, sklearn_model, data_adapter: NICEExperimentsDataAdapter):
        """
        Parameters
        ----------
        sklearn_model : sklearn estimator
            Trained model (e.g., RandomForestClassifier)
        data_adapter : NICEExperimentsDataAdapter
            Data adapter object
        """
        self.model = sklearn_model
        self.data = data_adapter
        self._model_trained = True
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for input DataFrame
        """
        # Remove 'y' if present
        if 'y' in df.columns:
            df = df.drop(columns=['y'])
        
        # Ensure correct feature order
        df = df[self.data.feature_names]
        
        # Convert to numpy and predict
        X = df.values
        return self.model.predict_proba(X)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for input DataFrame
        """
        if 'y' in df.columns:
            df = df.drop(columns=['y'])
        
        df = df[self.data.feature_names]
        X = df.values
        return self.model.predict(X)
    
    def get_ordered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has features in correct order
        """
        feature_cols = self.data.feature_names
        
        if 'y' in df.columns:
            return df[feature_cols + ['y']]
        else:
            return df[feature_cols]