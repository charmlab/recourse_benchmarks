"""
Data loader from NICE_experiments repository
Fetches and preprocesses data exactly as the original NICE paper did
"""
from pmlb import fetch_data
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


class PmlbFetcher:
    """
    Fetch data from PMLB and preprocess exactly as NICE_experiments does
    
    This replicates the data preprocessing from:
    https://github.com/DBrughmans/NICE_experiments
    """
    
    def __init__(self, name, test_size=0.2, explain_n=200):
        """
        Parameters
        ----------
        name : str
            Dataset name (e.g., 'adult')
        test_size : float
            Test set proportion
        explain_n : int
            Number of instances to explain
        """

        # Fetch data which is originally from from pmlb
        print(f"Fetching {name} dataset from PMLB...")
        # "dataset = fetch_data(name)" wont work in "THE ENVIRONMENT"
        csv_file = os.path.join(os.path.dirname(__file__), f'{name}_pmlb.csv')
        dataset = pd.read_csv(csv_file)
        print(f"Loaded {name} from local file: {dataset.shape}")

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        y = y.values

            
        # Get feature types
        con_feat, cat_feat = self._get_feature_types(X)
        
        # Reorder: categorical first, then continuous
        feature_names = cat_feat + con_feat
        X = X[feature_names].copy()
        X = X.values
        
        # Update indices after reordering
        cat_feat = list(range(len(cat_feat)))
        con_feat = list(range(len(cat_feat), len(cat_feat) + len(con_feat)))

        # Label encode categorical features
        feature_map = {}
        for feat in cat_feat:
            local_map = np.unique(X[:, feat])
            feature_map[feat] = local_map
            for i, v in enumerate(local_map):
                X[X[:, feat] == v, feat] = i

        # Train-test split
        if X.shape[0] * test_size <= explain_n:
            test_size = explain_n
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=test_size, random_state=42
            )
            X_explain = X_test.copy()
            y_explain = y_test.copy()
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=test_size, random_state=42
            )
            _, X_explain, _, y_explain = train_test_split(
                X_test, y_test, stratify=y_test, test_size=explain_n, random_state=42
            )

        print(f'{name}  train:{X_train.shape[0]}  test:{X_test.shape[0]}  explain:{X_explain.shape[0]}')
        
        # Store in dict (same format as NICE_experiments)
        self.dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_explain': X_explain,
            'y_explain': y_explain,
            'feature_names': feature_names,
            'cat_feat': cat_feat,
            'con_feat': con_feat,
            'feature_map': feature_map
        }

    def _get_feature_types(self, X):
        """
        Determine which features are categorical vs continuous
        
        Logic from NICE_experiments:
        - Continuous if has non-integer values
        - Continuous if has > 10 unique values
        - Otherwise categorical
        """
        feature_names = list(X.columns)
        X_na = X.dropna()
        con_feat = []
        cat_feat = []
        
        for feature_name in feature_names:
            x = X_na[feature_name].copy()
            
            # Check if all values are integers
            if not all(float(i).is_integer() for i in x.unique()):
                con_feat.append(feature_name)
            elif x.nunique() > 10:
                con_feat.append(feature_name)
            else:
                cat_feat.append(feature_name)
                
        return con_feat, cat_feat

    def save(self):
        """Save dataset to cache"""
        save_path = os.path.join(self.folder_path, 'data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"Dataset saved to {save_path}")
    
    def load(self):
        """Load dataset from cache"""
        load_path = os.path.join(self.folder_path, 'data.pkl')
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.dataset = pickle.load(f)
            print(f"Dataset loaded from {load_path}")
            return True
        return False