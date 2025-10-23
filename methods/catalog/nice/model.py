from typing import Dict
import pandas as pd
import numpy as np

from methods.api import RecourseMethod
from methods.processing import merge_default_parameters
from models.api import MLModel

# Import from our library folder
from .library.data import data_NICE
from .library.distance import HEOM, MinMaxDistance, NearestNeighbour
from .library.heuristic import best_first
from .library.reward import SparsityReward, ProximityReward, PlausibilityReward
from .library.autoencoder import AutoEncoder  # We'll create this


class NICE(RecourseMethod):
    """
    NICE: Nearest Instance Counterfactual Explanations
    
    Implementation of the NICE algorithm from:
    Brughmans et al. (2024) "NICE: an algorithm for nearest instance 
    counterfactual explanations"
    
    Parameters
    ----------
    mlmodel : MLModel
        Black-box classifier
    hyperparams : dict
        - "optimization": str, default: "sparsity"
            One of ["none", "sparsity", "proximity", "plausibility"]
        - "distance_metric": str, default: "HEOM"
            Distance metric to use
        - "num_normalization": str, default: "minmax"
            Normalization for numerical features ("minmax" or "std")
        - "justified_cf": bool, default: True
            If True, only use correctly classified training instances
    """
    
    _DEFAULT_HYPERPARAMS = {
        "optimization": "sparsity",
        "distance_metric": "HEOM",
        "num_normalization": "minmax",
        "justified_cf": True,
    }
    
    def __init__(self, mlmodel: MLModel, hyperparams: Dict = None):
        # Check backend compatibility
        # NICE is model-agnostic, so we accept all backends
        super().__init__(mlmodel)
        
        # Merge hyperparameters
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        
        self.optimization = checked_hyperparams["optimization"]
        self.distance_metric_name = checked_hyperparams["distance_metric"]
        self.num_normalization = checked_hyperparams["num_normalization"]
        self.justified_cf = checked_hyperparams["justified_cf"]
        
        # Get training data
        df_train = mlmodel.data.df_train
        X_train = df_train.drop(columns=["y"]).values  # Convert to numpy
        y_train = df_train["y"].values
        
        # Get feature information from benchmark's data catalog
        self.categorical_features = mlmodel.data.categorical
        self.continuous_features = mlmodel.data.continuous
        
        # Convert feature names to indices
        feature_names = mlmodel.data.df_train.drop(columns=["y"]).columns.tolist()
        self.cat_feat_idx = [feature_names.index(f) for f in self.categorical_features]
        self.num_feat_idx = [feature_names.index(f) for f in self.continuous_features]
        
        # Store for later use
        self.feature_names = feature_names
        self.mlmodel = mlmodel
        
        # Initialize NICE data object
        self.data = data_NICE(
            X_train=X_train,
            y_train=y_train,
            cat_feat=self.cat_feat_idx,
            num_feat=self.num_feat_idx,
            predict_fn=self._predict_fn_wrapper,
            justified_cf=self.justified_cf,
            eps=1e-10
        )
        
        # Initialize distance metric
        if self.num_normalization == "minmax":
            from .library.distance import MinMaxDistance as NumDistance
        else:
            from .library.distance import StandardDistance as NumDistance
        
        self.distance_metric = HEOM(self.data, NumDistance)
        
        # Initialize nearest neighbor finder
        self.nearest_neighbour = NearestNeighbour(self.data, self.distance_metric)
        
        # Initialize optimizer if needed
        if self.optimization != "none":
            # Initialize reward function
            if self.optimization == "sparsity":
                self.reward_function = SparsityReward(self.data)
            elif self.optimization == "proximity":
                self.reward_function = ProximityReward(
                    self.data,
                    distance_metric=self.distance_metric
                )
            elif self.optimization == "plausibility":
                # Train autoencoder on training data
                ae = AutoEncoder(X_train, self.cat_feat_idx, self.num_feat_idx)
                self.reward_function = PlausibilityReward(
                    self.data,
                    auto_encoder=ae
                )
            else:
                raise ValueError(f"Unknown optimization: {self.optimization}")
            
            # Initialize optimizer
            self.optimizer = best_first(self.data, self.reward_function)
    
    def _predict_fn_wrapper(self, X):
        """
        Wrapper to convert numpy arrays to DataFrames for mlmodel.predict_proba
        """
        # Convert numpy to DataFrame
        df = pd.DataFrame(X, columns=self.feature_names)
        
        # Get predictions
        proba = self.mlmodel.predict_proba(df)
        
        return proba
    
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual explanations for given factuals
        
        Parameters
        ----------
        factuals : pd.DataFrame
            Instances to explain (with 'y' column)
        
        Returns
        -------
        pd.DataFrame
            Counterfactual instances
        """
        counterfactuals_list = []
        
        for index, row in factuals.iterrows():
            # Remove target column if present
            factual = row.drop("y") if "y" in row.index else row
            
            # Convert to numpy array
            X = factual.values.reshape(1, -1)
            
            # Fit data object to this instance
            self.data.fit_to_X(X, target_class='other')
            
            # Find nearest unlike neighbor
            NN = self.nearest_neighbour.find_neighbour(self.data.X)
            
            # Optimize if needed
            if self.optimization != "none":
                CF = self.optimizer.optimize(NN)
            else:
                CF = NN
            
            # Convert back to DataFrame
            cf_df = pd.DataFrame(CF, columns=self.feature_names)
            counterfactuals_list.append(cf_df)
        
        # Concatenate all counterfactuals
        df_cfs = pd.concat(counterfactuals_list, ignore_index=True)
        
        # Ensure correct feature order for model
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        
        return df_cfs