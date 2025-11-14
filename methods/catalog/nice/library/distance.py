"""
Distance metrics for NICE algorithm

Adapted from nice/utils/distance.py in NICE repository:
https://github.com/DBrughmans/NICE
"""
import numpy as np
from abc import ABC, abstractmethod


class NumericDistance(ABC):
    """Abstract base class for numerical distance metrics"""
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def measure(self):
        pass


class DistanceMetric(ABC):
    """Abstract base class for distance metrics"""
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def measure(self):
        pass


class StandardDistance(NumericDistance):
    """
    Standard deviation normalization for numerical features.
    
    Distance = |x1 - x2| / std(X_train)
    """
    
    def __init__(self, X_train: np.ndarray, num_feat: list, eps: float):
        self.num_feat = num_feat
        self.scale = X_train[:, num_feat].std(axis=0, dtype=np.float64)
        self.scale[self.scale < eps] = eps
    
    def measure(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate standardized distance for numerical features.
        
        Parameters
        ----------
        X1 : np.ndarray
            First instance (1 x M)
        X2 : np.ndarray
            Instances to compare with (N x M)
        
        Returns
        -------
        np.ndarray
            Distances (N,)
        """
        distance = X2[:, self.num_feat].copy()
        distance = np.abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance, axis=1)
        return distance


class MinMaxDistance(NumericDistance):
    """
    Min-max normalization for numerical features.
    
    Distance = |x1 - x2| / (max - min)
    """
    
    def __init__(self, X_train: np.ndarray, num_feat: list, eps: float):
        self.num_feat = num_feat
        self.scale = (
            X_train[:, num_feat].max(axis=0) - 
            X_train[:, num_feat].min(axis=0)
        )
        self.scale[self.scale < eps] = eps
    
    def measure(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate min-max normalized distance for numerical features.
        
        Parameters
        ----------
        X1 : np.ndarray
            First instance (1 x M)
        X2 : np.ndarray
            Instances to compare with (N x M)
        
        Returns
        -------
        np.ndarray
            Distances (N,)
        """
        distance = X2[:, self.num_feat].copy()
        distance = np.abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance, axis=1)
        return distance


class HEOM(DistanceMetric):
    """
    Heterogeneous Euclidean Overlap Metric.
    
    Combines:
    - Normalized distance for numerical features
    - Overlap metric (0/1) for categorical features
    
    Total distance = numerical_distance + categorical_distance
    """
    
    def __init__(self, data, numeric_distance: NumericDistance):
        self.data = data
        self.numeric_distance = numeric_distance(
            data.X_train, 
            data.num_feat, 
            data.eps
        )
    
    def measure(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate HEOM distance between instances.
        
        Parameters
        ----------
        X1 : np.ndarray
            First instance (1 x M)
        X2 : np.ndarray
            Instances to compare with (N x M)
        
        Returns
        -------
        np.ndarray
            HEOM distances (N,)
        """
        # Numerical distance (normalized)
        num_distance = self.numeric_distance.measure(X1, X2)
        
        # Categorical distance (count of mismatches)
        cat_distance = np.sum(
            X2[:, self.data.cat_feat] != X1[0, self.data.cat_feat],
            axis=1
        )
        
        # Total distance
        distance = num_distance + cat_distance
        return distance


class NearestNeighbour:
    """
    Finds nearest unlike neighbor using a distance metric.
    """
    
    def __init__(self, data, distance_metric: DistanceMetric):
        self.data = data
        self.distance_metric = distance_metric
    
    def find_neighbour(self, X: np.ndarray) -> np.ndarray:
        """
        Find nearest unlike neighbor for instance X.
        
        Parameters
        ----------
        X : np.ndarray
            Instance to find neighbor for (1 x M)
        
        Returns
        -------
        np.ndarray
            Nearest unlike neighbor (1 x M)
        """
        # Calculate distances to all candidates
        distances = self.distance_metric.measure(X, self.data.candidates_view)
        
        # Find minimum
        min_idx = distances.argmin()
        
        # Return nearest neighbor (as copy to avoid reference issues)
        return self.data.candidates_view[min_idx, :].copy()[np.newaxis, :]