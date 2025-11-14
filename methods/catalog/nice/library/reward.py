"""
Reward functions for NICE optimization

Adapted from nice/utils/optimization/reward.py NICE repository:
https://github.com/DBrughmans/NICE
"""
import numpy as np
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_reward(self):
        pass


class SparsityReward(RewardFunction):
    """
    Reward function that optimizes for sparsity.
    
    Reward = score_improvement
    
    Picks the feature that gives biggest prediction score improvement
    toward the target class.
    """
    
    def __init__(self, data, **kwargs):
        self.data = data
    
    def calculate_reward(
        self, 
        X_prune: np.ndarray, 
        previous_CF_candidate: np.ndarray
    ) -> np.ndarray:
        """
        Calculate reward for candidate counterfactuals.
        
        Parameters
        ----------
        X_prune : np.ndarray
            Candidate counterfactuals (N x M)
        previous_CF_candidate : np.ndarray
            Previous best candidate (1 x M)
        
        Returns
        -------
        np.ndarray
            Best candidate (1 x M)
        """
        # Calculate score improvement
        score_prune = (
            -self.data.predict_fn(previous_CF_candidate) + 
            self.data.predict_fn(X_prune)
        )
        
        # Get improvement toward target class(es)
        score_diff = score_prune[:, self.data.target_class]
        score_diff = score_diff.max(axis=1)
        
        # Pick candidate with highest improvement
        idx_max = np.argmax(score_diff)
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        
        return CF_candidate


class ProximityReward(RewardFunction):
    """
    Reward function that optimizes for proximity.
    
    Reward = score_improvement / distance_increase
    
    Picks the feature that gives best score improvement per unit distance.
    """
    
    def __init__(self, data, distance_metric, **kwargs):
        self.data = data
        self.distance_metric = distance_metric
    
    def calculate_reward(
        self,
        X_prune: np.ndarray,
        previous_CF_candidate: np.ndarray
    ) -> np.ndarray:
        """
        Calculate reward for candidate counterfactuals.
        
        Parameters
        ----------
        X_prune : np.ndarray
            Candidate counterfactuals (N x M)
        previous_CF_candidate : np.ndarray
            Previous best candidate (1 x M)
        
        Returns
        -------
        np.ndarray
            Best candidate (1 x M)
        """
        # Calculate score improvement
        score_diff = (
            self.data.predict_fn(X_prune)[:, self.data.target_class] -
            self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]
        )
        
        # Calculate distance increase
        distance = (
            self.distance_metric.measure(self.data.X, X_prune) -
            self.distance_metric.measure(self.data.X, previous_CF_candidate)
        )
        
        # Reward = score improvement / distance increase
        # Add eps to avoid division by zero
        reward = score_diff / (distance + self.data.eps)[:, np.newaxis]
        
        # Pick candidate with highest reward
        idx_max = np.argmax(reward)
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        
        return CF_candidate


class PlausibilityReward(RewardFunction):
    """
    Reward function that optimizes for plausibility.
    
    Reward = score_improvement * AE_error_decrease
    
    Picks the feature that gives score improvement while keeping
    reconstruction error low (staying on data manifold).
    """
    
    def __init__(self, data, auto_encoder, **kwargs):
        self.data = data
        self.auto_encoder = auto_encoder
    
    def calculate_reward(
        self,
        X_prune: np.ndarray,
        previous_CF_candidate: np.ndarray
    ) -> np.ndarray:
        """
        Calculate reward for candidate counterfactuals.
        
        Parameters
        ----------
        X_prune : np.ndarray
            Candidate counterfactuals (N x M)
        previous_CF_candidate : np.ndarray
            Previous best candidate (1 x M)
        
        Returns
        -------
        np.ndarray
            Best candidate (1 x M)
        """
        # Calculate score improvement
        score_diff = (
            self.data.predict_fn(X_prune)[:, self.data.target_class] -
            self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]
        )
        
        # Calculate AE error difference
        # Negative means error decreased (good!)
        AE_loss_diff = (
            self.auto_encoder(previous_CF_candidate) -
            self.auto_encoder(X_prune)
        )
        
        # Reward = score improvement * error decrease
        # Higher reward when score improves AND error decreases
        reward = score_diff * AE_loss_diff[:, np.newaxis]
        
        # Pick candidate with highest reward
        idx_max = np.argmax(reward)
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        
        return CF_candidate