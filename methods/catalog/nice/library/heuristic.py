"""
Optimization heuristics for NICE algorithm

Adapted from nice/utils/optimization/heuristic.py in NICE repository:
https://github.com/DBrughmans/NICE
"""
import numpy as np
from abc import ABC, abstractmethod


class optimization(ABC):
    """Abstract base class for optimization strategies"""
    
    @abstractmethod
    def optimize(self):
        pass


class best_first(optimization):
    """
    Best-first greedy search optimization.
    
    At each iteration:
    1. Try changing each remaining feature
    2. Pick the feature with highest reward
    3. Check if class flipped
    4. Repeat until class flips
    
    Parameters
    ----------
    data : data_NICE
        Data object containing instance info
    reward_function : RewardFunction
        Reward function to guide search
    """
    
    def __init__(self, data, reward_function):
        self.reward_function = reward_function
        self.data = data
    
    def optimize(self, NN: np.ndarray) -> np.ndarray:
        """
        Optimize counterfactual starting from nearest neighbor.
        
        Parameters
        ----------
        NN : np.ndarray
            Nearest unlike neighbor (1 x M)
        
        Returns
        -------
        np.ndarray
            Optimized counterfactual (1 x M)
        """
        # Start with original instance
        CF_candidate = self.data.X.copy()
        
        stop = False
        while not stop:
            # Find features that differ from NN
            diff = np.where(CF_candidate != NN)[1]
            
            # If no more features to change, return NN
            if len(diff) == 0:
                return NN
            
            # Create candidates by changing each differing feature
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]
            
            # Calculate reward and pick best
            CF_candidate = self.reward_function.calculate_reward(
                X_prune, 
                CF_candidate
            )
            
            # Check if class flipped
            if self.data.predict_fn(CF_candidate).argmax() in self.data.target_class:
                return CF_candidate
        
        # Should never reach here due to loop logic
        return CF_candidate