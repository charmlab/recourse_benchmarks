"""
This file contains all relevant logic to perform the LARR method
The original source code can be found at https://github.com/kshitij-kayastha/learning-augmented-robust-recourse
"""
from abc import ABC, abstractmethod

import numpy as np


class RecourseCost:

    def __init__(self, x_0, lamb, const_fn):
        """
        - x_0: The original input vector (the one we want to change).
        
        - lamb: A hyperparameter that balances two competing costs.
        
        - cost_fn: The function used to measure the cost of changing x_0 to a new point x. 
        This defaults to l1_cost (L1-norm, or "Manhattan distance"), which just sums the absolute changes to each feature.
        """
    
    def eval(self, x, weights, bias):
        ...

    def eval_nonlinear(self, x, model):
        ...



class Recourse(ABC):
    
    def __init__(self, weights, bias, alpha, lamb, imm_features, y_target, seed):
        super.__init__()
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        self.lamb = lamb
        self.y_target = y_target
        self.rng = np.random.default_rng(seed=seed)
        self.imm_features = imm_features
        self.name = "Base"
    
    def calc_theta_adv(self, x: np.ndarray):
        weights_adv = self.weights - (self.alpha * np.sign(x))
        for i in range(len(x)):
            if np.sign(x[i]) == 0:
                weights_adv[i] = weights_adv[i] - (self.alpha * np.sign(weights_adv[i]))
            bias_adv = self.bias - self.alpha
        
        return weights_adv, bias_adv
        

    @abstractmethod
    def get_recourse(self, x, *args, **kwargs):
        ...
    
    @abstractmethod
    def set_weights(self, weights):
        ...

    @abstractmethod
    def set_bais(self, bias):
        ...

class LARRecourse(Recourse):
    def __init__(self, weights, bias, alpha, lamb, imm_features, y_target, seed):
        super().__init__(weights, bias, alpha, lamb, imm_features, y_target, seed)
        self.name = "Alg1"
    
    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

    def calc_delta(self, w: float, c: float):
        if (w > self.lamb):
            delta = ((np.log((w - self.lamb)/self.lamb) - c) / w)
            if delta < 0: delta = 0.
        elif (w < -self.lamb):
            delta = ((np.log((-w - self.lamb)/self.lamb) - c) / w)
            if delta > 0: delta = 0.
        else:
            delta = 0.
        return delta  
    
    def calc_augmented_delta(self, x: np.ndarray, i: int, theta: tuple[np.ndarray, np.ndarray], theta_p: tuple[np.ndarray, np.ndarray], beta: float, J: RecourseCost):
        n = 201
        delta = 10
        deltas = np.linspace(-delta, delta, n)
        
        x_rs = np.tile(x, (n, 1))
        x_rs[:, i] += deltas
        vals = beta*J.eval(x_rs, *theta) + (1-beta)*J.eval(x_rs, *theta_p)
        min_i = np.argmin(vals)
        return deltas[min_i]
    
    