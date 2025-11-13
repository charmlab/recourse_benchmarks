"""
This file contains all relevant logic to perform the LARR method
The original source code can be found at https://github.com/kshitij-kayastha/learning-augmented-robust-recourse
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, List

import lime
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import tqdm


class RecourseCost:

    def __init__(self, x_0, lamb, cost_fn):
        """
        - x_0: The original input vector (the one we want to change).
        
        - lamb: A hyperparameter that balances two competing costs.
        
        - cost_fn: The function used to measure the cost of changing x_0 to a new point x. 
        This defaults to l1_cost (L1-norm, or "Manhattan distance"), which just sums the absolute changes to each feature.
        """
        self.x_0 = x_0
        self.lamb = lamb
        self.cost_fn = cost_fn
    
    def eval(self, x, weights, bias, breakdown = False):
        f_x = 1 / (1 + np.exp(-(np.matmul(x, weights) + bias)))
        bce_loss = -np.log(f_x)
        cost = self.cost_fn(self.x_0, x)
        recourse_cost = bce_loss + self.lamb * cost

        if breakdown is True:
            return bce_loss, cost, recourse_cost
        return recourse_cost

    def eval_nonlinear(self, x, model, breakdown = False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(deepcopy(x)).float()
        
        f_x = model(x)
        loss_fn = torch.nn.BCELoss(reduction='mean')
        bce_loss = loss_fn(f_x, torch.ones(f_x.shape).float()) # This lines assumes our target is "1"
        cost = torch.dist(x, torch.tensor(self.x_0).float(), 1) # taking l1 distance, TODO make sure all tensors are on same device
        recourse_cost = bce_loss + self.lamb * cost

        if breakdown is True:
            return bce_loss.detach().item(), cost.detach().item(), recourse_cost.detach().item()
        return recourse_cost.detach().item()


class Recourse(ABC):
    
    def __init__(self, weights: np.ndarray, bias: np.ndarray, alpha: float, lamb: float, imm_features: List, y_target: float = 1, seed: int|None = None):
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
    def __init__(self, weights: np.ndarray, bias: np.ndarray, alpha: float, lamb: float = 0.1, imm_features: List = [], y_target: float = 1, seed: int|None = None):
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
    
    def sign(self, x):
        s = np.sign(x)
        return 1 if s == 0 else s
    
    def sign_x(self, x: np.float64, direction: int):
        return np.sign(x) if x != 0 else direction
    
    def find_directions(self, weights: np.ndarray):
        directions = np.zeros(weights.size)

        for i, val in enumerate(weights):
            if val > 0:
                directions[i] = 1
            elif val < 0:
                directions[i] = -1
        
        return directions
    
    def get_max_idx(self, weights: np.ndarray, changed: List):
        weights_copy = deepcopy(weights)

        # Can this ever get stuck in infinite loop?
        # can add a fall back just in case that may happen
        current_sum = np.sum(np.abs(weights_copy))

        while True:
            idx = np.argmax(np.abs(weights_copy))
            if not changed[idx]:
                return idx
            else:
                if weights_copy[idx] != 0.0: # fall back check
                    current_sum -= np.abs(weights_copy[idx])

                weights_copy[idx] = 0.0

            if current_sum == 0.0: # fall back just in case
                return None
            
    def get_recourse(self, x_0: np.ndarray, beta: float = 1, theta_p: tuple[np.ndarray, np.ndarray] = None):
        if beta == 1.0:
            return self.get_robust_recourse(x_0)
        elif beta == 0.0:
            return self.get_consistent_recourse(x_0, theta_p)
        else:
            return self.get_augmented_recourse(x_0, theta_p, beta)
        
    def get_robust_recourse(self, x_0: np.ndarray):
        x = deepcopy(x_0)
        weights = np.zeros(self.weights.size)
        active = np.arange(0, self.weights.size)
        imm_features = deepcopy(self.imm_features)
        bias = self.bias - self.alpha

        for i in range(weights.size):
            if x_0[i] != 0:
                weights[i] = self.weights[i] - (self.alpha * np.sign(x_0[i]))
            else:
                if np.abs(self.weights[i]) > self.alpha:
                    weights[i] = self.weights - (self.alpha * np.sign(self.weights[i]))
                else:
                    imm_features.append(i)

        active = np.delete(active, imm_features)
        directions = self.find_directions(weights)

        while active.size != 0:
            i_active = np.argmax(np.abs(weights[active]))

            i = active[i_active]
            c = (x @ weights) + bias
            delta = self.calc_delta(weights[i], c)

            if self.sign_x(x[i] + delta, directions[i]) == self.sign_x(x[i], directions[i]):
                x[i] += delta
                break
            else:
                x[i] = 0
                if np.abs(self.weights[i]) > self.alpha:
                    weights[i] = self.weights[i] + (self.alpha * np.sign(x_0[i]))
                else:
                    active = np.delete(active, i_active)
        
        return x
    
    def get_consistent_recourse(self,  x_0: np.ndarray, theta_p: tuple[np.ndarray, np.ndarray]):
        x = deepcopy(x_0)
        weights, bias = theta_p
        weights_c = np.abs(weights)
        while True:
            i = np.argmax(np.abs(weights_c))
            if i in self.imm_features:
                weights_c[i] = 0
            else:
                break
        
        x_i, w_i = x[i], weights[i]
        c = np.matmul(x, weights) + bias
        delta = self.calc_delta(w_i, c)
        x[i] = x_i + delta

        return x
    
    def get_augmented_recourse(self, x_0: np.ndarray, theta_p: tuple[np.ndarray, np.ndarray], beta: float, eps=1e-5):
        x = deepcopy(x_0)
        J = RecourseCost(x_0, self.lamb)

        for i in range(len(x)):
            if x[i] == 0:
                x[i] += self.rng.normal(0, eps)
        
        weights, bias = self.calc_theta_adv(x)
        weights_p, bias_p = theta_p

        while True:
            min_val = np.inf
            min_i = 0

            for i in range(len(x)):
                if i in self.imm_features:
                    continue
                
                delta = self.calc_augmented_delta(x, i, (weights, bias), (weights_p, bias_p), beta, J)
                if (x[i] == 0) and (x[i] != x_0[i]) and (self.sign(x_0[i]) == self.sign(delta)):
                    delta = 0
                
                x_new = deepcopy(x)
                x_new[i] += delta

                val = (beta * J.eval(x_new, weights, bias)) + ((1 - beta) * J.eval(x_new, weights_p, bias_p))

                if val < min_val:
                    min_val = val
                    min_i = i
                    min_delta = delta

            i = min_i
            delta = min_delta
            x_i = x[i]

            if np.abs(delta) < 1e-9:
                break
            if (np.sign(x_i + delta) == np.sign(x_i)) or (x_i == 0):
                x[i] = x_i + delta
            else:
                x[i] = 0
                weights[i] = self.weights[i] + (self.alpha * np.sign(x_0[i]))
        
        return x
    
    def lime_explanation(self, predict_proba_fn: Callable, X: np.ndarray, x: np.ndarray):
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X, mode='regression', discretize_continuous=False, feature_selection='none')
        exp = explainer.explain_instance(x, predict_proba_fn, num_features=X.shape[1], model_regressor=LogisticRegression())
        weights = exp.local_exp[1][0][1]
        bias = exp.intercept[1]
        return weights, bias
    
    def recourse_validity(self, predict_fn: Callable, recourses: np.ndarray, y_target: float|int = 1):
        return sum(predict_fn(recourses) == y_target) / len(recourses)

    def recourse_expectation(self, predict_proba_fn: Callable, recourses: np.ndarray):
        return sum(predict_proba_fn(recourses)[:,1]) / len(recourses)

    def choose_lambda(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        lambdas = np.arange(0.1, 1.1, 0.1).round(1)
        v_old = 0
        print('Choosing lambda')
        for i in range(len(lambdas)):
            lamb = lambdas[i]
            self.lamb = lamb
            recourses = []
            for xi in tqdm.trange(len(recourse_needed_X), desc=f'lambda={lamb}'):
                x = recourse_needed_X[xi]
                if self.weights is None and self.bias is None:
                    # set seed for lime
                    np.random.seed(xi)
                    weights, bias = self.lime_explanation(predict_fn, X_train, x)
                    weights, bias = np.round(weights, 4), np.round(bias, 4)
                    self.weights = weights
                    self.bias = bias

                    x_r = self.get_robust_recourse(x)

                    self.weights = None
                    self.bias = None
                else:
                    x_r = self.get_robust_recourse(x)
                recourses.append(x_r)

            if predict_proba_fn:
                v = self.recourse_expectation(predict_proba_fn, recourses)
            else:
                v = self.recourse_validity(predict_fn, recourses, self.y_target)
            if v >= v_old:
                v_old = v
            else:
                li = max(0, i - 1)
                return lambdas[li]
        return lamb
    