from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.api import MLModel
from recourse_methods.api import RecourseMethod
from recourse_methods.processing import merge_default_parameters



class ClaPROAR(RecourseMethod):
    """
    Implemention of ClaPROAR Recourse Algorithm

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    compute_costs:
        Compute sum of all costs
    compute_yloss:
        Compute the outcome loss between model's prediction for x_prime and the desired outcome y_star.
    compute_individual_cost:
        Compute the individual cost. (Euclidean distance between x and x_prime)
    compute_external_cost:
        Compute the external cost. (The change in model loss when the new point x_prime is added)
    
    Notes
    -----
    - Restriction
        * Currently working only with Pytorch models

    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.
        * "individual_cost_lambda": float, default: 0.1
            Controls the weight of the individual cost.
        * "external_cost_lambda": float, default: 0.1
            Controls the weight of the external cost.
        * "learning_rate": ifloat, default: 0.01
            Controls how large the steps taken by the optimizer.
        * "max_iter": int, default: 100
            Maximum number of iterations.
        * "tol": float, default: 1e-4
            This is the tolerance for convergence, which sets a threshold for the gradient norm. If the gradient norm falls below this value, 
            the optimization process will stop
        * "target_class": int (0 or 1), default: 1
            Desired output class.

    Implemented from:
        "Endogenous Macrodynamics in Algorithmic Recourse"
        Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. 
    """

    _DEFAULT_HYPERPARAMS = {
        "individual_cost_lambda": 0.1,
        "external_cost_lambda": 0.1,
        "learning_rate": 0.01,
        "max_iter": 100,
        "tol": 1e-4,
        "target_class": 1,
    }

    def __init__(self, mlmodel: MLModel = None, hyperparams: Dict = None):
        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.mlmodel = mlmodel
        self.individual_cost_lambda = checked_hyperparams["individual_cost_lambda"]
        self.external_cost_lambda = checked_hyperparams["external_cost_lambda"]
        self.learning_rate = checked_hyperparams["learning_rate"]
        self.max_iter = checked_hyperparams["max_iter"]
        self.tol = checked_hyperparams["tol"]
        self.target_class = checked_hyperparams["target_class"]

        self.criterion = nn.BCELoss()

    def compute_yloss(self, x_prime, y_star):
        output = self.mlmodel(x_prime)
        yloss = self.criterion(output, y_star)
        return yloss

    def compute_individual_cost(self, x, x_prime):
        return torch.norm(x - x_prime)

    def compute_external_cost(self, x_prime, y_true):
        output = self.mlmodel(x_prime)
        ext_cost = self.criterion(output, y_true)
        return ext_cost

    def compute_costs(self, x, x_prime, y_star, y_true):
        yloss = self.compute_yloss(x_prime, y_star)
        individual_cost = self.compute_individual_cost(x, x_prime)
        external_cost = self.compute_external_cost(x_prime, y_true)
        
        return yloss + self.individual_cost_lambda * individual_cost + self.external_cost_lambda * external_cost

    def get_counterfactuals(self, factuals: pd.DataFrame):

        x = torch.tensor(factuals.values, dtype=torch.float32)
        y_star = torch.tensor([[self.target_class]], dtype=torch.float32)
        y_true = torch.tensor([[1 - self.target_class]], dtype=torch.float32)

        x_prime = x.clone().detach().requires_grad_(True)
        optimizer_cf = optim.Adam([x_prime], lr=self.learning_rate)
        
        for i in range(self.max_iter):
            optimizer_cf.zero_grad()
            
            objective = self.compute_costs(x, x_prime, y_star, y_true)
            
            objective.backward()
            
            optimizer_cf.step()
            
            if torch.norm(x_prime.grad) < self.tol:
                print(f'Converged at iteration {i+1}')
                break

        cfs = x_prime.detach()
        cfs_df = pd.DataFrame(cfs.numpy(), columns=factuals.columns)

        return cfs_df