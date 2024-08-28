from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.api import MLModel
from recourse_methods.api import RecourseMethod
from recourse_methods.processing import (
    merge_default_parameters,
)

class Gravitational(RecourseMethod):
    """
    Implemention of Gravitational Recourse Algorithm 

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.
    x_center : numpy.array
        A central or sensible point in the feature space of the target class.
        By default, the mean of the instances belonging to the target class will be assign to x_center.
    
    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    prediction_loss:
        Calculates the loss that measures how well the counterfactual is classified as the target class by the model.
    cost:
        Calculates the distance between the original input instance and the generated counterfactua.
    gravitational_penalty:
        Calculates the distance between the generated counterfactual and the central point.
    set_x_center:
        Sets x_center entry to x_center.
    reset_x_center:
        Sets the mean of the instances belonging to the target class to the x_center and return it.
    
    Notes
    -----
    - Restriction
        * Currently working only with PyTorch models
    
    -Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.
        * "prediction_loss_lambda": float, default: 1
            Controls the weight of the prediction loss in the total loss.
        * "original_dist_lambda": float, default: 0.5
            Controls the weight of the original distance in the total loss.
        * "grav_penalty_lambda": float, default: 1.5
            Controls the weight of the gravitational penalty in the total loss.
        * "learning_rate": float, default: 0.01
            Specifies the learning rate for the optimization algorithm.
        * "num_steps": int, default: 500
            Specifies the number of iterations for the optimization process.
        * "target_class": int (0 or 1), default: 1:
            Specifies the desired class for the counterfactual.
        * "scheduler_step_size": int, default: 100
            Step_size for "torch.optim.lr_scheduler.StepLR". Specifies the number of steps or epochs after 
            which the learning rate should be decreased.
        * "scheduler_gamma": float, default: 0.5
            Gamma for "torch.optim.lr_scheduler.StepLR". Specifies the factor by which the learning rate is multiplied 
            at each step when the scheduler is applied.
    
    Implemented from:
        "Endogenous Macrodynamics in Algorithmic Recourse"
        Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. Liem
    """

    _DEFAULT_HYPERPARAMS = {
        "prediction_loss_lambda": 1,
        "original_dist_lambda": 0.5,
        "grav_penalty_lambda": 1.5,
        "learning_rate": 0.01,
        "num_steps": 500,
        "target_class": 1,
        "scheduler_step_size": 100,
        "scheduler_gamma": 0.5
    }

    def __init__(self, mlmodel: MLModel = None, hyperparams: Dict = None, x_center: np.array = None):
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
        self.prediction_loss_lambda = checked_hyperparams["prediction_loss_lambda"]
        self.original_dist_lambda = checked_hyperparams["original_dist_lambda"]
        self.grav_penalty_lambda = checked_hyperparams["grav_penalty_lambda"]
        self.learning_rate = checked_hyperparams["learning_rate"]
        self.num_steps = checked_hyperparams["num_steps"]
        self.target_class = checked_hyperparams["target_class"]
        self.scheduler_step_size = checked_hyperparams["scheduler_step_size"]
        self.scheduler_gamma = checked_hyperparams["scheduler_gamma"]

        self.x_center = x_center

        if self.x_center is None:
            data = self.mlmodel.data
            x_train = data.df_train.drop(data.target, axis=1)
            y_train = data.df_train[data.target]
            self.x_center = np.mean(x_train[y_train == self.target_class], axis=0)

        self.criterion = nn.CrossEntropyLoss()
    
    def prediction_loss(self, model, x_cf, target_class):
        output = model.predict(x_cf.unsqueeze(0))
        loss = self.criterion(output, torch.tensor([target_class], dtype=torch.long))
        return loss
    
    def cost(x_original, x_cf):
        return torch.norm(x_original - x_cf)

    def gravitational_penalty(x_cf, x_center):
        return torch.norm(x_cf - torch.tensor(x_center, dtype=torch.float32))

    def get_counterfactuals(self, factuals: pd.DataFrame):
        x_cf = torch.tensor(factuals.values.flatten(), dtype=torch.float32, requires_grad=True)
            
        optimizer = optim.Adam([x_cf], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            prediction_loss_value = self.prediction_loss(self.mlmodel, x_cf, self.target_class)
            original_dist = self.cost(torch.tensor(factuals.values.flatten(), dtype=torch.float32), x_cf)
            grav_penalty = self.gravitational_penalty(x_cf, self.x_center)

            loss = self.prediction_loss_lambda * prediction_loss_value + self.original_dist_lambda * original_dist + self.grav_penalty_lambda * grav_penalty

            loss.backward()
            optimizer.step()
            scheduler.step()
        
        x_cf_df = pd.DataFrame([x_cf.detach().numpy()], columns=factuals.columns)

        return x_cf_df
    
    def set_x_center(self, x_center):
        self.x_center = x_center
    
    def reset_x_center(self):
        data = self.mlmodel.data()
        x_train = data.df_train().drop(data.target())
        y_train = data.df_train()[data.target()]
        self.x_center = np.mean(x_train[y_train == self.target_class], axis=0)
        return self.x_center