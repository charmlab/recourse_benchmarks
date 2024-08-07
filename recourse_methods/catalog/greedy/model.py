from typing import Dict

import pandas as pd
import tensorflow as tf

from models.api import MLModel
from recourse_methods.api import RecourseMethod
from recourse_methods.processing import (
    merge_default_parameters,
)


class Greedy(RecourseMethod):
    """
    Implemention of Greedy Recourse Algorithm 

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
    
    Notes
    -----
    - Restriction
        * Currently working only with Tensorflow models
    
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.
        * "lambda_param": float, default: 0.05
            Hyperparameter to make balance between distance loss and classification loss.
        * "step_size": float, default: 0.05
            Learning rate of updates that applies to input features.
        * "max_iter": int, default: 500
            Maximum number of iterations.
        * "locked_features": list of string, default: []
            List of features that should not change.
        * "target_class": int (0 or 1), default: 1
            Desired output class.
    
    Implented from:
        "Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties"
        Lisa Schut, Oscar Key, Rory McGrathz, Luca Costabelloz, Bogdan Sacaleanuz, Medb Corcoranz, Yarin Galy.
    """

    _DEFAULT_HYPERPARAMS = {"lambda_param": 0.05, 
                            "step_size": 0.05, 
                            "max_iter": 500, 
                            "locked_features": [], 
                            "target_class": 1}

    def __init__(self, mlmodel: MLModel = None, hyperparams: Dict = None):
        supported_backends = ["tensorflow"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )
        
        super.__init__(mlmodel)
        
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.mlmodel = mlmodel
        self.lambda_param = checked_hyperparams["lambda_param"]
        self.step_size = checked_hyperparams["step_size"]
        self.max_iter = checked_hyperparams["max_iter"]
        self.locked_features = checked_hyperparams["locked_features"]
        self.target_class = checked_hyperparams["target_class"]

    def cross_entropy_loss(self, predictions, target_class):
        target = tf.constant([[target_class]], dtype=tf.float32)
        return tf.keras.losses.binary_crossentropy(target, predictions)
    
    def get_counterfactuals(self, factuals: pd.DataFrame):
        original_instance = factuals.values[0]
        feature_names = factuals.columns

        x = tf.Variable(original_instance.reshape(1, -1), dtype=tf.float32)

        for i in range(self.max_iter):
            with tf.GradientTape() as tape:
                tape.watch(x)
                predictions = self.model(x)
                distance_loss = tf.reduce_sum(tf.square(x - original_instance))
                classification_loss = self.cross_entropy_loss(predictions, self.target_class)
                loss = self.lambda_param * distance_loss + classification_loss

            gradients = tape.gradient(loss, x)
            x.assign_sub(self.step_size * gradients)

            for feature in self.locked_features:
                x[:, feature].assign(original_instance[feature])

            if (self.model.predict(x) >= 0.5 and self.target_class == 1) or (self.model.predict(x) < 0.5 and self.target_class == 0):
                break
        
        counterfactual_array = x.numpy().flatten()
        counterfactual_df = pd.DataFrame([counterfactual_array], columns=feature_names)

        return counterfactual_df