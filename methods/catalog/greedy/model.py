from typing import Dict

import pandas as pd
import tensorflow as tf

from methods.api import RecourseMethod
from methods.processing import merge_default_parameters, check_counterfactuals
from models.api import MLModel


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

    _DEFAULT_HYPERPARAMS = {
        "lambda_param": 0.05,
        "step_size": 0.05,
        "max_iter": 100,
        "locked_features": [],
        "target_class": 1,
    }

    def __init__(self, mlmodel: MLModel = None, hyperparams: Dict = None):
        supported_backends = ["tensorflow"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)

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
        prediction_val = tf.constant([[predictions]], dtype=tf.float32)

        # TensorFlow 1.x equivalent of binary crossentropy
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target, logits=prediction_val
        )

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = self._mlmodel.get_ordered_features(factuals)

        # Initialize a list to collect the counterfactuals
        counterfactuals_list = []

        # Iterate over each row in the DataFrame
        for index, row in factuals.iterrows():
            # Prepare the original instance
            original_instance = row.drop("y")
            feature_names = original_instance.index
            original_instance = original_instance.values

            # Define variable for optimization
            x = tf.Variable(original_instance.reshape(1, -1), dtype=tf.float32)

            # Initialize a TensorFlow session
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(self.max_iter):
                    predictions = self.mlmodel.raw_model(x)
                    distance_loss = tf.reduce_sum(tf.square(x - original_instance))
                    predictions = sess.run(predictions)
                    classification_loss = self.cross_entropy_loss(
                        predictions[0, 1], self.target_class
                    )
                    loss = self.lambda_param * distance_loss + classification_loss

                    gradients = tf.gradients(loss, [x])[
                        0
                    ]  # Use list to get gradient for tf.Variable
                    sess.run(x.assign_sub(self.step_size * gradients))

                    for feature in self.locked_features:
                        sess.run(tf.assign(x[:, feature], original_instance[feature]))

                    # Check stopping condition
                    current_prediction = sess.run(self.mlmodel.raw_model(x))[0, 1]
                    if (current_prediction >= 0.5 and self.target_class == 1) or (
                        current_prediction < 0.5 and self.target_class == 0
                    ):
                        break

                counterfactual_array = sess.run(x).flatten()
                counterfactual_df = pd.DataFrame(
                    [counterfactual_array], columns=feature_names
                )

                # Append the counterfactual result to the list
                counterfactuals_list.append(counterfactual_df)

        # Concatenate all counterfactuals into a single DataFrame
        final_counterfactuals_df = pd.concat(counterfactuals_list, ignore_index=True)

        df_cfs = check_counterfactuals(self._mlmodel, final_counterfactuals_df, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)

        return df_cfs
