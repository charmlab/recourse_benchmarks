from typing import List
import pandas as pd
from sklearn.base import BaseEstimator

from ...api import RecourseMethod
from methods.catalog.probe.library import probe_recourse
from methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)

class Probe(RecourseMethod):
    """
    Implementation of Probe framework using Wachter recourse generation from Pawelczyk et.al. [1]_.

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    data: data.Data
        Dataset to perform on
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    .. [1] Martin Pawelczyk,Teresa Datta, Johan Van den Heuvel, Gjergji Kasneci, Himabindu Lakkaraju.2023
            Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse
            https://openreview.net/pdf?id=sC-PmTsiTB(2023).
    """
    _DEFAULT_HYPERPARAMS = {
        "feature_cost": "_optional_",
        "lr": 0.001,
        "lambda_": 0.01,
        "n_iter": 1000,
        "t_max_min": 1.0,
        "norm": 1,
        "clamp": True,
        "loss_type": "MSE",
        "y_target": [0, 1],
        "binary_cat_features": True,
        "noise_variance": 0.01,
        "invalidation_target": 0.45,
        "inval_target_eps": 0.005,
    }

    def __init__(self, mlmodel, hyperparams):
        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self._feature_costs = checked_hyperparams["feature_cost"]
        self._lr = checked_hyperparams["lr"]
        self._lambda_param = checked_hyperparams["lambda_"]
        self._n_iter = checked_hyperparams["n_iter"]
        self._t_max_min = checked_hyperparams["t_max_min"]
        self._norm = checked_hyperparams["norm"]
        self._clamp = checked_hyperparams["clamp"]
        self._loss_type = checked_hyperparams["loss_type"]
        self._y_target = checked_hyperparams["y_target"]
        self._binary_cat_features = checked_hyperparams["binary_cat_features"]
        self._noise_variance = checked_hyperparams["noise_variance"]
        self._invalidation_target = checked_hyperparams["invalidation_target"]
        self._inval_target_eps = checked_hyperparams["inval_target_eps"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Normalize and encode data
        # df_enc_norm_fact = self.encode_normalize_order_factuals(factuals)
        
        factuals = self._mlmodel.get_ordered_features(factuals)

        encoded_feature_names = self._mlmodel.data.categorical
        cat_features_indices = [
            factuals.columns.get_loc(feature) for feature in encoded_feature_names
        ]

        df_cfs = factuals.apply(
            lambda x: probe_recourse(
                self._mlmodel.raw_model,
                x.reshape((1, -1)),
                cat_features_indices,
                binary_cat_features=self._binary_cat_features,
                feature_costs=self._feature_costs,
                lr=self._lr,
                lambda_param=self._lambda_param,
                n_iter=self._n_iter,
                t_max_min=self._t_max_min,
                norm=self._norm,
                clamp=self._clamp,
                loss_type=self._loss_type,
                invalidation_target=self._invalidation_target,
                inval_target_eps=self._inval_target_eps,
                noise_variance=self._noise_variance
            ),
            raw=True,
            axis=1,
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
