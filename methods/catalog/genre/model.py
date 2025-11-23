"""
GenRe (Generative Recourse) Implementation

This module implements GenRe as a RecourseMethod for the recourse_benchmarks repo.

Paper: "From Search to Sampling: Generative Models for Robust Algorithmic Recourse"
       ICLR 2025
"""

import os
import sys
from typing import Dict

import pandas as pd
import torch

from methods.catalog.genre.library.recourse.genre import GenRe as GenReOriginal

# Repo imports
from methods.api import RecourseMethod
from methods.processing.counterfactuals import merge_default_parameters
from models.catalog.catalog import ModelCatalog

# Add author's library to path
GENRE_ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRARY_PATH = os.path.join(GENRE_ROOT, "library")
sys.path.insert(0, LIBRARY_PATH)


class GenRe(RecourseMethod):
    """
    GenRe wrapper for recourse_benchmarks

    Parameters
    ----------
    mlmodel : ModelCatalog or nn.Module
        Black-Box-Model. Can be either:
        - Repo's ModelCatalog (FUTURE)
        - Author's BinaryClassifier nn.Module (CURRENT)

    hyperparams : dict
        Dictionary containing hyperparameters

    Hyperparameters
    ---------------
    transformer : torch.nn.Module
        Pretrained GenRe Transformer model
    cat_mask : torch.Tensor
        Binary mask for categorical features
    temp : float, default=10.0
        Temperature for GenRe sampling
    sigma : float, default=0.0
        Noise level for GenRe sampling
    best_k : int, default=10
        Number of candidates to generate
    device : str, default="cpu"
        Device for computation
    """

    _DEFAULT_HYPERPARAMS = {
        "transformer": None,
        "cat_mask": None,
        "temp": 10.0,
        "sigma": 0.0,
        "best_k": 10,
        "device": "cpu",
    }

    def __init__(self, mlmodel, hyperparams: Dict) -> None:
        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self._mlmodel = mlmodel

        # Auto-detect mlmodel type and extract the actual nn.Module
        if isinstance(mlmodel, ModelCatalog):
            # FUTURE: Repo's ModelCatalog - extract internal nn.Module
            self._ann_clf = mlmodel.raw_model
            # Get cat_mask from data if not provided
            if checked_hyperparams["cat_mask"] is not None:
                self._cat_mask = checked_hyperparams["cat_mask"]
            else:
                # Build cat_mask from ModelCatalog's data
                self._cat_mask = torch.tensor(
                    [1 if f in mlmodel.data.categorical else 0
                     for f in mlmodel.feature_input_order]
                )
        else:
            # CURRENT: Author's BinaryClassifier - use directly
            self._ann_clf = mlmodel
            self._cat_mask = checked_hyperparams["cat_mask"]

        # Transformer must be provided in hyperparams (repo doesn't have it yet)
        self._transformer = checked_hyperparams["transformer"]

        self._temp = checked_hyperparams["temp"]
        self._sigma = checked_hyperparams["sigma"]
        self._best_k = checked_hyperparams["best_k"]
        self._device = torch.device(checked_hyperparams["device"])

        # Initialize GenRe recourse module
        self._genre_recourse = GenReOriginal(
            pair_model=self._transformer,
            temp=self._temp,
            sigma=self._sigma,
            best_k=self._best_k,
            ann_clf=self._ann_clf,  # Use extracted nn.Module
            ystar=1.0,
            cat_mask=self._cat_mask,
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate counterfactual explanations

        Parameters
        ----------
        factuals : pd.DataFrame
            Factual instances

        Returns
        -------
        pd.DataFrame
            Counterfactual instances
        """
        # Convert to tensor
        factuals_array = factuals.values
        factuals_tensor = torch.FloatTensor(factuals_array).to(self._device)

        # Generate counterfactuals
        with torch.no_grad():
            cf_tensor = self._genre_recourse(factuals_tensor)
            cf_tensor = cf_tensor.squeeze().cpu().float()

        # Handle single instance
        if cf_tensor.dim() == 1:
            cf_tensor = cf_tensor.unsqueeze(0)

        # Convert back to DataFrame
        df_cfs = pd.DataFrame(
            cf_tensor.numpy(), columns=factuals.columns, index=factuals.index
        )

        return df_cfs