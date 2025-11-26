"""
GenRe (Generative Recourse) Implementation

This module implements GenRe as a RecourseMethod for the recourse_benchmarks repo.

Paper: "From Search to Sampling: Generative Models for Robust Algorithmic Recourse"
       ICLR 2025
"""

import os
import sys
from typing import Dict, Optional

import pandas as pd
import torch

# Add author's library to path first
GENRE_ROOT = os.path.dirname(os.path.abspath(__file__))
LIBRARY_PATH = os.path.join(GENRE_ROOT, "library")
sys.path.insert(0, LIBRARY_PATH)

# Import after adding to path
from library.recourse.genre import GenRe as GenReOriginal
from library.models import binnedpm

# Repo imports
from methods.api import RecourseMethod
from methods.processing.counterfactuals import merge_default_parameters


class GenRe(RecourseMethod):
    """
    GenRe wrapper for recourse_benchmarks

    Parameters
    ----------
    mlmodel : nn.Module or ModelCatalog
        Black-Box-Model (typically an ANN classifier)

    hyperparams : dict
        Dictionary containing hyperparameters

    Hyperparameters
    ---------------
    data : DataCatalog, optional
        DataCatalog object containing dataset information.
        If provided, cat_mask will be calculated from data.categorical
    cat_mask : torch.Tensor, optional
        Binary mask for categorical features (1=categorical, 0=continuous).
        If not provided, will be calculated from 'data' parameter.
        Note: Either 'data' or 'cat_mask' must be provided.
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
        self._device = torch.device(checked_hyperparams["device"])
        
        # Use mlmodel directly
        self._ann_clf = mlmodel

        # Get cat_mask: either directly provided or calculated from data
        if "cat_mask" in hyperparams and hyperparams["cat_mask"] is not None:
            # Option 1: cat_mask provided directly
            self._cat_mask = hyperparams["cat_mask"]
            input_dim = len(self._cat_mask)
        elif "data" in hyperparams and hyperparams["data"] is not None:
            # Option 2: calculate cat_mask from DataCatalog object
            data = hyperparams["data"]
            all_features = data.continuous + data.categorical
            self._cat_mask = torch.tensor(
                [1 if f in data.categorical else 0 for f in all_features]
            )
            input_dim = len(all_features)
        else:
            raise ValueError(
                "Either 'cat_mask' or 'data' must be provided in hyperparams"
            )

        # Load transformer from library/transformer/
        self._transformer = self._load_transformer(input_dim)

        self._temp = checked_hyperparams["temp"]
        self._sigma = checked_hyperparams["sigma"]
        self._best_k = checked_hyperparams["best_k"]

        # Initialize GenRe recourse module
        self._genre_recourse = GenReOriginal(
            pair_model=self._transformer,
            temp=self._temp,
            sigma=self._sigma,
            best_k=self._best_k,
            ann_clf=self._ann_clf,
            ystar=1.0,
            cat_mask=self._cat_mask,
        )

    def _load_transformer(self, input_dim: int):
        """Load pretrained transformer from library/transformer/"""
        transformer_path = os.path.join(
            GENRE_ROOT, "library", "transformer", "genre_transformer.pth"
        )

        if not os.path.exists(transformer_path):
            raise FileNotFoundError(
                f"Transformer not found at {transformer_path}. "
                "Please ensure genre_transformer.pth is in library/transformer/"
            )

        # Create transformer architecture
        transformer = binnedpm.PairedTransformerBinned(
            n_bins=50,
            num_inputs=input_dim,
            num_labels=1,
            num_encoder_layers=16,
            num_decoder_layers=16,
            emb_size=32,
            nhead=8,
            dim_feedforward=32,
            dropout=0.1,
        )

        # Load state dict
        state_dict = torch.load(transformer_path, map_location=self._device)

        # Handle both direct state_dict and nested format
        if "state_dict" in state_dict:
            transformer.load_state_dict(state_dict["state_dict"])
        else:
            transformer.load_state_dict(state_dict)

        transformer.to(self._device)
        transformer.eval()

        return transformer

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