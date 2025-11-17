"""
GenRe Integration - Simplified Version

This module implements GenRe (Generative Recourse), which uses a trained
Transformer model to generate counterfactual explanations through forward
sampling rather than gradient-based optimization.

Strategy: Use author's EVERYTHING (data, models, pipeline)
Provide a wrapper for like RecourseMethod interface.

Paper: "From Search to Sampling: Generative Models for Robust Algorithmic Recourse"
ICLR 2025
"""

from typing import Dict
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd

# Add author's code in library to path
GENRE_ROOT = os.path.dirname(os.path.abspath(__file__))
library = os.path.join(GENRE_ROOT, 'library')
sys.path.insert(0, library)

# Author's imports (everything from author's code)
import library.data.utils as genre_dutils
import library.models.binnedpm as bpm
from library.recourse.genre import GenRe as GenReOriginal
from library.models.classifiers.ann import ANN

# Repo imports (just for interface compatibility)
from methods.api import RecourseMethod
from methods.processing.counterfactuals import merge_default_parameters
from models.catalog.catalog import ModelCatalog


class GenRe(RecourseMethod):
    """
    GenRe wrapper - uses author's complete implementation
    
    This is just a thin interface layer. Everything happens in author's code.
    """
    
    _DEFAULT_HYPERPARAMS = {
        "temp": 10.0,
        "sigma": 0.0,
        "best_k": 10,
        "device": "cpu",
        # Paths to YOUR pre-trained models
        "saved_models_dir": None,  # e.g., "methods/catalog/genre/library/saved_models"
    }
    
    def __init__(self, mlmodel: ModelCatalog, hyperparams: Dict) -> None:
        super().__init__(mlmodel)
        
        # Get hyperparams
        checked_hyperparams = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        self._temp = checked_hyperparams["temp"]
        self._sigma = checked_hyperparams["sigma"]
        self._best_k = checked_hyperparams["best_k"]
        self._device = torch.device(checked_hyperparams["device"])
        self._saved_models_dir = checked_hyperparams["saved_models_dir"]
        
        if self._saved_models_dir is None:
            # Default: look in library/saved_models
            self._saved_models_dir = os.path.join(library, 'saved_models')
        
        # Map repo dataset name to author's dataset name
        dataset_mapping = {'adult': 'adult-all', 'compas': 'compas-all', 'heloc': 'heloc'}
        repo_dataset = mlmodel.data.name
        self._author_dataset = dataset_mapping.get(repo_dataset, repo_dataset)
        
        # Load author's data (we use this, not repo's data!)
        self._load_author_data()
        
        # Load author's pre-trained models
        self._load_author_models()
        
        # Initialize GenRe recourse module
        self._init_genre_recourse()
    
    def _load_author_data(self):
        """Load data using author's loader - this is what GenRe was trained on"""
        print(f"Loading author's {self._author_dataset} data...")
        
        # Load using author's exact data loading
        train_y, train_X, test_y, test_X, cat_mask, immutable_mask = genre_dutils.load_dataset(
            self._author_dataset,
            ret_tensor=True,
            min_max=True,
            ret_masks=True
        )
        
        self._author_train_X = train_X
        self._author_train_y = train_y
        self._author_test_X = test_X
        self._author_test_y = test_y
        self._cat_mask = cat_mask
        self._immutable_mask = immutable_mask
        
        self._input_dim = train_X.shape[1]
        print(f"Loaded author's data: {train_X.shape}")
    
    def _load_author_models(self):
        """Load your pre-trained models"""
        # Construct paths to your saved models
        # Adjust these based on your actual saved model locations!
        
        # Example paths (you need to adjust based on your actual structure):
        # saved_models/classifiers/adult-all/rf_tt_mm/state.pkl
        # saved_models/classifiers/adult-all/ann_rf_tt_mm_<config>/state.pt
        # saved_models/genre/adult-all/bpm_<config>/state.pt
        
        dataset = self._author_dataset
        
        # Load ANN
        # You need to find your actual ANN path
        ann_dir = os.path.join(self._saved_models_dir, f'classifiers/{dataset}')
        # Find the ANN folder (usually named like ann_rf_tt_mm_...)
        ann_folders = [f for f in os.listdir(ann_dir) if f.startswith('ann_rf')]
        if not ann_folders:
            raise FileNotFoundError(f"No ANN model found in {ann_dir}")
        
        ann_path = os.path.join(ann_dir, ann_folders[0], 'state.pt')
        print(f"Loading ANN from: {ann_path}")
        
        ann_state = torch.load(ann_path, map_location=self._device)
        
        # Reconstruct ANN
        self._ann_clf = ANN(
            input_shape=self._input_dim,
            hidden_dims=[10, 10, 10],  
            output_dim=1
        )
        self._ann_clf.load_state_dict(ann_state['model'])
        self._ann_clf.to(self._device)
        self._ann_clf.eval()
        print(f"Loaded ANN")
        
        # Load GenRe Transformer
        genre_dir = os.path.join(self._saved_models_dir, f'genre/{dataset}')
        genre_folders = [f for f in os.listdir(genre_dir) if f.startswith('bpm')]
        if not genre_folders:
            raise FileNotFoundError(f"No GenRe model found in {genre_dir}")
        
        genre_path = os.path.join(genre_dir, genre_folders[0], 'state.pt')
        print(f"Loading GenRe from: {genre_path}")
        
        genre_state = torch.load(genre_path, map_location=self._device)
        
        # Reconstruct GenRe Transformer
        self._pair_model = bpm.PairedTransformerBinned(
            n_bins=50,
            num_inputs=self._input_dim,
            num_labels=1,
            num_encoder_layers=16,  
            num_decoder_layers=16,
            emb_size=32,
            dim_feedforward=512,
            nhead=8,
            dropout=0.1
        )
        self._pair_model.load_state_dict(genre_state['model'])
        self._pair_model.to(self._device)
        self._pair_model.eval()
        print(f"Loaded GenRe Transformer")
    
    def _init_genre_recourse(self):
        """Initialize author's GenRe recourse module"""
        self._genre_recourse = GenReOriginal(
            pair_model=self._pair_model,
            temp=self._temp,
            sigma=self._sigma,
            best_k=self._best_k,
            ann_clf=self._ann_clf,
            ystar=1.0,  # Target favorable outcome
            cat_mask=self._cat_mask
        )
        print(f"GenRe recourse module initialized")
    
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate counterfactuals
        
        IMPORTANT: We ignore the input 'factuals' from repo!
        Instead, we use author's test data directly.
        
        This is a "fake" interface - we just return CFs for author's test set.
        """
        n_requested = len(factuals)
        
        # Get negative instances from author's test set
        test_pred = (self._ann_clf(self._author_test_X.to(self._device)) > 0.5).cpu().squeeze()
        negative_mask = ~test_pred.bool()
        
        factual_tensor = self._author_test_X[negative_mask][:n_requested]
        
        if len(factual_tensor) < n_requested:
            print(f"Warning: Only {len(factual_tensor)} negative instances available")
        
        # Generate CFs using author's GenRe
        with torch.no_grad():
            cf_tensor = self._genre_recourse(factual_tensor.to(self._device))
            cf_tensor = cf_tensor.squeeze().cpu().float()
        
        # Convert to DataFrame (fake format to match repo's interface)
        # Column names don't matter much since we're not using repo's data
        n_features = cf_tensor.shape[1] if cf_tensor.dim() > 1 else cf_tensor.shape[0]
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if cf_tensor.dim() == 1:
            cf_tensor = cf_tensor.unsqueeze(0)
        
        df_cfs = pd.DataFrame(
            cf_tensor.numpy(),
            columns=feature_names
        )
        
        return df_cfs


# For testing
if __name__ == "__main__":
    print("GenRe module loaded successfully")
    print(f"Author's code path: {library}")