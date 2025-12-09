"""
CARE: Coherent Actionable Recourse based on sound counterfactual Explanations

Implementation of CARE method for recourse benchmarks.
"""

from typing import Dict
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add library to path
CURRENT_FILE = os.path.abspath(__file__)
CARE_DIR = os.path.dirname(CURRENT_FILE)
LIBRARY_DIR = os.path.join(CARE_DIR, 'library')
sys.path.insert(0, LIBRARY_DIR)

from prepare_datasets import PrepareAdult # type: ignore
from user_preferences import userPreferences # type: ignore
from care.care import CARE # type: ignore

from methods.processing import check_counterfactuals
from models.catalog.catalog import ModelCatalog
from tools.log import log
from ...api import RecourseMethod
from ...processing.counterfactuals import merge_default_parameters


class Care(RecourseMethod):  # ← Changed from CARERecourse to Care
    """
    Implementation of CARE [1]_.

    Parameters
    ----------
    mlmodel : ModelCatalog
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "config": str, default: "{1,2,3,4}"
            CARE configuration. Options: "{1}", "{1,2}", "{1,2,3}", "{1,2,3,4}"
            - {1}: Validity only
            - {1,2}: Validity + Soundness  
            - {1,2,3}: Validity + Soundness + Coherency
            - {1,2,3,4}: Validity + Soundness + Coherency + Actionability
        * "n_cf": int, default: 10
            Number of counterfactuals to generate per instance
        * "dataset_name": str, default: "adult"
            Dataset name for loading preprocessing metadata

    - Restrictions
        * Currently only supports Adult dataset
        * Requires ordinal-encoded features
        * Configuration {1,2,3,4} uses default user preferences (age≥current, fix sex/race/country)

    .. [1] Rasouli, P., & Yu, I. C. (2022). CARE: Coherent Actionable Recourse based on 
           sound Counterfactual Explanations. arXiv preprint arXiv:2203.06850.
    """

    _DEFAULT_HYPERPARAMS = {
        "config": "{1,2,3,4}",
        "n_cf": 10,
        "dataset_name": "adult",
    }

    def __init__(
        self,
        mlmodel: ModelCatalog,
        hyperparams: Dict,
    ) -> None:
        super().__init__(mlmodel)
        self._mlmodel = mlmodel
        self._data = mlmodel.data

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self._config = checked_hyperparams["config"]
        self._n_cf = checked_hyperparams["n_cf"]
        self._dataset_name = checked_hyperparams["dataset_name"]

        # Parse config
        self._SOUNDNESS = '2' in self._config
        self._COHERENCY = '3' in self._config
        self._ACTIONABILITY = '4' in self._config

        # Will be initialized in _initialize_care()
        self._care_model = None
        self._dataset = None
        self._task = 'classification'

    def _load_dataset_metadata(self):
        """
        Load CARE's dataset preprocessing metadata.

        Returns
        -------
        dict
            Dataset dictionary with encoders and metadata
        """
        dataset_path = os.path.join(CARE_DIR, 'datasets/')

        if self._dataset_name == 'adult':
            self._dataset = PrepareAdult(dataset_path, 'adult.csv')
        else:
            raise NotImplementedError(
                f"Dataset {self._dataset_name} not yet supported"
            )

        return self._dataset

    def _initialize_care(self):
        """
        Initialize CARE model with the black-box model.
        """
        if self._care_model is not None:
            return  # Already initialized

        log.info(f"Initializing CARE with config {self._config}")

        # Load dataset metadata
        self._load_dataset_metadata()

        # Define prediction functions
        predict_fn = lambda x: self._mlmodel.predict(
            pd.DataFrame(x, columns=self._mlmodel.feature_input_order)
        ).ravel()

        predict_proba_fn = lambda x: self._mlmodel.predict_proba(
            pd.DataFrame(x, columns=self._mlmodel.feature_input_order)
        )

        # Initialize CARE
        self._care_model = CARE(
            self._dataset,
            task=self._task,
            predict_fn=predict_fn,
            predict_proba_fn=predict_proba_fn,
            SOUNDNESS=self._SOUNDNESS,
            COHERENCY=self._COHERENCY,
            ACTIONABILITY=self._ACTIONABILITY,
            n_cf=self._n_cf
        )

        # Fit CARE with training data
        X_train = self._dataset['X_ord']
        y_train = self._dataset['y']

        # Use train/test split from original preprocessing
        X_tr, _, y_tr, _ = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        log.info("Fitting CARE model...")
        self._care_model.fit(X_tr, y_tr)
        log.info("CARE model fitted successfully")

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate counterfactual examples for given factuals.

        Parameters
        ----------
        factuals : pd.DataFrame
            Instances for which to generate counterfactual examples.

        Returns
        -------
        pd.DataFrame
            Counterfactual examples.
        """
        # Initialize CARE if not already done
        self._initialize_care()

        factuals = factuals.reset_index(drop=True)
        factuals = self._mlmodel.get_ordered_features(factuals)

        # Convert to ordinal encoding (CARE's expected format)
        factuals_ord = factuals.values

        log.info(f"Generating counterfactuals for {len(factuals)} instances...")

        cfs = []
        for index, row in enumerate(factuals_ord):
            try:
                # Get user preferences if actionability is enabled
                user_preferences = None
                if self._ACTIONABILITY:
                    user_preferences = userPreferences(self._dataset, row)

                # Generate explanation
                if self._ACTIONABILITY:
                    explanation = self._care_model.explain(
                        row, user_preferences=user_preferences
                    )
                else:
                    explanation = self._care_model.explain(row)

                # Extract best counterfactual
                best_cf = explanation['best_cf_ord']
                cfs.append(best_cf)

                if (index + 1) % 10 == 0:
                    log.info(f"  Progress: {index + 1}/{len(factuals)}")

            except Exception as e:
                log.warning(
                    f"Failed to generate counterfactual for instance {index}: {e}"
                )
                # Return original instance as fallback
                cfs.append(row)

        # Convert output into correct format
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)

        # Check counterfactuals validity
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)

        log.info("Counterfactual generation completed")

        return df_cfs