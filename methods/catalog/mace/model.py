from typing import Dict, Optional

import numpy as np
import pandas as pd

import data.catalog.loadData as loadData
from methods import RecourseMethod
from methods.catalog.mace.library.mace import generateExplanations
from methods.processing import check_counterfactuals, merge_default_parameters
from models.catalog import ModelCatalog

# Custom recourse implementations need to
# inherit from the RecourseMethod interface


class MACE(RecourseMethod):
    """
    Implementation of MACE from Karimi et.al. [1]_.

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
    - Hyperparams
      Hyperparameter contains important information for the recourse method to initialize.
      Please make sure to pass all values as dict with the following keys.

      * "num": int, default: 1
          Number of counterfactuals per factual to generate
      * "desired_class": int, default: 1
          Given a binary class label, the desired class a counterfactual should have (e.g., 0 or 1)
      * "posthoc_sparsity_param": float, default: 0.1
          Fraction of post-hoc preprocessing steps.

    .. [1] A.-H. Karimi, G. Barthe, B. Balle, and I. Valera, “Model-Agnostic counterfactual explanations for consequential decisions,”
        arXiv.org, May 27, 2019. https://arxiv.org/abs/1905.11190
    """

    _DEFAULT_HYPERPARAMS = {"norm_type": ["zero_norm"]}

    def __init__(self, mlmodel: ModelCatalog, hyperparameters: Optional[Dict] = None):
        supported_backends = ["sklearn"]

        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )
        super().__init__(mlmodel)
        self._continuous = mlmodel.data.continuous
        self._categorical = mlmodel.data.categorical
        self._target = mlmodel.data.target
        self._model = mlmodel

        checked_hyperparams = merge_default_parameters(
            hyperparameters, self._DEFAULT_HYPERPARAMS
        )

        self._approach_string = "MACE_eps_1e-5"
        self._explanation_file_name = "mace"
        self._norm_type = checked_hyperparams["norm_type"]

    # Generate and return encoded and
    # scaled counterfactual examples
    def get_counterfactuals(self, factuals: pd.DataFrame):
        # Prepare factuals
        querry_instances = factuals.copy()
        querry_instances = self._model.get_ordered_features(querry_instances)
        dataset_string = self._model.data.name

        # check if querry_instances are not empty
        if not querry_instances.shape[0] > 0:
            raise ValueError("Factuals should not be empty")

        # Generate counterfactuals

        explanation_folder_name = f"{self._explanation_file_name}/__explanation_log"

        # get the predicted labels (only test set)
        dataset_obj = loadData.loadDataset(
            dataset_string, return_one_hot=True, load_from_cache=False, debug_flag=False
        )
        X_test_pred_labels = self._model.predict(querry_instances)

        all_pred_data_df = querry_instances
        # IMPORTANT: note that 'y' is actually 'pred_y', not 'true_y'
        all_pred_data_df["y"] = X_test_pred_labels

        explanation_counter = 1
        iterate_over_data_df = all_pred_data_df
        iterate_over_data_dict = iterate_over_data_df.T.to_dict()
        recourse_counterfactuals = []

        for norm_type_string in self._norm_type:
            for factual_sample_index, factual_sample in list(
                iterate_over_data_dict.items()
            ):
                factual_sample["y"] = bool(factual_sample["y"])
                print(
                    "\t\t\t\t"
                    f"Generating explanation for\t"
                    f"sample #{explanation_counter}/{len(iterate_over_data_dict.keys())}\t"
                    f"(sample index {factual_sample_index}): \n",
                    end="",
                )  # , file=log_file)
                explanation_counter = explanation_counter + 1
                explanation_file_name = (
                    f"{explanation_folder_name}/sample_{factual_sample_index}.txt"
                )

                mace_counterfactuals = generateExplanations(
                    self._approach_string,
                    explanation_file_name,
                    self._model._model,
                    dataset_obj,
                    factual_sample,
                    norm_type_string,
                )

                cfe_sample = mace_counterfactuals.get("cfe_sample", None)
                if not cfe_sample:
                    cfe_sample = {
                        feature: factual_sample.get(feature, np.nan)
                        for feature in self._mlmodel.feature_input_order
                    }
                else:
                    for feature in self._mlmodel.feature_input_order:
                        cfe_sample.setdefault(
                            feature, factual_sample.get(feature, np.nan)
                        )

                recourse_counterfactuals.append(pd.DataFrame([cfe_sample]))

        if recourse_counterfactuals:
            recourse_counterfactuals = pd.concat(
                recourse_counterfactuals, ignore_index=True
            )
        else:
            recourse_counterfactuals = pd.DataFrame(
                columns=self._mlmodel.feature_input_order
            )

        recourse_counterfactuals = recourse_counterfactuals.drop(
            columns=[self._mlmodel.data.target], errors="ignore"
        )
        recourse_counterfactuals = recourse_counterfactuals.reindex(
            columns=self._mlmodel.feature_input_order
        )

        df_cfs = check_counterfactuals(
            self._mlmodel, recourse_counterfactuals, factuals.index
        )
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)

        return df_cfs
