from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# import torch
from lime.lime_tabular import LimeTabularExplainer

# from sklearn.linear_model import LogisticRegression
from methods.api.recourse_method import RecourseMethod
from methods.catalog.larr.library.larr import LARRecourse
from methods.processing.counterfactuals import (
    check_counterfactuals,
    merge_default_parameters,
)
from models.catalog.catalog import ModelCatalog


class Larr(RecourseMethod):
    """
    Implementation of LARR (Learning-Augmented Robust Recourse) [1]_.

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.
    coeffs : np.ndArray, optional
        Coefficients. Will be approximated by LIME if None
    intercepts: np.ndArray, optional
        Intercepts. Will be approximated by LIME if None

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "beta": float, default: 0.5 (max 1.0, min 0.0)
            The ratio between Robustness and Consistency. Higher beta means more Robust, Lower means more Consistent
        * "alpha": float, default: 0.5 (max 1.0, min 0.0)
            The parameter used to shift model weights for adversarial training

    .. [1] Kayastha, K., Gkatzelis, V., Jabbari, S. (2025). Learning-Augmented Robust Algorithmic Recourse. Drexel University. (https://arxiv.org/pdf/2410.01580)
    """

    _DEFAULT_HYPERPARAMS = {
        "feature_cost": "_optional_",
        "alpha": 0.5,
        "loss_type": "BCE",
        "lime_seed": 0,
        "beta": 0.5,
    }

    def __init__(
        self,
        mlmodel: ModelCatalog,
        hyperparams: Dict,
        coeffs: Optional[np.ndarray] = None,
        intercepts: Optional[np.ndarray] = None,
    ):
        super().__init__(mlmodel)

        self._data = mlmodel.data
        self._mlmodel = mlmodel
        self._coeffs = coeffs
        self._intercepts = intercepts

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.alpha = checked_hyperparams["alpha"]
        self.lime_seed = checked_hyperparams["lime_seed"]
        self.beta = checked_hyperparams["beta"]

        self.method = LARRecourse(
            weights=self._coeffs, bias=self._intercepts, alpha=self.alpha
        )

    def _get_lime_coefficients(
        self, factuals: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        LAR Recourse is only defined on linear models. To make it work for arbitrary non-linear networks
        we need to find the lime coefficients for every instance.

        Parameters
        ----------
        factuals : pd.DataFrame
            Instances we want to get lime coefficients

        Returns
        -------
        coeffs : np.ndArray
        intercepts : np.ndArray

        """

        np.random.seed(self.lime_seed)

        coeffs = np.zeros(factuals.shape)
        intercepts = []
        lime_data = self._data.df[self._mlmodel.feature_input_order]
        lime_label = self._data.df[self._data.target]

        lime_exp = LimeTabularExplainer(
            training_data=lime_data.values,
            training_labels=lime_label,
            mode="regression",
            discretize_continuous=False,
            feature_selection="none",
        )

        for index, row in factuals.iterrows():
            factual = row.values
            explanations = lime_exp.explain_instance(
                factual,
                self._mlmodel.predict_proba,
                num_features=len(self._mlmodel.feature_input_order),
                # model_regressor=LogisticRegression(),
            )
            intercepts.append(explanations.intercept[1])

            for tpl in explanations.local_exp[1]:
                coeffs[index][tpl[0]] = tpl[1]

        return coeffs, np.array(intercepts)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = factuals.reset_index()
        factuals = self._mlmodel.get_ordered_features(factuals)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoded_feature_names = self._mlmodel.data.categorical
        # cat_features_indices = [
        #     factuals.columns.get_loc(feature) for feature in encoded_feature_names
        # ]

        coeffs = self._coeffs
        intercepts = self._intercepts

        if (coeffs is None) or (intercepts is None):
            if self._mlmodel.model_type == "linear":
                coeffs_neg = (
                    # self._mlmodel.raw_model.output.weight.cpu().detach()[0].numpy()
                    self._mlmodel.raw_model.linear.weight.cpu()
                    .detach()[0]
                    .numpy()
                )
                coeffs_pos = (
                    self._mlmodel.raw_model.linear.weight.cpu().detach()[1].numpy()
                )

                intercepts_neg = np.array(
                    self._mlmodel.raw_model.linear.bias.cpu().detach()[0].numpy()
                )
                intercepts_pos = np.array(
                    self._mlmodel.raw_model.linear.bias.cpu().detach()[1].numpy()
                )

                self._coeffs = coeffs_pos - coeffs_neg
                self._intercepts = intercepts_pos - intercepts_neg

                # self._coeffs = self._mlmodel._model.coef_[0]
                # self._intercepts = self._mlmodel._model.intercept_[0]

                # Local explanations via LIME generate coeffs and intercepts per instance, while global explanations
                # via input parameter need to be set into correct shape [num_of_instances, num_of_features]
                coeffs = np.vstack([self._coeffs] * factuals.shape[0])
                intercepts = np.vstack([self._intercepts] * factuals.shape[0]).squeeze(
                    axis=1
                )
                self.method.weights = self._coeffs
                self.method.bias = self._intercepts
            elif self._mlmodel.model_type == "mlp":
                coeffs, intercepts = self._get_lime_coefficients(factuals)

        # we now need to find the optimal Lambda value
        # print(self._data.df_train.head())

        df_train_processed = self._data.df_train[self._mlmodel.feature_input_order]

        # X_train_t = torch.from_numpy(df_train_processed.values).float().to(device)

        preds_gpu_probs = self._mlmodel.predict_proba(df_train_processed)
        preds_gpu_labels = preds_gpu_probs.argmax(
            axis=1
        )  # since the models use softmax, we need argmax to see which class was predicted
        # preds_cpu_labels = preds_gpu_labels.cpu().numpy()

        # print(preds_gpu_labels)

        recourse_needed_X_train = df_train_processed[preds_gpu_labels == 0].values
        # print(recourse_needed_X_train)

        if len(recourse_needed_X_train) == 0:
            raise Exception(
                "The model did not predict any failures in the original training data. It cannot search for the Lambda parameter"
            )

        # recourse_needed_X_train = df_train_processed.values[:5]

        # first choose the optimal lambda value
        self.method.choose_lambda(
            recourse_needed_X_train,
            predict_fn=self._mlmodel.predict,
            predict_proba_fn=self._mlmodel.predict_proba,
            X_train=df_train_processed.values,
        )  # self._data.df_train.values)

        cfs = []
        for index, row in factuals.iterrows():
            coeff = coeffs[index]
            intercept = intercepts[index]

            cf = self.method.run_method(
                # self._mlmodel.raw_model,
                row.to_numpy().reshape((1, -1)),
                coeff,
                intercept,
                self.beta,
            )
            cfs.append(cf)

        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
