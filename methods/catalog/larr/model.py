from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from methods.api.recourse_method import RecourseMethod
from methods.catalog.larr.library.larr import LARRecourse
from methods.processing.counterfactuals import check_counterfactuals, merge_default_parameters
from models.catalog.catalog import ModelCatalog
from lime.lime_tabular import LimeTabularExplainer



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

        * "lr": float, default: 0.01
            Learning rate for gradient descent.


    .. [1] Kayastha, K., Gkatzelis, V., Jabbari, S. (2025). Learning-Augmented Robust Algorithmic Recourse. Drexel University. (https://arxiv.org/pdf/2410.01580)
    """
    _DEFAULT_HYPERPARAMS = {
        "feature_cost": "_optional_",
        "lr": 0.01,
        "alpha": 0.5,
        "norm": 1,
        "lambda_": None,
        "loss_type": "BCE",
        "y_target": [0, 1],
        "lime_seed": 0,
        "binary_cat_features": True,
        
    }

    def __init__(self, 
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

        self.feature_cost = checked_hyperparams["feature_cost"]
        self.lr = checked_hyperparams["lr"]
        self.alpha = checked_hyperparams["alpha"]
        self.norm = checked_hyperparams["norm"]
        self.lambda_ = checked_hyperparams["labmda_"]
        self.loss_type = checked_hyperparams["loss_type"]
        self.y_target = checked_hyperparams["y_target"]
        self.lime_seed = checked_hyperparams["lime_seed"]
        self.binary_cat_features = checked_hyperparams["binary_cat_features"]

        self.method = LARRecourse(weights=self._coeffs, 
                                  bias=self._intercepts, 
                                  alpha=self.alpha)
        
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
            feature_selection='none',
        )

        for index, row in factuals.iterrows():
            factual = row.values
            explanations = lime_exp.explain_instance(
                factual,
                self._mlmodel.predict_proba,
                num_features=len(self._mlmodel.feature_input_order),
                model_regressor=LogisticRegression(),
            )
            intercepts.append(explanations.intercept[1])

            for tpl in explanations.local_exp[1]:
                coeffs[index][tpl[0]] = tpl[1]

        return coeffs, np.array(intercepts)


    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = self._mlmodel.get_ordered_features(factuals)

        encoded_feature_names = self._mlmodel.data.categorical
        cat_features_indices = [
            factuals.columns.get_loc(feature) for feature in encoded_feature_names
        ]

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
            elif self._mlmodel.model_type == "mlp":
                coeffs, intercepts = self._get_lime_coefficients(factuals)


        # we now need to find the optimal Lambda value

        recourse_needed_X_train = self._data.df_train.values[np.where(self._mlmodel.raw_model.predict_fn(self._data.df_train.values) == 0)]

        # first choose the optmial lambda value
        if self.lambda_ == None:
            self.method.choose_lambda(recourse_needed_X_train, self._mlmodel.raw_model.predict_fn, self._data.df_train.values)

        cfs = []
        for index, row in factuals.iterrows():
            coeff = coeffs[index]
            intercept = intercepts[index]

            cf = self.method.run_method(
                self._mlmodel.raw_model,
                row.to_numpy().reshape((1, -1)),
                coeff,
                intercept,
            )
            cfs.append(cf)

        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self._mlmodel.feature_input_order)
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs