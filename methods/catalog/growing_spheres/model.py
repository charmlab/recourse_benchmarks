import pandas as pd

from data.api import Data
from methods.api import RecourseMethod
from methods.catalog.growing_spheres.library import growing_spheres_search
from methods.utils import check_counterfactuals, encode_feature_names
from models.api import MLModel


class GrowingSpheres(RecourseMethod):
    """
    Implementation of Growing Spheres from Laugel et.al. [1]_.

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Growing Spheeres needs no hyperparams.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Restrictions
        Growing Spheres works at the moment only for data with dropped first column of binary categorical features.

    .. [1] Thibault Laugel, Marie-Jeanne Lesot, Christophe Marsala, Xavier Renard, and Marcin Detyniecki. 2017.
            Inverse Classification for Comparison-based Interpretability in Machine Learning.
            arXiv preprint arXiv:1712.08443(2017).
    """

    def __init__(self, data: Data, mlmodel: MLModel, hyperparams=None) -> None:
        supported_backends = ["tensorflow", "pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(data, mlmodel)

        self._immutables = encode_feature_names(
            self._data.immutables, self._data.feature_input_order
        )
        self._mutables = [
            feature
            for feature in self._data.feature_input_order
            if feature not in self._immutables
        ]
        self._continuous = self._data.continuous
        self._categorical_enc = encode_feature_names(
            self._data.categorical, self._data.feature_input_order
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = self._data.get_ordered_features(factuals)

        list_cfs = []
        for index, row in factuals.iterrows():
            counterfactual = growing_spheres_search(
                row,
                self._mutables,
                self._immutables,
                self._continuous,
                self._categorical_enc,
                self._data.feature_input_order,
                self._mlmodel,
            )
            list_cfs.append(counterfactual)

        df_cfs = check_counterfactuals(self._mlmodel, list_cfs, factuals.index)
        df_cfs = self._data.get_ordered_features(df_cfs)
        return df_cfs
