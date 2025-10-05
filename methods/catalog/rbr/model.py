from typing import Dict, Optional

import pandas as pd

from methods.catalog.rbr.library.rbr_loss import robust_bayesian_recourse
from methods.processing.counterfactuals import merge_default_parameters

from ...api import RecourseMethod
from methods.processing import check_counterfactuals
from sklearn.utils import check_random_state

RANDOM_SEED = 54321

class RBR(RecourseMethod):
    """
    Implementation of Robust Bayesian Recourse [1]_.


    .. [1] Nguyen, Tuan-Duy Hien, Ngoc Bui, Duy Nguyen, Man-Chung Yue, and Viet Anh Nguyen. 2022. "Robust Bayesian Recourse." (UAI 2022)
    """
    _DEFAULT_HYPERPARAMS = {
        "num_samples": 50,
        "perturb_radius": 0.1,
        "delta_plus": 0.2,
        "sigma": 1.0,
        "epsilon_op": 0.0,
        "epsilon_pe": 0.0,
        "max_iter": 500,
        "device": "cpu",
        "clamp": True,
    }

    def __init__(self, mlmodel, hyperparams: Optional[Dict] = None):
        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(f"{mlmodel.backend} not supported (RBR supports: {supported_backends})")

        super().__init__(mlmodel)
        checked = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        self._num_samples = checked["num_samples"]
        self._perturb_radius = checked["perturb_radius"]
        self._delta_plus = checked["delta_plus"]
        self._sigma = checked["sigma"]
        self._epsilon_op = checked["epsilon_op"]
        self._epsilon_pe = checked["epsilon_pe"]
        self._max_iter = checked["max_iter"]
        self._device = checked["device"]
        self._clamp = checked["clamp"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = self._mlmodel.get_ordered_features(factuals)

        # categorical encoded feature indices (if MLModel provides list)
        encoded_feature_names = self._mlmodel.data.categorical
        cat_features_indices = [factuals.columns.get_loc(f) for f in encoded_feature_names] if len(encoded_feature_names) > 0 else None

        # train_data passed from MLModel wrapper
        # CARLA commonly stores training data in mlmodel.data.train or similar; if not present, expect user to pass via hyperparams or external config.
        if hasattr(self._mlmodel.data, "train"):
            train_data = self._mlmodel.data.train
        else:
            # try raw training data as fallback
            train_data = getattr(self._mlmodel.data, "X", None)
            if train_data is None:
                raise ValueError("RBR needs training data available via mlmodel.data.train or mlmodel.data.X")

        # ensure numpy arrays:
        if isinstance(train_data, pd.DataFrame):
            train_np = train_data.values
        else:
            train_np = train_data

        def apply_rbr(x_row: pd.Series) -> pd.Series:
            x_np = x_row.reshape((1, -1)).astype(float)
            cf = robust_bayesian_recourse(
                self._mlmodel.raw_model,
                x_np.squeeze(),
                cat_features_indices=cat_features_indices,
                train_data=train_np,
                num_samples=self._num_samples,
                perturb_radius=self._perturb_radius,
                delta_plus=self._delta_plus,
                sigma=self._sigma,
                epsilon_op=self._epsilon_op,
                epsilon_pe=self._epsilon_pe,
                max_iter=self._max_iter,
                device=self._device,
                random_state=RANDOM_SEED,
                verbose=False,
            )
            # optional final clamp (0,1) if requested
            if self._clamp:
                cf = cf.clip(0.0, 1.0)
            return cf

        df_cfs = factuals.apply(lambda row: apply_rbr(row.values), raw=True, axis=1)
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs