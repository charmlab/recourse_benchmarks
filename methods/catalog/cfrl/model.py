from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

from data.catalog.loadData import loadDataset
from methods.api import RecourseMethod
from methods.processing import merge_default_parameters
from models.api import MLModel
from tools.logging import log

from .cfrl_tabular import CounterfactualRLTabular as CFRLExplainer
from .cfrl_tabular import get_he_preprocessor


class HeterogeneousEncoder(nn.Module):
    """Simple feed-forward encoder used for the heterogeneous auto-encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.network(x)


class HeterogeneousDecoder(nn.Module):
    """Decoder with one head for numerical features and one head per categorical feature."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dims: List[int]) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.num_dim = output_dims[0] if output_dims else 0
        self.num_head = (
            nn.Linear(hidden_dim, self.num_dim) if self.num_dim > 0 else None
        )
        self.cat_heads = nn.ModuleList(
            nn.Linear(hidden_dim, dim) for dim in output_dims[1:]
        )

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:  # noqa: D401
        h = self.shared(z)
        outputs: List[torch.Tensor] = []
        if self.num_head is not None:
            outputs.append(self.num_head(h))
        outputs.extend(head(h) for head in self.cat_heads)
        return outputs


@dataclass
class _FeatureMetadata:
    feature_names: List[str]
    long_to_short: Dict[str, str]
    short_to_long: Dict[str, str]
    attr_bounds: Dict[str, List[float]]
    categorical_indices: List[int]
    numerical_indices: List[int]
    raw_to_idx: Dict[str, Dict[int, int]]
    idx_to_raw: Dict[str, Dict[int, int]]
    category_map: Dict[int, List[str]]
    feature_types: Dict[str, type]


class CFRL(RecourseMethod):
    """
    Implementation of CFRL [1]_.

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-box model to explain.
    hyperparams : dict, optional
        Hyper-parameter dictionary.

    Notes
    -----
    - Hyperparams

        * ``latent_dim``: :class:`int`, default ``15`` \
          Latent dimension of the auto-encoder.
        * ``encoder_hidden_dim``: :class:`int`, default ``128`` \
          Hidden dimension of encoder/decoder.
        * ``autoencoder_epochs``: :class:`int`, default ``50`` \
          Number of training epochs for the auto-encoder.
        * ``autoencoder_batch_size``: :class:`int`, default ``128`` \
          Batch size used when training the auto-encoder.
        * ``autoencoder_lr``: :class:`float`, default ``1e-3`` \
          Learning rate for the auto-encoder optimiser.
        * ``coeff_sparsity``: :class:`float`, default ``0.5`` \
          Sparsity loss coefficient for the CFRL explainer.
        * ``coeff_consistency``: :class:`float`, default ``0.5`` \
          Consistency loss coefficient for the CFRL explainer.
        * ``train_steps``: :class:`int`, default ``10000`` \
          Number of reinforcement learning optimisation steps.
        * ``batch_size``: :class:`int`, default ``100`` \
          Batch size used by the CFRL explainer.
        * ``seed``: :class:`int`, default ``0`` \
          Seed forwarded to the CFRL explainer.
        * ``immutable_features``: :class:`List[str]`, optional \
          Override list of immutable features (long names). \
          Defaults to dataset metadata when available.
        * ``ranges``: :class:`Dict[str, List[float]]`, optional \
          Override numeric ranges (long names) passed to CFRL.
        * ``train``: :class:`bool`, default ``True`` \
          Train explainer automatically after construction.

    .. [1] Samoilescu RF et al., *Model-agnostic and Scalable Counterfactual
           Explanations via Reinforcement Learning*, 2021.
    """

    _DEFAULT_HYPERPARAMS: Dict[str, object] = {
        "latent_dim": 15,
        "encoder_hidden_dim": 128,
        "autoencoder_epochs": 50,
        "autoencoder_batch_size": 128,
        "autoencoder_lr": 1e-3,
        "coeff_sparsity": 0.5,
        "coeff_consistency": 0.5,
        "train_steps": 10000,
        "batch_size": 100,
        "seed": 0,
        "immutable_features": "_optional_",
        "ranges": "_optional_",
        "train": True,
    }
    _OPTIONAL_HYPERPARAMS = {"immutable_features", "ranges"}

    def __init__(self, mlmodel: MLModel, hyperparams: Optional[Dict] = None) -> None:
        supported_backends = {"pytorch"}
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)

        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        for key in self._OPTIONAL_HYPERPARAMS:
            if self._params[key] == "_optional_":
                self._params[key] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._metadata = self._prepare_metadata()

        self._encoder: Optional[HeterogeneousEncoder] = None
        self._decoder: Optional[HeterogeneousDecoder] = None
        self._encoder_preprocessor = None
        self._decoder_inv_preprocessor = None
        self._cf_model: Optional[CFRLExplainer] = None
        self._trained = False

        if self._params["train"]:
            self.train()

    def _prepare_metadata(self) -> _FeatureMetadata:
        dataset_name = getattr(self._mlmodel.data, "name", None)
        if dataset_name is None:
            raise ValueError(
                "Unable to infer dataset name."
            )

        dataset = loadDataset(
            dataset_name,
            return_one_hot=False,
            load_from_cache=True,
            debug_flag=False,
        )

        feature_names = list(dataset.getInputAttributeNames("long"))
        long_to_short = {
            name: dataset.attributes_long[name].attr_name_kurz for name in feature_names
        }
        short_to_long = {v: k for k, v in long_to_short.items()}

        attr_bounds: Dict[str, List[float]] = {}
        categorical_indices: List[int] = []
        numerical_indices: List[int] = []
        raw_to_idx: Dict[str, Dict[int, int]] = {}
        idx_to_raw: Dict[str, Dict[int, int]] = {}
        category_map: Dict[int, List[str]] = {}
        feature_types: Dict[str, type] = {}

        df = dataset.data_frame_long[feature_names]

        for idx, name in enumerate(feature_names):
            attr = dataset.attributes_long[name]
            attr_bounds[name] = [attr.lower_bound, attr.upper_bound]

            if attr.attr_type in {"numeric-int", "numeric-real"}:
                numerical_indices.append(idx)
                if attr.attr_type == "numeric-int":
                    feature_types[name] = int
                else:
                    feature_types[name] = float
                continue

            categorical_indices.append(idx)
            unique_vals = sorted(df[name].dropna().unique())
            mapped_unique = [int(round(v)) for v in unique_vals]
            raw_to_idx[name] = {val: i for i, val in enumerate(mapped_unique)}
            idx_to_raw[name] = {i: val for i, val in enumerate(mapped_unique)}
            category_map[idx] = [str(val) for val in mapped_unique]

        return _FeatureMetadata(
            feature_names=feature_names,
            long_to_short=long_to_short,
            short_to_long=short_to_long,
            attr_bounds=attr_bounds,
            categorical_indices=categorical_indices,
            numerical_indices=numerical_indices,
            raw_to_idx=raw_to_idx,
            idx_to_raw=idx_to_raw,
            category_map=category_map,
            feature_types=feature_types,
        )

    # --------------------------------------------------------------------- #
    # Helper conversions between representations                            #
    # --------------------------------------------------------------------- #
    def _raw_df_to_zero_array(self, df_raw: pd.DataFrame) -> np.ndarray:
        arr = df_raw[self._metadata.feature_names].to_numpy(dtype=np.float32, copy=True)
        for col_idx in self._metadata.categorical_indices:
            name = self._metadata.feature_names[col_idx]
            mapping = self._metadata.raw_to_idx[name]
            raw_values = df_raw[name].round().astype(int).to_numpy()
            arr[:, col_idx] = np.vectorize(mapping.__getitem__)(raw_values)
        return arr

    def _zero_array_to_raw_df(self, arr_zero: np.ndarray) -> pd.DataFrame:
        arr = np.asarray(arr_zero).copy()
        for col_idx in self._metadata.categorical_indices:
            name = self._metadata.feature_names[col_idx]
            mapping = self._metadata.idx_to_raw[name]
            values = np.clip(
                np.round(arr[:, col_idx]).astype(int),
                0,
                len(mapping) - 1,
            )
            arr[:, col_idx] = np.vectorize(mapping.__getitem__)(values)

        df = pd.DataFrame(arr, columns=self._metadata.feature_names)
        for name, dtype in self._metadata.feature_types.items():
            df[name] = df[name].astype(dtype)
        return df

    def _normalize_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        for name in self._metadata.feature_names:
            lower, upper = self._metadata.attr_bounds[name]
            if np.isclose(upper, lower):
                df[name] = 0.0
                continue
            df[name] = (df[name] - lower) / (upper - lower)
            df[name] = df[name].clip(0.0, 1.0)
        return df

    def _denormalize_df(self, df_norm: pd.DataFrame) -> pd.DataFrame:
        df = df_norm.copy()
        for name in self._metadata.feature_names:
            lower, upper = self._metadata.attr_bounds[name]
            df[name] = df[name] * (upper - lower) + lower
            if name in self._metadata.raw_to_idx:
                df[name] = df[name].round().clip(lower, upper)
        return df

    # --------------------------------------------------------------------- #
    # Predictor and auto-encoder training                                   #
    # --------------------------------------------------------------------- #
    def _build_predictor(self):
        def predictor(x: np.ndarray) -> np.ndarray:
            array = np.atleast_2d(x)
            df_zero = pd.DataFrame(array, columns=self._metadata.feature_names)
            df_raw = self._zero_array_to_raw_df(df_zero.to_numpy())
            df_norm = self._normalize_df(df_raw)
            df_short = df_norm.rename(columns=self._metadata.long_to_short)
            ordered = self._mlmodel.get_ordered_features(df_short)
            preds = self._mlmodel.predict_proba(ordered)
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            return preds

        return predictor

    def _train_autoencoder(self, X_pre: np.ndarray, X_zero: np.ndarray) -> None:
        epochs = int(self._params["autoencoder_epochs"])
        batch_size = int(self._params["autoencoder_batch_size"])
        lr = float(self._params["autoencoder_lr"])

        inputs = torch.tensor(X_pre, dtype=torch.float32, device=self._device)

        num_dim = (
            len(self._metadata.numerical_indices)
            if self._metadata.numerical_indices
            else 0
        )
        if num_dim > 0:
            num_targets = inputs[:, :num_dim]
        else:
            num_targets = None

        cat_targets = [
            torch.tensor(
                X_zero[:, idx].astype(np.int64),
                dtype=torch.long,
                device=self._device,
            )
            for idx in self._metadata.categorical_indices
        ]

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optimiser = optim.Adam(params, lr=lr)

        num_samples = inputs.size(0)
        for epoch in range(epochs):
            perm = torch.randperm(num_samples, device=self._device)
            epoch_loss = 0.0

            for start in range(0, num_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch_x = inputs[idx]
                outputs = self._decoder(self._encoder(batch_x))

                loss = torch.zeros((), device=self._device)

                output_offset = 0
                if num_dim > 0 and num_targets is not None:
                    recon_num = outputs[0]
                    target_num = num_targets[idx]
                    loss = loss + F.mse_loss(recon_num, target_num)
                    output_offset = 1

                cat_outputs = outputs[output_offset:]
                cat_batches = [target[idx] for target in cat_targets]

                if cat_outputs and cat_batches:
                    weight = 1.0 / len(cat_outputs)
                    for logits, target in zip(cat_outputs, cat_batches):
                        loss = loss + weight * F.cross_entropy(logits, target)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()

            if epochs <= 10 or (epoch + 1) % max(1, epochs // 5) == 0:
                log.info(
                    "CFRL autoencoder epoch %s/%s | loss %.4f",
                    epoch + 1,
                    epochs,
                    epoch_loss / max(1, num_samples // batch_size),
                )

        self._encoder.eval()
        self._decoder.eval()

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def train(self) -> None:  # noqa: D401
        dataset_name = getattr(self._mlmodel.data, "name", None)

        dataset = loadDataset(
            dataset_name,
            return_one_hot=False,
            load_from_cache=True,
            debug_flag=False,
        )
        df_raw = dataset.data_frame_long[self._metadata.feature_names].copy()

        X_zero = self._raw_df_to_zero_array(df_raw).astype(np.float32)
        self._encoder_preprocessor, self._decoder_inv_preprocessor = get_he_preprocessor(
            X=X_zero,
            feature_names=self._metadata.feature_names,
            category_map=self._metadata.category_map,
            feature_types=self._metadata.feature_types,
        )

        X_pre = self._encoder_preprocessor(X_zero).astype(np.float32)
        input_dim = X_pre.shape[1]

        latent_dim = int(self._params["latent_dim"])
        hidden_dim = int(self._params["encoder_hidden_dim"])

        self._encoder = HeterogeneousEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self._device)

        output_dims = [len(self._metadata.numerical_indices)]
        output_dims += [
            len(self._metadata.category_map[idx])
            for idx in self._metadata.categorical_indices
        ]

        self._decoder = HeterogeneousDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dims=output_dims,
        ).to(self._device)

        log.info("Training CFRL heterogeneous autoencoder.")
        self._train_autoencoder(X_pre, X_zero)

        predictor = self._build_predictor()

        immutable_features = self._params.get("immutable_features")
        if immutable_features is None:
            data_immutables = getattr(self._mlmodel.data, "immutables", [])
            immutable_features = [
                self._metadata.short_to_long.get(name, name) for name in data_immutables
            ]
        else:
            immutable_features = [
                self._metadata.short_to_long.get(name, name)
                for name in immutable_features
            ]

        ranges = self._params.get("ranges") or {}
        ranges_long = {
            self._metadata.short_to_long.get(name, name): bounds
            for name, bounds in ranges.items()
        }

        log.info("Training CFRL explainer (train_steps=%s).", self._params["train_steps"])
        self._cf_model = CFRLExplainer(
            predictor=predictor,
            encoder=self._encoder,
            decoder=self._decoder,
            latent_dim=latent_dim,
            encoder_preprocessor=self._encoder_preprocessor,
            decoder_inv_preprocessor=self._decoder_inv_preprocessor,
            coeff_sparsity=float(self._params["coeff_sparsity"]),
            coeff_consistency=float(self._params["coeff_consistency"]),
            feature_names=self._metadata.feature_names,
            category_map=self._metadata.category_map,
            immutable_features=immutable_features,
            ranges=ranges_long,
            train_steps=int(self._params["train_steps"]),
            batch_size=int(self._params["batch_size"]),
            seed=int(self._params["seed"]),
        )
        self._cf_model.fit(X_zero.astype(np.float32))
        self._trained = True

    def _generate_counterfactual(self, factual_row: pd.DataFrame) -> pd.DataFrame:
        factual_ordered = self._mlmodel.get_ordered_features(factual_row)
        long_named = factual_ordered.rename(columns=self._metadata.short_to_long)
        raw_df = self._denormalize_df(long_named)
        zero_input = self._raw_df_to_zero_array(raw_df)

        preds = self._mlmodel.predict_proba(factual_ordered)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        target_class = 1 - int(np.argmax(preds, axis=1)[0])

        explanation = self._cf_model.explain(
            X=zero_input.astype(np.float32),
            Y_t=np.array([target_class]),
        )
        cf_data = explanation.data.get("cf", {}).get("X")
        if cf_data is None:
            log.warning("CFRL failed to produce a counterfactual; falling back to input.")
            return factual_ordered

        cf_array = np.asarray(cf_data)
        if cf_array.ndim == 3:
            cf_array = cf_array[:, 0, :]
        cf_array = np.atleast_2d(cf_array)

        cf_raw = self._zero_array_to_raw_df(cf_array)
        cf_norm = self._normalize_df(cf_raw)
        cf_short = cf_norm.rename(columns=self._metadata.long_to_short)
        cf_ordered = self._mlmodel.get_ordered_features(cf_short)
        cf_ordered.index = factual_ordered.index
        return cf_ordered

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        assert self._trained, "Error: run train() first."
        results: List[pd.DataFrame] = []

        for index, row in factuals.iterrows():
            cf_row = self._generate_counterfactual(pd.DataFrame([row]))
            cf_row.index = [index]
            results.append(cf_row)

        counterfactuals = pd.concat(results, axis=0)
        return self._mlmodel.get_ordered_features(counterfactuals)
