import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

from data.catalog.loadData import loadDataset
from methods.api import RecourseMethod
from methods.processing import check_counterfactuals, merge_default_parameters
from models.api import MLModel
from tools.logging import log


class _CFVAE(nn.Module):
    def __init__(self, feature_num: int, encoded_size: int):
        super().__init__()
        self.feature_num = feature_num
        self.encoded_size = encoded_size

        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid(),
        )

        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size + 1, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.feature_num),
            nn.Sigmoid(),
        )

    def encoder(self, x: torch.Tensor):
        x = x.float()
        mean = self.encoder_mean(x)
        var = 0.5 + self.encoder_var(x)  # Quirk
        return mean, var

    def sample_latent_code(self, mean: torch.Tensor, var: torch.Tensor):
        mean, var = mean.float(), var.float()
        eps = torch.randn_like(var)
        return mean + torch.sqrt(var) * eps

    def decoder(self, z: torch.Tensor):
        z = z.float()
        return self.decoder_mean(z)

    def normal_likelihood(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor):
        x = x.float()
        mean, var = mean.float(), var.float()
        return torch.sum(
            -0.5 * ((x - mean) * (1.0 / var) * (x - mean) + torch.log(var)),
            dim=1,
        )

    def forward(self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True):
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        z = self.sample_latent_code(mean, var) if sample else mean
        return self.decoder(torch.cat((z, conditions), dim=1))

    def forward_with_kl(
        self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True
    ):
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        kl_divergence = 0.5 * torch.mean(mean**2 + var - torch.log(var) - 1, dim=1)
        z = self.sample_latent_code(mean, var) if sample else mean
        x_pred = self.decoder(torch.cat((z, conditions), 1))
        return x_pred, kl_divergence


class CFVAE(RecourseMethod):
    """
    Implementation of CFVAE [1]_

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    train:
        Train CFVAE with custom args and optional saving path.
    load:
        Load CFVAE from saving path.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "encoded_size": int, default: 10
            Size of VAE latent code.
        * "train": bool, default: True
            Whether to train when init.


    .. [1] Preserving causal constraints in counterfactual explanations for machine learning classifiers
            D Mahajan, C Tan, A Sharma - arXiv preprint arXiv:1912.03277, 2019..
    """

    _DEFAULT_HYPERPARAMS = {
        "encoded_size": 10,
        "train": True,
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Optional[Dict] = None) -> None:
        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        feature_num = len(self._mlmodel.feature_input_order)
        encoded_size = self._params["encoded_size"]
        self._cf_model = _CFVAE(feature_num, encoded_size)
        self._trained = False

        if self._params["train"]:
            self.train()

    def train(
        self,
        save_path: Optional[str] = None,
        batch_size: int = 2048,
        epoch: int = 50,
        learning_rate: float = 1e-2,
        constraint_loss_func=None,
        preference_dataset=None,
        n_samples: int = 50,
        margin: float = 0.2,
        validity_reg: float = 20,
        constraint_reg: float = 1,
        preference_reg: float = 1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"  # pyright: ignore[reportAttributeAccessIssue]
        ),
    ):
        """
        Train CFVAE with custom args and optional saving path.

        Parameters
        ----------
        save_path: str | None, default: None
            If provided, save .pth to save_path. By default will not save .pth.
        batch_size: int, default: 2048
            Batch size for dataloader.
        epoch: int, default: 50
            Num of training epochs.
        learning_rate: float, default: 1e-2
            Learning rate (for SGD with weight decay 1e-2).
        constraint_loss_func: function | None, default: None
            A function taking a tensor (train_x in feature_input_order) as input, returning a single-value tensor of constraint loss.
            If provided, will apply ModelApproxCF's loss as written in the original paper.
            See constraint_loss_func_example() for example, which contraints train_x[0] to go upwards.
        preference_dataset: dict | None, default: None
            Dict[str:Dict[torch.Tensor:List[torch.Tensor]]].
            preference_dataset["x_prefer"][train_x][idx] and preference_dataset["y_prefer"][train_x][idx] are corresponding user/oracle data points for train_x.
            If provided, will apply ExampleBasedCF's loss as written in the original paper.
        n_samples: int, default: 50
            For one train_x, how many times we should sample from the latent distribution when training.
        margin: float, default: 0.2
            margin for the reconstruction hinge loss.
        validity_reg: float, default: 20
            Lambda for validity loss term. See the original paper.
        constraint_reg: float, default: 1
            Lambda for constraint loss term. See the original paper.
            Only valid when constraint_loss_func is not None.
        preference_reg: float, default: 1
            Lambda for preference/example-based loss term. See the original paper.
            Only valid when preference_dataset is not None.
        device: torch.device: torch.device, default: will auto choose torch.device("cuda") when available
            Which device we should train on. Will auto choose cuda:0/the first available NVIDIA GPU by default.
        """
        train_dataset: pd.Dataframe = self._mlmodel.data.df_train[
            self._mlmodel.feature_input_order
        ].values
        train_dataset = torch.tensor(train_dataset).float()
        dataset_size = train_dataset.size(0)
        train_loader = torch.utils.data.DataLoader(  # pyright: ignore[reportAttributeAccessIssue]
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
        )

        feature_order = self._mlmodel.feature_input_order
        data_catalog = self._mlmodel.data
        continuous_features = [
            feature for feature in feature_order if feature in data_catalog.continuous
        ]
        categorical_features = [
            feature for feature in feature_order if feature in data_catalog.categorical
        ]
        categorical_indices = [
            feature_order.index(feature) for feature in categorical_features
        ]

        # Prepare categorical sibling groups for enforcing one-hot plausibility
        dataset_one_hot = None
        encoded_categorical_feature_indexes: List[List[int]] = []
        try:
            dataset_one_hot = loadDataset(
                data_catalog.name,
                return_one_hot=True,
                load_from_cache=True,
                debug_flag=False,
            )
            dict_of_siblings = dataset_one_hot.getDictOfSiblings("kurz")
            for siblings in dict_of_siblings.get("cat", {}).values():
                indices = [feature_order.index(col) for col in siblings if col in feature_order]
                if indices:
                    encoded_categorical_feature_indexes.append(indices)
        except Exception:
            encoded_categorical_feature_indexes = []

        # Calculate original value ranges for continuous features
        normalise_weights: Dict[int, Tuple[float, float]] = {}
        attribute_lookup = {}
        if dataset_one_hot is not None:
            raw_attributes = getattr(dataset_one_hot, "attributes_kurz", {})
            if isinstance(raw_attributes, dict):
                attribute_lookup = raw_attributes
        decoded_train = None
        for feature in continuous_features:
            idx = feature_order.index(feature)
            attr = attribute_lookup.get(feature)
            if attr is not None:
                min_val = float(attr.lower_bound)
                max_val = float(attr.upper_bound)
            else:
                if decoded_train is None:
                    decoded_train = data_catalog.inverse_transform(
                        data_catalog.df_train[feature_order]
                    )
                column = decoded_train[feature]
                min_val = float(column.min())
                max_val = float(column.max())
            normalise_weights[idx] = (min_val, max_val)

        self._cf_model.train().to(device)
        optimizer = optim.SGD(
            self._cf_model.parameters(), lr=learning_rate, weight_decay=1e-2
        )

        for _ in tqdm(range(epoch)):
            with tqdm(total=dataset_size, desc="loss: N/A") as pbar:
                for train_x in train_loader:
                    train_x = train_x.float().to(device)
                    optimizer.zero_grad()

                    with torch.no_grad():
                        train_y = 1.0 - torch.argmax(
                            self._mlmodel.forward(train_x), dim=1
                        )

                    reconstruction_loss = torch.zeros(train_x.size(0)).to(device)
                    kl_loss = torch.zeros(train_x.size(0)).to(device)
                    validity_loss = torch.zeros(1).to(device)
                    constraint_loss = torch.zeros(1).to(device)  # Optional
                    preference_loss = torch.zeros(1).to(device)  # Optional
                    for _ in range(n_samples):
                        # kl_loss won't change for the same train_x
                        x_pred, kl_loss = self._cf_model.forward_with_kl(
                            train_x, train_y
                        )

                        # Reconstruction Term
                        reconstruction_increment = torch.zeros(
                            train_x.size(0), device=device
                        )

                        # Categorical features
                        if categorical_indices:
                            reconstruction_increment += -torch.sum(
                                torch.abs(
                                    train_x[:, categorical_indices]
                                    - x_pred[:, categorical_indices]
                                ),
                                dim=1,
                            )

                        # Continuous features
                        for key, (min_val, max_val) in normalise_weights.items():
                            range_val = max_val - min_val
                            if range_val <= 0:
                                range_val = 1.0
                            reconstruction_increment += -range_val * torch.abs(
                                train_x[:, key] - x_pred[:, key]
                            )

                        # Sum to 1 over the categorical indexes of a feature
                        for index_group in encoded_categorical_feature_indexes:
                            if index_group:
                                reconstruction_increment += -torch.abs(
                                    1.0 - torch.sum(x_pred[:, index_group], dim=1)
                                )

                        reconstruction_loss += reconstruction_increment

                        # Validity
                        y_pred = self._mlmodel.forward(x_pred)
                        y_pred_pos = y_pred[train_y == 1, :]
                        y_pred_neg = y_pred[train_y == 0, :]
                        if torch.sum(train_y == 1) > 0:
                            validity_loss += F.hinge_embedding_loss(
                                y_pred_pos[:, 1] - y_pred_pos[:, 0],
                                torch.tensor(-1).to(device),
                                margin,
                                reduction="mean",
                            )
                        if torch.sum(train_y == 0) > 0:
                            validity_loss += F.hinge_embedding_loss(
                                y_pred_neg[:, 0] - y_pred_neg[:, 1],
                                torch.tensor(-1).to(device),
                                margin,
                                reduction="mean",
                            )

                        # Optional Constraint Loss
                        if constraint_loss_func != None:
                            constraint_loss += constraint_loss_func(
                                train_x=train_x, x_pred=x_pred
                            )

                        # Optional Preference Loss
                        if preference_dataset != None:
                            pos_preference = []
                            neg_preference = []
                            for x in train_x:
                                for x_prefer, y_prefer in zip(
                                    preference_dataset["x_prefer"][x],
                                    preference_dataset["y_prefer"][x],
                                ):
                                    if y_prefer == 1:
                                        pos_preference.append(
                                            torch.exp(
                                                -0.5
                                                * (
                                                    (x_prefer - x_pred)
                                                    * (x_prefer - x_pred)
                                                )
                                            )
                                        )
                                    else:
                                        neg_preference.append(
                                            torch.exp(
                                                -0.5
                                                * (
                                                    (x_prefer - x_pred)
                                                    * (x_prefer - x_pred)
                                                )
                                            )
                                        )
                            if len(pos_preference):
                                preference_loss += 1 - torch.mean(
                                    torch.stack(pos_preference)
                                ).to(device)
                            if len(neg_preference):
                                preference_loss += torch.mean(
                                    torch.stack(neg_preference)
                                ).to(device)

                    reconstruction_loss = reconstruction_loss / n_samples
                    kl_loss = kl_loss / n_samples
                    validity_loss = -1 * validity_reg * validity_loss / n_samples
                    constraint_loss = constraint_reg * constraint_loss / n_samples
                    preference_loss = preference_reg * preference_loss / n_samples

                    loss = (
                        -torch.mean(reconstruction_loss - kl_loss)
                        - validity_loss
                        + constraint_loss
                        + preference_loss
                    )

                    loss.backward()
                    optimizer.step()

                    pbar.set_description(
                        "Reconstruction: "
                        + str(-torch.mean(reconstruction_loss).item())
                        + " KL: "
                        + str(torch.mean(kl_loss).item())
                        + " Validity: "
                        + str(torch.mean(-validity_loss).item())
                    )
                    pbar.update(len(train_x))

        if save_path:
            path = os.path.join(
                save_path,
                (
                    "CFVAE"
                    + "-margin-"
                    + str(margin)
                    + "-validity_reg-"
                    + str(validity_reg)
                    + "-epoch-"
                    + str(epoch)
                    + "-"
                    + (
                        "ExampleBasedCF"
                        if preference_dataset != None
                        else (
                            "ModelApproxCF"
                            if constraint_loss_func != None
                            else "ModelBasedCF" + ".pth"
                        )
                    )
                ),
            )
            torch.save(self._cf_model.state_dict(), path)
            log.info(f"Saved to: {path}")
        self._trained = True

    def load(self, load_path: str):
        """
        Load CFVAE from saving path.

        Parameters
        ----------
        load_path: str
            Path to the .pth.
        """
        self._cf_model.load_state_dict(
            torch.load(load_path, map_location=torch.device("cpu"))
        )
        log.info(f"Loaded from: {load_path}")
        self._trained = True

    def get_counterfactuals(
        self,
        factuals: pd.DataFrame,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"  # pyright: ignore[reportAttributeAccessIssue]
        ),
    ) -> pd.DataFrame:
        assert self._trained, "Error: Run train() or load() first!"
        self._cf_model.eval().to(device)

        factuals = self._mlmodel.get_ordered_features(factuals)
        cat_features_indices = [
            factuals.columns.get_loc(feature)
            for feature in self._mlmodel.data.categorical
        ]

        def generate(test_x, sample: bool = False):
            with torch.no_grad():
                test_x = torch.tensor(test_x).float().view(1, -1).to(device)
                test_y = 1.0 - torch.argmax(self._mlmodel.forward(test_x), dim=1)
                x_pred = self._cf_model(test_x, test_y, sample=sample)

                # Round categorical features like reconstruct_encoding_constraints does with binary_cat=True
                # Reimplement here since reconstruct_encoding_constraints is faulty  # TODO
                for idx in cat_features_indices:
                    x_pred[:, idx] = torch.round(x_pred[:, idx])

                return x_pred.view(-1).cpu().numpy()

        df_cfs = factuals.apply(
            lambda x: generate(x),
            raw=True,
            axis=1,
        )

        # df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)  # disabled due to faulty arg: negative_label
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs

    @staticmethod
    def constraint_loss_func_example(train_x: torch.Tensor, x_pred: torch.Tensor):
        return F.hinge_embedding_loss(
            x_pred[:, 0] - train_x[:, 0], torch.tensor(-1).to(train_x.device), 0
        )
