# util functions from the original RBR implementation
# meant mostly for the reproduce.py script

# Following instructions by professor Karimi,
# this file will contain all the preprocesing/postprocessing functions
# and model creation, that exists in this repo, but modified to function
# with the implementation of RBR in methods/catalog/rbr to get the results
# as close as possible to the original paper.
from typing import Any, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split

from data.api.data import Data
from methods.catalog.rbr.library.utils_general import Transformer, get_transformer
from models.api.mlmodel import MLModel

# define the model used in the paper
# Custom Pytorch Module for Neural Networks
class PyTorchNeuralNetworkTemp(torch.nn.Module):
    """
    Initializes a PyTorch neural network model with specified number of inputs, outputs, and neurons.

    Parameters
    ----------
    n_inputs (int): Number of input features.

    Returns
    -------
    PyTorchNeuralNetwork.

    Raises
    -------
    None.
    """

    # Constructor
    def __init__(self, n_inputs):
        super(PyTorchNeuralNetworkTemp, self).__init__()
        self.fc1 = torch.nn.Linear(n_inputs, 20)
        self.fc2 = torch.nn.Linear(20, 50)
        self.fc3 = torch.nn.Linear(50, 20)
        self.out = torch.nn.Linear(20, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    # Predictions
    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Parameters
        -------
        x (torch.Tensor): Input tensor to the neural network.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x

    def fit(self, x_train, y_train):
        """
        Fits the neural network to the training data.

        Parameters
        ----------
        x_train (array-like): Input training data.
        y_train (array-like): Target training data.

        Returns
        -------
        PyTorchNeuralNetwork: Trained neural network instance.

        Raises
        ------
        None.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = next(self.parameters()).device
        x_train_tensor = torch.from_numpy(np.array(x_train, dtype=np.float32))
        y_train_tensor = torch.from_numpy(np.array(y_train, dtype=np.float32))
        # print(f"x_train_tensor shape: {x_train_tensor[:5]}")
        # print(f"y_train_tensor shape: {y_train_tensor.view(-1,1)[:5]}")

        # train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
        # train_loader = torch.utils.data.DataLoader(
        #     dataset=train_dataset, batch_size=1000, shuffle=True
        # )
        self.train()
        # defining the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.1)
        # defining Cross-Entropy loss
        criterion = torch.nn.BCELoss(reduction="sum")

        loss_diff = 1.0
        prev_loss = 0.0
        num_stable_iter = 0
        max_stable_iter = 3

        epochs = 1000  # TODO increase epochs because paper base is 1000
        for i in range(epochs):
            # for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = self(x_train_tensor.to(device))
            loss = criterion(output, y_train_tensor.view(-1, 1).to(device))

            loss.backward()
            optimizer.step()

            # print("Iter %d: loss: %f" % (i, loss.data.item()))

            loss_diff = prev_loss - loss.data.item()

            if loss_diff <= 1e-7:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0

            prev_loss = loss.data.item()

        self.eval()

        return self


    def predict(self, test):
        """
        Predicts using the trained neural network.

        Parameters
        -------
        test (torch.Tensor): Input tensor for prediction.

        Returns
        -------
        torch.Tensor: Predicted output tensor.

        Raises
        -------
        None.
        """
        device = next(self.parameters()).device
        self.eval()
        y_train_pred = []
        with torch.no_grad():
            output = self(test.to(device))
            # print(f"output shape in predict: {output[:5]}")
            y_train_pred.extend(output)

        y_train_pred = torch.stack(y_train_pred)
        # print(f"y_train_pred shape: {y_train_pred[:5]}")
        return y_train_pred


# implement my own version of the DataCatalog that uses this model
class DataTemp(Data):
    """
    Custom Data class for handling dataset operations.

    Parameters
    ----------
    df: pd.DataFrame
        The input dataframe containing the dataset.
    continuous: List[str]
        List of continuous feature names.
    categorical: List[str]
        List of categorical feature names.
    immutable: List[str]
        List of immutable feature names.
    target: str
        The target variable name.

    Returns
    -------
    DataTemp
    """

    def __init__(
        self,
        df_name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        continuous: List[str],
        categorical: List[str],
        immutable: List[str],
        transformer: Transformer,
        target: str,
    ):
        # self._df = df.copy()
        self._continuous = continuous
        self._categorical = categorical
        self._immutable = immutable
        self._target = target
        # create train/test split
        # for simplicity, we will just do a simple split here

        X_train = transformer.transform(X_train)
        X_test = transformer.transform(X_test)

        # dataset_obj = pd.concat([X_train, X_test], ignore_index=True)
        # output_merge = pd.concat([y_train, y_test], ignore_index=True)
        # dataset_obj["y"] = output_merge
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

        X_train["y"] = y_train.values
        X_test["y"] = y_test.values

        self._df_train = X_train
        self._df_test = X_test

    @property
    def categorical(self) -> List[str]:
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        return self._categorical

    @property
    def continuous(self) -> List[str]:
        """
        Provides the column names of continuous data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all continuous columns
        """
        return self._continuous

    @property
    def df(self) -> pd.DataFrame:
        """
        The full Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        return None

    @property
    def df_train(self) -> pd.DataFrame:
        """
        The training split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        """
        The testing split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        return self._df_test.copy()

    @property
    def immutables(self) -> Union[List[str], None]:
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        return None

    @property
    def target(self) -> str:
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        return self._target

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to keep correct encodings and normalization

        Parameters
        ----------
        df : pd.DataFrame
            Contains raw (not normalized and not encoded) data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input normalized and encoded

        """
        output = df.copy()
        return output

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms output after prediction back into original form.
        Only possible for DataFrames with preprocessing steps.

        Parameters
        ----------
        df : pd.DataFrame
            Contains normalized and encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction output denormalized and decoded

        """
        output = df.copy()
        return output


# Make my own verion of ModelCatalog that uses this model
class ModelCatalogTemp(MLModel):
    """
    Use pretrained classifier.

    Parameters
    ----------
    data : data.catalog.DataCatalog Class
        Correct dataset for ML model.

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model

    Returns
    -------
    None
    """

    def __init__(
        self,
        data: Data,
        # train_data: pd.DataFrame,
        model_type: str = "mlp",
        backend: str = "pytorch",
        # model_type: str, # we are just using the mlp for this paper
        **kws,
    ) -> None:
        """
        Constructor for pretrained ML models from the catalog.

        Possible backends are currently "pytorch", "tensorflow" for "ann" and "linear" models.
        Possible backends are currently "sklearn", "xgboost" for "forest" models.

        """
        super().__init__(data)
        self._model_type = model_type
        self._backend = backend

        self._continuous = data.continuous
        self._categorical = data.categorical

        self._feature_input_order = data.df_train.drop(
            columns=[data.target]
        ).columns.tolist()

        self._model = PyTorchNeuralNetworkTemp(n_inputs=len(self._feature_input_order))

        if self.backend == "pytorch":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device)

        tmp_text = (
            f"Training {self._model_type} model using {self._backend} backend on device {device}"
            + f" with {len(data.df_train)} training samples."
        )
        print(tmp_text)

        # kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # for i, (train_index, cross_index) in enumerate(kf.split(data.df_train.drop(columns=[data.target]))):
        #     # Get the training data for this fold
        #     X_training = data.df_train[train_index].drop(columns=[data.target])
        #     y_training = data.df_train[train_index][data.target]

        self._model = self._model.fit(
            x_train=data.df_train.drop(columns=[data.target]),
            y_train=data.df_train[data.target],
        )

    def _test_accuracy(self):
        # get preprocessed data
        df_test = self.data.df_test

        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]

        prediction = (self.predict(x_test) > 0.5).flatten()
        correct = prediction == y_test
        print(f"test accuracy for model: {correct.mean()}")

    @property
    def feature_input_order(self) -> List[str]:
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        ordered_features : list of str
            Correct order of input features for ml model
        """
        return self._feature_input_order

    @property
    def model_type(self) -> str:
        """
        Describes the model type

        E.g., ann, linear

        Returns
        -------
        backend : str
            model type
        """
        return self._model_type

    @property
    def backend(self) -> str:
        """
        Describes the type of backend which is used for the ml model.

        E.g., tensorflow, pytorch, sklearn, ...

        Returns
        -------
        backend : str
            Used framework
        """
        return self._backend

    @property
    def raw_model(self) -> Any:
        """
        Returns the raw ML model built on its framework

        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        return self._model

    def predict(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]:
        """
        One-dimensional prediction of ml model for an output interval of [0, 1]

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        if len(x.shape) != 2:
            raise ValueError(
                "Input shape has to be two-dimensional, (instances, features)."
            )

        if self._backend == "pytorch":
            return self.predict_proba(x)[:, 1].reshape((-1, 1))
        elif self._backend == "tensorflow":
            # keep output in shape N x 1
            # order data (column-wise) before prediction
            x = self.get_ordered_features(x)
            return self._model.predict(x)[:, 1].reshape((-1, 1))
        elif self._backend == "sklearn" or self._backend == "xgboost":
            return self._model.predict(self.get_ordered_features(x))
        else:
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
            )

    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor, tf.Tensor]:
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction with shape N x 2
        """

        # order data (column-wise) before prediction
        x = self.get_ordered_features(x)

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        if self._backend == "pytorch":
            # Keep model and input on the same device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)

            if isinstance(x, pd.DataFrame):
                _x = x.values
            elif isinstance(x, torch.Tensor):
                _x = x.clone()
            else:
                _x = x.copy()

            # If the input was a tensor, return a tensor. Else return a np array.
            tensor_output = torch.is_tensor(x)
            if not tensor_output:
                _x = torch.Tensor(_x)

            # input, tensor_output = (
            #     (torch.Tensor(x), False) if not torch.is_tensor(x) else (x, True)
            # )

            _x = _x.to(device)
            output = self._model.predict(_x)

            if tensor_output:
                return output
            else:
                return output.detach().cpu().numpy()

        elif self._backend == "tensorflow":
            return self._model.predict(x)
        elif self._backend == "sklearn" or self._backend == "xgboost":
            return self._model.predict_proba(x)
        else:
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
            )
