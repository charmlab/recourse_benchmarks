import os
import pathlib
from abc import ABC
from typing import Any, Dict, List

import pandas as pd

from data.api import Data
from data.catalog.loadData import loadDataset
from data.load_catalog import load


class DataCatalog(Data, ABC):
    """
    Implements DataCatalog using already implemented datasets. These datasets are loaded from the _data_main folder.

    Parameters
    ----------
    data_name : {'adult', 'compass', 'credit'}
        Used to get the correct dataset from online repository.
    model_type : {'mlp', 'linear', 'forest'}
        The model architecture. Multi-Layer Perceptron, Logistic Regression, and Random Forest respectively.
    train_split : float
        Specifies the split of the available data used for training the model.

    Returns
    -------
    DataCatalog
    """

    def __init__(
        self,
        data_name: str,
        model_type: str,
        train_split: float,
    ):
        catalog_content = ["default", "one-hot"]
        lib_path = pathlib.Path(__file__).parent.resolve()
        catalog = load(os.path.join(lib_path, "data_catalog.yaml"), data_name, catalog_content)  # type: ignore

        catalog_content = "default"
        if model_type in {"mlp", "linear"}:
            catalog_content = "one-hot"

        self.catalog: Dict[str, Any] = catalog[catalog_content]

        for key in ["continuous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        # Load the raw data
        return_one_hot = False
        if model_type in {"mlp", "linear"}:
            return_one_hot = True
        dataset_obj = loadDataset(
            data_name,
            return_one_hot=return_one_hot,
            load_from_cache=True,
            debug_flag=False,
        )
        
        train_raw, test_raw, y_train, y_test = dataset_obj.getTrainTestSplit(
            preprocessing="normalize", train_split=train_split
        )
        dataset_obj = pd.concat([train_raw, test_raw], ignore_index=True)
        output_merge = pd.concat([y_train, y_test], ignore_index=True)
        dataset_obj["y"] = output_merge
        train_raw["y"] = y_train
        test_raw["y"] = y_test

        self.name = data_name
        self._df = dataset_obj
        self._df_train = train_raw
        self._df_test = test_raw

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
        return self.catalog["categorical"]

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
        return self.catalog["continuous"]

    @property
    def df(self) -> pd.DataFrame:
        """
        The full Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        return self._df.copy()

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
    def immutables(self) -> List[str]:
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        return self.catalog["target"]

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
