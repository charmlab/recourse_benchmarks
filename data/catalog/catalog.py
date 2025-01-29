import os
import pathlib
import re
from abc import ABC
from typing import Any, Dict, List

import pandas as pd

from data.api import Data
from data.load_data import loadDataset
from data.utils.load_catalog import load


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
        catalog = load(os.path.join(lib_path, "catalog.yaml"), data_name, catalog_content)  # type: ignore

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
    def X_train(self) -> List[str]:
        """
        Provides the X_train of data.

        Returns
        -------
        pd.DataFrame
        """
        return self._df_train.copy().drop(columns=["y"])

    @property
    def X_test(self) -> List[str]:
        """
        Provides the X_test of data.

        Returns
        -------
        pd.DataFrame
        """
        return self._df_test.copy().drop(columns=["y"])

    @property
    def y_train(self) -> List[str]:
        """
        Provides the y_train of data.

        Returns
        -------
        pd.DataFrame
        """
        return self._df_train.copy()["y"]

    @property
    def y_test(self) -> List[str]:
        """
        Provides the y_test of data.

        Returns
        -------
        pd.DataFrame
        """
        return self._df_test.copy()["y"]

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
        all_features = self.catalog.get("continuous", []) + self.catalog.get(
            "categorical", []
        )

        # Remove duplicates and sort using custom logic
        unique_features = list(
            dict.fromkeys(all_features)
        )  # Preserve original order for deduplication
        sorted_features = sorted(unique_features, key=self.custom_sort_key)

        return sorted_features

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

    def custom_sort_key(self, feature):
        """
        Custom sorting function for features.
        Sorts primarily by 'xn' (numerical value of n) and secondarily by 'm' if present.

        Args:
            feature (str): Feature name.

        Returns:
            tuple: Sorting key.
        """
        # Match patterns like 'xn' or 'xn_cat_m'
        match = re.match(r"x(\\d+)(?:_\\w+)?(?:_(\\d+))?", feature)
        if match:
            n = int(match.group(1))  # Extract 'n'
            m = (
                int(match.group(2)) if match.group(2) else -1
            )  # Extract 'm' or use -1 if not present
            return (n, m)
        return (
            float("inf"),
            float("inf"),
        )  # Default for features not matching the pattern
