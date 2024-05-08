from typing import Any, Dict, List

from carla.data.catalog import DataCatalog
from carla.data.load_catalog import load

from .load_data import load_dataset

import pandas as pd

class OnlineCatalog(DataCatalog):
    """
    Implements DataCatalog using already implemented datasets. These datasets are loaded from an online repository.

    Parameters
    ----------
    data_name : {'adult', 'compass', 'credit'}
        Used to get the correct dataset from online repository.
    model_type : {'mlp', 'linear', 'forest'}
        The model architecture. Multi-Layer Perceptron, Logistic Regression, and Random Forest respectively.

    Returns
    -------
    DataCatalog
    """

    def __init__(
        self,
        data_name: str,
        model_type: str,
    ):
        catalog_content = ["default", "one-hot"]
        catalog = load(  # type: ignore
            "data_catalog.yaml", data_name, catalog_content
        )

        catalog_content = "default"
        if model_type in {"mlp", "linear"}:
            catalog_content = "one-hot"

        self.catalog: Dict[str, Any] = catalog[catalog_content]

        for key in ["continuous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        # Load the raw data
        from carla.recourse_methods.catalog.mace import loadDataset
        return_one_hot = False
        if model_type in {"mlp", "linear"}:
            return_one_hot = True
        dataset_obj = loadDataset(data_name, return_one_hot = return_one_hot, load_from_cache = True, debug_flag = False)
        train_raw, test_raw, y_train, y_test = dataset_obj.getTrainTestSplit(preprocessing = 'normalize')
        dataset_obj = pd.concat([train_raw, test_raw], ignore_index = True)
        output_merge = pd.concat([y_train, y_test], ignore_index = True)
        dataset_obj["y"] = output_merge

        super().__init__(
            data_name, dataset_obj, train_raw, test_raw
        )

    @property
    def categorical(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continuous(self) -> List[str]:
        return self.catalog["continuous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]
