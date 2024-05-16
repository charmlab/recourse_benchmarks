import os
from typing import List

import yaml

def load(filename: str, dataset: str, keys: List[str]):
    """
    Loads a YAML file containing a catalog of datasets and their associated metadata.

    Parameters
    ----------
    filename (str): A string specifying the name of the YAML file to load.
    dataset (str): A string specifying the name of the dataset to retrieve from the catalog.
    keys (List[str]): A list of strings specifying the keys that should be present in the dataset's metadata.

    Returns
    -------
    dict: A dictionary containing the metadata associated with the specified dataset.

    Raises
    -------
    KeyError: If the specified dataset is not found in the catalog or if any of the specified keys are not present in the dataset's metadata.
    """
    print(filename)
    with open(filename, "r") as f:
        catalog = yaml.safe_load(f)

    if dataset not in catalog:
        raise KeyError("Dataset not in catalog.")

    for key in keys:
        if key not in catalog[dataset].keys():
            raise KeyError("Important key {} is not in Catalog".format(key))
        if catalog[dataset][key] is None:
            catalog[dataset][key] = []

    return catalog[dataset]
