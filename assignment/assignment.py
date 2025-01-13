"""
This file has been provided for you to work off of for your assignment of adding the implementation of the new recourse method to the repository.
Specifically, the file should help you understand how to reference your implementation and generate counterfactuals against factual samples.
There are assertions that are specifcally designed to instruct you on whether your implementation is correct or not.

Currently, the file uses the Dice recourse method as an example to show you what to do. Replace this with your custom implementation class instead.
Follow the TO-DOs as hints on what to change and subsitute with your implementation.
If all tests pass, then you should see 'TEST PASSED' printed at the end of your console.
"""

import os
import warnings
from random import seed
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import yaml

# flake8: noqa
from data.catalog import DataCatalog
from methods import Dice  # TO-DO: Replace with implemented recourse method
from methods.processing import create_hash_dataframe
from models.catalog import ModelCatalog
from models.predict_factuals import predict_negative_instances
from tools.logging import log

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter(action="ignore", category=FutureWarning)

RANDOM_SEED = 54321
NUMBER_OF_SAMPLES = 20
BACKEND = (
    "tensorflow"  # TO-DO: Replace with backend type of the implemented recourse method
)
DATA_NAME = "adult"
METHOD_NAME = "dice"  # TO-DO: Replace with implemented recourse method
MODEL_NAME = "linear"
TRAIN_SPLIT = 0.7

seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_setup() -> Dict:
    """
    Loads experimental setup information from a YAML file and returns a dictionary
    containing the recourse methods specified in the setup.

    Parameters
    ----------
    None

    Returns
    -------
    Dict: A dictionary containing the recourse methods specified in the experimental setup.

    Raises
    -------
    FileNotFoundError: If the experimental setup file ("experimental_setup.yaml") is not found.
    yaml.YAMLError: If there is an error while parsing the YAML file.
    """
    with open("./experiments/experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    return setup_catalog["recourse_methods"]


if __name__ == "__main__":
    """
    Generates and validates Counterfactual Explanations using recourse methods from academic literature on specified datasets and model types.
    It loads experimental setup details related to the chosen recourse method.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Workflow:
    -------
    1. Load experimental setup details.
    2. Initialize necessary variables and containers for storing results.
    3. Iterate over recourse methods, datasets, and model types:
      - Log the current recourse method, dataset, and model type details.
      - Initialize the dataset, machine learning model, and recourse method.
      - Select factual instances from the dataset that are predicted as the negative class.
      - Generate counterfactual explanations for these factual instances using the recourse method.
      - Assert the functional and algorithmic correctness of the generated counterfactuals.

    Raises
    -------
    Error if there are inconsistencies in the assertion.
    """

    setup = (
        load_setup()
    )  # TO-DO: Include the hyperparamemters of the implemented recourse method in ./experimental_setup.yaml, to access from here.

    log.info("=====================================")
    log.info("Recourse method: {}".format(METHOD_NAME))
    log.info("Dataset: {}".format(DATA_NAME))
    log.info("Model type: {}".format(MODEL_NAME))

    dataset = DataCatalog(DATA_NAME, MODEL_NAME, TRAIN_SPLIT)
    hyperparameters = setup[METHOD_NAME]["hyperparams"]
    mlmodel = ModelCatalog(dataset, MODEL_NAME, BACKEND)
    recourse_method = Dice(
        mlmodel, hyperparameters
    )  # TO-DO: Replace with implemented recourse method

    factual_test_set = predict_negative_instances(mlmodel, dataset)
    factual_test_set = factual_test_set.sample(
        n=NUMBER_OF_SAMPLES, random_state=RANDOM_SEED
    )
    factuals = factual_test_set.reset_index(drop=True)

    counterfactuals = recourse_method.get_counterfactuals(factuals)
    log.info("Generated Counterfactual: {}".format(counterfactuals))

    # Hash the counterfactuals and prepare dataframe from the new implementation
    df_hashes = create_hash_dataframe(counterfactuals)

    # Load the hashes of the pre-generated counterfactuals
    pre_generated_df = pd.read_csv("counterfactual_hashes.csv")
    pre_generated_hash_df = pre_generated_df[["id", "hash"]].rename(
        columns={"hash": "pre_generated_hash"}
    )

    # Merge DataFrames for comparison
    merged_df = pd.merge(df_hashes, pre_generated_hash_df, on="id")

    # Check if all new hashes match pre-generated counterfactual hashes
    if (merged_df["hash"] == merged_df["pre_generated_hash"]).all():
        log.info("All counterfactuals match the pre-generated counterfactuals.")
        log.info("=====================================")
        log.info("TEST PASSED")
        log.info("=====================================")
    else:
        mismatches = merged_df[merged_df["hash"] != merged_df["pre_generated_hash"]]
        log.info("Some counterfactuals do not match the pre-generated counterfactuals.")
        log.info(mismatches)
        log.info("=====================================")
        log.info("TEST FAILED")
        log.info("=====================================")
