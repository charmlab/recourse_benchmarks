"""
This file has been provided for you to work off of for your assignment of adding the implementation of the new recourse method to the repository.
Specifically, the file should help you understand how to reference your implementation and generate counterfactuals against factual samples.
There are assertions that are specifcally designed to instruct you on whether your implementation is correct or not.

Currently, the file uses the Dice recourse method as an example to show you what to do. Replace this with your custom implementation class instead.
Follow the TO-DOs as hints on what to change and subsitute with your implementation.
If all tests pass, then you should see 'TEST PASSED' printed at the end of your console.
"""

# flake8: noqa
from data.catalog import DataCatalog
from logging_carla import log
from models.catalog import ModelCatalog
from models.negative_instances import predict_negative_instances
from random import seed
from recourse_methods import Dice # TO-DO: Replace with implemented recourse method
from recourse_methods.processing import check_counterfactuals
from typing import Dict, Tuple, Union

import numpy as np
import os
import pandas as pd
import yaml
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.simplefilter(action="ignore", category=FutureWarning)

RANDOM_SEED = 54321
NUMBER_OF_SAMPLES = 20
BACKEND = "tensorflow" # TO-DO: Replace with backend type of the implemented recourse method
DATA_NAME = "adult" 
METHOD_NAME = "dice" # TO-DO: Replace with implemented recourse method
MODEL_NAME = "linear"
TRAIN_SPLIT = 0.7

seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
  with open("./experimental_setup.yaml", "r") as f:
      setup_catalog = yaml.safe_load(f)

  return setup_catalog["recourse_methods"]

def assert_counterfactuals(
    counterfactuals: pd.DataFrame, factuals: pd.DataFrame
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
  """
  Remove instances for which a counterfactual could not be found,
  and returns an error if such instances exist.

  Parameters
  ----------
  counterfactuals (pd.DataFrame): Has to be the same shape as factuals.
  factuals (pd.DataFrame): Has to be the same shape as counterfactuals. (optional)

  Returns
  -------
  pd.DataFrame: An instance of a pd.DataFrame without any nan/null instances.
  
  Raises
  -------
  ValueError: If any nan/null instances are found in the counterfactual DataFrame, and it no longer matches the number of factuals.
  """
  # get indices of unsuccessful counterfactuals
  nan_idx = counterfactuals.index[counterfactuals.isnull().any(axis=1)]
  output_counterfactuals = counterfactuals.copy()
  output_counterfactuals = output_counterfactuals.drop(index=nan_idx)

  if factuals.shape[0] != counterfactuals.shape[0]:
    raise ValueError(
    "Counterfactuals and factuals should contain the same amount of samples"
  )

  return output_counterfactuals


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
    - Assert the functional correctness of the generated counterfactuals:
      - Verify that counterfactuals cause a change in the target class.
      - Replace any counterfactuals that do not cause a class change with NaNs using the `check_counterfactuals` function.
      - Ensure the number of valid counterfactuals matches the number of factuals; otherwise, an error is raised.
    - Store the valid counterfactuals in a DataFrame.
  4. Print the valid counterfactuals without NaN/Null values.

  Raises
  -------
  Error if there are inconsistencies in the assertion.
  """

  setup = load_setup()  # TO-DO: Include the hyperparamemters of the implemented recourse method in ./experimental_setup.yaml, to access from here.

  log.info("=====================================")
  log.info("Recourse method: {}".format(METHOD_NAME))
  log.info("Dataset: {}".format(DATA_NAME))
  log.info("Model type: {}".format(MODEL_NAME))

  dataset = DataCatalog(DATA_NAME, MODEL_NAME, TRAIN_SPLIT)
  hyperparameters = setup[METHOD_NAME]["hyperparams"]
  mlmodel = ModelCatalog(dataset, MODEL_NAME, BACKEND)
  recourse_method = Dice(mlmodel, hyperparameters)

  factual_test_set = predict_negative_instances(mlmodel, dataset)
  factual_test_set = factual_test_set.sample(
      n=NUMBER_OF_SAMPLES, random_state=RANDOM_SEED
  )
  factuals = factual_test_set.reset_index(drop=True)

  counterfactuals = recourse_method.get_counterfactuals(factuals)
  log.info("Generated Counterfactual: {}".format(counterfactuals))

  # Functional Correctness Assertion
  # Check if generated counterfactuals cause a change in the target class
  # Counterfactuals that do not cause a class change are replaced with NaNs in the check_counterfactuals function
  # After passing through check_counterfactuals, the number of valid counterfactuals must match the factuals; otherwise, an error is raised.
  df_cfs = check_counterfactuals(mlmodel, counterfactuals, factuals.index)
  counterfactuals_without_nans = assert_counterfactuals(
    df_cfs, factuals
  )
  log.info("Counterfactuals without Nan/Null Values: {}".format(counterfactuals_without_nans))
  log.info("=====================================")
  log.info("TEST PASSED")
