from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import torch

from data.catalog.online_catalog import DataCatalog
from methods.catalog.rbr.model import RBR
from models.catalog.catalog import ModelCatalog
from ...api import RecourseMethod

RANDOM_SEED = 54321

RecourseResult = namedtuple("RecourseResults", ["l1_cost", "cur_valid", "fut_valid", "feasible"])

def lp_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=p)

def calc_future_validity(x, shifted_models):
    preds = []
    for model in shifted_models:
        pred = model.predict(x)
        preds.append(pred)
    preds = np.array(preds)
    return np.mean(preds)

def run_single_instance(
        idx: int,
        method_object: RecourseMethod, # we will only use RBR
        x0_numpy: np.ndarray,
        x0_df: pd.DataFrame,
        shifted_models: list,
):
    """
    Runs recourse on a single instance using the implemented RBR method
    """
    counterfactual_df = method_object.get_counterfactuals(x0_df)

    if counterfactual_df.empty:
        print(f"error for {idx}: no counterfactual found")
        return RecourseResult(np.inf, 0, 0, False)
    
    counterfactual_df = method_object._mlmodel.get_ordered_features(counterfactual_df)

    x_cf_numpy = counterfactual_df.iloc[0].to_numpy()

    # l1 cost
    l1_cost = lp_dist(x0_numpy, x_cf_numpy, p=1)

    # current validity
    cur_valid = method_object._mlmodel.predict(counterfactual_df)[0]

    # future validity
    fut_valid = calc_future_validity(x_cf_numpy, shifted_models)

    return RecourseResult(l1_cost, float(cur_valid), fut_valid, True)


@pytest.mark.parametrize(
    "dataset_name, model_type, backend",
    [
        ("german", "mlp", "pytorch"),
    ],
)
def test_rbr(dataset_name, model_type, backend):

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # load the dataset and model
    dataset = DataCatalog('german', "mlp", 0.8)

    # df = dataset.df()
    # X_df = df.drop(columns=['y'], axis=1)
    # y_s = df['y']

    model = ModelCatalog(dataset, "mlp", "tensorflow")

    rbr = RBR(model, hyperparams={})

    X_test = dataset._df_test.drop(columns=['y'], axis=1)
    y_test = dataset._df_test['y']

    X_test = X_test[y_test == 0]  # only negative class

    factuals = X_test.sample(n=10, random_state=RANDOM_SEED)

    for idx in range(len(factuals)):
        x0_df = factuals.iloc[[idx]]
        x0_numpy = x0_df.to_numpy()

        result = run_single_instance(
            idx,
            rbr,
            x0_numpy,
            x0_df,
            shifted_models=[],
        )

        print(f"Instance {idx}: L1 cost = {result.l1_cost}, Current Validity = {result.cur_valid}, Future Validity = {result.fut_valid}, Feasible = {result.feasible}")


if __name__ == "__main__":
    test_rbr("german", "mlp", "pytorch")