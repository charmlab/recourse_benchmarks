from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import torch

from methods.catalog.rbr.library.utils_reproduce import DataTemp, ModelCatalogTemp
from methods.catalog.rbr.model import RBR
from ...api import RecourseMethod

RANDOM_SEED = 54321

RecourseResult = namedtuple("RecourseResults", ["l1_cost", "cur_valid", "fut_valid", "feasible"])

def lp_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=p)


def calc_future_validity(x, shifted_models):
    preds = []
    for model in shifted_models:
        pred = model.raw_model.predict(x)
        pred = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred
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

    print(f"Counterfactual: {counterfactual_df}")

    if counterfactual_df.empty:
        print(f"error for {idx}: no counterfactual found")
        return RecourseResult(np.inf, 0, 0, False)
    
    counterfactual_df = method_object._mlmodel.get_ordered_features(counterfactual_df)

    print(f"Counterfactual after get feature order: {counterfactual_df}")

    x_cf_numpy = counterfactual_df.iloc[0].to_numpy()

    print(f"x0_numpy: {x0_numpy}, x_cf_numpy: {x_cf_numpy}")

    # l1 cost
    l1_cost = lp_dist(x0_numpy, x_cf_numpy, p=1)

    cf_tensor  = torch.from_numpy(counterfactual_df.values.astype(np.float32))

    # current validity
    cur_valid = method_object._mlmodel.raw_model.predict(cf_tensor)

    # future validity
    # fut_valid = calc_future_validity(x_cf_numpy, shifted_models)
    fut_valid = calc_future_validity(cf_tensor, shifted_models)

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

    # load the csv as a pandas DataFrame
    dataset = pd.read_csv(f"methods/catalog/rbr/library/{dataset_name}.csv")
    dataset_shifted = pd.read_csv(f"methods/catalog/rbr/library/{dataset_name}_modified.csv")
    
    num_feat = ["duration", "amount", "age"]
    cat_feat = ["personal_status_sex"]
    target = "credit_risk"
    
    df1 = dataset.drop(columns=[c for c in list(dataset) if c not in num_feat+cat_feat+[target]])
    df2 = dataset_shifted.drop(columns=[c for c in list(dataset_shifted) if c not in num_feat+cat_feat+[target]])

    df1.rename(columns={'credit_risk': 'y'}, inplace=True)
    df2.rename(columns={'credit_risk': 'y'}, inplace=True)

    # load the dataset and model
    dataset = DataTemp(df_name=dataset_name, df=df1, continuous=num_feat, categorical=cat_feat, immutable=[], target='y') # these are temporary classes for testing/reproducing
    dataset_shifted = DataTemp(df_name=dataset_name+'_modified', df=df2, continuous=num_feat, categorical=cat_feat, immutable=[], target='y')

    # df = dataset.df()
    # X_df = df.drop(columns=['y'], axis=1)
    # y_s = df['y']

    model = ModelCatalogTemp(dataset, model_type, backend) # these are temporary classes for testing/reproducing
    model_shifted = ModelCatalogTemp(dataset_shifted, model_type, backend) # these are temporary classes for testing/reproducing
    # model._test_accuracy()

    rbr = RBR(model, hyperparams={'train_data': dataset.df_train.drop(columns=['y'], axis=1), 'reproduce': True})

    X_test = dataset.df_test.drop(columns=['y'], axis=1)
    y_test = dataset.df_test['y']

    # X_test = X_test[y_test == 0]  # only negative class
    # want a few samples that the original model classifies as negative
    preds_test = model.raw_model.predict(torch.from_numpy(X_test.values.astype(np.float32)))
    # print(f"Predictions on test set: {preds_test.flatten()}")
    mask = (preds_test.flatten() < 0.5).detach().cpu().numpy()
    X_test = X_test[mask]

    factuals = X_test.sample(n=5, random_state=RANDOM_SEED)

    for idx in range(len(factuals)):
        x0_df = factuals.iloc[[idx]]
        x0_numpy = x0_df.to_numpy()

        result = run_single_instance(
            idx,
            rbr,
            x0_numpy,
            x0_df,
            shifted_models=[model_shifted],
        )

        print(f"Instance {idx}: L1 cost = {result.l1_cost}, Current Validity = {result.cur_valid}, Future Validity = {result.fut_valid}, Feasible = {result.feasible}")


if __name__ == "__main__":
    test_rbr("german", "mlp", "pytorch")
