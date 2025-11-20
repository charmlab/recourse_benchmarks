from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from methods.catalog.rbr.library.utils_general import get_transformer
from methods.catalog.rbr.library.utils_reproduce import DataTemp, ModelCatalogTemp
from methods.catalog.rbr.model import RBR

from ...api import RecourseMethod

RANDOM_SEED = 54321

RecourseResult = namedtuple(
    "RecourseResults", ["l1_cost", "cur_valid", "fut_valid", "feasible"]
)


def lp_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=p)


def calc_future_validity(x, shifted_models):
    preds = []
    for model in shifted_models:
        pred = model.raw_model.predict(x)
        pred = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred
        pred = 1 if pred >= 0.5 else 0
        preds.append(pred)
    preds = np.array(preds)
    return np.mean(preds)


def run_single_instance(
    idx: int,
    method_object: RecourseMethod,  # we will only use RBR
    x0_numpy: np.ndarray,
    x0_df: pd.DataFrame,
    shifted_models: list,
):
    """
    Runs recourse on a single instance using the implemented RBR method
    """
    counterfactual_df = method_object.get_counterfactuals(x0_df)

    # print(f"Counterfactual: {counterfactual_df}")

    if counterfactual_df.empty:
        print(f"error for {idx}: no counterfactual found")
        return RecourseResult(np.inf, 0, 0, False)

    counterfactual_df = method_object._mlmodel.get_ordered_features(counterfactual_df)

    # print(f"Counterfactual after get feature order: {counterfactual_df}")

    x_cf_numpy = counterfactual_df.iloc[0].to_numpy()

    print(f"x0_numpy: {x0_numpy}, x_cf_numpy: {x_cf_numpy}")

    # l1 cost
    l1_cost = lp_dist(x0_numpy, x_cf_numpy, p=1)

    cf_tensor = torch.from_numpy(counterfactual_df.values.astype(np.float32))

    # current validity
    cur_valid = method_object._mlmodel.raw_model.predict(cf_tensor)
    cur_valid = 1 if cur_valid >= 0.5 else 0

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
    dataset_shifted = pd.read_csv(
        f"methods/catalog/rbr/library/{dataset_name}_modified.csv"
    )

    num_feat = ["duration", "amount", "age"]
    cat_feat = ["personal_status_sex"]
    target = "credit_risk"

    df1 = dataset.drop(
        columns=[c for c in list(dataset) if c not in num_feat + cat_feat + [target]]
    )
    temp = dataset_shifted.drop(
        columns=[
            c for c in list(dataset_shifted) if c not in num_feat + cat_feat + [target]
        ]
    )

    df1.rename(columns={"credit_risk": "y"}, inplace=True)
    temp.rename(columns={"credit_risk": "y"}, inplace=True)

    X = df1.drop(columns=["y"])
    y = df1["y"]

    X_temp = temp.drop(columns=["y"])
    y_temp = temp["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42, stratify=y
    )

    combined_df = pd.concat([df1, temp], ignore_index=True)

    transformer = get_transformer(dataset_name, combined_df.copy())

    dataset_org = DataTemp(
        df_name=dataset_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        continuous=num_feat,
        categorical=cat_feat,
        immutable=[],
        transformer=transformer,
        target="y",
    )

    X_train_temp, _, y_train_temp, _ = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=1, stratify=y_temp
    )
    future_X = pd.concat([X_train, X_train_temp], ignore_index=True)
    future_y = pd.concat([y_train, y_train_temp], ignore_index=True)
    dataset_shifted_1 = DataTemp(
        df_name=dataset_name + "_modified",
        X_train=future_X,
        X_test=X_test,
        y_train=future_y,
        y_test=y_test,
        continuous=num_feat,
        categorical=cat_feat,
        immutable=[],
        transformer=transformer,
        target="y",
    )

    X_train_temp, _, y_train_temp, _ = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=2, stratify=y_temp
    )
    future_X = pd.concat([X_train, X_train_temp], ignore_index=True)
    future_y = pd.concat([y_train, y_train_temp], ignore_index=True)
    dataset_shifted_2 = DataTemp(
        df_name=dataset_name + "_modified",
        X_train=future_X,
        X_test=X_test,
        y_train=future_y,
        y_test=y_test,
        continuous=num_feat,
        categorical=cat_feat,
        immutable=[],
        transformer=transformer,
        target="y",
    )

    X_train_temp, _, y_train_temp, _ = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=3, stratify=y_temp
    )
    future_X = pd.concat([X_train, X_train_temp], ignore_index=True)
    future_y = pd.concat([y_train, y_train_temp], ignore_index=True)
    dataset_shifted_3 = DataTemp(
        df_name=dataset_name + "_modified",
        X_train=future_X,
        X_test=X_test,
        y_train=future_y,
        y_test=y_test,
        continuous=num_feat,
        categorical=cat_feat,
        immutable=[],
        transformer=transformer,
        target="y",
    )

    X_train_temp, _, y_train_temp, _ = train_test_split(
        X_temp, y_temp, train_size=0.5, random_state=4, stratify=y_temp
    )
    future_X = pd.concat([X_train, X_train_temp], ignore_index=True)
    future_y = pd.concat([y_train, y_train_temp], ignore_index=True)
    dataset_shifted_4 = DataTemp(
        df_name=dataset_name + "_modified",
        X_train=future_X,
        X_test=X_test,
        y_train=future_y,
        y_test=y_test,
        continuous=num_feat,
        categorical=cat_feat,
        immutable=[],
        transformer=transformer,
        target="y",
    )

    # load the dataset and model
    # these are temporary classes for testing/reproducing
    # dataset_shifted_2 = DataTemp(df_name=dataset_name+'_modified', df=df3, continuous=num_feat, categorical=cat_feat, immutable=[], transformer=transformer, target='y')
    # dataset_shifted_3 = DataTemp(df_name=dataset_name+'_modified', df=df4, continuous=num_feat, categorical=cat_feat, immutable=[], transformer=transformer, target='y')
    # dataset_shifted_4 = DataTemp(df_name=dataset_name+'_modified', df=df5, continuous=num_feat, categorical=cat_feat, immutable=[], transformer=transformer, target='y')

    # df = dataset.df()
    # X_df = df.drop(columns=['y'], axis=1)
    # y_s = df['y']

    model = ModelCatalogTemp(
        data=dataset_org, model_type=model_type, backend=backend
    )  # these are temporary classes for testing/reproducing

    model_shifted_1 = ModelCatalogTemp(
        dataset_shifted_1, model_type, backend
    )  # these are temporary classes for testing/reproducing
    model_shifted_2 = ModelCatalogTemp(
        dataset_shifted_2, model_type, backend
    )  # these are temporary classes for testing/reproducing
    model_shifted_3 = ModelCatalogTemp(
        dataset_shifted_3, model_type, backend
    )  # these are temporary classes for testing/reproducing
    model_shifted_4 = ModelCatalogTemp(
        dataset_shifted_4, model_type, backend
    )  # these are temporary classes for testing/reproducing
    # model._test_accuracy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rbr = RBR(
        model,
        hyperparams={
            "device": device,
            "train_data": dataset_org.df_train.drop(columns=["y"], axis=1),
            "reproduce": True,
        },
    )

    real_x_test = dataset_org.df_test.drop(columns=["y"], axis=1)
    # print(real_x_test[:10])
    # y_test = dataset.df_test['y']

    # X_test = X_test[y_test == 0]  # only negative class
    # want a few samples that the original model classifies as negative
    preds_test = model.raw_model.predict(
        torch.from_numpy(real_x_test.values.astype(np.float32))
    )
    # print(f"Predictions on test set: {preds_test.flatten()}")
    mask = (preds_test.flatten() < 0.5).detach().cpu().numpy()
    real_x_test = real_x_test[mask]

    n = 5  # X_test.shape[0]

    factuals = real_x_test.sample(n=n, random_state=RANDOM_SEED)

    running_current_val = 0
    running_future_val = 0
    running_cost = 0

    for idx in range(len(factuals)):
        x0_df = factuals.iloc[[idx]]
        x0_numpy = x0_df.to_numpy()

        result = run_single_instance(
            idx,
            rbr,
            x0_numpy,
            x0_df,
            shifted_models=[
                model_shifted_1,
                model_shifted_2,
                model_shifted_3,
                model_shifted_4,
            ],
        )
        running_current_val += result.cur_valid
        running_future_val += result.fut_valid
        running_cost += result.l1_cost
        print(
            f"Instance {idx}: L1 cost = {result.l1_cost}, Current Validity = {result.cur_valid}, Future Validity = {result.fut_valid}, Feasible = {result.feasible}"
        )
    print(
        f"Average: L1 cost = {running_cost/n}, Current Validity = {running_current_val/n}, Future Validity = {running_future_val/n}"
    )

    assert running_current_val / n >= 0.9
    assert running_future_val / n >= 0.9


if __name__ == "__main__":
    test_rbr("german", "mlp", "pytorch")
