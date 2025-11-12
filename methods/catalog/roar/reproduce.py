import numpy as np
import pytest

from data.catalog.online_catalog import DataCatalog
from methods.catalog.roar.model import Roar
from models.catalog.catalog import ModelCatalog

RANDOM_SEED = 54321


# Find indices where recourse is needed
def recourse_needed(predict_fn, X, target=1):
    return np.where(predict_fn(X) == 1 - target)[0]


# Recourse validity
def recourse_validity(
    predict_fn, rs, target=0.5
):  # think it needs to 0.5 cause predict gives prob
    return sum(predict_fn(rs) > target) / len(rs)


# Functions to compute the cost of recourses
def l1_cost(xs, rs):
    # ensure numpy arrays
    xs_arr = xs.to_numpy()
    rs_arr = rs.to_numpy()

    print(xs_arr)
    print("-----------------------------------")
    print(rs_arr)

    # compute row-wise L1 norms
    cost = np.linalg.norm(rs_arr - xs_arr, ord=1, axis=1)

    return np.mean(cost)


# because running the experiment just the same as the paper would require breaking tweaks to the implemented ROAR
# the reproduce will be a partial one, showing us the validation results just on the first models (M1)s of the paper


# TODO make sure to update mlmodel_catalog.yaml
# TODO make sure to update reproduce to match above statement, get permission for this as well.

# This will only test with the german dataset as it is the easiest to run
@pytest.mark.parametrize(
    "dataset_name, model_type, backend",
    [
        ("german", "linear", "sklearn"),
        ("german", "mlp", "pytorch"),
    ],
)
def test_roar(dataset_name, model_type, backend):

    # results = {}

    args = {
        "cost": "l1",
        "lamb": 0.1,
    }

    print("Training %s models" % model_type)
    if model_type == "linear":
        data = DataCatalog(dataset_name, model_type="linear", train_split=0.8)
        m1 = ModelCatalog(data, model_type="linear", backend=backend)  # m1 = LR()
        data2 = DataCatalog(
            dataset_name, model_type="linear", train_split=0.8, modified=True
        )
        m2 = ModelCatalog(data2, model_type="linear", backend=backend)
    if model_type == "mlp":
        data = DataCatalog(dataset_name, model_type="mlp", train_split=0.8)
        m1 = ModelCatalog(
            data, model_type="mlp", backend=backend
        )  # m1 = NN(X1_train.shape[1])
        data2 = DataCatalog(
            dataset_name, model_type="mlp", train_split=0.8, modified=True
        )
        m2 = ModelCatalog(data2, model_type="mlp", backend=backend)
        # m2 = NN(X1_train.shape[1])

    # print(data._df_train)
    m1._test_accuracy()  # TODO best practice to remove, as they are private
    m2._test_accuracy()

    print("Using %s cost" % args["cost"])
    # if args["cost"] == "l1":
    #     feature_costs = None

    coefficients = intercept = None

    roar = Roar(mlmodel=m1, hyperparams={}, coeffs=coefficients, intercepts=intercept)

    # lamb = args["lamb"]

    recourses = []
    # deltas = []

    factuals = (data._df_test).sample(n=10, random_state=RANDOM_SEED)

    factuals = factuals.drop("y", axis=1)

    r = roar.get_counterfactuals(factuals=factuals)
    recourses.append(r)

    # results_i["recourses"] = recourses

    recourses = np.array(recourses)
    print(factuals.columns)
    print(r.columns)

    # For model 1, ran on og dataset
    m1_validity = recourse_validity(m1.predict, r)
    # results_i["m1_validity"] = m1_validity
    print("M1 validity: %f" % m1_validity)

    # For model 2, ran on modified dataset
    m2_validity = recourse_validity(m2.predict, r)
    # results_i["m2_validity"] = m2_validity
    print("M2 validity: %f" % m2_validity)

    if args["cost"] == "l1":
        cost = l1_cost(factuals, r)

    # results_i["cost"] = cost
    print("%s cost: %f" % (args["cost"], cost))

    if model_type == "linear":
        assert m1_validity >= 0.99
        assert m2_validity >= 0.92
        # assert cost >= 2.80 and cost <= 4
    elif model_type == "mlp":
        assert m1_validity >= 0.79
        assert m2_validity >= 0.62
        # assert cost >= 1.60 and cost <= 2.10


if __name__ == "__main__":
    test_roar("german", "linear", "sklearn")
    test_roar("german", "mlp", "pytorch")
