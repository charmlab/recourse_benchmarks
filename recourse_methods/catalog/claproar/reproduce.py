import pytest
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from data.catalog import DataCatalog
from models.catalog import ModelCatalog
from recourse_methods import ClaPROAR
from models.negative_instances import predict_negative_instances


@pytest.mark.parametrize("dataset_name", [
    ("credit")
])
def test_claproar_counterfactuals_standard_deviation(dataset_name):
    data = DataCatalog(dataset_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    claproar = ClaPROAR(mlmodel=model)

    total_factuals = predict_negative_instances(model, data)

    factuals = total_factuals.iloc[:5]

    counterfactuals = claproar.get_counterfactuals(factuals)

    factuals_np = factuals.drop("y", axis=1).to_numpy()
    counterfactuals_np = counterfactuals.to_numpy()

    differences = np.abs(counterfactuals_np - factuals_np)

    std_deviation = np.std(differences, axis=0)

    expected_std_deviation = np.array([0.03] * len(std_deviation))

    tolerance = 0.2

    assert np.allclose(std_deviation, expected_std_deviation, atol=tolerance), \
        "Standard deviation mismatch."

@pytest.mark.parametrize("dataset_name", [
    ("credit")
])
def test_claproar_distribution_shift(dataset_name):
    data = DataCatalog(dataset_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    claproar = ClaPROAR(mlmodel=model)

    total_factuals = predict_negative_instances(model, data)

    factuals = total_factuals.iloc[:5]

    counterfactuals = claproar.get_counterfactuals(factuals)

    negative_instances = predict_negative_instances(model, data).iloc[:5]

    original_np = negative_instances.drop("y", axis=1).to_numpy()
    counterfactual_np = counterfactuals.to_numpy()
    mmd_value = compute_mmd(original_np, counterfactual_np)

    expected_mmd_value = 0.03

    tolerance = 0.03

    assert abs(mmd_value - expected_mmd_value) <= tolerance or abs(expected_mmd_value - mmd_value) <= tolerance, \
        f"MMD value mismatch."

@pytest.mark.parametrize("dataset_name", [
    ("credit")
])
def test_claproar_individual_cost(dataset_name):
    data = DataCatalog(dataset_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    claproar = ClaPROAR(mlmodel=model)

    total_factuals = predict_negative_instances(model, data)

    factuals = total_factuals.iloc[:5]

    counterfactuals = claproar.get_counterfactuals(factuals)

    negative_instances = predict_negative_instances(model, data).iloc[:5]

    factuals_np = negative_instances.drop("y", axis=1).to_numpy()
    counterfactuals_np = counterfactuals.to_numpy()
    individual_cost = np.linalg.norm(counterfactuals_np - factuals_np, axis=1).mean()

    expected_cost = 0.5

    assert individual_cost <= expected_cost, \
        f"Individual cost too high."

def compute_mmd(X, Y, gamma=0.05):
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
        return np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)