import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

from data.catalog import DataCatalog
from methods import ClaPROAR
from models.catalog import ModelCatalog
from models.negative_instances import predict_negative_instances

"""
This test is designed to replicate the standard deviation results described in the research paper.
By comparing the calculated standard deviation with the expected range,
the test ensures that the generated counterfactuals are consistent with paper's findings.

Implemented from:
"Endogenous Macrodynamics in Algorithmic Recourse"
Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. Liem
"""


@pytest.mark.parametrize("dataset_name", [("credit")])
def test_claproar_counterfactuals_standard_deviation(dataset_name):
    data = DataCatalog(dataset_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    claproar = ClaPROAR(mlmodel=model)

    total_factuals = predict_negative_instances(model, data)

    factuals = total_factuals.iloc[:5]

    counterfactuals = claproar.get_counterfactuals(factuals, raw_output=True)

    factuals_np = claproar._mlmodel.get_ordered_features(factuals).to_numpy()
    counterfactuals_np = counterfactuals.to_numpy()

    differences = np.abs(counterfactuals_np - factuals_np)

    std_deviation = np.std(differences, axis=0)

    expected_std_deviation = np.array([0.03] * len(std_deviation))

    tolerance = 0.2

    assert np.allclose(
        std_deviation, expected_std_deviation, atol=tolerance
    ), "Standard deviation mismatch."


"""
This test focuses on measuring distribution shifts between the original data and
the counterfactual data using the Maximum Mean Discrepancy (MMD) metric as described in the research paper.
By comparing the calculated  Maximum Mean Discrepancy (MMD) with the expected range,
the test ensures that the generated counterfactuals are consistent with paper's findings.

Implemented from:
"Endogenous Macrodynamics in Algorithmic Recourse"
Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. Liem
"""

# flaky test

# @pytest.mark.parametrize("dataset_name", [("credit")])
# def test_claproar_distribution_shift(dataset_name):
#     data = DataCatalog(dataset_name, "linear", 0.7)
#     model = ModelCatalog(data, "linear", backend="pytorch")

#     claproar = ClaPROAR(mlmodel=model)

#     total_factuals = predict_negative_instances(model, data)

#     factuals = total_factuals.iloc[:5]

#     counterfactuals = claproar.get_counterfactuals(factuals)

#     negative_instances = predict_negative_instances(model, data).iloc[:5]

#     original_np = negative_instances.drop("y", axis=1).to_numpy()
#     counterfactual_np = counterfactuals.to_numpy()
#     mmd_value = compute_mmd(original_np, counterfactual_np)

#     expected_mmd_value = 0.03

#     tolerance = 0.03

#     assert abs(mmd_value - expected_mmd_value) <= tolerance, "MMD value mismatch."


"""
This test measures the individual cost of applying recourse,
specifically focusing on the Euclidean distance between the original factual instances
and the counterfactuals.
By comparing the calculated indivdual cost with the expected range,
the test ensures that the generated counterfactuals are consistent with paper's findings.

Implemented from:
"Endogenous Macrodynamics in Algorithmic Recourse"
Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. Liem
"""


@pytest.mark.parametrize("dataset_name", [("credit")])
def test_claproar_individual_cost(dataset_name):
    data = DataCatalog(dataset_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    claproar = ClaPROAR(mlmodel=model)

    total_factuals = predict_negative_instances(model, data)

    factuals = total_factuals.iloc[:5]

    counterfactuals = claproar.get_counterfactuals(factuals, raw_output=True)

    negative_instances = predict_negative_instances(model, data).iloc[:5]

    factuals_np = claproar._mlmodel.get_ordered_features(negative_instances).to_numpy()
    counterfactuals_np = counterfactuals.to_numpy()
    individual_cost = np.linalg.norm(counterfactuals_np - factuals_np, axis=1).mean()

    expected_cost = 0.5

    assert individual_cost <= expected_cost, "Individual cost too high."


def compute_mmd(X, Y, gamma=0.05):
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    return np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
