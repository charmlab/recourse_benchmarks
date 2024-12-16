import numpy as np
import pytest

from data.catalog import DataCatalog
from methods import Gravitational
from models.catalog import ModelCatalog
from models.negative_instances import predict_negative_instances

"""
The test is designed to replicate the standard deviation results described in the research paper.
By comparing the calculated standard deviation with the expected range,
the test ensures that the generated counterfactuals are consistent with paper's findings.

Implemented from:
"Endogenous Macrodynamics in Algorithmic Recourse"
Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. Liem
"""


def calculate_std_deviation(counterfactuals):
    return np.std(counterfactuals, axis=0)


@pytest.mark.parametrize("dataset_name", [("credit")])
def test_gravitationalon_datasets(dataset_name):
    data = DataCatalog(dataset_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    gravitational = Gravitational(mlmodel=model)

    total_factuals = predict_negative_instances(model, data)
    factuals = total_factuals.iloc[:5]

    counterfactuals = gravitational.get_counterfactuals(factuals)

    std_deviation = calculate_std_deviation(counterfactuals)

    expected_std_range = (0, 0.1)

    assert np.all(std_deviation >= expected_std_range[0]) and np.all(
        std_deviation <= expected_std_range[1]
    ), f"Standard deviation out of expected range: {std_deviation}"
