import pytest
import numpy as np

from data.catalog import DataCatalog
from models.catalog import ModelCatalog
from recourse_methods import Gravitational
from models.negative_instances import predict_negative_instances


def calculate_std_deviation(counterfactuals):
    return np.std(counterfactuals, axis=0)

@pytest.mark.parametrize("dataset_name", [
    ("credit")
])
def test_gravitationalon_datasets(dataset_name):
    data = DataCatalog(dataset_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    gravitational = Gravitational(mlmodel=model)

    total_factuals = predict_negative_instances(model, data)
    factuals = total_factuals.iloc[:5]

    counterfactuals = gravitational.get_counterfactuals(factuals)

    std_deviation = calculate_std_deviation(counterfactuals)

    expected_std_range = (0, 0.1)

    assert np.all(std_deviation >= expected_std_range[0]) and np.all(std_deviation <= expected_std_range[1]), \
        f"Standard deviation out of expected range: {std_deviation}"
