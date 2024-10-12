import pytest
import numpy as np

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

    tolerance = 0.01

    assert np.allclose(std_deviation, expected_std_deviation, atol=tolerance), \
        "Standard deviation mismatch."
