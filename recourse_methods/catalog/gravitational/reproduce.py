import pytest
import numpy as np
from data.catalog import DataCatalog
from models.catalog import ModelCatalog
from recourse_methods import Gravitational
from models.negative_instances import predict_negative_instances
import matplotlib.pyplot as plt

def plot_counterfactual(x_original, x_cf, x_center):
    plt.figure(figsize=(12, 6))
    plt.plot(x_original, label='Original Instance', marker='o')
    plt.plot(x_cf, label='Counterfactual', marker='x')
    plt.plot(x_center, label='Centeral Point', marker='*')
    plt.legend()
    plt.title('Feature Values: Original vs Counterfactual')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.grid(True)
    plt.show()

@pytest.mark.parametrize("data_name", ["adult", "german", "compass"])
def test_gravitational_method(data_name):
    data = DataCatalog(data_name, "linear", 0.7)
    model = ModelCatalog(data, "linear", backend="pytorch")

    hyperparams = {
        "prediction_loss_lambda": 1,
        "original_dist_lambda": 0.5,
        "grav_penalty_lambda": 1.5,
        "learning_rate": 0.01,
        "num_steps": 500,
        "target_class": 1,
        "scheduler_step_size": 100,
        "scheduler_gamma": 0.5
    }

    gravitational_method = Gravitational(mlmodel=model, hyperparams=hyperparams)

    total_factuals = predict_negative_instances(model, data)
    factuals = total_factuals.iloc[:5]

    for index, row in factuals.iterrows():
        expected_negative_prediction = 0
        prediction = model.predict_proba(row.to_frame().T)
        actual_negative_prediction = np.argmax(prediction)
        print("pred", data_name)
        # plot_counterfactual(((row.to_frame().T).values)[0])
        assert expected_negative_prediction == actual_negative_prediction

    counterfactuals = gravitational_method.get_counterfactuals(factuals)

    for index, row in counterfactuals.iterrows():
        expected_positive_prediction = 1
        prediction = model.predict_proba(row.to_frame().T)
        actual_positive_prediction = np.argmax(prediction)
        assert expected_positive_prediction == actual_positive_prediction


if __name__ == "__main__":
    pytest.main()
