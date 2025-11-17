# in order to accurately reproduce the work in the paper
# i will run the experiments similar to them, but I
# will not make use of the carla model method directly, but rather run
# a modified version of the run_method funtion found in larr.py
from copy import deepcopy
from typing import Callable

import numpy as np
import pytest
import tqdm

from data.catalog.online_catalog import DataCatalog
from methods.catalog.larr.library.larr import LARRecourse, RecourseCost
from methods.catalog.larr.model import Larr
from methods.catalog.larr.utils_reproduce import NN, GermanDataset
from models.catalog.catalog import ModelCatalog

RANDOM_SEED = 54321


def test_run_experiment():
    dataset = GermanDataset()
    beta = 0.5

    running_robustness = 0
    running_consistency = 0
    counter = 0

    for seed in range(5):
        (train_data, test_data) = dataset.get_data(seed)
        X_train, y_train = train_data
        X_test, _ = test_data

        base_model = NN(X_train.shape[1])
        base_model.train(X_train.values, y_train.values)

        def recourse_needed(predict_fn: Callable, X: np.ndarray, y_target: float = 1):
            indices = np.where(predict_fn(X) == 1 - y_target)
            return X[indices]

        recourse_needed_X_train = recourse_needed(base_model.predict, X_train.values)
        recourse_needed_X_test = recourse_needed(base_model.predict, X_test.values)

        weights, bias = None, None
        lar_recourse = LARRecourse(weights, bias, alpha=0.5)

        lar_recourse.choose_lambda(
            recourse_needed_X_train, base_model.predict, X_train.values
        )

        n = 5

        for i in tqdm.trange(
            n,
            desc=f"Evaluating recourse | alpha={lar_recourse.alpha}; lambda={lar_recourse.lamb}",
            colour="#0091ff",
        ):
            x_0 = recourse_needed_X_test[i]

            J = RecourseCost(x_0, lar_recourse.lamb)

            np.random.seed(i)
            weights_0, bias_0 = lar_recourse.lime_explanation(
                base_model.predict, X_train, x_0
            )
            weights_0, bias_0 = np.round(weights_0, 4), np.round(bias_0, 4)
            # theta_0 = np.hstack((weights_0, bias_0))

            lar_recourse.weights = weights_0
            lar_recourse.bias = bias_0

            x_r = lar_recourse.get_recourse(x_0, beta=1.0)
            weights_r, bias_r = lar_recourse.calc_theta_adv(x_r)
            # theta_r = np.hstack((weights_r, bias_r))
            J_r_opt = J.eval(x_r, weights_r, bias_r)

            weights_p = deepcopy(weights_r)
            bias_p = deepcopy(bias_r)
            theta_p = (weights_p, bias_p)

            x_c = lar_recourse.get_recourse(x_0, beta=0.0, theta_p=theta_p)
            J_c_opt = J.eval(x_c, *theta_p)

            x = lar_recourse.get_recourse(x_0, beta=beta, theta_p=theta_p)
            weights_r, bias_r = lar_recourse.calc_theta_adv(x)
            # theta_r = np.hstack((weights_r, bias_r))

            J_r = J.eval(x, weights_r, bias_r)
            J_c = J.eval(x, weights_p, bias_p)
            robustness = J_r - J_r_opt
            consistency = J_c - J_c_opt

            running_robustness += robustness
            running_consistency += consistency
            counter += 1

            # print(f"alpha {lar_recourse.alpha}, lamb {lar_recourse.lamb}, i {i}, x_0 {x_0}, theta_0 {theta_0}, beta {beta}, x {x}, robustness[0] {robustness[0]}, consistency[0] {consistency[0]}")

    avge_robustness = running_robustness / counter
    avge_consistency = running_consistency / counter
    print(
        f"Avge Robustness: {avge_robustness} and Avge Consistency: {avge_consistency}"
    )  # we should have around 25

    assert avge_robustness > 0.27 and avge_robustness < 0.29
    assert avge_consistency > 0.40 and avge_consistency < 0.41


@pytest.mark.parametrize(
    "dataset_name, model_type, backend",
    [
        ("german", "mlp", "pytorch"),
    ],
)
def test_with_carla(dataset_name, model_type, backend):

    dataset = DataCatalog(dataset_name, model_type, 0.8)

    # load artificial neural network from catalog
    model = ModelCatalog(dataset, model_type, backend)

    # get factuals from the data to generate counterfactual examples
    factuals = (dataset._df_train).sample(n=5, random_state=RANDOM_SEED)

    # load a recourse model and pass black box model
    larr = Larr(model, hyperparams={"beta": 0.5})

    # generate counterfactual examples
    counterfactuals = larr.get_counterfactuals(factuals)
    print(counterfactuals)


if __name__ == "__main__":
    test_run_experiment()
    test_with_carla("german", "mlp", "pytorch")
