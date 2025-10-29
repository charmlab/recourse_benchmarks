import os
import pytest
import numpy as np
import pandas as pd
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from data.catalog import DataCatalog
from models.catalog import ModelCatalog
from models.negative_instances import predict_negative_instances
from methods.catalog.probe import Probe

"""
This file aims to reproduce the results obtained by Pawelczyk et.al. as reported in the main paper.
This file structurely tries to follow the format in the original codebase.
"""


training_params_linear = {
    "adult": {"lr": 0.002,
              "epochs": 100,
              "batch_size": 1024},
    "compass": {"lr": 0.002,
               "epochs": 25,
               "batch_size": 128},
    "credit": {"lr": 0.002,
                "epochs": 50,
                "batch_size": 2048},
}
training_params_mlp = {
    "adult": {"lr": 0.002,
              "epochs": 30,
              "batch_size": 1024},
    "compass": {"lr": 0.002,
               "epochs": 25,
               "batch_size": 25},
    "credit": {"lr": 0.002,
                "epochs": 50,
                "batch_size": 2048},
}

training_params = {"linear": training_params_linear,
                   "mlp": training_params_mlp}

def expect(model, test_factual, sigma2, invalidation_target):
    hyperparams = {"loss_type": "BCE",
                   "binary_cat_features": False,
                   "invalidation_target": invalidation_target,
                   "inval_target_eps": 0.010,
                   "noise_variance": sigma2,
                   "n_iter": 200,
                   "t_max_min": 0.50}
    df_cfs = Probe(model, hyperparams).get_counterfactuals(test_factual)
    return df_cfs

def perturb_sample(x, n_samples, sigma2):
    
    # stack copies of this sample, i.e. n rows of x.
    X = x.repeat(n_samples, 1)
    
    # sample normal distributed values
    Sigma = torch.eye(x.shape[1]) * sigma2
    eps = MultivariateNormal(
        loc=torch.zeros(x.shape[1]), covariance_matrix=Sigma
    ).sample((n_samples,))

    return X + eps, Sigma


def run_experiment(cf_method,
                   hidden_width,
                   data_name,
                   model_type,
                   backend,
                   sigma2,
                   invalidation_target,
                   n_cfs=100,
                   n_samples=10000,
                   ):
    
    print(
        f"Running experiments with: {cf_method} {data_name} {model_type} {hidden_width}"
    )

    data = DataCatalog(data_name=data_name, model_type=model_type, train_split=0.8)

    # params = training_params[model_type][data_name]
    model = ModelCatalog(data=data, 
                         model_type=model_type, 
                         backend=backend,
                         batch_size=training_params[model_type][data_name]["batch_size"],
                         epochs=training_params[model_type][data_name]["epochs"],
                         learning_rate=training_params[model_type][data_name]["lr"],)
    # model.train(
    #     learning_rate=params["lr"],
    #     epochs=params["epochs"],
    #     batch_size=params["batch_size"],
    #     hidden_size=hidden_width,
    # )
    # model.use_pipeline = False

    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:n_cfs]

    print(test_factual)

    # print the list of factuals
    # for test_factual in test_factual.itertuples():
    #     print(test_factual)
    
    if cf_method == 'probe':
        df_cfs = expect(model,
                        test_factual,
                        sigma2=sigma2,
                        invalidation_target=invalidation_target)
    else:
        raise ValueError(f"cf_method {cf_method} not recognized")

    # print(df_cfs)
    # df_cfs = df_cfs.drop(columns=data.target)

    result = []
    cf_predictions = []
    for i, x in df_cfs.iterrows():
        x = torch.Tensor(x).unsqueeze(0)
        X_pert, _ = perturb_sample(x, n_samples, sigma2=sigma2)
        if backend == "pytorch":
            prediction = (model.predict(x).squeeze() > 0.5).int()
            cf_predictions.append(prediction.item())
            delta_M = torch.mean(
                (1 - (model.predict(X_pert).squeeze() > 0.5).int()).float()
            ).item()
        else:
            prediction = (model.predict(x).squeeze() > 0.5).astype(int)
            cf_predictions.append(prediction)
            delta_M = np.mean(
                1 - (model.predict(X_pert).squeeze() > 0.5).astype(int)
            )

        result.append(delta_M)
    df_cfs["prediction"] = cf_predictions

    folder_name = f"{cf_method}_{data_name}_{model_type}_{hidden_width[0]}_sigma2_{sigma2}_intarget_{invalidation_target}"
    if not os.path.exists(f"recourse_invalidation_results/{folder_name}"):
        os.makedirs(f"recourse_invalidation_results/{folder_name}")

    # normalize factual
    factual_predictions = test_factual[data.target]
    # test_factual = model.perform_pipeline(test_factual)
    test_factual["prediction"] = factual_predictions

    df = pd.DataFrame(result)

    ar = df_cfs["prediction"].mean()
    air = df.mean().values[0]

    # this is the average Invalidation rate
    print("Average Recourse Accuracy (AR): ", ar)
    print("Average Invalidation Rate (AIR): ", air)
    # compare each row of factual and cf to get the l1 cost
    test_factual = test_factual.drop(columns=['y'])[df_cfs.columns]
    print("test_factual columns ", test_factual.columns)
    print("df_cf columns ", df_cfs.columns)
    cost = np.mean(np.abs(test_factual.drop(columns=["prediction"]).values - df_cfs.drop(columns=["prediction"]).values).sum(axis=1))
    print("Average Cost (AC): ", cost)

    return ar, air

@pytest.mark.parametrize(
    "dataset_name, model_type, backend",
    [
        ("compass", "mlp", "pytorch"),
        ("compass", "linear", "pytorch"),
    ],
)
def test_probe(dataset_name, model_type, backend):
    # below is to recreate the results in the table just for the PROBE method
    sigmas2 = [0.01] #[0.005, 0.01, 0.015, 0.02, 0.025]
    invalidation_target = 0.35 #[0.15, 0.20, 0.25, 0.30]
    # cost_weights = [0.0, 0.25, 0.5, 0.75, 1] unused in the experiment
    hidden_widths = [[50]]
    backend = backend
    method = "probe"  # "arar", "roar", "wachter"
    dataset = dataset_name # ["compass", "credit", "adult"]
    model = model_type

    n_cfs = 5 # just try 5 for faster results
    n_samples = 10000

    if model == "mlp":
        for hidden_width in hidden_widths:
            for sigma2 in sigmas2:
                print(f'Generating recourses for sigma2={sigma2}')
                ar, air = run_experiment(
                    method,
                    hidden_width,
                    dataset,
                    model,
                    backend,
                    n_cfs=n_cfs,
                    n_samples=n_samples,
                    sigma2=sigma2,
                    invalidation_target=invalidation_target)
    else:
        for sigma2 in sigmas2:
            hidden_width = [0]
            ar, air = run_experiment(
                method,
                hidden_width,
                dataset,
                model,
                backend,
                n_cfs=n_cfs,
                n_samples=n_samples,
                sigma2=sigma2,
                invalidation_target=invalidation_target)
    
    assert ar == 1.0
    assert air <= (invalidation_target + 0.01)

if __name__ == '__main__':
    test_probe("compass", "linear", "pytorch")
    test_probe("compass", "mlp", "pytorch")
    
