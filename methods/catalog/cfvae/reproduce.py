import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import _CFVAE
from .resources import DataLoader, load_adult_income_dataset, load_pretrained_binaries


class BlackBox(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dim = 10
        self.predict_net = nn.Sequential(
            nn.Linear(self.input_shape, self.hidden_dim),
            nn.Linear(self.hidden_dim, 2),  # Still Binary Classification
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.predict_net(x)


def target_class_validity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    sample_sizes: List[int],
):
    """
    Target-Class Validity: % of CFs whose predicted class is the target class
    """

    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        valid_cf_count = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)
            valid_cf_count += np.sum(test_y.cpu().numpy() != cf_label.cpu().numpy())
        dataset_size = test_x.shape[0]
        valid_cf_count = valid_cf_count / sample_size

        results.append(100 * valid_cf_count / dataset_size)

    return results


def compare_results(results: Dict, ref: Dict, tolerance: float = 1.0) -> None:
    print(f"Comparing results to reference with tolerance ±{tolerance}")
    for dataset_name, ref_methods in ref.items():
        dataset_results = results.get(dataset_name)
        if dataset_results is None:
            print(f"[MISSING] Dataset `{dataset_name}` not found in results.")
            continue
        for method_name, ref_metrics in ref_methods.items():
            method_results = dataset_results.get(method_name)
            if method_results is None:
                print(
                    f"[MISSING] Method `{dataset_name}/{method_name}` not found in results."
                )
                continue
            for metric_name, ref_value in ref_metrics.items():
                result_value = method_results.get(metric_name)
                if result_value is None:
                    print(
                        f"[MISSING] Metric `{dataset_name}/{method_name}/{metric_name}` not found in results."
                    )
                    continue
                result_array = np.array(result_value, dtype=float)
                ref_array = np.array(ref_value, dtype=float)
                if result_array.shape != ref_array.shape:
                    print(
                        f"[SHAPE DIFF] `{dataset_name}/{method_name}/{metric_name}` result shape {result_array.shape} != reference shape {ref_array.shape}."
                    )
                    continue
                diff = np.abs(result_array - ref_array)
                max_diff = float(np.max(diff)) if diff.size else 0.0
                if np.all(diff <= tolerance):
                    print(
                        f"[OK] `{dataset_name}/{method_name}/{metric_name}` max diff {max_diff:.6f}"
                    )
                else:
                    print(
                        f"[DIFF] `{dataset_name}/{method_name}/{metric_name}` max diff {max_diff:.6f} exceeds tolerance"
                    )


def constraint_feasibility_score_age(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
):
    """
    Constraint Feasibility Score: % of CFs satisfying age constraint
    """

    results_valid = []
    results_invalid = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        valid_change = 0
        invalid_change = 0
        dataset_size = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            age_idx = x_ori.columns.get_loc("age")
            for i in range(x_ori.shape[0]):
                if cf_label[i] == 0:
                    continue
                dataset_size += 1

                if x_pred.iloc[i, age_idx] >= x_ori.iloc[i, age_idx]:
                    valid_change += 1
                else:
                    invalid_change += 1
        valid_change = valid_change / sample_size
        invalid_change = invalid_change / sample_size
        dataset_size = dataset_size / sample_size

        results_valid.append(100 * valid_change / dataset_size)
        results_invalid.append(100 * invalid_change / dataset_size)

    return results_valid, results_invalid


def constraint_feasibility_score_age_ed(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
):
    """
    Constraint Feasibility Score: % of CFs satisfying age-ed constraint
    """

    education_score = {
        "HS-grad": 0,
        "School": 0,
        "Bachelors": 1,
        "Assoc": 1,
        "Some-college": 1,
        "Masters": 2,
        "Prof-school": 2,
        "Doctorate": 3,
    }

    results_valid = []
    results_invalid = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        valid_change = 0
        invalid_change = 0
        dataset_size = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)
            cf_label = torch.argmax(target_model(x_pred), dim=1)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            age_idx = x_ori.columns.get_loc("age")
            ed_idx = x_ori.columns.get_loc("education")
            for i in range(x_ori.shape[0]):
                if cf_label[i] == 0:
                    continue
                dataset_size += 1

                if (
                    education_score[x_pred.iloc[i, ed_idx]]
                    < education_score[x_ori.iloc[i, ed_idx]]
                ):
                    invalid_change += 1
                elif (
                    education_score[x_pred.iloc[i, ed_idx]]
                    == education_score[x_ori.iloc[i, ed_idx]]
                ):
                    if x_pred.iloc[i, age_idx] >= x_ori.iloc[i, age_idx]:
                        valid_change += 1
                    else:
                        invalid_change += 1
                elif (
                    education_score[x_pred.iloc[i, ed_idx]]
                    > education_score[x_ori.iloc[i, ed_idx]]
                ):
                    if x_pred.iloc[i, age_idx] > x_ori.iloc[i, age_idx]:
                        valid_change += 1
                    else:
                        invalid_change += 1
        valid_change = valid_change / sample_size
        invalid_change = invalid_change / sample_size

        dataset_size = dataset_size / sample_size
        results_valid.append(100 * valid_change / dataset_size)
        results_invalid.append(100 * invalid_change / dataset_size)

    return results_valid, results_invalid


def cat_proximity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    sample_sizes: List[int],
):
    """
    Cat-Proximity: Proximity for categorical features as the total number of mismatches on categorical value between xcf and x for each feature
    """

    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        diff_count = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            for column in dataloader.categorical_feature_names:
                diff_count += np.sum(
                    np.array(x_ori[column], dtype=pd.Series)
                    != np.array(x_pred[column], dtype=pd.Series)
                )
        dataset_size = test_x.shape[0]
        diff_count = diff_count / sample_size

        results.append(-1 * diff_count / dataset_size)

    return results


def cont_proximity(
    cf_model: _CFVAE,
    target_model: nn.Module,
    test_dataset: torch.Tensor,
    dataloader: DataLoader,
    mad_feature_weights: Dict[str, float],
    sample_sizes: List[int],
):
    """
    Cont-Proximity: Proximity for continuous features as the average L1-distance between xcf and x in units of median absolute deviation for each feature
    """

    results = []
    for sample_size in sample_sizes:
        test_x = test_dataset.float()
        test_y = torch.argmax(target_model(test_x), dim=1)

        diff_amount = 0
        for _ in range(sample_size):
            x_pred = cf_model(test_x, 1.0 - test_y)

            x_pred = dataloader.de_normalize_data(
                dataloader.get_decoded_data(x_pred.detach().cpu().numpy())
            )
            x_ori = dataloader.de_normalize_data(
                dataloader.get_decoded_data(test_x.detach().cpu().numpy())
            )

            for column in dataloader.continuous_feature_names:
                diff_amount += (
                    np.sum(np.abs(x_ori[column] - x_pred[column]))
                    / mad_feature_weights[column]
                )
        dataset_size = test_x.shape[0]
        diff_amount = diff_amount / sample_size

        results.append(-1 * diff_amount / dataset_size)

    return results


def eval_adult(
    methods: Dict[str, str],
    encoded_size: int,
    target_model: nn.Module,
    val_dataset_np: np.ndarray,
    dataloader: DataLoader,
    mad_feature_weights: Dict[str, float],
    sample_sizes: List[int],
    constraint: str,
    n_test: int = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    with torch.no_grad():
        results = {}

        np.random.shuffle(val_dataset_np)
        x_sample = val_dataset_np[0, :]
        x_sample = np.reshape(x_sample, (1, val_dataset_np.shape[1]))
        print(
            "Input Data Sample: ",
            dataloader.de_normalize_data(dataloader.get_decoded_data(x_sample)),
        )
        val_dataset = torch.tensor(val_dataset_np).to(device)

        target_model.eval().to(device)

        for name, path in methods.items():
            cf_val = {}

            cf_vae = _CFVAE(len(dataloader.encoded_feature_names), encoded_size)
            cf_vae.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
            cf_vae.eval().to(device)

            # Evaluate only Low to High Income CF
            test_x = val_dataset.float().to(device)
            test_y = torch.argmax(target_model(test_x), dim=1).to(device)
            val_dataset = val_dataset[test_y == 0]

            for _ in range(n_test):
                cf_val["target_class_validity"] = cf_val.get(
                    "target_class_validity", []
                )
                cf_val["target_class_validity"].append(
                    target_class_validity(
                        cf_vae, target_model, val_dataset, sample_sizes
                    )
                )
                if constraint == "age":
                    val, inval = constraint_feasibility_score_age(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        sample_sizes,
                    )
                    cf_val["constraint_feasibility_score"] = cf_val.get(
                        "constraint_feasibility_score", []
                    )
                    cf_val["constraint_feasibility_score"].append(
                        100 * np.array(val) / (np.array(val) + np.array(inval))
                    )
                elif constraint == "age-ed":
                    val, inval = constraint_feasibility_score_age_ed(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        sample_sizes,
                    )
                    cf_val["constraint_feasibility_score"] = cf_val.get(
                        "constraint_feasibility_score", []
                    )
                    cf_val["constraint_feasibility_score"].append(
                        100 * np.array(val) / (np.array(val) + np.array(inval))
                    )
                cf_val["cont_proximity"] = cf_val.get("cont_proximity", [])
                cf_val["cont_proximity"].append(
                    cont_proximity(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        mad_feature_weights,
                        sample_sizes,
                    )
                )
                cf_val["cat_proximity"] = cf_val.get("cat_proximity", [])
                cf_val["cat_proximity"].append(
                    cat_proximity(
                        cf_vae,
                        target_model,
                        val_dataset,
                        dataloader,
                        sample_sizes,
                    )
                )
            for key, value in cf_val.items():
                cf_val[key] = np.mean(np.array(value), axis=0)

            results[name] = cf_val
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Allowed absolute difference when comparing against reference.",
    )
    args = parser.parse_args()

    # Seed for Reproducibility
    torch.manual_seed(10000000)

    results = {}

    dataset = load_adult_income_dataset()
    params = {
        "dataframe": dataset.copy(),
        "continuous_features": ["age", "hours_per_week"],
        "outcome_name": "income",
    }
    dataloader = DataLoader(params)

    vae_test_dataset = np.load(load_pretrained_binaries("adult-test-set.npy"))
    vae_test_dataset = vae_test_dataset[
        vae_test_dataset[:, -1] == 0, :
    ]  # Use only Low to High Income CF
    vae_test_dataset = vae_test_dataset[:, :-1]  # Remove labels

    mad_feature_weights = {
        "age": 10.0,
        "hours_per_week": 3.0,
    }  # Median absolute deviation

    # Load Black Box Prediction Model
    data_size = len(dataloader.encoded_feature_names)
    target_model = BlackBox(data_size)
    target_model.load_state_dict(
        torch.load(
            load_pretrained_binaries("adult-target-model.pth"),
            map_location=torch.device("cpu"),
        )
    )
    target_model.eval()

    dataset_name = "adult-age"
    results[dataset_name] = {}
    methods = {
        "BaseCVAE": load_pretrained_binaries(
            "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth"
        ),
        "BaseVAE": load_pretrained_binaries(
            "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth"
        ),
        "ModelApprox": load_pretrained_binaries(
            "adult-margin-0.764-constraint-reg-192.0-validity_reg-29.0-epoch-25-unary-gen.pth"
        ),
        "ExampleBased": load_pretrained_binaries(
            "adult-eval-case-0-supervision-limit-100-const-case-0-margin-0.084-oracle_reg-5999.0-validity_reg-159.0-epoch-50-oracle-gen.pth"
        ),
    }
    results[dataset_name] = eval_adult(
        methods,
        encoded_size=10,
        target_model=target_model,
        val_dataset_np=vae_test_dataset,
        dataloader=dataloader,
        mad_feature_weights=mad_feature_weights,
        sample_sizes=[1, 2, 3],
        constraint="age",
    )

    dataset_name = "adult-age-ed"
    results[dataset_name] = {}
    methods = {
        "BaseCVAE": load_pretrained_binaries(
            "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth"
        ),
        "BaseVAE": load_pretrained_binaries(
            "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth"
        ),
        "ModelApprox": load_pretrained_binaries(
            "adult-margin-0.344-constraint-reg-87.0-validity_reg-76.0-epoch-25-unary-ed-gen.pth"
        ),
        "ExampleBased": load_pretrained_binaries(
            "adult-eval-case-0-supervision-limit-100-const-case-1-margin-0.117-oracle_reg-3807.0-validity_reg-175.0-epoch-50-oracle-gen.pth"
        ),
    }
    results[dataset_name] = eval_adult(
        methods,
        encoded_size=10,
        target_model=target_model,
        val_dataset_np=vae_test_dataset,
        dataloader=dataloader,
        mad_feature_weights=mad_feature_weights,
        sample_sizes=[1, 2, 3],
        constraint="age-ed",
    )

    print(results)
    ref = {
    "adult-age": {
        "BaseCVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([56.82554814, 56.93040991, 56.9399428]),
            "cont_proximity": ([-2.24059021, -2.254801, -2.24498223]),
            "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
        },
        "BaseVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([42.85033365, 43.11248808, 42.99014935]),
            "cont_proximity": ([-2.66647616, -2.66112855, -2.66558379]),
            "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
        },
        "ModelApprox": {
            "target_class_validity": ([99.5900858, 99.5900858, 99.55513187]),
            "constraint_feasibility_score": ([84.26668079, 83.60122942, 83.58960991]),
            "cont_proximity": ([-2.73294234, -2.73510212, -2.73455902]),
            "cat_proximity": ([-3.26167779, -3.26172545, -3.26151891]),
        },
        "ExampleBased": {
            "target_class_validity": ([99.52335558, 99.52812202, 99.46298062]),
            "constraint_feasibility_score": ([74.03769743, 74.18632186, 74.21309514]),
            "cont_proximity": ([-6.80110826, -6.7996033, -6.80052672]),
            "cat_proximity": ([-3.72411821, -3.7242612, -3.72443597]),
        },
    },
    "adult-age-ed": {
        "BaseCVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([57.01620591, 57.20209724, 56.80012711]),
            "cont_proximity": ([-2.25337728, -2.24797755, -2.24245525]),
            "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
        },
        "BaseVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([42.59294566, 43.06482364, 43.02192564]),
            "cont_proximity": ([-2.6661057, -2.66556556, -2.6650458]),
            "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
        },
        "ModelApprox": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([79.5042898, 79.32793136, 79.30092151]),
            "cont_proximity": ([-2.90320554, -2.90264891, -2.90204051]),
            "cat_proximity": ([-3.22097235, -3.22054337, -3.21916111]),
        },
        "ExampleBased": {
            "target_class_validity": ([99.93326978, 99.89513823, 99.92691452]),
            "constraint_feasibility_score": ([66.35181132, 66.50929669, 66.46958292]),
            "cont_proximity": ([-3.20324281, -3.2077721, -3.20357766]),
            "cat_proximity": ([-3.5914204, -3.59394662, -3.59634573]),
        },
    },
}
    compare_results(results, ref, tolerance=args.tolerance)
