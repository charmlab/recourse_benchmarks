import sys

sys.path.append("./")

import os
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import os
import pickle
import time
import traceback

import baselines.bridge as bridge
import numpy as np
import pandas as pd
import torch
import utils

# data support
from carla.recourse_methods.catalog.cchvae import CCHVAE
from carla.recourse_methods.catalog.crud import CRUD
from carla.recourse_methods.catalog.dice_diverse import DICEDiv
from carla.recourse_methods.catalog.growing_spheres import GrowingSpheres
from carla.recourse_methods.catalog.revise import Revise
from carla.recourse_methods.catalog.roar import ROAR

# Robust
from carla.recourse_methods.catalog.w_rip import Wachter_rip  # PROBE

# mincost
from carla.recourse_methods.catalog.wachter import Wachter
from torch.distributions.multivariate_normal import MultivariateNormal

import data.utils as dutils

# disable GPU, some implementation don't work well GPU
# torch.cuda.is_available = lambda: False


def get_recourse(
    exp_config,
    method,
    mlmodel,
    data,
    lambda_=None,
    base_dir="./saved_models",
    EPOCHS=None,
):
    """
    Initializes and returns an instance of a recourse method based on the specified recourse method,
    """

    hyperparams = exp_config[method]["hyperparams"]
    INPUT_SHAPE = len(data.df_train.columns) - 1
    if method == "cchvae":
        os.environ["CF_MODELS"] = os.path.join(
            base_dir, "carla", f"{method}", "autoencoders"
        )
        hyperparams["data_name"] = data.name
        hyperparams["vae_params"]["layers"] = [INPUT_SHAPE] + hyperparams["vae_params"][
            "layers"
        ]
        hyperparams["vae_params"]["train"] = True
        hyperparams["vae_params"]["epochs"] = EPOCHS
        # doesn't have a concept of lambda
        # always returns
        return CCHVAE(mlmodel, hyperparams)

    elif method == "cruds":
        os.environ["CF_MODELS"] = os.path.join(
            base_dir, "carla", f"{method}", "autoencoders"
        )
        hyperparams["data_name"] = data.name
        hyperparams["vae_params"]["layers"] = [INPUT_SHAPE] + hyperparams["vae_params"][
            "layers"
        ]
        hyperparams["vae_params"]["train"] = True
        hyperparams["vae_params"]["epochs"] = EPOCHS
        if lambda_ is not None:
            hyperparams["lambda_param"] = lambda_
        rec_method = CRUD(mlmodel, hyperparams)

        return rec_method
    elif method == "revise":
        os.environ["CF_MODELS"] = os.path.join(
            base_dir, "carla", f"{method}", "autoencoders"
        )
        hyperparams["data_name"] = data.name
        hyperparams["vae_params"]["layers"] = [INPUT_SHAPE] + hyperparams["vae_params"][
            "layers"
        ]
        hyperparams["vae_params"]["train"] = True
        hyperparams["vae_params"]["epochs"] = EPOCHS
        if lambda_ is not None:
            hyperparams["lambda"] = lambda_

        rec_method = Revise(mlmodel, data, hyperparams)

        return rec_method
    elif method == "roar":
        if lambda_ is not None:
            hyperparams["delta"] = (
                1 / lambda_
            )  # interpreted different more delta ---> far away ----> less lambda_

        return ROAR(mlmodel, hyperparams, coeffs=None, intercept=None)

    elif method == "probe":
        if lambda_ is not None:
            hyperparams[
                "invalidation_target"
            ] = lambda_  # interpreted different more delta ---> far away ----> less lambda_

        return Wachter_rip(mlmodel, hyperparams)
    elif method == "dice":
        if lambda_ is not None:
            hyperparams["lambda_param"] = lambda_

        return DICEDiv(mlmodel, hyperparams)
    elif method == "gs":

        return GrowingSpheres(mlmodel)
    elif "wachter" in method:
        if lambda_ is not None:
            hyperparams[
                "lambda_param"
            ] = lambda_  # doesn't matter too much since they schedule it
        return Wachter(mlmodel, hyperparams)
    else:
        raise ValueError("Recourse method not known")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Baselines")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
    parser.add_argument("--lamb", type=float, help="cost factor")

    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[
            "heloc",
            "compas-all",
            "adult-all",
        ],
        help="dataset(s) to run on",
        type=str,
    )
    parser.add_argument(
        "--method",
        nargs="+",
        default=["roar", "gs", "wachter", "revise", "cruds"],
        type=str,
        help="method(s) to run",
    )
    parser.add_argument("--override", action="store_true", help="override existing")
    args = parser.parse_args()

    SEED = args.seed
    OVERRIDE = args.override
    config_path = "results/exp1_config.yaml"
    LABEL_SRC = "rf"
    lambda_cost = args.lamb
    experiment_name = f"paper_experiments"
    # experiment_name = utils.curr_time_hash()

    exp_config = utils.load_config(config_path)

    _all_real_datasets = [
        "heloc",
        "compas-all",
        "adult-all",
    ]  # TODO: to be replaced by the compass-all, adult-all
    _all_synth_datasets = ["moons", "corr", "circles"]
    _all_datasets = _all_real_datasets + _all_synth_datasets

    if "all" in args.dataset:
        datasets_to_run = _all_datasets
    elif "synth" in args.dataset:
        datasets_to_run = _all_synth_datasets
    elif "real" in args.dataset:
        datasets_to_run = _all_real_datasets
    else:
        datasets_to_run = args.dataset

    # Supported methods: ['roar','probe','wachter','gs','cchvae','revise','cruds','dice']
    # _all_rec_methods = ["roar", "gs", "wachter", "revise", "cruds"]

    for rec_method in args.method:
        for DATASET_STR in datasets_to_run:

            if DATASET_STR in _all_synth_datasets:
                EPOCHS = 100
            else:
                EPOCHS = 20

            print(
                f" =========================== [INFO] Executing  {rec_method} for dataset: {DATASET_STR} ==========================="
            )
            RF_FOLDER = utils.get_rf_folder(DATASET_STR, **exp_config["common"])
            (
                train_y,
                train_X,
                test_y,
                test_X,
                cat_mask,
                immutable_mask,
            ) = dutils.load_dataset(
                DATASET_STR,
                cust_labels_path=RF_FOLDER,
                ret_tensor=True,
                min_max=exp_config["common"]["MIN_MAX"],
                ret_masks=True,
            )
            INPUT_SHAPE = train_X.shape[1]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ann_clf, ann_folder = utils.load_ann(
                INPUT_SHAPE=INPUT_SHAPE,
                DATASET_STR=DATASET_STR,
                LABEL_SRC="rf",
                **exp_config["common"],
                **exp_config["ann"][DATASET_STR],
            )
            ann_clf = ann_clf.to(device)
            print(
                f"[INFO] loaded ann model from {ann_folder}"
            )  # move inside load function?

            MODEL_NAME = "ann"
            BACKEND = "pytorch"
            YSTAR = 1
            # construct relevant constructs for carla recourse methods
            data = bridge.CustomDataCatalog(
                train_y,
                train_X,
                test_y,
                test_X,
                DATASET_STR,
                YSTAR=YSTAR,
                immutable_mask=immutable_mask,
                cat_mask=cat_mask,
            )
            model = bridge.CustomModelCatalog(
                ann_clf, data=data, model_type=MODEL_NAME, backend=BACKEND, YSTAR=YSTAR
            )

            model._test_accuracy()
            common_dir = f"./results/{experiment_name}/{DATASET_STR}"
            with open(f"{common_dir}/xf_r", "rb") as fp:
                xf_r = pickle.load(fp).cpu().numpy()

            column_names = [f"x{i}" for i in range(INPUT_SHAPE)]
            test_factual = pd.DataFrame(data=xf_r, columns=column_names)
            test_factual[data.target] = (1 - data.ystar) * np.ones(len(test_factual))

            try:
                utils.set_seed(SEED)
                if lambda_cost is None:
                    output_dir = f"{common_dir}/{rec_method}"
                else:
                    output_dir = f"{common_dir}/{rec_method}_{lambda_cost}"

                if os.path.isfile(f"{output_dir}/xcf") and not (OVERRIDE):
                    print("[INFO] found existing recourse: skipping")
                    continue

                sttt = time.time()
                recourse_module = get_recourse(
                    exp_config,
                    rec_method,
                    model,
                    data,
                    EPOCHS=EPOCHS,
                    lambda_=lambda_cost,
                )
                finnn = time.time()
                print("time required for training", finnn - sttt)

                sttt = time.time()
                xcf = recourse_module.get_counterfactuals(test_factual)
                finnn = time.time()
                print("time required for inference", finnn - sttt)

                xcf.drop(data.target, axis=1, inplace=True)
                xcf = xcf.to_numpy(dtype=np.float32)
                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/xcf", "wb") as fp:
                    pickle.dump(torch.from_numpy(xcf).cpu(), fp)

            except Exception as e:
                print(
                    f"[CRITICAL] exception encountered while executing {rec_method} for dataset {DATASET_STR}:",
                    e,
                )
                print(traceback.format_exc())
                print(f"moving on ....")
