# flake8: noqa
import argparse
import os
import warnings
from random import seed
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import yaml
from tensorflow import Graph, Session
from tensorflow.python.keras.backend import set_session

import evaluation.catalog as evaluation_catalog
from data.api import Data
from data.catalog import DataCatalog
from evaluation import Benchmark
from methods import *
from methods.api import RecourseMethod
from models.api import MLModel
from models.catalog import ModelCatalog
from models.negative_instances import predict_negative_instances
from tools.log import log

RANDOM_SEED = 54321

np.random.seed(RANDOM_SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
seed(
    RANDOM_SEED
)  # set the random seed so that the random permutations can be reproduced again
tf.set_random_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
warnings.simplefilter(action="ignore", category=FutureWarning)


def load_setup() -> Dict:
    """
    Loads experimental setup information from a YAML file and returns a dictionary
    containing the recourse methods specified in the setup.

    Parameters
    ----------
    None

    Returns
    -------
    Dict: A dictionary containing the recourse methods specified in the experimental setup.

    Raises
    -------
    FileNotFoundError: If the experimental setup file ("experimental_setup.yaml") is not found.
    yaml.YAMLError: If there is an error while parsing the YAML file.
    """
    with open("./experiments/experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    return setup_catalog["recourse_methods"]


def initialize_recourse_method(
    method: str,
    mlmodel: MLModel,
    data: Data,
    data_name: str,
    model_type: str,
    setup: Dict,
    sess: Session = None,
) -> RecourseMethod:
    """
    Initializes and returns an instance of a recourse method based on the specified recourse method,
    machine learning model, data, and an optional TensorFlow session.

    Parameters
    ----------
    method (str): The name of the recourse method to initialize.
    mlmodel (MLModel): The machine learning model used for recourse.
    data (Data): The dataset used for recourse.
    data_name (str): The name of the dataset.
    model_type (str): The type of machine learning model.
    setup (Dict): The experimental setup containing hyperparameters for the recourse methods.
    sess (Session, optional): Optional TensorFlow session. Defaults to None.

    Returns
    -------
    RecourseMethod: An instance of the initialized recourse method.

    Raises
    -------
    KeyError: If the specified method name is not found in the experimental setup.
    ValueError: If the specified method name is not recognized as a known recourse method.
    """
    if method not in setup.keys():
        raise KeyError("Method not in experimental setup")

    hyperparams = setup[method]["hyperparams"]
    if method == "causal_recourse":
        return CausalRecourse(mlmodel, hyperparams)
    elif method == "ar":
        coeffs, intercepts = None, None
        if model_type == "linear":
            # get weights and bias of linear layer for negative class 0
            coeffs_neg = mlmodel.raw_model.layers[0].get_weights()[0][:, 0]
            intercepts_neg = np.array(mlmodel.raw_model.layers[0].get_weights()[1][0])

            # get weights and bias of linear layer for positive class 1
            coeffs_pos = mlmodel.raw_model.layers[0].get_weights()[0][:, 1]
            intercepts_pos = np.array(mlmodel.raw_model.layers[0].get_weights()[1][1])

            coeffs = -(coeffs_neg - coeffs_pos)
            intercepts = -(intercepts_neg - intercepts_pos)

        ar = ActionableRecourse(mlmodel, hyperparams, coeffs, intercepts)
        act_set = ar.action_set
        ar.action_set = act_set

        return ar
    elif method == "cchvae":
        hyperparams["data_name"] = data_name
        hyperparams["vae_params"]["layers"] = [
            sum(mlmodel.get_mutable_mask())
        ] + hyperparams["vae_params"]["layers"]
        return CCHVAE(mlmodel, hyperparams)
    elif "cem" in method:
        hyperparams["data_name"] = data_name
        return CEM(sess, mlmodel, hyperparams)
    elif method == "claproar":
        hyperparams["data_name"] = data_name
        return ClaPROAR(mlmodel, hyperparams)
    elif method == "clue":
        hyperparams["data_name"] = data_name
        return Clue(data, mlmodel, hyperparams)
    elif method == "cruds":
        hyperparams["data_name"] = data_name
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
            sum(mlmodel.get_mutable_mask())
        ] + hyperparams["vae_params"]["layers"]
        return CRUD(mlmodel, hyperparams)
    elif method == "dice":
        return Dice(mlmodel, hyperparams)
    elif "face" in method:
        return Face(mlmodel, hyperparams)
    elif method == "feature_tweak":
        return FeatureTweak(mlmodel)
    elif method == "focus":
        return FOCUS(mlmodel)
    elif method == "gravitational":
        return Gravitational(mlmodel, hyperparams)
    elif method == "greedy":
        return Greedy(mlmodel, hyperparams)
    elif method == "gs":
        return GrowingSpheres(mlmodel)
    elif method == "mace":
        return MACE(mlmodel)
    elif method == "revise":
        hyperparams["data_name"] = data_name
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
            sum(mlmodel.get_mutable_mask())
        ] + hyperparams["vae_params"]["layers"]
        return Revise(mlmodel, data, hyperparams)
    elif method == "wachter":
        return Wachter(mlmodel, hyperparams)
    elif method == "cfvae":
        return CFVAE(mlmodel, hyperparams)
    elif method == "cfrl":
        return CFRL(mlmodel, hyperparams)
    elif method == "probe":
        return Probe(mlmodel, hyperparams)
    elif method == "roar":
        return Roar(mlmodel, hyperparams)
    else:
        raise ValueError("Recourse method not known")


def create_parser():
    """
    Creates and configures an argument parser for running experiments on the implemented recourse methods.
    It defines command-line arguments for specifying datasets, model types,
    recourse methods, number of samples, and output file path.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser: An argument parser object configured with the specified command-line arguments.

    Command-line Arguments
    -------
    -d, --dataset: Specifies datasets for the experiment.
        Default: ["adult", "compass", "credit", "german", "mortgage", "twomoon", "breast_cancer", "boston_housing"].
        Choices: ["adult", "compass", "credit", "german", "mortgage", "twomoon", "breast_cancer", "boston_housing"].
    -t, --type: Specifies model types for the experiment.
        Default: ["linear"].
        Choices: ["mlp", "linear", "forest"].
    -r, --recourse_method: Specifies recourse methods for the experiment.
        Default: ["dice", "ar", "causal_recourse", "cchvae", "cem", "cem_vae", "claproar", "clue", "cruds", "face_knn", "face_epsilon", "feature_tweak",
            "focus", "gravitational", "greedy", "gs", "mace", "revise", "wachter", "cfvae", "cfrl", "probe", "roar"].
        Choices: ["dice", "ar", "causal_recourse", "cchvae", "cem", "cem_vae", "claproar", "clue", "cruds", "face_knn", "face_epsilon", "feature_tweak",
            "focus", "gravitational", "greedy", "gs", "mace", "revise", "wachter", "cfvae", "cfrl", "probe", "roar"].
    -n, --number_of_samples: Specifies the number of instances per dataset.
        Default: 20.
    -s, --train_split: Specifies the split of the available data used for training.
        Default: 0.7.
    -p, --path: Specifies the save path for the output CSV file. If None, the output is written to the cache.
        Default: None.

    Raises
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Run experiments from paper")
    parser.add_argument(
        "-d",
        "--dataset",
        nargs="*",
        default=[
            "adult",
            "compass",
            "credit",
            "german",
            "mortgage",
            "twomoon",
            "breast_cancer",
            "boston_housing",
        ],
        choices=[
            "adult",
            "compass",
            "credit",
            "german",
            "mortgage",
            "twomoon",
            "breast_cancer",
            "boston_housing",
        ],
        help="Datasets for experiment",
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="*",
        default=["linear"],
        choices=["mlp", "linear", "forest"],
        help="Model type for experiment",
    )
    parser.add_argument(
        "-r",
        "--recourse_method",
        nargs="*",
        default=[
            "dice",
            "ar",
            "causal_recourse",
            "cchvae",
            "cem",
            "cem_vae",
            "claproar",
            "clue",
            "cruds",
            "face_knn",
            "face_epsilon",
            "feature_tweak",
            "focus",
            "gravitational",
            "greedy",
            "gs",
            "mace",
            "revise",
            "wachter",
            "cfvae",
            "cfrl",
            "probe",
            "roar",
        ],
        choices=[
            "dice",
            "ar",
            "causal_recourse",
            "cchvae",
            "cem",
            "cem_vae",
            "claproar",
            "clue",
            "cruds",
            "face_knn",
            "face_epsilon",
            "feature_tweak",
            "focus",
            "gravitational",
            "greedy",
            "gs",
            "mace",
            "revise",
            "wachter",
            "cfvae",
            "cfrl",
            "probe",
            "roar",
        ],
        help="Recourse methods for experiment",
    )
    parser.add_argument(
        "-n",
        "--number_of_samples",
        type=int,
        default=20,
        help="Number of instances per dataset",
    )
    parser.add_argument(
        "-s",
        "--train_split",
        type=float,
        default=0.7,
        help="Training Data Split",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Save path for the output csv. If None, the output is written to the cache.",
    )
    return parser


def _csv_has_data(path: str) -> bool:
    with open(path, "r") as csv_file:
        for line in csv_file:
            if line.strip():
                return True

    # Empty or whitespace-only file; clear contents to keep CSV detection consistent.
    open(path, "w").close()
    return False


if __name__ == "__main__":
    """
    Runs experiments on recourse methods extracted from academic papers, iterating over different combinations of datasets, model types, and recourse methods.
    It loads experimental setup information, performs experiments, and saves the results to CSV files.

    Parameters
    ----------
    None

    Returns
    -------
    .csv file: Generated benchmark results are written to a .csv file.

    Workflow:
    -------
    1. Parse command-line arguments specifying datasets, model types, recourse methods, number of samples, and output file path.
    2. Load experimental setup information.
    3. Initialize necessary variables and containers for storing results.
    4. Iterate over recourse methods, datasets, and model types:
        - Check if dataset information exists in the results data CSV file. If not, add it.
        - Check if results for the combination of recourse method, dataset, and model type already exist. If yes, skip.
        - Perform experiments:
            - If recourse method requires TensorFlow session, initialize TensorFlow session and related objects.
            - Initialize machine learning model and factual instances.
            - Initialize recourse method.
            - Run benchmark on recourse method.
        - Store benchmark results in a DataFrame.
    5. Save benchmark results to a CSV file.

    Raises
    -------
    None
    """

    args = create_parser().parse_args()
    setup = load_setup()

    path = file_path = os.path.join(os.path.dirname(__file__), "results.csv")
    if os.path.isfile(path) and _csv_has_data(path):
        results = pd.read_csv(path)
    else:
        results = pd.DataFrame()

    session_models = ["cem", "cem_vae", "greedy"]
    torch_methods = [
        "cchvae",
        "claproar",
        "clue",
        "cruds",
        "gravitational",
        "wachter",
        "revise",
        "cfvae",
        "cfrl",
        "probe",
        "roar",
    ]
    sklearn_methods = ["feature_tweak", "focus", "mace"]

    for method_name in args.recourse_method:
        if method_name in torch_methods:
            backend = "pytorch"
        elif method_name in sklearn_methods:
            backend = "sklearn"
        else:
            backend = "tensorflow"
        log.info("Recourse method: {}".format(method_name))
        for data_name in args.dataset:
            for model_name in args.type:
                log.info("=====================================")
                log.info("Recourse method: {}".format(method_name))
                log.info("Dataset: {}".format(data_name))
                log.info("Model type: {}".format(model_name))

                exists_already = (
                    len(
                        results.query(
                            "Recourse_Method == @method_name and Dataset == @data_name and ML_Model == @model_name"
                        )
                    )
                    > 0
                )
                # face_knn requires datasets with immutable features.
                if exists_already or (
                    "face" in method_name
                    and (data_name == "mortgage" or data_name == "twomoon")
                ):
                    continue

                dataset = DataCatalog(data_name, model_name, args.train_split)

                if method_name in session_models:
                    graph = Graph()
                    ann_sess = Session()
                    session_graph = tf.get_default_graph()
                    init = tf.global_variables_initializer()
                    ann_sess.run(init)
                    with graph.as_default():
                        with session_graph.as_default():
                            set_session(ann_sess)
                            mlmodel_sess = ModelCatalog(dataset, model_name, backend)

                            factuals_sess = predict_negative_instances(
                                mlmodel_sess, dataset
                            )

                            recourse_method_sess = initialize_recourse_method(
                                method_name,
                                mlmodel_sess,
                                dataset,
                                data_name,
                                model_name,
                                setup,
                                sess=ann_sess,
                            )
                            factuals_len = len(factuals_sess)
                            if factuals_len == 0:
                                continue
                            elif factuals_len > args.number_of_samples:
                                factuals_sess = factuals_sess.sample(
                                    n=args.number_of_samples, random_state=RANDOM_SEED
                                )

                            factuals_sess = factuals_sess.reset_index(drop=True)
                            benchmark = Benchmark(
                                mlmodel_sess, recourse_method_sess, factuals_sess
                            )
                            evaluation_measures = [
                                evaluation_catalog.YNN(
                                    benchmark.mlmodel, {"y": 5, "cf_label": 1}
                                ),
                                evaluation_catalog.Distance(benchmark.mlmodel),
                                evaluation_catalog.SuccessRate(),
                                evaluation_catalog.Redundancy(
                                    benchmark.mlmodel, {"cf_label": 1}
                                ),
                                evaluation_catalog.ConstraintViolation(
                                    benchmark.mlmodel
                                ),
                                evaluation_catalog.AvgTime({"time": benchmark.timer}),
                            ]
                            df_benchmark = benchmark.run_benchmark(evaluation_measures)
                else:
                    mlmodel = ModelCatalog(dataset, model_name, backend)
                    factuals = predict_negative_instances(mlmodel, dataset)

                    factuals_len = len(factuals)
                    if factuals_len == 0:
                        continue
                    elif factuals_len > args.number_of_samples:
                        factuals = factuals.sample(
                            n=args.number_of_samples, random_state=RANDOM_SEED
                        )

                    factuals = factuals.reset_index(drop=True)
                    recourse_method = initialize_recourse_method(
                        method_name, mlmodel, dataset, data_name, model_name, setup
                    )

                    benchmark = Benchmark(mlmodel, recourse_method, factuals)
                    evaluation_measures = [
                        evaluation_catalog.YNN(
                            benchmark.mlmodel, {"y": 5, "cf_label": 1}
                        ),
                        evaluation_catalog.Distance(benchmark.mlmodel),
                        evaluation_catalog.SuccessRate(),
                        evaluation_catalog.Redundancy(
                            benchmark.mlmodel, {"cf_label": 1}
                        ),
                        evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
                        evaluation_catalog.AvgTime({"time": benchmark.timer}),
                    ]
                    df_benchmark = benchmark.run_benchmark(evaluation_measures)

                df_benchmark["Recourse_Method"] = method_name
                df_benchmark["Dataset"] = data_name
                df_benchmark["ML_Model"] = model_name
                df_benchmark.rename(dict(avg_time="Average_Time"), axis=1, inplace=True)
                df_benchmark = df_benchmark[
                    [
                        "Recourse_Method",
                        "Dataset",
                        "ML_Model",
                        "L0_distance",  # "Distance_1",
                        "L1_distance",  # "Distance_2",
                        "L2_distance",  # "Distance_3",
                        "Linf_distance",  # "Distance_4",
                        "Constraint_Violation",
                        "Redundancy",
                        "y-Nearest-Neighbours",
                        "Success_Rate",
                        "Average_Time",
                    ]
                ]

                results = pd.concat([results, df_benchmark], axis=0)
                log.info(
                    f"==={method_name}==={data_name}==============================="
                )

                results.to_csv(path, index=False)
                # deliberately saving this after every addition
                # save_result(results, path)
