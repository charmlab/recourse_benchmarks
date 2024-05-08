# flake8: noqa
import os

from carla import log

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from typing import Dict, Optional

import numpy as np
import yaml
import tensorflow as tf
from tensorflow import Graph, Session
from tensorflow.python.keras.backend import set_session

import carla.evaluation.catalog as evaluation_catalog
from carla.data.api import Data
from carla.data.catalog import OnlineCatalog
from carla.evaluation import Benchmark
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods import *
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.mace import MACE


# def save_result(result: pd.DataFrame, alt_path: Optional[str]) -> None:
#     data_home = os.environ.get("CF_DATA", os.path.join("~", "carla", "results"))
#
#     data_home = os.path.expanduser(data_home)
#     if not os.path.exists(data_home):
#         os.makedirs(data_home)
#
#     path = os.path.join(data_home, "results.csv") if alt_path is None else alt_path

    # result.to_csv(path, index=False)


def load_setup() -> Dict:
    with open("../experimental_setup.yaml", "r") as f:
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
    if method not in setup.keys():
        raise KeyError("Method not in experimental setup")

    hyperparams = setup[method]["hyperparams"]
    if method == "casual_recourse":
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
    elif "wachter" in method:
        return Wachter(mlmodel, hyperparams)
    else:
        raise ValueError("Recourse method not known")

def create_parser():
    parser = argparse.ArgumentParser(description="Run experiments from paper")
    parser.add_argument(
        "-d",
        "--dataset",
        nargs="*",
        default=["adult", "compass", "credit", "german", "mortgage", "twomoon"],
        choices=["adult", "compass", "credit", "german", "mortgage", "twomoon"],
        help="Datasets for experiment",
    )
    parser.add_argument(
        "-t",
        "--type",
        nargs="*",
        default=["mlp", "linear"],
        choices=["mlp", "linear", "forest"],
        help="Model type for experiment",
    )
    parser.add_argument(
        "-r",
        "--recourse_method",
        nargs="*",
        default=[
            "dice",
            "cchvae",
            "cem",
            "cem-vae",
            "clue",
            "cruds",
            "face_knn",
            "face_epsilon",
            "gs",
            "revise",
            "wachter",
        ],
        choices=[
            "dice",
            "ar",
            "casual_recourse",
            "cchvae",
            "cem",
            "cem-vae",
            "clue",
            "cruds",
            "face_knn",
            "face_epsilon",
            "feature_tweak",
            "focus",
            "gs",
            "mace",
            "revise",
            "wachter",
        ],
        help="Recourse methods for experiment",
    )
    parser.add_argument(
        "-n",
        "--number_of_samples",
        type=int,
        default=100,
        help="Number of instances per dataset",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Save path for the output csv. If None, the output is written to the cache.",
    )
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    setup = load_setup()

    path = "results.csv"  # args.path
    path_data_results = "results_data.csv"
    results_data = pd.read_csv(path_data_results)
    if os.path.isfile(path):
        results = pd.read_csv(path)
    else:
        results = pd.DataFrame()

    session_models = ["cem", "cem-vae"]
    torch_methods = ["cchvae", "clue", "cruds", "wachter", "revise"]
    sklearn_methods = ["mace", "feature_tweak", "focus"]
    
    for rm in args.recourse_method:
        if rm in torch_methods:
            backend = "pytorch"
        elif rm in sklearn_methods:
            backend = "sklearn"
        else:
            backend = "tensorflow"
        log.info("Recourse method: {}".format(rm))
        for data_name in args.dataset: 
           
            for model_type in args.type:
                log.info("=====================================")
                log.info("Recourse method: {}".format(rm))
                log.info("Dataset: {}".format(data_name))
                log.info("Model type: {}".format(model_type))

                dataset = OnlineCatalog(data_name, model_type)
                dataset_catalog = dataset.catalog
                query_string = " & ".join([f"{key.strip()} == '{value.strip()}'" for key, value in dataset_catalog.items() if type(value) is not list])
                dataset_info_exists = len(
                    results_data.query(f"{query_string} & dataset == '{data_name}'")
                ) > 0

                if not (dataset_info_exists):
                    df_dataset = pd.DataFrame.from_dict(dataset.catalog, orient='index')
                    df_dataset = df_dataset.transpose()
                    df_dataset["dataset"] = data_name
                    df_dataset = df_dataset[
                        [
                            "dataset",
                            "continuous",
                            "categorical",
                            "immutable",
                            "target",
                        ]
                    ]

                    results_data = pd.concat([results_data, df_dataset], axis=0)
                    results_data.to_csv(path_data_results, index=False)

                exists_already = len(
                    results.query("Recourse_Method == @rm and Dataset == @data_name and ML_Model == @model_type")
                ) > 0
                if exists_already:
                    continue

                if rm in session_models:
                    graph = Graph()
                    ann_sess = Session()
                    session_graph = tf.get_default_graph()
                    init = tf.global_variables_initializer()
                    ann_sess.run(init)
                    with graph.as_default():
                        with session_graph.as_default():
                            set_session(ann_sess)
                            mlmodel_sess = MLModelCatalog(dataset, model_type, backend)

                            factuals_sess = predict_negative_instances(
                                mlmodel_sess, dataset
                            )
                            factuals_sess = factuals_sess.iloc[: args.number_of_samples]
                            factuals_sess = factuals_sess.reset_index(drop=True)

                            recourse_method_sess = initialize_recourse_method(
                                rm,
                                mlmodel_sess,
                                dataset,
                                data_name,
                                model_type,
                                setup,
                                sess=ann_sess,
                            )
                            if len(factuals_sess) == 0:
                                continue
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
                                evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
                                evaluation_catalog.AvgTime({"time": benchmark.timer}),
                            ]
                            df_benchmark = benchmark.run_benchmark(evaluation_measures)
                else:
                    mlmodel = MLModelCatalog(dataset, model_type, backend)
                    factuals = predict_negative_instances(mlmodel, dataset)
                    factuals = factuals.iloc[: args.number_of_samples]
                    factuals = factuals.reset_index(drop=True)

                    if len(factuals) == 0:
                        continue
                    recourse_method = initialize_recourse_method(
                        rm, mlmodel, dataset, data_name, model_type, setup
                    )

                    benchmark = Benchmark(mlmodel, recourse_method, factuals)
                    evaluation_measures = [
                        evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
                        evaluation_catalog.Distance(benchmark.mlmodel),
                        evaluation_catalog.SuccessRate(),
                        evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
                        evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
                        evaluation_catalog.AvgTime({"time": benchmark.timer}),
                    ]
                    df_benchmark = benchmark.run_benchmark(evaluation_measures)

                df_benchmark["Recourse_Method"] = rm
                df_benchmark["Dataset"] = data_name
                df_benchmark["ML_Model"] = model_type
                df_benchmark.rename(dict(avg_time="Average_Time"),axis=1,inplace=True)
                df_benchmark = df_benchmark[
                    [
                        "Recourse_Method",
                        "Dataset",
                        "ML_Model",
                        "L0_distance",#"Distance_1",
                        "L1_distance",#"Distance_2",
                        "L2_distance",#"Distance_3",
                        "Linf_distance",#"Distance_4",
                        "Constraint_Violation",
                        "Redundancy",
                        "y-Nearest-Neighbours",
                        "Success_Rate",
                        "Average_Time",
                    ]
                ]

                results = pd.concat([results, df_benchmark], axis=0)
                log.info(f"==={rm}==={data_name}===============================")

                results.to_csv(path, index=False)
                # deliberately saving this after every addition
                # save_result(results, path)
