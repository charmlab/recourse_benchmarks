import pickle
import sys
import warnings
from random import seed

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import tools.utils.memoize as utils
from models.utils.implement_framework_model import (
    PyTorchLogisticRegression,
    PyTorchNeuralNetwork,
    TensorflowLogisticRegression,
    TensorflowNeuralNetwork,
)


def warn(*args, **kwargs):
    pass


warnings.warn = warn  # to ignore all warnings.


RANDOM_SEED = 54321
seed(
    RANDOM_SEED
)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)

try:
    import treeUtils
except Exception as e:
    print(f"[ENV WARNING] treeUtils not available. Error: {e}")

SIMPLIFY_TREES = False


@utils.Memoize
def loadModelForDataset(
    model_class,
    dataset_obj,
    backend="sklearn",
    scm_class=None,
    experiment_folder_name=None,
):
    """
    Loads and returns a model with trained data.

    Parameters
    ----------
    model_class : str
      Class of model to be trained.
    dataset_obj : DataCatalog
        Dataset to train model with.
    backend : str
        The backend of the model.
    scm_class : object
        SCM Class of the retrieved dataset.
    experiment_folder_name : str
        Folder name to save model in.

    Returns
    -------
    model :  The trained model.
    """
    log_file = (
        sys.stdout
        if experiment_folder_name is None
        else open(f"{experiment_folder_name}/log_training.txt", "w")
    )

    if not (model_class in {"linear", "mlp", "tree", "forest"}):
        raise Exception(f"{model_class} not supported.")

    if not (
        dataset_obj.name
        in {
            "synthetic",
            "mortgage",
            "twomoon",
            "german",
            "credit",
            "compass",
            "adult",
            "test",
            "breast_cancer",
            "boston_housing",
        }
    ):
        raise Exception(f"{dataset_obj.name} not supported.")

    # if model_class in {"tree", "forest"}:
    #     one_hot = False
    # elif model_class in {"mlp", "linear"}:
    #     one_hot = True
    # else:
    #     raise Exception(f"{model_class} not recognized as a valid `model_class`.")

    # dataset_obj = loadData.loadDataset(
    #     dataset_string,
    #     return_one_hot=one_hot,
    #     load_from_cache=True,
    #     meta_param=scm_class,
    # )
    # X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit(
    #     preprocessing="normalize"
    # )
    y_all = dataset_obj.df["y"]
    assert sum(y_all) / len(y_all) == 0.5, "Expected class balance should be 50/50%."

    logisticRegressionMap = {
        "sklearn": LogisticRegression(
            solver="liblinear"
        ),  # IMPORTANT: The default solver changed from ‘liblinear’ to ‘lbfgs’ in 0.22; therefore, results may differ slightly from paper.
        "pytorch": PyTorchLogisticRegression(dataset_obj.X_train.shape[1], 2),
        "tensorflow": TensorflowLogisticRegression(dataset_obj.X_train.shape[1], 2),
    }

    neuralNetworksMap = {
        "sklearn": MLPClassifier(hidden_layer_sizes=(10, 10)),
        "pytorch": PyTorchNeuralNetwork(dataset_obj.X_train.shape[1], 2, 10),
        "tensorflow": TensorflowNeuralNetwork(dataset_obj.X_train.shape[1], 2, 10),
    }

    forestMap = {
        "sklearn": RandomForestClassifier(),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            subsample=0.9,
            colsample_bynode=0.2,
            tree_method="hist",
            early_stopping_rounds=2,
        ),
    }

    if model_class == "tree":
        model_pretrain = DecisionTreeClassifier()
    elif model_class == "forest":
        model_pretrain = forestMap[backend]
    elif model_class == "linear":
        model_pretrain = logisticRegressionMap[backend]
    elif model_class == "mlp":
        model_pretrain = neuralNetworksMap[backend]

    tmp_text = (
        f"[INFO] Training `{model_class}` on {dataset_obj.X_train.shape[0]:,} samples "
        + f"(%{100 * dataset_obj.X_train.shape[0] / (dataset_obj.X_train.shape[0] + dataset_obj.X_test.shape[0]):.2f} "
        + f"of {dataset_obj.X_train.shape[0] + dataset_obj.X_test.shape[0]:,} samples)..."
    )
    print(tmp_text)
    print(tmp_text, file=log_file)

    model_trained = model_pretrain.fit(
        dataset_obj.X_train.values, dataset_obj.y_train.values
    )
    # visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, experiment_folder_name)

    if model_class == "tree":
        if SIMPLIFY_TREES:
            print("[INFO] Simplifying decision tree...", end="", file=log_file)
            model_trained.tree_ = treeUtils.simplifyDecisionTree(model_trained, False)
            print("\tdone.", file=log_file)
        # treeUtils.saveTreeVisualization(model_trained, model_class, '', X_test, feature_names, experiment_folder_name)
    elif model_class == "forest":
        for tree_idx in range(len(model_trained.estimators_)):
            if SIMPLIFY_TREES:
                print(
                    f"[INFO] Simplifying decision tree (#{tree_idx + 1}/{len(model_trained.estimators_)})...",
                    end="",
                    file=log_file,
                )
                model_trained.estimators_[
                    tree_idx
                ].tree_ = treeUtils.simplifyDecisionTree(
                    model_trained.estimators_[tree_idx], False
                )
                print("\tdone.", file=log_file)
            # treeUtils.saveTreeVisualization(model_trained.estimators_[tree_idx], model_class, f'tree{tree_idx}', X_test, feature_names, experiment_folder_name)

    if experiment_folder_name:
        pickle.dump(
            model_trained, open(f"{experiment_folder_name}/_model_trained", "wb")
        )

    return model_trained
