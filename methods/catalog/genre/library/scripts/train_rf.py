import sys

# sys.path.append('./') # to be called from root directory as script/train_ann.py --args
sys.path.insert(0, ".")  # from library
import argparse
import logging
import os
import pickle

import numpy as np
import utils
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

import data.utils as dutils


def _train_rf(SEED, DATASET_STR, MIN_MAX, TRAIN_ON_TEST, FORCE_RETRAIN):
    CLF_FOLDER = utils.get_rf_folder(DATASET_STR, TRAIN_ON_TEST, MIN_MAX)
    os.makedirs(CLF_FOLDER, exist_ok=True)
    STATE_PATH = os.path.join(CLF_FOLDER, "state.pkl")
    LOG_PATH = os.path.join(CLF_FOLDER, "train.log")

    if os.path.exists(STATE_PATH) and not FORCE_RETRAIN:
        # load the model and print the hidden params of this model
        print(
            "[INFO] trained model already exists. Do FORCE_RETRAIN=True to train again"
        )
        with open(STATE_PATH, "rb") as f:
            rf_model = pickle.load(f)
        return rf_model
    else:
        logger = logging.getLogger(__name__)
        utils.configLogger(
            logger,
            level=logging.DEBUG,
            format="%(asctime)s :: %(levelname)8s :: %(filename)s:%(funcName)s :: %(message)s",
            handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
        )
        train_y, train_X, test_y, test_X = dutils.load_dataset(
            DATASET_STR, cust_labels_path=None, ret_tensor=False, min_max=MIN_MAX
        )
        if TRAIN_ON_TEST:
            logger.warning(
                "RF model will be trained on both train and test gold labels"
            )
            train_X_og = train_X
            train_y_og = train_y
            train_X = np.concatenate([train_X, test_X])
            train_y = np.concatenate([train_y, test_y])
        utils.set_seed(SEED)
        logger.info(f"seed set to {SEED}")

        sample_weights = compute_sample_weight(class_weight="balanced", y=train_y)
        rf_model = CalibratedClassifierCV(
            RandomForestClassifier(random_state=SEED), method="isotonic"
        )
        rf_model.fit(train_X, train_y, sample_weight=sample_weights)

        test_predprob = rf_model.predict_proba(test_X)[:, 1]
        test_pred = rf_model.predict(test_X)
        utils.clf_eval(test_y, test_pred, test_predprob, logger)
        logger.info(
            f"Briar score for the rf model: {brier_score_loss(test_y, test_predprob)}"
        )
        with open(STATE_PATH, "wb") as f:
            pickle.dump(rf_model, f)

        logger.info(f"saved trained model at:{STATE_PATH}")

        logger.warning(
            "Dumping rf labels to clf folder, randomly generated using calibrated probabilities"
        )
        train_rfpred = np.random.binomial(1, rf_model.predict_proba(train_X_og)[:, 1])
        test_rfpred = np.random.binomial(1, rf_model.predict_proba(test_X)[:, 1])

        np.save(f"{CLF_FOLDER}/train_labels.npy", train_rfpred)
        np.save(f"{CLF_FOLDER}/test_labels.npy", test_rfpred)
        logger.info(f"saved rf labels at: {CLF_FOLDER}")
    return rf_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to train XGB model")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
    parser.add_argument("--dataset", type=str, help="dataset to train on")
    parser.add_argument(
        "--train-on-test",
        action="store_true",
        help="flag to specify if model should be trained on test data",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="flag to specify if model should be retrained",
    )
    # parser.add_argument('--min-max', action='store_true', help='flag to specify if min-max scaling should be used')

    args = parser.parse_args()
    SEED = args.seed
    DATASET_STR = args.dataset
    TRAIN_ON_TEST = True
    FORCE_RETRAIN = args.force_retrain
    MIN_MAX = True
    rf_model = _train_rf(SEED, DATASET_STR, MIN_MAX, TRAIN_ON_TEST, FORCE_RETRAIN)
