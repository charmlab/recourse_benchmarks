import pandas as pd
import numpy as np
import math
import torch
import sys
from sklearn.datasets import (
    make_circles,
    make_moons,
)  # https://stackoverflow.com/questions/41467570/sklearn-doesnt-have-attribute-datasets
from sklearn.model_selection import train_test_split

sys.path.append("../")

import utils
import os


def binary_gaussian(
    mean_0, mean_1, cov_0, cov_1, prior_prob_1, N_SAMPLES=1000, seed=42
):
    utils.set_seed(seed)
    N_1 = np.random.binomial(N_SAMPLES, p=prior_prob_1)
    N_0 = N_SAMPLES - N_1

    X_0 = np.random.multivariate_normal(mean_0, cov_0, size=N_0)
    X_1 = np.random.multivariate_normal(mean_1, cov_1, size=N_1)
    X = np.vstack((X_0, X_1))
    y = np.ones((N_SAMPLES))
    y[:N_0] = 0
    return X, y


def uniformOverDisk(N_SAMP, r1=0, r2=1):
    """returns uniform distribution of points over disks of radius r1 and r2"""
    r = np.sqrt(np.random.uniform(r1**2, r2**2, N_SAMP))
    # sqrt needed because of change of formula:
    # https://stats.stackexchange.com/questions/481543/generating-random-points-uniformly-on-a-disk
    theta = np.random.uniform(0, 2 * np.pi, N_SAMP)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X_pos = np.stack((x1, x2)).T
    return X_pos


def uniformOverHalfDisk(N_SAMP, r1=0, r2=1):
    """returns uniform distribution of points over half disks of radius r1 and r2"""
    r = np.sqrt(np.random.uniform(r1**2, r2**2, N_SAMP))
    # sqrt needed because of change of formula:
    # https://stats.stackexchange.com/questions/481543/generating-random-points-uniformly-on-a-disk
    theta = np.random.uniform(0, np.pi, N_SAMP)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X_pos = np.stack((x1, x2)).T
    return X_pos


def load_dataset(
    DATASET,
    cust_labels_path=None,
    ret_tensor=False,
    min_max=False,
    DATA_DIR=os.path.join(os.path.dirname(__file__), '..', 'datasets'), # changed for recourse benchmark paths
    ret_norm_dict=False,
    ret_masks = False,
    **kwargs,
):

    immutable_idx = []
    cat_idx = []
# ADD THIS NEW SECTION at the beginning
    if DATASET == "compas-all-preprocessed":
        """
        Load preprocessed COMPASS data from recourse_benchmarks.
        Data is already one-hot encoded and normalized.
        """
        # Load CSV files
        train_df = pd.read_csv(DATA_DIR + "/compas/train.csv")
        test_df = pd.read_csv(DATA_DIR + "/compas/test.csv")
        
        # Expected columns: x0_ord_0, x0_ord_1, x0_ord_2, x1, x2, x3, x4, score
        target = "score"
        
        # Split features and target
        train_y = train_df[target]
        train_X = train_df.drop(columns=[target])
        test_y = test_df[target]
        test_X = test_df.drop(columns=[target])
        
        # Define categorical and immutable features by index
        # x0_ord_0, x0_ord_1, x0_ord_2 (indices 0,1,2) - age (immutable)
        # x1 (index 3) - race (immutable, categorical)
        # x2 (index 4) - sex (immutable, categorical)
        # x3 (index 5) - priors count (mutable, integer)
        # x4 (index 6) - charge degree (mutable, categorical)
        
        cat_idx = [0, 1, 2, 3, 4, 6]  # All except x3 (priors count)
        immutable_idx = [0, 1, 2, 3, 4]  # Age, race, sex
        
        # Convert to numpy
        train_X = train_X.to_numpy(dtype=np.float32)
        test_X = test_X.to_numpy(dtype=np.float32)
        train_y = train_y.to_numpy(dtype=np.float32)
        test_y = test_y.to_numpy(dtype=np.float32)
        
        # Data is already normalized to [0,1], no need to normalize again
        
    elif DATASET in ["adult-all", "compas-all","heloc"]:
        if DATASET == "adult-all":
            dataset = "adult"
            target = "income"
        elif DATASET == "compas-all":
            dataset = "compas"
            target = "score"
        elif DATASET == "heloc":
            dataset = "heloc"
            target = "RiskPerformance"

        # load the data
        train_df = pd.read_csv(DATA_DIR + f"/{dataset}/train.csv")
        test_df = pd.read_csv(DATA_DIR + f"/{dataset}/test.csv")

        cat_cols = {
            "compas": ["two_year_recid", "c_charge_degree", "race", "sex"],
            "adult": [
                "workclass",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ],
            "heloc": [],
        }

        immutable_cols = {
            "compas": ["race", "sex"],
            "adult": ["race","sex"],
            "heloc": [],
        }

        for i,col in enumerate(train_df.columns):
            if col in cat_cols[dataset]:
                cat_idx.append(i)

            if col in immutable_cols[dataset]:
                immutable_idx.append(i)

        train_df[cat_cols[dataset]] = train_df[cat_cols[dataset]].astype("category")
        test_df[cat_cols[dataset]] = test_df[cat_cols[dataset]].astype("category")
        for col_name in cat_cols[dataset]:
            assert (
                train_df[col_name].cat.categories is train_df[col_name].cat.categories
            )

        train_df[cat_cols[dataset]] = train_df[cat_cols[dataset]].apply(
            lambda x: x.cat.codes
        )
        test_df[cat_cols[dataset]] = test_df[cat_cols[dataset]].apply(
            lambda x: x.cat.codes
        )
        # target/covariate split
        train_y = train_df[target]
        train_X = train_df.drop(columns=[target])
        test_y = test_df[target]
        test_X = test_df.drop(columns=[target])

        # # normalize the data

        # WARNING: this is not std normalisation
        assert min_max, f"Only min max normalisation supported for dataset {dataset}"

        train_mean = train_X.min()
        train_std = (
            train_X.max() - train_X.min()
        )  # train_X.range(): dne in DataFrame object
        # # pandas gives column wise data only
        train_X = (train_X - train_mean) / train_std
        test_X = (test_X - train_mean) / train_std

        train_X = train_X.to_numpy(dtype=np.float32)
        test_X = test_X.to_numpy(dtype=np.float32)
        train_y = train_y.to_numpy(dtype=np.float32)
        test_y = test_y.to_numpy(dtype=np.float32)

    elif DATASET in ["circles"]:
        SEED = 42
        utils.set_seed(SEED)
        TEST_TRAIN_SPLIT = 0.1
        N_SAMPLES = 1000
        NOISE_STD = 0.04
        SIZE_RATIO = 0.7  # inner to outer circle ratio

        X, y = make_circles(
            n_samples=N_SAMPLES, noise=NOISE_STD, random_state=SEED, factor=SIZE_RATIO
        )
        X = np.float32(X)
        y = np.float32(y)
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=TEST_TRAIN_SPLIT, random_state=SEED
        )
        train_y = 1.0 * train_y
        test_y = 1.0 * test_y
        train_mean = np.mean(train_X, axis=0)
        train_std = np.std(train_X, axis=0)
        train_X = (train_X - train_mean) / train_std
        test_X = (test_X - train_mean) / train_std

    elif DATASET in ["moons"]:
        SEED = 42
        utils.set_seed(SEED)
        TEST_TRAIN_SPLIT = 0.1
        N_SAMPLES = 1000
        NOISE_STD = 0.04

        X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE_STD, random_state=SEED)
        X = np.float32(X)
        y = np.float32(y)
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=TEST_TRAIN_SPLIT, random_state=SEED
        )
        train_y = 1.0 * train_y
        test_y = 1.0 * test_y
        train_mean = np.mean(train_X, axis=0)
        train_std = np.std(train_X, axis=0)
        train_X = (train_X - train_mean) / train_std
        test_X = (test_X - train_mean) / train_std

    elif DATASET in ["corr"]:
        SEED = 42
        utils.set_seed(SEED)
        TEST_TRAIN_SPLIT = 0.1
        N_SAMPLES = 1000
        pho = 0.5

        def sample_x1(n: int):  # sample x1
            return np.random.rand(n) - 0.5

        def sample_x2__x1(n: int, x1):  # sample x2 given x1
            # return (2*(x1>0)*1.0 - 1)* np.random.rand(n)/2
            return np.tanh(20 * x1) * np.random.rand(n) / 2

        # true labeller
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def true_label(x1, x2):
            p = sigmoid((x2 + 60 * x1**3) / pho)
            return np.random.binomial(n=1, p=p) * 2.0 - 1

        np.random.seed(SEED)

        X1 = sample_x1(N_SAMPLES)
        X2 = sample_x2__x1(N_SAMPLES, X1)
        X = np.c_[X1, X2]
        y = (true_label(X1, X2) > 0) * 1.0

        X = np.float32(X)
        y = np.float32(y)
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=TEST_TRAIN_SPLIT, random_state=SEED
        )
        train_y = 1.0 * train_y
        test_y = 1.0 * test_y
        train_mean = np.mean(train_X, axis=0)
        train_std = np.std(train_X, axis=0)
        train_X = (train_X - train_mean) / train_std
        test_X = (test_X - train_mean) / train_std

    else:
        raise ValueError(
            f"Invalid dataset name: {DATASET} is not one of the recognized dataset"
        )

    if cust_labels_path is not None:
        # raise NotImplementedError
        cust_train_y = np.load(f"{cust_labels_path}/train_labels.npy")
        cust_test_y = np.load(f"{cust_labels_path}/test_labels.npy")
        assert train_y.shape == cust_train_y.shape
        assert test_y.shape == cust_test_y.shape
        train_y = np.float32(cust_train_y * 1.0)
        test_y = np.float32(cust_test_y * 1.0)

    if ret_norm_dict:
        norm_dict = {}
        if not isinstance(train_mean, np.ndarray):
            train_mean = train_mean.to_numpy(np.float32)
            train_std = train_std.to_numpy(np.float32)
        norm_dict["mean"] = train_mean
        norm_dict["std"] = train_std

    if min_max:
        train_min = np.min(train_X, axis=0)
        train_range = np.max(train_X, axis=0) - train_min
        train_X = (train_X - train_min) / train_range
        test_X = (test_X - train_min) / train_range
        if ret_norm_dict:
            norm_dict["min"] = train_min
            norm_dict["range"] = train_range

    if ret_tensor:
        train_y = torch.from_numpy(train_y)
        train_X = torch.from_numpy(train_X)
        test_y = torch.from_numpy(test_y)
        test_X = torch.from_numpy(test_X)

    immutable_mask = [False]*train_X.shape[1]
    for idx in immutable_idx: immutable_mask[idx] = True

    cat_mask = [False]*train_X.shape[1] 
    for idx in cat_idx: cat_mask[idx] = True

    if ret_masks:
        assert not(ret_norm_dict)
        return train_y, train_X, test_y, test_X, cat_mask, immutable_mask

    if ret_norm_dict:
        return train_y, train_X, test_y, test_X, norm_dict
    else:
        return train_y, train_X, test_y, test_X


def circles_bayes():
    def density_normal_sampled(X,noise_std):
        # X is nxD
        def density(x1,x2):
            x = np.array([x1,x2]).reshape(1,-1)
            return (np.exp(-((X-x)**2).sum(axis=1)/(2*noise_std**2))/(noise_std*math.sqrt(2*math.pi))).mean()
        return density

    factor = 0.7
    n_samples_out = 500
    n_samples_in = 500
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circ_x = np.cos(linspace_out)
    outer_circ_y = np.sin(linspace_out)
    inner_circ_x = np.cos(linspace_in) * factor
    inner_circ_y = np.sin(linspace_in) * factor

    X_neg = np.vstack([outer_circ_x, outer_circ_y]).T
    X_pos = np.vstack([inner_circ_x, inner_circ_y]).T
    pos_prob = 0.5
    pos_density = density_normal_sampled(X_pos,0.04)
    neg_density = density_normal_sampled(X_neg,0.04)
    def bayesfunc(x1,x2, eps):
        return (pos_prob*(eps+pos_density(x1,x2)))/(eps + pos_prob*pos_density(x1,x2) + (1 - pos_prob)*neg_density(x1,x2))
    return bayesfunc

def moons_bayes():
    def density_normal_sampled(X,noise_std):
        # X is nxD
        def density(x1,x2):
            x = np.array([x1,x2]).reshape(1,-1)
            return (np.exp(-((X-x)**2).sum(axis=1)/(2*noise_std**2))/(noise_std*math.sqrt(2*math.pi))).mean()
        return density

    n_samples_out = 500
    n_samples_in = 500
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5


    X_neg = np.vstack([outer_circ_x, outer_circ_y]).T
    X_pos = np.vstack([inner_circ_x, inner_circ_y]).T
    pos_prob = 0.5
    pos_density = density_normal_sampled(X_pos,0.04)
    neg_density = density_normal_sampled(X_neg,0.04)
    def bayesfunc(x1,x2, eps):
        return (pos_prob*(eps+pos_density(x1,x2)))/(eps + pos_prob*pos_density(x1,x2) + (1 - pos_prob)*neg_density(x1,x2))
    return bayesfunc

def corr_bayes_density():
    def uniform_over_interval(start, end):
        def func(x):
            if start<x<end:
                return 1/(end - start)
            else:
                return 0
        return func


    def sigmoid(x):
        return 1/(1+math.exp(-x))

    def bayesfunc(x1, x2, *args):
        # ignores eps
        return sigmoid((x2 + 60*x1**3)/0.5)

    # def density(x1,x2):
    #     unif = uniform_over_interval(0,1)
    #     return unif(x2/math.tanh(x1))*unif(x1) 

    def density(x1,x2):
        tol = 0
        u_x2 = uniform_over_interval(0,1)
        u_x1 = uniform_over_interval(-0.5,0.5)
        factor = (math.tanh(20*x1)+tol)/2
        return (u_x2(x2/(factor))/factor)*u_x1(x1)
    
    return bayesfunc, density