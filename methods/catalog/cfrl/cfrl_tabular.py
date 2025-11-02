from functools import partial
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from .cfrl_backend import split_ohe
from .cfrl_base import CounterfactualRL, Postprocessing


def get_conditional_dim(
    feature_names: List[str], category_map: Dict[int, List[str]]
) -> int:
    """
    Computes the dimension of the conditional vector.

    Parameters
    ----------
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values. This should be provided by the dataset.

    Returns
    -------
    Dimension of the conditional vector
    """
    cat_feat = int(np.sum([len(vals) for vals in category_map.values()]))
    num_feat = len(feature_names) - len(category_map)
    return 2 * num_feat + cat_feat


def generate_numerical_condition(
    X_ohe: np.ndarray,
    feature_names: List[str],
    category_map: Dict[int, List[str]],
    ranges: Dict[str, List[float]],
    immutable_features: List[str],
    conditional: bool = True,
) -> np.ndarray:
    """
    Generates numerical features conditional vector. For numerical features with a minimum value `a_min` and a
    maximum value `a_max`, we include in the conditional vector the values `-p_min`, `p_max`, where `p_min, p_max`
    are in [0, 1]. The range `[-p_min, p_max]` encodes a shift and scale-invariant representation of the interval
    `[a - p_min * (a_max - a_min), a + p_max * (a_max - a_min)], where `a` is the original feature value. During
    training, `p_min` and `p_max` are sampled from `Beta(2, 2)` for each unconstrained feature. Immutable features
    can be encoded by `p_min = p_max = 0` or listed in `immutable_features` list. Features allowed to increase or
    decrease only correspond to setting `p_min = 0` or `p_max = 0`, respectively. For example, allowing the ``'Age'``
    feature to increase by up to 5 years is encoded by taking `p_min = 0`, `p_max=0.1`, assuming the minimum age of
    10 and the maximum age of 60 years in the training set: `5 = 0.1 * (60 - 10)`.

    Parameters
    ----------
    X_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        This argument is used to extract the number of conditional vector. The choice of `X_ohe` instead of a
        `size` argument is for consistency purposes with `categorical_cond` function.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.
    ranges:
        Dictionary of ranges for numerical features. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        Dictionary of immutable features. The keys are the column indexes and the values are booleans: ``True`` if
        the feature is immutable, ``False`` otherwise.
    conditional
        Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any
        restrictions on the feature value.

    Returns
    -------
    Conditional vector for numerical features.
    """
    num_cond = []
    size = X_ohe.shape[0]

    for feature_id, feature_name in enumerate(feature_names):
        # skip categorical features
        if feature_id in category_map:
            continue

        if feature_name in immutable_features:
            # immutable feature
            range_low, range_high = 0.0, 0.0
        else:
            range_low = ranges[feature_name][0] if feature_name in ranges else -1.0
            range_high = ranges[feature_name][1] if feature_name in ranges else 1.0

        # Check if the ranges are valid.
        if range_low > 0:
            raise ValueError(
                f"Lower bound range for {feature_name} should be negative."
            )
        if range_high < 0:
            raise ValueError(
                f"Upper bound range for {feature_name} should be positive."
            )

        # Generate lower and upper bound coefficients.
        coeff_lower = (
            np.random.beta(  # pyright: ignore[reportAttributeAccessIssue]
                a=2, b=2, size=size
            ).reshape(
                -1, 1
            )
            if conditional
            else np.ones((size, 1))
        )
        coeff_upper = (
            np.random.beta(  # pyright: ignore[reportAttributeAccessIssue]
                a=2, b=2, size=size
            ).reshape(
                -1, 1
            )
            if conditional
            else np.ones((size, 1))
        )

        # Generate lower and upper bound conditionals.
        num_cond.append(coeff_lower * range_low)
        num_cond.append(coeff_upper * range_high)

    # Construct numerical conditional vector by concatenating all numerical conditions.
    return np.concatenate(num_cond, axis=1)


def generate_categorical_condition(
    X_ohe: np.ndarray,
    feature_names: List[str],
    category_map: Dict[int, List],
    immutable_features: List[str],
    conditional: bool = True,
) -> np.ndarray:
    """
    Generates categorical features conditional vector. For a categorical feature of cardinality `K`, we condition the
    subset of allowed feature through a binary mask of dimension `K`. When training the counterfactual generator,
    the mask values are sampled from `Bern(0.5)`. For immutable features, only the original input feature value is
    set to one in the binary mask. For example, the immutability of the ``'marital_status'`` having the current
    value ``'married'`` is encoded through the binary sequence [1, 0, 0], given an ordering of the possible feature
    values `[married, unmarried, divorced]`.

    Parameters
    ----------
    X_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        The elements are required since some features can be immutable. In that case, the mask vector is the
        one-hot encoding itself for that particular feature.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.
    immutable_features
        List of immutable features.
    conditional
        Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any
        restrictions on the feature value.

    Returns
    -------
    Conditional vector for categorical feature.
    """

    C_cat = []  # define list of conditional vector for each feature
    cat_idx = 0  # categorical feature index

    # Split the one-hot representation into a list where each element corresponds to an feature.
    _, X_ohe_cat_split = split_ohe(X_ohe, category_map)

    # Create mask for each categorical column.
    for feature_id, feature_name in enumerate(feature_names):
        # Skip numerical features
        if feature_id not in category_map:
            continue

        # Initialize mask with the original value
        mask = X_ohe_cat_split[cat_idx].copy()

        # If the feature is not immutable, add noise to modify the mask
        if feature_name not in immutable_features:
            mask += (
                np.random.rand(*mask.shape)  # pyright: ignore[reportAttributeAccessIssue]
                if conditional
                else np.ones_like(mask)
            )

        # Construct binary mask
        mask = (mask > 0.5).astype(np.float32)
        C_cat.append(mask)

        # Move to the next categorical index
        cat_idx += 1

    return np.concatenate(C_cat, axis=1)


def generate_condition(
    X_ohe: np.ndarray,
    feature_names: List[str],
    category_map: Dict[int, List[str]],
    ranges: Dict[str, List[float]],
    immutable_features: List[str],
    conditional: bool = True,
) -> np.ndarray:
    """
    Generates conditional vector.

    Parameters
    ----------
    X_ohe
        One-hot encoding representation of the element(s) for which the conditional vector will be generated.
        This method assumes that the input array, `X_ohe`, is has the first columns corresponding to the
        numerical features, and the rest are one-hot encodings of the categorical columns. The numerical and the
        categorical columns are ordered by the original column index( e.g., `numerical = (1, 4)`,
        `categorical=(0, 2, 3)`).
    feature_names
        List of feature names.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.
    ranges
        Dictionary of ranges for numerical features. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        List of immutable map features.
    conditional
        Boolean flag to generate a conditional vector. If ``False`` the conditional vector does not impose any
        restrictions on the feature value.

    Returns
    -------
    Conditional vector.
    """
    # Define conditional vector buffer
    C = []

    # Generate numerical condition vector.
    if len(feature_names) > len(category_map):
        C_num = generate_numerical_condition(
            X_ohe=X_ohe,
            feature_names=feature_names,
            category_map=category_map,
            ranges=ranges,
            immutable_features=immutable_features,
            conditional=conditional,
        )
        C.append(C_num)

    # Generate categorical condition vector.
    if len(category_map):
        C_cat = generate_categorical_condition(
            X_ohe=X_ohe,
            feature_names=feature_names,
            category_map=category_map,
            immutable_features=immutable_features,
            conditional=conditional,
        )
        C.append(C_cat)

    # Concatenate numerical and categorical conditional vectors.
    return np.concatenate(C, axis=1)


def sample_numerical(
    X_hat_num_split: List[np.ndarray],
    X_ohe_num_split: List[np.ndarray],
    C_num_split: Optional[List[np.ndarray]],
    stats: Dict[int, Dict[str, float]],
) -> List[np.ndarray]:
    """
    Samples numerical features according to the conditional vector. This method clips the values between the
    desired ranges specified in the conditional vector, and ensures that the values are between the minimum and
    the maximum values from train training datasets stored in the dictionary of statistics.

    Parameters
    ----------
    X_hat_num_split
        List of reconstructed numerical heads from the auto-encoder. This list should contain a single element
        as all the numerical features are part of a singe linear layer output.
    X_ohe_num_split
        List of original numerical heads. The list should contain a single element as part of the convention
        mentioned in the description of `X_ohe_hat_num`.
    C_num_split
        List of conditional vector for numerical heads. The list should contain a single element as part of the
        convention mentioned in the description of `X_ohe_hat_num`.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing ``'min'`` and ``'max'`` keys.

    Returns
    -------
    X_ohe_hat_num
        List of clamped input vectors according to the conditional vectors and the dictionary of statistics.
    """
    num_cols = X_hat_num_split[0].shape[1]  # number of numerical columns
    sorted_cols = sorted(stats.keys())  # ensure that the column ids are sorted

    for i, col_id in zip(range(num_cols), sorted_cols):
        # Extract the minimum and the maximum value for the current column from the training set.
        min, max = stats[col_id]["min"], stats[col_id]["max"]

        if C_num_split is not None:
            # Extract the minimum and the maximum value according to the conditional vector.
            lhs = X_ohe_num_split[0][:, i] + C_num_split[0][:, 2 * i] * (max - min)
            rhs = X_ohe_num_split[0][:, i] + C_num_split[0][:, 2 * i + 1] * (max - min)

            # Clamp output according to the conditional vector.
            X_hat_num_split[0][:, i] = np.clip(
                X_hat_num_split[0][:, i], a_min=lhs, a_max=rhs
            )

        # Clamp output according to the minimum and maximum value from the training set.
        X_hat_num_split[0][:, i] = np.clip(
            X_hat_num_split[0][:, i], a_min=min, a_max=max
        )

    return X_hat_num_split


def sample_categorical(
    X_hat_cat_split: List[np.ndarray], C_cat_split: Optional[List[np.ndarray]]
) -> List[np.ndarray]:
    """
    Samples categorical features according to the conditional vector. This method sample conditional according to
    the masking vector the most probable outcome.

    Parameters
    ----------
    X_hat_cat_split
        List of reconstructed categorical heads from the auto-encoder. The categorical columns contain logits.
    C_cat_split
        List of conditional vector for categorical heads.

    Returns
    -------
    X_ohe_hat_cat
        List of one-hot encoded vectors sampled according to the conditional vector.
    """
    X_ohe_hat_cat = []  # initialize the returning list
    rows = np.arange(X_hat_cat_split[0].shape[0])  # initialize the returning list

    for i in range(len(X_hat_cat_split)):
        # compute probability distribution
        proba = softmax(X_hat_cat_split[i], axis=1)
        proba = proba * C_cat_split[i] if (C_cat_split is not None) else proba

        # sample the most probable outcome conditioned on the conditional vector
        cols = np.argmax(proba, axis=1)
        samples = np.zeros_like(proba)
        samples[rows, cols] = 1  # pyright: ignore[reportIndexIssue]
        X_ohe_hat_cat.append(samples)

    return X_ohe_hat_cat


def sample(
    X_hat_split: List[np.ndarray],
    X_ohe: np.ndarray,
    C: Optional[np.ndarray],
    category_map: Dict[int, List[str]],
    stats: Dict[int, Dict[str, float]],
) -> List[np.ndarray]:
    """
    Samples an instance from the given reconstruction according to the conditional vector and
    the dictionary of statistics.

    Parameters
    ----------
    X_hat_split
        List of reconstructed columns from the auto-encoder. The categorical columns contain logits.
    X_ohe
        One-hot encoded representation of the input.
    C
        Conditional vector.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for a feature.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing ``'min'`` and ``'max'`` keys.

    Returns
    -------
    X_ohe_hat_split
        Most probable reconstruction sample according to the auto-encoder, sampled according to the conditional vector
        and the dictionary of statistics. This method assumes that the input array, `X_ohe` , has the first columns
        corresponding to the numerical features, and the rest are one-hot encodings of the categorical columns.
    """
    X_ohe_num_split, X_ohe_cat_split = split_ohe(X_ohe, category_map)
    C_num_split, C_cat_split = (
        split_ohe(C, category_map) if (C is not None) else (None, None)
    )

    X_ohe_hat_split = (
        []
    )  # list of sampled numerical columns and sampled categorical columns
    num_feat, cat_feat = len(X_ohe_num_split), len(X_ohe_cat_split)

    if num_feat > 0:
        # Sample numerical columns
        X_ohe_hat_split += sample_numerical(
            X_hat_num_split=X_hat_split[:num_feat],
            X_ohe_num_split=X_ohe_num_split,
            C_num_split=C_num_split,
            stats=stats,
        )

    if cat_feat > 0:
        # Sample categorical columns
        X_ohe_hat_split += sample_categorical(
            X_hat_cat_split=X_hat_split[-cat_feat:], C_cat_split=C_cat_split
        )

    return X_ohe_hat_split


def get_he_preprocessor(
    X: np.ndarray,
    feature_names: List[str],
    category_map: Dict[int, List[str]],
    feature_types: Optional[Dict[str, type]] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Heterogeneous dataset preprocessor. The numerical features are standardized and the categorical features
    are one-hot encoded.

    Parameters
    ----------
    X
        Data to fit.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values. This should be provided by the dataset.
    feature_types
        Dictionary of type for the numerical features.

    Returns
    -------
    preprocessor
        Data preprocessor.
    inv_preprocessor
        Inverse data preprocessor (e.g., `inv_preprocessor(preprocessor(x)) = x` )
    """
    if feature_types is None:
        feature_types = dict()

    # Separate columns in numerical and categorical
    categorical_ids = list(category_map.keys())
    numerical_ids = [
        i for i in range(len(feature_names)) if i not in category_map.keys()
    ]

    # Define standard scaler and one-hot encoding transformations
    num_transf = StandardScaler()
    cat_transf = OneHotEncoder(
        categories=[
            range(len(x)) for x in category_map.values()
        ],  # pyright: ignore[reportArgumentType]
        handle_unknown="ignore",
    )

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transf, numerical_ids),
            ("cat", cat_transf, categorical_ids),
        ],
        sparse_threshold=0,
    )
    preprocessor.fit(X)

    num_feat_ohe = len(numerical_ids)  # number of numerical columns
    cat_feat_ohe = sum(
        [len(v) for v in category_map.values()]
    )  # number of categorical columns

    # Define inverse preprocessor
    def get_inv_preprocessor(X_ohe: np.ndarray):
        X_inv = []

        if "num" in preprocessor.named_transformers_ and len(numerical_ids):
            num_transf = preprocessor.named_transformers_["num"]
            X_ohe_num = (
                X_ohe[:, :num_feat_ohe]
                if preprocessor.transformers[0][0] == "num"
                else X_ohe[:, -num_feat_ohe:]
            )
            X_inv.append(num_transf.inverse_transform(X_ohe_num))

        if "cat" in preprocessor.named_transformers_ and len(categorical_ids):
            cat_transf = preprocessor.named_transformers_["cat"]
            X_ohe_cat = (
                X_ohe[:, :cat_feat_ohe]
                if preprocessor.transformers[0][0] == "cat"
                else X_ohe[:, -cat_feat_ohe:]
            )
            X_inv.append(cat_transf.inverse_transform(X_ohe_cat))

        # Concatenate all columns. At this point the columns are not ordered correctly
        np_X_inv = np.concatenate(X_inv, axis=1)

        # Construct permutation to order the columns correctly
        perm = [i for i in range(len(feature_names)) if i not in category_map.keys()]
        perm += [i for i in range(len(feature_names)) if i in category_map.keys()]

        inv_perm = [0] * len(perm)
        for i in range(len(perm)):
            inv_perm[perm[i]] = i

        # Permute columns and cast to object
        np_X_inv = np_X_inv[
            :, inv_perm
        ].astype(  # pyright: ignore[reportCallIssue, reportArgumentType]
            object
        )

        # Cast numerical features to desired data types
        for i, fn in enumerate(feature_names):
            if i in numerical_ids:
                ft_type = feature_types[fn] if fn in feature_types else float

                # Round `int` type features to the closest integer number to avoid rounding error when casting to `int`.
                # The casting to `np.float32` is due to previous casting to `object` which raises an error when
                # applying `np.rint` (i.e., 'TypeError: loop of ufunc does not support argument 0 of type float which
                # has no callable rint method')
                if ft_type == int:
                    np_X_inv[:, i] = np.rint(
                        np_X_inv[:, i].astype(np.float32)
                    )  # pyright: ignore[reportCallIssue]

            else:
                ft_type = int  # for categorical features

            np_X_inv[:, i] = np_X_inv[:, i].astype(ft_type)

        return np_X_inv

    return preprocessor.transform, get_inv_preprocessor


def get_statistics(
    X: np.ndarray,
    preprocessor: Callable[[np.ndarray], np.ndarray],
    category_map: Dict[int, List[str]],
) -> Dict[int, Dict[str, float]]:
    """
    Computes statistics.

    Parameters
    ----------
    X
        Instances for which to compute statistic.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values. This should be provided by the dataset.

    Returns
    -------
    Dictionary of statistics. For each numerical column, the minimum and maximum value is returned.
    """
    stats = dict()

    # Extract numerical features
    num_features_ids = [id for id in range(X.shape[1]) if id not in category_map]

    # Preprocess data (standardize + one-hot encoding)
    X_ohe = preprocessor(X)

    for i, feature_id in enumerate(num_features_ids):
        min, max = np.min(X_ohe[:, i]), np.max(X_ohe[:, i])
        stats[feature_id] = {"min": min, "max": max}

    return stats


def get_numerical_conditional_vector(
    X: np.ndarray,
    condition: Dict[str, List[Union[float, str]]],
    preprocessor: Callable[[np.ndarray], np.ndarray],
    feature_names: List[str],
    category_map: Dict[int, List[str]],
    stats: Dict[int, Dict[str, float]],
    ranges: Optional[Dict[str, List[float]]] = None,
    immutable_features: Optional[List[str]] = None,
    diverse=False,
) -> List[np.ndarray]:
    """
    Generates a conditional vector. The condition is expressed a a delta change of the feature.
    For numerical features, if the ``'Age'`` feature is allowed to increase up to 10 more years, the delta change is
    [0, 10].  If the ``'Hours per week'`` is allowed to decrease down to -5 and increases up to +10, then the
    delta change is [-5, +10]. Note that the interval must go include 0.

    Parameters
    ----------
    X
        Instances for which to generate the conditional vector in the original input format.
    condition
        Dictionary of conditions per feature. For numerical features it expects a range that contains the original
        value. For categorical features it expects a list of feature values per features that includes the original
        value.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values. This should be provided by the dataset.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing ``'min'`` and ``'max'`` keys.
    ranges
        Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        List of immutable features.
    diverse
        Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
        a diverse set of counterfactuals for a given input instance.

    Returns
    -------
    List of conditional vectors for each numerical feature.
    """
    if ranges is None:
        ranges = dict()

    if immutable_features is None:
        immutable_features = list()

    # Extract numerical features
    num_features_ids = [id for id in range(X.shape[1]) if id not in category_map]
    num_features_names = [feature_names[id] for id in num_features_ids]

    # Need to standardize numerical features. Thus, we use the preprocessor
    X_low, X_high = X.copy(), X.copy()

    for feature_id, feature_name in enumerate(feature_names):
        if feature_id in category_map:
            continue

        if feature_name in condition:
            if (
                int(condition[feature_name][0]) > 0
            ):  # int conversion because of mypy error (the value can be str too)
                raise ValueError(
                    f"Lower bound on the conditional vector for {feature_name} should be negative."
                )

            if (
                int(condition[feature_name][1]) < 0
            ):  # int conversion because of mypy error (the value can be str too)
                raise ValueError(
                    f"Upper bound on the conditional vector for {feature_name} should be positive."
                )

            X_low[:, feature_id] += condition[feature_name][0]
            X_high[:, feature_id] += condition[feature_name][1]

    # Preprocess the vectors (standardize + one-hot encoding)
    X_low_ohe = preprocessor(X_low)
    X_high_ohe = preprocessor(X_high)
    X_ohe = preprocessor(X)

    # Initialize conditional vector buffer.
    C = []

    # Scale the numerical features in [0, 1] and add them to the conditional vector
    for i, (feature_id, feature_name) in enumerate(
        zip(num_features_ids, num_features_names)
    ):
        if feature_name in immutable_features:
            range_low, range_high = 0.0, 0.0
        elif feature_name in ranges:
            range_low, range_high = ranges[feature_name][0], ranges[feature_name][1]
        else:
            range_low, range_high = -1.0, 1.0

        if (feature_name in condition) and (feature_name not in immutable_features):
            # Mutable feature with conditioning
            min, max = stats[feature_id]["min"], stats[feature_id]["max"]
            X_low_ohe[:, i] = (X_low_ohe[:, i] - X_ohe[:, i]) / (max - min)
            X_high_ohe[:, i] = (X_high_ohe[:, i] - X_ohe[:, i]) / (max - min)

            # Clip in [0, 1]
            X_low_ohe[:, i] = np.clip(X_low_ohe[:, i], a_min=range_low, a_max=0)
            X_high_ohe[:, i] = np.clip(X_high_ohe[:, i], a_min=0, a_max=range_high)
        else:
            # This means no conditioning
            X_low_ohe[:, i] = range_low
            X_high_ohe[:, i] = range_high

        if diverse:
            # Note that this is still a feasible counterfactual
            X_low_ohe[
                :, i
            ] *= np.random.rand(  # pyright: ignore[reportAttributeAccessIssue]
                *X_low_ohe[:, i].shape
            )
            X_high_ohe[
                :, i
            ] *= np.random.rand(  # pyright: ignore[reportAttributeAccessIssue]
                *X_high_ohe[:, i].shape
            )

        # Append feature conditioning
        C += [X_low_ohe[:, i].reshape(-1, 1), X_high_ohe[:, i].reshape(-1, 1)]

    return C


def get_categorical_conditional_vector(
    X: np.ndarray,
    condition: Dict[str, List[Union[float, str]]],
    preprocessor: Callable[[np.ndarray], np.ndarray],
    feature_names: List[str],
    category_map: Dict[int, List[str]],
    immutable_features: Optional[List[str]] = None,
    diverse=False,
) -> List[np.ndarray]:
    """
    Generates a conditional vector. The condition is expressed a a delta change of the feature.
    For categorical feature, if the ``'Occupation'`` can change to ``'Blue-Collar'`` or ``'White-Collar'``, the delta
    change is ``['Blue-Collar', 'White-Collar']``. Note that the original value is optional as it is
    included by default.

    Parameters
    ----------
    X
        Instances for which to generate the conditional vector in the original input format.
    condition
        Dictionary of conditions per feature. For numerical features it expects a range that contains the original
        value. For categorical features it expects a list of feature values per features that includes the original
        value.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.  This should be provided by the dataset.
    immutable_features
        List of immutable features.
    diverse
        Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
        a diverse set of counterfactuals for a given input instance.

    Returns
    -------
    List of conditional vectors for each categorical feature.
    """
    if immutable_features is None:
        immutable_features = list()

    # Define conditional vector buffer
    C = []

    # extract categorical features
    cat_features_ids = [id for id in range(X.shape[1]) if id in category_map]
    cat_feature_names = [feature_names[id] for id in cat_features_ids]

    # Extract list of categorical one-hot encoded columns
    X_ohe = preprocessor(X)
    _, X_ohe_cat_split = split_ohe(X_ohe, category_map)

    # For each categorical feature add the masking vector
    for i, (feature_id, feature_name) in enumerate(
        zip(cat_features_ids, cat_feature_names)
    ):
        mask = np.zeros_like(X_ohe_cat_split[i])

        if feature_name not in immutable_features:
            if feature_name in condition:
                indexes = [
                    category_map[feature_id].index(str(feature_value))
                    for feature_value in condition[feature_name]
                ]  # conversion to str because of mypy (can be also float)
                mask[:, indexes] = 1  # pyright: ignore[reportIndexIssue]
            else:
                # Allow any value
                mask[:] = 1  # pyright: ignore[reportIndexIssue]

        if diverse:
            # Note that by masking random entries we still have a feasible counterfactual
            mask *= np.random.randint(  # pyright: ignore[reportAttributeAccessIssue]
                low=0,
                high=2,
                size=mask.shape,  # pyright: ignore[reportAttributeAccessIssue]
            )

        # Ensure that the original value is a possibility
        mask = ((mask + X_ohe_cat_split[i]) > 0).astype(int)

        # Append feature conditioning
        C.append(mask)
    return C


def get_conditional_vector(
    X: np.ndarray,
    condition: Dict[str, List[Union[float, str]]],
    preprocessor: Callable[[np.ndarray], np.ndarray],
    feature_names: List[str],
    category_map: Dict[int, List[str]],
    stats: Dict[int, Dict[str, float]],
    ranges: Optional[Dict[str, List[float]]] = None,
    immutable_features: Optional[List[str]] = None,
    diverse=False,
) -> np.ndarray:
    """
    Generates a conditional vector. The condition is expressed a a delta change of the feature.

    For numerical features, if the ``'Age'`` feature is allowed to increase up to 10 more years, the delta change is
    [0, 10].  If the ``'Hours per week'`` is allowed to decrease down to -5 and increases up to +10, then the
    delta change is [-5, +10]. Note that the interval must go include 0.

    For categorical feature, if the ``'Occupation'`` can change to ``'Blue-Collar'`` or ``'White-Collar'``,
    the delta change is ``['Blue-Collar', 'White-Collar']``. Note that the original value is optional as it is
    included by default.

    Parameters
    ----------
    X
        Instances for which to generate the conditional vector in the original input format.
    condition
        Dictionary of conditions per feature. For numerical features it expects a range that contains the original
        value. For categorical features it expects a list of feature values per features that includes the original
        value.
    preprocessor
        Data preprocessor. The preprocessor should standardize the numerical values and convert categorical ones
        into one-hot encoding representation. By convention, numerical features should be first, followed by the
        rest of categorical ones.
    feature_names
        List of feature names. This should be provided by the dataset.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the
        possible feature values.  This should be provided by the dataset.
    stats
        Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
        feature in the training set. Each key is an index of the column and each value is another dictionary
        containing ``'min'`` and ``'max'`` keys.
    ranges
        Dictionary of ranges for numerical feature. Each value is a list containing two elements, first one
        negative and the second one positive.
    immutable_features
        List of immutable features.
    diverse
        Whether to generate a diverse set of conditional vectors. A diverse set of conditional vector can generate
        a diverse set of counterfactuals for a given input instance.

    Returns
    -------
    Conditional vector.
    """
    if ranges is None:
        ranges = dict()

    if immutable_features is None:
        immutable_features = list()

    # Reshape the vector.
    X = X.reshape(1, -1) if len(X.shape) == 1 else X

    # Check that the second dimension matches the number of features.
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Unexpected number of features. The expected number "
            f"is {len(feature_names)}, but the input has {X.shape[1]} features."
        )

    # Get list of numerical conditional vectors.
    C_num = get_numerical_conditional_vector(
        X=X,
        condition=condition,
        preprocessor=preprocessor,
        feature_names=feature_names,
        category_map=category_map,
        stats=stats,
        ranges=ranges,
        immutable_features=immutable_features,
        diverse=diverse,
    )

    # Get list of categorical conditional vectors.
    C_cat = get_categorical_conditional_vector(
        X=X,
        condition=condition,
        preprocessor=preprocessor,
        feature_names=feature_names,
        category_map=category_map,
        immutable_features=immutable_features,
        diverse=diverse,
    )

    # concat all conditioning
    return np.concatenate(C_num + C_cat, axis=1)


def apply_category_mapping(
    X: np.ndarray, category_map: Dict[int, List[str]]
) -> np.ndarray:
    """
    Applies a category mapping for the categorical feature in the array. It transforms ints back to strings
    to be readable.

    Parameters
    -----------
    X
        Array containing the columns to be mapped.
    category_map
        Dictionary of category mapping. Keys are columns index, and values are list of feature values.

    Returns
    -------
    Transformed array.
    """
    pd_X = pd.DataFrame(X)

    for key in category_map:
        pd_X[key].replace(  # pyright: ignore[reportOptionalMemberAccess]
            range(len(category_map[key])), category_map[key], inplace=True
        )

    return pd_X.to_numpy()


class SampleTabularPostprocessing(Postprocessing):
    """
    Tabular sampling post-processing. Given the output of the heterogeneous auto-encoder the post-processing
    functions samples the output according to the conditional vector. Note that the original input instance
    is required to perform the conditional sampling.
    """

    def __init__(
        self, category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]]
    ):
        """
        Constructor.

        Parameters
        ----------
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the
            possible feature values.
        stats
            Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
            feature in the training set. Each key is an index of the column and each value is another dictionary
            containing ``'min'`` and ``'max'`` keys.
        """
        super().__init__()
        self.category_map = category_map
        self.stats = stats

    def __call__(
        self, X_cf: List[np.ndarray], X: np.ndarray, C: Optional[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Performs counterfactual conditional sampling according to the conditional vector and the original input.

        Parameters
        ----------
        X_cf
            Decoder reconstruction of the counterfactual instance. The decoded instance is a list where each
            element in the list correspond to the reconstruction of a feature.
        X
            Input instance.
        C
            Conditional vector.

        Returns
        -------
        Conditional sampled counterfactual instance.
        """
        return sample(
            X_hat_split=X_cf,
            X_ohe=X,
            C=C,
            stats=self.stats,
            category_map=self.category_map,
        )


class ConcatTabularPostprocessing(Postprocessing):
    """Tabular feature columns concatenation post-processing."""

    def __call__(
        self, X_cf: List[np.ndarray], X: np.ndarray, C: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Performs a concatenation of the counterfactual feature columns along the axis 1.

        Parameters
        ----------
        X_cf
            List of counterfactual feature columns.
        X
            Input instance. Not used. Included for consistency.
        C
            Conditional vector. Not used. Included for consistency.

        Returns
        -------
        Concatenation of the counterfactual feature columns.
        """
        return np.concatenate(X_cf, axis=1)


class CounterfactualRLTabular(CounterfactualRL):
    """Counterfactual Reinforcement Learning Tabular."""

    def __init__(
        self,
        predictor: Callable[[np.ndarray], np.ndarray],
        encoder: "torch.nn.Module",  # pyright: ignore[reportAttributeAccessIssue]
        decoder: "torch.nn.Module",  # pyright: ignore[reportAttributeAccessIssue]
        encoder_preprocessor: Callable,
        decoder_inv_preprocessor: Callable,
        coeff_sparsity: float,
        coeff_consistency: float,
        feature_names: List[str],
        category_map: Dict[int, List[str]],
        immutable_features: Optional[List[str]] = None,
        ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        weight_num: float = 1.0,
        weight_cat: float = 1.0,
        latent_dim: Optional[int] = None,
        seed: int = 0,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        predictor
            A callable that takes a `numpy` array of `N` data points as inputs and returns `N` outputs. For
            classification task, the second dimension of the output should match the number of classes. Thus, the
            output can be either a soft label distribution or a hard label distribution (i.e. one-hot encoding)
            without affecting the performance since `argmax` is applied to the predictor's output.
        encoder
            Pretrained heterogeneous encoder network.
        decoder
            Pretrained heterogeneous decoder network. The output of the decoder must be a list of tensors.
        encoder_preprocessor
            Auto-encoder data pre-processor. Depending on the input format, the pre-processor can normalize
            numerical attributes, transform label encoding to one-hot encoding etc.
        decoder_inv_preprocessor
            Auto-encoder data inverse pre-processor. This is the inverse function of the pre-processor. It can
            denormalize numerical attributes, transform one-hot encoding to label encoding, feature type casting etc.
        coeff_sparsity
           Sparsity loss coefficient.
        coeff_consistency
           Consistency loss coefficient.
        feature_names
            List of feature names. This should be provided by the dataset.
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
            values for a feature. This should be provided by the dataset.
        immutable_features
            List of immutable features.
        ranges
            Numerical feature ranges. Note that exist numerical features such as ``'Age'``, which are  allowed to
            increase only. We denote those by ``'inc_feat'``. Similarly, there exist features  allowed to decrease only.
            We denote them by ``'dec_feat'``. Finally, there are some free feature, which we denote by ``'free_feat'``.
            With the previous notation, we can define ``range = {'inc_feat': [0, 1], 'dec_feat': [-1, 0],
            'free_feat': [-1, 1]}``. ``'free_feat'`` can be omitted, as any unspecified feature is considered free.
            Having the ranges of a feature `{'feat': [a_low, a_high}`, when sampling is performed the numerical value
            will be clipped between `[a_low * (max_val - min_val), a_high * [max_val - min_val]]`, where `a_low` and
            `a_high` are the minimum and maximum values the feature ``'feat'``. This implies that `a_low` and `a_high`
            are not restricted to ``{-1, 0}`` and ``{0, 1}``, but can be any float number in-between `[-1, 0]` and
            `[0, 1]`.
        weight_num
            Numerical loss weight.
        weight_cat
            Categorical loss weight.
        latent_dim
            Auto-encoder latent dimension. Can be omitted if the actor network is user specified.
        seed
            Seed for reproducibility.
        **kwargs
            Used to replace any default parameter from :py:data:`alibi.explainers.cfrl_base.DEFAULT_BASE_PARAMS`.
        """
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            predictor=predictor,
            coeff_sparsity=coeff_sparsity,
            coeff_consistency=coeff_consistency,
            seed=seed,
            **kwargs,
        )

        # Set encoder preprocessor and decoder inverse preprocessor.
        self.params["encoder_preprocessor"] = encoder_preprocessor
        self.params["decoder_inv_preprocessor"] = decoder_inv_preprocessor

        # Set dataset specific arguments.
        self.params["category_map"] = category_map
        self.params["feature_names"] = feature_names
        self.params["ranges"] = ranges if (ranges is not None) else dict()
        self.params["immutable_features"] = (
            immutable_features if (immutable_features is not None) else list()
        )
        self.params["weight_num"] = weight_num
        self.params["weight_cat"] = weight_cat

        # Set sparsity loss if not user-specified.
        if "sparsity_loss" not in kwargs:
            self.params["sparsity_loss"] = partial(
                self.backend.sparsity_loss_tabular,
                category_map=self.params[
                    "category_map"
                ],  # pyright: ignore[reportCallIssue]
                weight_num=weight_num,  # pyright: ignore[reportCallIssue]
                weight_cat=weight_cat,  # pyright: ignore[reportCallIssue]
            )

        # Set consistency loss if not user-specified.
        if "consistency_loss" not in kwargs:
            self.params["consistency_loss"] = self.backend.consistency_loss_tabular

        # Set training conditional function generator if not user-specified.
        if "conditional_func" not in kwargs:
            self.params["conditional_func"] = partial(
                generate_condition,
                feature_names=self.params["feature_names"],
                category_map=self.params["category_map"],
                ranges=self.params["ranges"],  # pyright: ignore[reportArgumentType]
                immutable_features=self.params["immutable_features"],
            )

        # Set testing conditional function generator if not user-specified.
        if "conditional_vector" not in kwargs:
            self.params["conditional_vector"] = partial(
                get_conditional_vector,
                preprocessor=self.params["encoder_preprocessor"],
                feature_names=self.params["feature_names"],
                category_map=self.params["category_map"],
                ranges=self.params["ranges"],  # pyright: ignore[reportArgumentType]
                immutable_features=self.params["immutable_features"],
            )

    def _validate_input(self, X: np.ndarray):
        """
        Validates the input instances by checking the appropriate dimensions.

        Parameters
        ----------
        X
            Input instances.
        """
        if len(X.shape) != 2:
            raise ValueError(
                f"The input should be a 2D array. Found {len(X.shape)}D instead."
            )

        # Check if the number of features matches the expected one.
        if X.shape[1] != len(self.params["feature_names"]):
            raise ValueError(
                f"Unexpected number of features. The expected number "
                f"is {len(self.params['feature_names'])}, but the input has {X.shape[1]} features."
            )

        return X

    def fit(self, X: np.ndarray):
        # Compute vector of statistics to clamp numerical values between the minimum and maximum
        # value from the training set.
        self.params["stats"] = get_statistics(
            X=X,
            preprocessor=self.params["encoder_preprocessor"],
            category_map=self.params["category_map"],
        )

        # Set postprocessing functions. Needs `stats`.
        self.params["postprocessing_funcs"] = [
            SampleTabularPostprocessing(
                stats=self.params["stats"], category_map=self.params["category_map"]
            ),
            ConcatTabularPostprocessing(),
        ]

        # validate dataset
        self._validate_input(X)

        # call base class fit
        return super().fit(X)

    def explain(
        self,  # type: ignore[override]
        X: np.ndarray,
        Y_t: np.ndarray,
        C: Optional[List[Dict[str, List[Union[str, float]]]]] = None,
        batch_size: int = 100,
        diversity: bool = False,
        num_samples: int = 1,
        patience: int = 1000,
        tolerance: float = 1e-3,
    ):
        """
        Computes counterfactuals for the given instances conditioned on the target and the conditional vector.

        Parameters
        ----------
        X
            Input instances to generate counterfactuals for.
        Y_t
            Target labels.
        C
            List of conditional dictionaries. If ``None``, it means that no conditioning was used during training
            (i.e. the `conditional_func` returns ``None``). If conditioning was used during training but no
            conditioning is desired for the current input, an empty list is expected.
        diversity
            Whether to generate diverse counterfactual set for the given instance. Only supported for a single
            input instance.
        num_samples
            Number of diversity samples to be generated. Considered only if ``diversity=True``.
        batch_size
            Batch size to use when generating counterfactuals.
        patience
            Maximum number of iterations to perform diversity search stops. If -1, the search stops only if
            the desired number of samples has been found.
        tolerance
            Tolerance to distinguish two counterfactual instances.

        Returns
        -------
        explanation
            `Explanation` object containing the counterfactual with additional metadata as attributes. \
            See usage `CFRL examples`_ for details.

            .. _CFRL examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/CFRL.html
        """
        # General validation.
        self._validate_input(X)
        self._validate_target(Y_t)

        # Check if diversity flag is on.
        if diversity:
            return self._diversity(
                X=X,
                Y_t=Y_t,
                C=C,
                num_samples=num_samples,
                batch_size=batch_size,
                patience=patience,
                tolerance=tolerance,
            )

        # Get conditioning for a zero input. Used for a sanity check of the user-specified conditioning.
        X_zeros = np.zeros((1, X.shape[1]))
        C_zeros = self.params["conditional_func"](X_zeros)

        # If the conditional vector is `None`. This is equivalent of no conditioning at all, not even during training.
        if C is None:
            # Check if the conditional function actually a `None` conditioning
            if C_zeros is not None:
                raise ValueError(
                    "A `None` conditioning is not a valid input when training with conditioning. "
                    "If no feature conditioning is desired for the given input, `C` is expected to be an "
                    "empty list. A `None` conditioning is valid only when no conditioning was used "
                    "during training (i.e. `conditional_func` returns `None`)."
                )

            return super().explain(X=X, Y_t=Y_t, C=C, batch_size=batch_size)

        elif C_zeros is None:
            raise ValueError(
                "Conditioning different than `None` is not a valid input when training without "
                "conditioning. If feature conditioning is desired, consider defining an appropriate "
                "`conditional_func` that does not return `None`."
            )

        # Define conditional vector if an empty list. This is equivalent of no conditioning, but the conditional
        # vector was used during training.
        if len(C) == 0:
            C = [dict()]

        # Check the number of conditions.
        if len(C) != 1 and len(C) != X.shape[0]:
            raise ValueError(
                "The number of conditions should be 1 or equals the number of samples in X."
            )

        # If only one condition is passed.
        if len(C) == 1:
            C_vec = self.params["conditional_vector"](
                X=X, condition=C[0], stats=self.params["stats"]
            )
        else:
            # If multiple conditions were passed.
            C_vecs = []

            for i in range(len(C)):
                # Generate conditional vector for each instance. Note that this depends on the input instance.
                C_vecs.append(
                    self.params["conditional_vector"](
                        X=np.atleast_2d(X[i]),
                        condition=C[i],
                        stats=self.params["stats"],
                    )
                )

            # Concatenate all conditional vectors.
            C_vec = np.concatenate(C_vecs, axis=0)

        explanation = super().explain(X=X, Y_t=Y_t, C=C_vec, batch_size=batch_size)
        explanation.data.update(  # pyright: ignore[reportAttributeAccessIssue]
            {"condition": C}
        )
        return explanation

    def _diversity(
        self,
        X: np.ndarray,
        Y_t: np.ndarray,
        C: Optional[List[Dict[str, List[Union[str, float]]]]],
        num_samples: int = 1,
        batch_size: int = 100,
        patience: int = 1000,
        tolerance: float = 1e-3,
    ):
        """
        Generates a set of diverse counterfactuals given a single instance, target and conditioning.

        Parameters
        ----------
        X
            Input instance.
        Y_t
            Target label.
        C
            List of conditional dictionaries. If ``None``, it means that no conditioning was used during training
            (i.e. the `conditional_func` returns ``None``).
        num_samples
            Number of counterfactual samples to be generated.
        batch_size
            Batch size used at inference.
        num_samples
            Number of diversity samples to be generated. Considered only if ``diversity=True``.
        batch_size
            Batch size to use when generating counterfactuals.
        patience
            Maximum number of iterations to perform diversity search stops. If -1, the search stops only if
            the desired number of samples has been found.
        tolerance
            Tolerance to distinguish two counterfactual instances.

        Returns
        -------
        Explanation object containing the diverse counterfactuals.
        """
        # Check if condition. If no conditioning was used during training, the method can not generate a diverse
        # set of counterfactual instances
        if C is None:
            raise ValueError(
                "A diverse set of counterfactual can not be generated if a `None` conditioning is "
                "used during training. Use the `explain` method to generate a counterfactual. The "
                "generation process is deterministic in its core. If conditioning is used during training "
                "a diverse set of counterfactual can be generated by restricting each feature condition "
                "to a subset to remain feasible."
            )

        # Check the number of inputs
        if X.shape[0] != 1:
            raise ValueError("Only a single input instance can be passed.")

        # Check the number of labels.
        if Y_t.shape[0] != 1:
            raise ValueError("Only a single label can be passed.")

        # Check the number of conditions.
        if (C is not None) and len(C) > 1:
            raise ValueError("At most, one condition can be passed.")

        # Generate a batch of data.
        X_repeated = np.tile(X, (batch_size, 1))
        Y_t = np.tile(np.atleast_2d(Y_t), (batch_size, 1))

        # Define counterfactual buffer.
        X_cf_buff = None

        for i in tqdm(count()):
            if i == patience:
                break

            if (X_cf_buff is not None) and (X_cf_buff.shape[0] >= num_samples):
                break

            # Generate conditional vector.
            C_vec = get_conditional_vector(
                X=X_repeated,
                condition=C[0] if len(C) else {},
                preprocessor=self.params["encoder_preprocessor"],
                feature_names=self.params["feature_names"],
                category_map=self.params["category_map"],
                stats=self.params["stats"],
                immutable_features=self.params["immutable_features"],
                diverse=True,
            )

            # Generate counterfactuals.
            results = self._compute_counterfactual(X=X_repeated, Y_t=Y_t, C=C_vec)
            X_cf, Y_m_cf, Y_t = results["X_cf"], results["Y_m_cf"], results["Y_t"]  # type: ignore[assignment]
            X_cf = cast(np.ndarray, X_cf)  # help mypy out

            # Select only counterfactuals where prediction matches the target.
            X_cf = X_cf[Y_t == Y_m_cf]
            X_cf = cast(np.ndarray, X_cf)  # help mypy out

            if X_cf.shape[0] == 0:
                continue

            # Find unique counterfactuals.
            _, indices = np.unique(
                np.floor(X_cf / tolerance).astype(  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
                    int
                ),
                return_index=True,
                axis=0,
            )

            # Add them to the unique buffer but make sure not to add duplicates.
            if X_cf_buff is None:
                X_cf_buff = X_cf[indices]
            else:
                X_cf_buff = np.concatenate([X_cf_buff, X_cf[indices]], axis=0)
                _, indices = np.unique(
                    np.floor(
                        X_cf_buff / tolerance
                    ).astype(  # pyright: ignore[reportCallIssue]
                        int
                    ),
                    return_index=True,
                    axis=0,
                )
                X_cf_buff = X_cf_buff[indices]

        # Construct counterfactuals to the explanation.
        X_cf = X_cf_buff[:num_samples] if (X_cf_buff is not None) else np.array([])

        # Compute model's prediction on the counterfactual instances
        Y_m_cf = self.params["predictor"](X_cf) if X_cf.shape[0] != 0 else np.array([])
        if self._is_classification(pred=Y_m_cf):
            Y_m_cf = np.argmax(Y_m_cf, axis=1)

        # Compute model's prediction on the original input.
        Y_m = self.params["predictor"](X)
        if self._is_classification(Y_m):
            Y_m = np.argmax(Y_m, axis=1)

        # Update target representation if necessary.
        if self._is_classification(Y_t):
            Y_t = np.argmax(Y_t, axis=1)

        return self._build_explanation(X=X, Y_m=Y_m, X_cf=X_cf, Y_m_cf=Y_m_cf, Y_t=Y_t, C=C)  # type: ignore[arg-type]
