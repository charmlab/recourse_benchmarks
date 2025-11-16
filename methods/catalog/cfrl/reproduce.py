from dataclasses import dataclass, field
from io import StringIO
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from requests import RequestException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import Bunch
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from tools.log import log

from .cfrl_backend import set_seed
from .cfrl_tabular import (
    CounterfactualRLTabular,
    apply_category_mapping,
    get_conditional_dim,
    get_he_preprocessor,
)
from .model import HeterogeneousDecoder, HeterogeneousEncoder


@dataclass
class ExperimentConfig:
    test_size: float = 0.2
    rf_max_depth: int = 15
    rf_min_samples_split: int = 10
    rf_n_estimators: int = 50
    seed: int = 0
    autoencoder_batch_size: int = 128
    autoencoder_target_steps: int = 100_000
    autoencoder_lr: float = 1e-3
    autoencoder_latent_dim: int = 15
    autoencoder_hidden_dim: int = 128
    cfrl_coeff_sparsity: float = 0.5
    cfrl_coeff_consistency: float = 0.5
    cfrl_train_steps: int = 100_000
    cfrl_batch_size: int = 128
    immutable_features: Tuple[str, ...] = (
        "Marital Status",
        "Relationship",
        "Race",
        "Sex",
    )
    constrained_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: {"Age": [0.0, 1.0]}
    )


def fetch_adult(
    features_drop: Optional[List[str]] = None,
    return_X_y: bool = False,
    url_id: int = 0,
) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads and pre-processes 'adult' dataset.
    More info: http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/

    Parameters
    ----------
    features_drop
        List of features to be dropped from dataset, by default drops ``["fnlwgt", "Education-Num"]``.
    return_X_y
        If ``True``, return features `X` and labels `y` as `numpy` arrays. If ``False`` return a `Bunch` object.
    url_id
        Index specifying which URL to use for downloading.

    Returns
    -------
    Bunch
        Dataset, labels, a list of features and a dictionary containing a list with the potential categories
        for each categorical feature where the key refers to the feature column.
    (data, target)
        Tuple if ``return_X_y=True``
    """
    if features_drop is None:
        features_drop = ["fnlwgt", "Education-Num"]

    adult_urls = [
        "https://storage.googleapis.com/seldon-datasets/adult/adult.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
    ]
    dataset_url = adult_urls[url_id]
    raw_features = [
        "Age",
        "Workclass",
        "fnlwgt",
        "Education",
        "Education-Num",
        "Marital Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Capital Gain",
        "Capital Loss",
        "Hours per week",
        "Country",
        "Target",
    ]
    try:
        resp = requests.get(dataset_url)
        resp.raise_for_status()
    except RequestException:
        log.exception("Could not connect, URL may be out of service")
        raise

    raw_data = pd.read_csv(
        StringIO(resp.text), names=raw_features, delimiter=", ", engine="python"
    ).fillna("?")  # pyright: ignore[reportAttributeAccessIssue]

    labels = (raw_data["Target"] == ">50K").astype(int).values  # pyright: ignore[reportOptionalSubscript, reportAttributeAccessIssue]
    features_drop += ["Target"]
    data = raw_data.drop(features_drop, axis=1)  # pyright: ignore[reportOptionalMemberAccess]
    features = list(data.columns)  # pyright: ignore[reportOptionalMemberAccess]

    education_map = {
        "10th": "Dropout",
        "11th": "Dropout",
        "12th": "Dropout",
        "1st-4th": "Dropout",
        "5th-6th": "Dropout",
        "7th-8th": "Dropout",
        "9th": "Dropout",
        "Preschool": "Dropout",
        "HS-grad": "High School grad",
        "Some-college": "High School grad",
        "Masters": "Masters",
        "Prof-school": "Prof-School",
        "Assoc-acdm": "Associates",
        "Assoc-voc": "Associates",
    }
    occupation_map = {
        "Adm-clerical": "Admin",
        "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar",
        "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Prof-specialty": "Professional",
        "Protective-serv": "Other",
        "Sales": "Sales",
        "Tech-support": "Other",
        "Transport-moving": "Blue-Collar",
    }
    country_map = {
        "Cambodia": "SE-Asia",
        "Canada": "British-Commonwealth",
        "China": "China",
        "Columbia": "South-America",
        "Cuba": "Other",
        "Dominican-Republic": "Latin-America",
        "Ecuador": "South-America",
        "El-Salvador": "South-America",
        "England": "British-Commonwealth",
        "France": "Euro_1",
        "Germany": "Euro_1",
        "Greece": "Euro_2",
        "Guatemala": "Latin-America",
        "Haiti": "Latin-America",
        "Holand-Netherlands": "Euro_1",
        "Honduras": "Latin-America",
        "Hong": "China",
        "Hungary": "Euro_2",
        "India": "British-Commonwealth",
        "Iran": "Other",
        "Ireland": "British-Commonwealth",
        "Italy": "Euro_1",
        "Jamaica": "Latin-America",
        "Japan": "Other",
        "Laos": "SE-Asia",
        "Mexico": "Latin-America",
        "Nicaragua": "Latin-America",
        "Outlying-US(Guam-USVI-etc)": "Latin-America",
        "Peru": "South-America",
        "Philippines": "SE-Asia",
        "Poland": "Euro_2",
        "Portugal": "Euro_2",
        "Puerto-Rico": "Latin-America",
        "Scotland": "British-Commonwealth",
        "South": "Euro_2",
        "Taiwan": "China",
        "Thailand": "SE-Asia",
        "Trinadad&Tobago": "Latin-America",
        "United-States": "United-States",
        "Vietnam": "SE-Asia",
    }
    married_map = {
        "Never-married": "Never-Married",
        "Married-AF-spouse": "Married",
        "Married-civ-spouse": "Married",
        "Married-spouse-absent": "Separated",
        "Separated": "Separated",
        "Divorced": "Separated",
        "Widowed": "Widowed",
    }
    mapping = {
        "Education": education_map,
        "Occupation": occupation_map,
        "Country": country_map,
        "Marital Status": married_map,
    }

    data_copy = data.copy()  # pyright: ignore[reportOptionalMemberAccess]
    for feat, feat_map in mapping.items():
        data_tmp = data_copy[feat].values  # pyright: ignore[reportOptionalMemberAccess]
        for key, value in feat_map.items():
            data_tmp[data_tmp == key] = value
        data[feat] = data_tmp  # pyright: ignore[reportOptionalSubscript]

    categorical_features = [f for f in features if data[f].dtype == "O"]  # pyright: ignore[reportOptionalSubscript, reportOptionalMemberAccess]
    category_map: Dict[int, List[str]] = {}
    for feat in categorical_features:
        encoder = LabelEncoder()
        data_tmp = encoder.fit_transform(data[feat].values)  # pyright: ignore[reportOptionalSubscript, reportOptionalMemberAccess]
        data[feat] = data_tmp  # pyright: ignore[reportOptionalSubscript]
        category_map[features.index(feat)] = list(encoder.classes_)

    data = data.values  # pyright: ignore[reportOptionalMemberAccess]
    target_names = ["<=50K", ">50K"]

    if return_X_y:
        return data, labels

    return Bunch(
        data=data,
        target=labels,
        feature_names=features,
        target_names=target_names,
        category_map=category_map,
    )


def _train_autoencoder(
    encoder: HeterogeneousEncoder,
    decoder: HeterogeneousDecoder,
    train_inputs: np.ndarray,
    categorical_targets: List[np.ndarray],
    num_dim: int,
    target_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    """Train heterogeneous autoencoder using PyTorch."""
    encoder.train()
    decoder.train()

    inputs = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    num_targets = inputs[:, :num_dim] if num_dim > 0 else None
    cat_tensors = [
        torch.tensor(target.astype(np.int64), dtype=torch.long, device=device)
        for target in categorical_targets
    ]

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimiser = optim.Adam(params, lr=lr)
    num_samples = inputs.size(0)
    if target_steps <= 0:
        raise ValueError("autoencoder_target_steps must be a positive integer.")

    steps_per_epoch = max(1, math.ceil(num_samples / batch_size))
    max_epochs = math.ceil(target_steps / steps_per_epoch)
    steps_run = 0

    with tqdm(total=target_steps, desc="AE steps", leave=False) as pbar:
        for epoch in range(max_epochs):
            perm = torch.randperm(num_samples, device=device)
            epoch_loss = 0.0
            steps_this_epoch = 0
            for start in range(0, num_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch_x = inputs[idx]
                outputs = decoder(encoder(batch_x))

                loss = torch.zeros((), device=device)

                if num_dim > 0 and num_targets is not None:
                    recon_num = outputs[0]
                    loss = loss + F.mse_loss(recon_num, num_targets[idx])
                    cat_outputs = outputs[1:]
                else:
                    cat_outputs = outputs

                if cat_tensors:
                    weight = 1.0 / len(cat_tensors)
                    for logits, target in zip(cat_outputs, cat_tensors):
                        loss = loss + weight * F.cross_entropy(logits, target[idx])

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                steps_run += 1
                steps_this_epoch += 1
                pbar.update(1)

                if steps_run >= target_steps:
                    break

            log.debug(
                "Autoencoder epoch %d/%d (~%d/%d steps) | loss %.4f",
                epoch + 1,
                max_epochs,
                steps_run,
                target_steps,
                epoch_loss / max(1, steps_this_epoch),
            )

            if steps_run >= target_steps:
                break

    encoder.eval()
    decoder.eval()

def _train_classifier(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    categorical_ids: List[int],
    numerical_ids: List[int],
    config: ExperimentConfig,
) -> Tuple[RandomForestClassifier, ColumnTransformer]:
    cat_transf = OneHotEncoder(
        categories=[range(int(X_train[:, idx].max()) + 1) for idx in categorical_ids],  # pyright: ignore[reportArgumentType]
        handle_unknown="ignore",
    )
    num_transf = StandardScaler()
    preprocessor = ColumnTransformer(
        [
            ("cat", cat_transf, categorical_ids),
            ("num", num_transf, numerical_ids),
        ],
        sparse_threshold=0,
    )
    preprocessor.fit(X_train)
    X_train_pre = preprocessor.transform(X_train)

    clf = RandomForestClassifier(
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        n_estimators=config.rf_n_estimators,
        random_state=config.seed,
    )
    clf.fit(X_train_pre, Y_train)
    return clf, preprocessor


def _prepare_autoencoder(
    X_train: np.ndarray,
    categorical_ids: List[int],
    adult_feature_names: List[str],
    category_map: Dict[int, List[str]],
    feature_types: Dict[str, type],
    config: ExperimentConfig,
) -> Tuple[
    HeterogeneousEncoder,
    HeterogeneousDecoder,
    np.ndarray,
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pyright: ignore[reportAttributeAccessIssue]
    torch.manual_seed(config.seed)

    heae_preprocessor, heae_inv_preprocessor = get_he_preprocessor(
        X=X_train,
        feature_names=adult_feature_names,
        category_map=category_map,
        feature_types=feature_types,
    )

    X_pre = heae_preprocessor(X_train).astype(np.float32)
    batch_size = config.autoencoder_batch_size
    target_steps = config.autoencoder_target_steps

    numerical_ids = [
        idx for idx in range(len(adult_feature_names)) if idx not in categorical_ids
    ]
    num_dim = len(numerical_ids)
    cat_targets = [X_train[:, idx] for idx in categorical_ids]

    output_dims: List[int] = []
    if num_dim > 0:
        output_dims.append(num_dim)
    output_dims.extend(len(category_map[idx]) for idx in categorical_ids)

    encoder = HeterogeneousEncoder(
        hidden_dim=config.autoencoder_hidden_dim,
        latent_dim=config.autoencoder_latent_dim,
        input_dim=X_pre.shape[1],
    ).to(device)
    decoder = HeterogeneousDecoder(
        hidden_dim=config.autoencoder_hidden_dim,
        output_dims=output_dims,
        latent_dim=config.autoencoder_latent_dim,
    ).to(device)

    _train_autoencoder(
        encoder=encoder,
        decoder=decoder,
        train_inputs=X_pre,
        categorical_targets=cat_targets,
        num_dim=num_dim,
        target_steps=target_steps,
        batch_size=batch_size,
        lr=config.autoencoder_lr,
        device=device,
    )

    return encoder, decoder, X_pre, heae_preprocessor, heae_inv_preprocessor


def _get_explainer(
    predictor,
    encoder: HeterogeneousEncoder,
    decoder: HeterogeneousDecoder,
    he_preprocessor,
    he_inv_preprocessor,
    dataset,
    config: ExperimentConfig,
    actor_input_dim: int,
) -> CounterfactualRLTabular:
    return CounterfactualRLTabular(
        predictor=predictor,
        encoder=encoder,
        decoder=decoder,
        latent_dim=config.autoencoder_latent_dim,
        encoder_preprocessor=he_preprocessor,
        decoder_inv_preprocessor=he_inv_preprocessor,
        coeff_sparsity=config.cfrl_coeff_sparsity,
        coeff_consistency=config.cfrl_coeff_consistency,
        category_map=dataset.category_map,
        feature_names=dataset.feature_names,
        immutable_features=config.immutable_features,  # pyright: ignore[reportArgumentType]
        ranges=config.constrained_ranges,  # pyright: ignore[reportArgumentType]
        train_steps=config.cfrl_train_steps,
        batch_size=config.cfrl_batch_size,
        seed=config.seed,
        actor_input_dim=actor_input_dim,
    )


def _evaluate_counterfactuals(
    orig: np.ndarray,
    cf: np.ndarray,
    predictor,
    target_labels: np.ndarray,
    feature_names: List[str],
    categorical_idx: List[int],
    immutable_features: Tuple[str, ...],
    numeric_stats: Dict[int, Tuple[float, float]],
    he_preprocessor: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    train_preds: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    preds = predictor(cf).argmax(axis=1)
    success = preds == target_labels
    categorical_idx_set = set(categorical_idx)
    immutable_idx = {feature_names.index(name) for name in immutable_features}
    numerical_idx = [i for i in range(len(feature_names)) if i not in categorical_idx]

    cat_l0: List[float] = []
    num_l1: List[float] = []
    violation_flags: List[bool] = []

    # Sparsity is computed only over valid counterfactuals (matching the paper definition).
    for row_orig, row_cf, is_valid in zip(orig, cf, success):
        if not is_valid:
            continue

        cat_changes = 0
        l1_sum = 0.0
        violation = False

        for idx, (val_o, val_c) in enumerate(zip(row_orig, row_cf)):
            name = feature_names[idx]
            if idx in categorical_idx_set:
                changed = val_o != val_c
                if changed:
                    cat_changes += 1
                if idx in immutable_idx:
                    violation |= changed
                continue

            # numerical feature
            val_o = float(val_o)
            val_c = float(val_c)
            diff_std = abs(val_o - val_c) / max(numeric_stats[idx][1], 1e-8)
            l1_sum += diff_std
            if idx in immutable_idx:
                violation |= diff_std > 1e-6

        cat_l0.append(cat_changes / max(1, len(categorical_idx)))
        num_l1.append(l1_sum / max(1, len(numerical_idx)))
        violation_flags.append(violation)

    # Target-conditional MMD using a random encoder and RBF kernel.
    def _random_encode(data: np.ndarray) -> np.ndarray:
        h1_dim, h2_dim, h3_dim = 32, 16, 5
        w1 = rng.standard_normal((data.shape[1], h1_dim)) / math.sqrt(
            max(1, data.shape[1])
        )
        w2 = rng.standard_normal((h1_dim, h2_dim)) / math.sqrt(h1_dim)
        w3 = rng.standard_normal((h2_dim, h3_dim)) / math.sqrt(h2_dim)
        h1 = np.maximum(0, data @ w1)  # pyright: ignore[reportCallIssue]
        h2 = np.maximum(0, h1 @ w2)  # pyright: ignore[reportCallIssue]
        return h2 @ w3

    def _rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
        def _pairwise_sq(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            u_norm = (u**2).sum(axis=1)[:, None]
            v_norm = (v**2).sum(axis=1)[None, :]
            return u_norm + v_norm - 2 * (u @ v.T)

        z = np.concatenate([x, y], axis=0)
        if z.shape[0] > 2000:  # pyright: ignore[reportAttributeAccessIssue]
            idx = rng.choice(z.shape[0], size=2000, replace=False)  # pyright: ignore[reportAttributeAccessIssue]
            z_sample = z[idx]
        else:
            z_sample = z
        dists = _pairwise_sq(z_sample, z_sample)
        median = np.median(dists[dists > 0])
        sigma = math.sqrt(median) if median > 0 else 1.0
        gamma = 1.0 / (2 * sigma**2)

        k_xx = np.exp(-gamma * _pairwise_sq(x, x))  # pyright: ignore[reportCallIssue]
        k_yy = np.exp(-gamma * _pairwise_sq(y, y))  # pyright: ignore[reportCallIssue]
        k_xy = np.exp(-gamma * _pairwise_sq(x, y))  # pyright: ignore[reportCallIssue]

        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()  # pyright: ignore[reportAttributeAccessIssue]
        return float(max(mmd, 0.0))

    # Target-conditional MMD: report per-class as in the paper (no validity filtering).
    mmd_weighted = 0.0
    mmd_per_class: Dict[Union[int, float], float] = {}
    weights: List[int] = []
    mmd_vals: List[float] = []
    if cf.shape[0] > 0:
        for cls in np.unique(target_labels):
            cls_mask = target_labels == cls
            cf_cls = cf[cls_mask]
            train_target = X_train[train_preds == cls]
            if train_target.shape[0] == 0 or cf_cls.shape[0] == 0:
                continue
            cf_enc = _random_encode(he_preprocessor(cf_cls))
            ref_enc = _random_encode(he_preprocessor(train_target))
            cls_mmd = _rbf_mmd(cf_enc, ref_enc)
            mmd_per_class[float(cls)] = cls_mmd
            weights.append(cf_cls.shape[0])
            mmd_vals.append(cls_mmd)
        if mmd_vals:
            total = sum(weights)
            mmd_weighted = float(sum(w * v for w, v in zip(weights, mmd_vals)) / total)

    metrics: Dict[str, float] = {
        "validity": float(success.mean()) if len(success) else 0.0,
        "sparsity_cat_l0": float(np.mean(cat_l0)) if cat_l0 else 0.0,
        "sparsity_num_l1": float(np.mean(num_l1)) if num_l1 else 0.0,
        "immutability_violation_rate": float(np.mean(violation_flags))
        if violation_flags
        else 0.0,
        # Keep the weighted aggregate for convenience/compatibility.
        "target_conditional_mmd": mmd_weighted,
    }

    # Expose per-class MMD values (e.g., target_conditional_mmd_cls_0) to mirror paper reporting.
    for cls, val in mmd_per_class.items():
        metrics[f"target_conditional_mmd_cls_{int(cls)}"] = val

    return metrics


def run_experiment() -> Dict[str, object]:
    config = ExperimentConfig()
    set_seed(config.seed)

    adult = fetch_adult()

    categorical_ids = list(adult.category_map.keys())  # pyright: ignore[reportAttributeAccessIssue]
    numerical_ids = [
        idx for idx in range(len(adult.feature_names)) if idx not in categorical_ids  # pyright: ignore[reportAttributeAccessIssue]
    ]

    X, Y = adult.data, adult.target  # pyright: ignore[reportAttributeAccessIssue]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=Y,
    )

    clf, preprocessor = _train_classifier(
        X_train, Y_train, categorical_ids, numerical_ids, config
    )
    predictor = lambda data: clf.predict_proba(preprocessor.transform(data))

    # Compute explicit actor/critic input dimensions.
    sample_pred = predictor(X_train[:1])
    num_classes = sample_pred.shape[1] if sample_pred.ndim == 2 else 1  # pyright: ignore[reportAttributeAccessIssue]
    cond_dim = get_conditional_dim(adult.feature_names, adult.category_map)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
    actor_input_dim = config.autoencoder_latent_dim + 2 * num_classes + cond_dim

    accuracy = accuracy_score(Y_test, predictor(X_test).argmax(axis=1))  # pyright: ignore[reportAttributeAccessIssue]
    log.info("Random forest accuracy on test split: %.3f", accuracy)

    feature_types = {
        "Age": int,
        "Capital Gain": int,
        "Capital Loss": int,
        "Hours per week": int,
    }
    encoder, decoder, X_pre, heae_preprocessor, heae_inv_preprocessor = _prepare_autoencoder(
        X_train,
        categorical_ids,
        adult.feature_names,  # pyright: ignore[reportAttributeAccessIssue]
        adult.category_map,  # pyright: ignore[reportAttributeAccessIssue]
        feature_types,
        config,
    )

    explainer = _get_explainer(
        predictor,
        encoder,
        decoder,
        heae_preprocessor,
        heae_inv_preprocessor,
        adult,
        config,
        actor_input_dim,
    )
    explainer.fit(X_train)

    predictions = predictor(X_test)
    num_samples = min(1000, X_test.shape[0])  # pyright: ignore[reportAttributeAccessIssue]
    rng = np.random.default_rng(config.seed)
    sample_idx = rng.choice(X_test.shape[0], size=num_samples, replace=False)  # pyright: ignore[reportAttributeAccessIssue]
    batch = X_test[sample_idx]
    # Assign a random target different from the model's predicted label.
    pred_labels = np.argmax(predictions, axis=1)
    num_classes = predictions.shape[1] if predictions.ndim == 2 else 2  # pyright: ignore[reportAttributeAccessIssue]
    if num_classes == 2:
        target_labels = 1 - pred_labels[sample_idx]
    else:
        target_labels = np.array(
            [
                rng.choice([c for c in range(num_classes) if c != pred_labels[i]])
                for i in sample_idx
            ]
        )
    explanation = explainer.explain(batch, Y_t=target_labels, C=[])

    orig = explanation["orig"]["X"]
    cf = explanation["cf"]["X"]
    numeric_stats = {
        idx: (float(X_train[:, idx].mean()), float(X_train[:, idx].std()))  # pyright: ignore[reportCallIssue, reportArgumentType]
        for idx in range(len(adult.feature_names))  # pyright: ignore[reportAttributeAccessIssue]
        if idx not in categorical_ids
    }
    train_preds = predictor(X_train).argmax(axis=1)  # pyright: ignore[reportAttributeAccessIssue]
    metric_summary = _evaluate_counterfactuals(
        orig=orig,
        cf=cf,
        predictor=predictor,
        target_labels=target_labels,
        feature_names=adult.feature_names,  # pyright: ignore[reportAttributeAccessIssue]
        categorical_idx=categorical_ids,
        immutable_features=config.immutable_features,
        numeric_stats=numeric_stats,
        he_preprocessor=heae_preprocessor,
        X_train=X_train,
        train_preds=train_preds,
        rng=rng,
    )

    feature_names = adult.feature_names + ["Label"]  # pyright: ignore[reportAttributeAccessIssue]
    category_map = adult.category_map.copy()  # pyright: ignore[reportAttributeAccessIssue]
    category_map[len(feature_names) - 1] = adult.target_names  # pyright: ignore[reportAttributeAccessIssue]
    orig_df = pd.DataFrame(
        apply_category_mapping(
            np.concatenate([explanation["orig"]["X"], explanation["orig"]["class"]], axis=1),
            category_map,
        ),
        columns=feature_names,
    )
    cf_df = pd.DataFrame(
        apply_category_mapping(
            np.concatenate([explanation["cf"]["X"], explanation["cf"]["class"]], axis=1),
            category_map,
        ),
        columns=feature_names,
    )

    log.info("Metrics: %s", metric_summary)
    return {
        "accuracy": float(accuracy),  # pyright: ignore[reportArgumentType]
        **metric_summary,
        "orig": orig_df,
        "counterfactuals": cf_df,
    }


if __name__ == "__main__":
    results = run_experiment()
    for key, value in results.items():
        if isinstance(value, float):
            log.info("%s: %.4f", key, value)
    log.info("Counterfactual preview:\n%s", results["counterfactuals"].head())  # pyright: ignore[reportAttributeAccessIssue]
