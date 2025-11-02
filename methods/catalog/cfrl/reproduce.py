import logging
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from requests import RequestException
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import Bunch
from torch import optim

from sklearn.ensemble import RandomForestClassifier

from .model import HeterogeneousDecoder, HeterogeneousEncoder
from .cfrl_tabular import (
    CounterfactualRLTabular,
    apply_category_mapping,
    get_he_preprocessor,
)

logger = logging.getLogger(__name__)

ADULT_URLS = ['https://storage.googleapis.com/seldon-datasets/adult/adult.data',
              'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
              'http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data']


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

    dataset_url = ADULT_URLS[url_id]
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
        logger.exception("Could not connect, URL may be out of service")
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
    epochs: int,
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

    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
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

        logger.debug(
            "Autoencoder epoch %d/%d | loss %.4f",
            epoch + 1,
            epochs,
            epoch_loss / max(1, num_samples // batch_size),
        )

    encoder.eval()
    decoder.eval()


def run_experiment() -> Dict[str, Union[float, pd.DataFrame]]:
    adult = fetch_adult()

    categorical_ids = list(adult.category_map.keys())  # pyright: ignore[reportAttributeAccessIssue]
    numerical_ids = [
        idx for idx in range(len(adult.feature_names)) if idx not in categorical_ids  # pyright: ignore[reportAttributeAccessIssue]
    ]

    X, Y = adult.data, adult.target  # pyright: ignore[reportAttributeAccessIssue]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=13
    )

    num_transf = StandardScaler()
    cat_transf = OneHotEncoder(
        categories=[range(len(x)) for x in adult.category_map.values()],  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        handle_unknown="ignore",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transf, categorical_ids),
            ("num", num_transf, numerical_ids),
        ],
        sparse_threshold=0,
    )

    preprocessor.fit(X_train)
    X_train_ohe = preprocessor.transform(X_train)

    clf = RandomForestClassifier(
        max_depth=15, min_samples_split=10, n_estimators=50, random_state=13
    )
    clf.fit(X_train_ohe, Y_train)

    predictor = lambda data: clf.predict_proba(preprocessor.transform(data))

    acc = accuracy_score(y_true=Y_test, y_pred=predictor(X_test).argmax(axis=1))  # pyright: ignore[reportAttributeAccessIssue]
    print(f"Accuracy: {acc:.3f}")

    feature_types = {
        "Age": int,
        "Capital Gain": int,
        "Capital Loss": int,
        "Hours per week": int,
    }

    heae_preprocessor, heae_inv_preprocessor = get_he_preprocessor(
        X=X_train,
        feature_names=adult.feature_names,  # pyright: ignore[reportAttributeAccessIssue]
        category_map=adult.category_map,  # pyright: ignore[reportAttributeAccessIssue]
        feature_types=feature_types,
    )

    trainset_input = heae_preprocessor(X_train).astype(np.float32)

    cat_targets = [X_train[:, cat_id] for cat_id in categorical_ids]  # pyright: ignore[reportCallIssue, reportArgumentType]
    num_dim = len(numerical_ids)

    output_dims: List[int] = []
    if num_dim > 0:
        output_dims.append(num_dim)
    output_dims.extend(len(adult.category_map[cat_id]) for cat_id in categorical_ids)  # pyright: ignore[reportAttributeAccessIssue]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pyright: ignore[reportAttributeAccessIssue]

    latent_dim = 15
    encoder = HeterogeneousEncoder(
        hidden_dim=128,
        latent_dim=latent_dim,
        input_dim=trainset_input.shape[1],
    ).to(device)
    decoder = HeterogeneousDecoder(
        hidden_dim=128,
        output_dims=output_dims,
        latent_dim=latent_dim,
    ).to(device)

    heae_path = Path("pytorch") / "ADULT_autoencoder"
    heae_path.mkdir(parents=True, exist_ok=True)
    encoder_path = heae_path / "encoder.pt"
    decoder_path = heae_path / "decoder.pt"

    if not encoder_path.exists() or not decoder_path.exists():
        _train_autoencoder(
            encoder=encoder,
            decoder=decoder,
            train_inputs=trainset_input,
            categorical_targets=cat_targets,
            num_dim=num_dim,
            epochs=50,
            batch_size=128,
            lr=1e-3,
            device=device,
        )
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
    else:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        encoder.eval()
        decoder.eval()

    COEFF_SPARSITY = 0.5
    COEFF_CONSISTENCY = 0.5
    TRAIN_STEPS = 10000
    BATCH_SIZE = 100

    immutable_features = ["Marital Status", "Relationship", "Race", "Sex"]
    ranges = {"Age": [0.0, 1.0]}

    explainer = CounterfactualRLTabular(
        predictor=predictor,
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        encoder_preprocessor=heae_preprocessor,
        decoder_inv_preprocessor=heae_inv_preprocessor,
        coeff_sparsity=COEFF_SPARSITY,
        coeff_consistency=COEFF_CONSISTENCY,
        category_map=adult.category_map,  # pyright: ignore[reportAttributeAccessIssue]
        feature_names=adult.feature_names,  # pyright: ignore[reportAttributeAccessIssue]
        ranges=ranges,  # pyright: ignore[reportArgumentType]
        immutable_features=immutable_features,
        train_steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
    )

    explainer = explainer.fit(X=X_train)

    predictions = predictor(X_test)
    X_positive = X_test[np.argmax(predictions, axis=1) == 1]

    if X_positive.shape[0] == 0:
        raise RuntimeError("No positive examples found to generate counterfactuals.")

    X_batch = X_positive[:1000]
    Y_t = np.array([0])
    C = [{"Age": [0, 20], "Workclass": ["State-gov", "?", "Local-gov"]}]

    explanation = explainer.explain(X_batch, Y_t, C)

    orig = np.concatenate(
        [explanation["orig"]["X"], explanation["orig"]["class"]], axis=1
    )
    cf = np.concatenate(
        [explanation["cf"]["X"], explanation["cf"]["class"]], axis=1
    )

    feature_names = adult.feature_names + ["Label"]  # pyright: ignore[reportAttributeAccessIssue]
    category_map = deepcopy(adult.category_map)  # pyright: ignore[reportAttributeAccessIssue]
    category_map.update({feature_names.index("Label"): adult.target_names})  # pyright: ignore[reportAttributeAccessIssue]

    orig_pd = pd.DataFrame(
        apply_category_mapping(orig, category_map),
        columns=feature_names,
    )
    cf_pd = pd.DataFrame(
        apply_category_mapping(cf, category_map),
        columns=feature_names,
    )

    X_single = X_positive[0].reshape(1, -1)
    explanation_diverse = explainer.explain(
        X=X_single,
        Y_t=Y_t,
        C=C,
        diversity=True,
        num_samples=100,
        batch_size=10,
    )

    orig_diverse = np.concatenate(
        [
            explanation_diverse["orig"]["X"],
            explanation_diverse["orig"]["class"],
        ],
        axis=1,
    )
    cf_diverse = np.concatenate(
        [
            explanation_diverse["cf"]["X"],
            explanation_diverse["cf"]["class"],
        ],
        axis=1,
    )

    orig_pd_diverse = pd.DataFrame(
        apply_category_mapping(orig_diverse, category_map),
        columns=feature_names,
    )
    cf_pd_diverse = pd.DataFrame(
        apply_category_mapping(cf_diverse, category_map),
        columns=feature_names,
    )

    print("Original sample preview:")
    print(orig_pd.head())
    print("Counterfactual sample preview:")
    print(cf_pd.head())
    print("Diverse counterfactual preview:")
    print(cf_pd_diverse.head())

    return {
        "accuracy": acc,  # pyright: ignore[reportReturnType]
        "orig": orig_pd,
        "counterfactuals": cf_pd,
        "orig_diverse": orig_pd_diverse,
        "counterfactuals_diverse": cf_pd_diverse,
    }


if __name__ == "__main__":
    run_experiment()
