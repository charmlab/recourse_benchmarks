"""
GenRe Reproduction Script

Reproduces GenRe paper results using the RecourseMethod interface.

Usage:
    python reproduce.py --dataset adult-all --device cpu
"""
import argparse
import os
import pickle

import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("Install huggingface-hub: pip install huggingface-hub")

from library.data import utils as dutils
from library.models.classifiers.ann import BinaryClassifier

# Import repo catalogs
from data.catalog import DataCatalog
from models.catalog import ModelCatalog

# Import utils and GenRe wrapper
from methods.catalog.genre import utils as genre_utils
from methods.catalog.genre import GenRe

RANDOM_SEED = 54321


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce GenRe results")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult-all",
        choices=["adult-all", "compas-all", "heloc"],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--temp", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--best_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_repo", type=str, default="jamie250/genrereproduce")
    parser.add_argument(
        "--use-data-object",
        action="store_true",
        help="Use DataCatalog object for cat_mask (instead of author's cat_mask)",
    )
    return parser.parse_args()


def load_author_data(dataset_name):
    """Load author's data with preprocessing"""
    train_y, train_X, test_y, test_X, cat_mask, immutable_mask = dutils.load_dataset(
        dataset_name, ret_tensor=True, min_max=True, ret_masks=True
    )
    return train_y, train_X, test_y, test_X, cat_mask, immutable_mask


def load_author_models_from_hf(hf_repo, input_dim, device):
    """Load author's pretrained models from HuggingFace (RF and ANN only)"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genre_models")
    os.makedirs(cache_dir, exist_ok=True)

    # Download models
    rf_path = hf_hub_download(
        repo_id=hf_repo, filename="rf_state.pkl", cache_dir=cache_dir
    )
    ann_path = hf_hub_download(
        repo_id=hf_repo, filename="ann_state.pth", cache_dir=cache_dir
    )

    # Load RF
    with open(rf_path, "rb") as f:
        rf_clf = pickle.load(f)

    # Load ANN
    ann_state = torch.load(ann_path, map_location=device)
    ann_clf = BinaryClassifier(layer_sizes=[input_dim, 10, 10, 10])
    ann_clf.load_state_dict(ann_state["state_dict"])
    ann_clf.to(device)
    ann_clf.eval()

    return rf_clf, ann_clf


def get_factuals(test_X, ann_clf, device, n_samples=None):
    """Get negative instances that need recourse"""
    # Use ANN to identify negative instances
    test_pred_prob = ann_clf(test_X.to(device)).cpu().detach()
    test_pred_binary = (test_pred_prob <= 0.5).squeeze()

    xf_r = test_X[test_pred_binary].to(torch.float32)

    if n_samples is not None and len(xf_r) > n_samples:
        xf_r = xf_r[:n_samples]

    return xf_r


def evaluate(xf_r, sample_xcf, rf_clf, train_X, test_X):
    """Evaluate counterfactuals"""
    xf_r_cpu = xf_r.cpu().to(torch.float32)
    sample_xcf_cpu = sample_xcf.cpu().to(torch.float32)
    n_features = xf_r_cpu.shape[1]

    # Cost
    cost = torch.abs(sample_xcf_cpu - xf_r_cpu).sum(dim=1).mean().item()

    # Validity
    rf_pred_proba = rf_clf.predict_proba(sample_xcf_cpu.numpy())[:, 1]
    validity = (rf_pred_proba > 0.5).mean()

    # LOF
    all_X = torch.cat((train_X, test_X)).cpu().numpy()
    all_pred_rf = (rf_clf.predict_proba(all_X)[:, 1] > 0.5) * 1.0
    positive_instances = all_X[all_pred_rf == 1.0]

    lof_clf = LocalOutlierFactor(n_neighbors=5, novelty=True)
    lof_clf.fit(positive_instances)
    lof_predictions = lof_clf.predict(sample_xcf_cpu.numpy())
    lof_score = (lof_predictions == 1).mean()

    # Score
    score = validity + lof_score - (cost / n_features)

    return {"cost": cost, "validity": validity, "lof": lof_score, "score": score}


def reproduce_results():
    args = parse_args()
    genre_utils.set_seed(args.seed)
    device = torch.device(args.device)

    print(f"GenRe Reproduction - {args.dataset}")

    # Load author's data (tensor format)
    print("Loading author's data...")
    train_y, train_X, test_y, test_X, cat_mask, _ = load_author_data(args.dataset)
    input_dim = train_X.shape[1]

    # Load author's models from HuggingFace (RF and ANN only)
    print("Loading author's models from HuggingFace...")
    rf_clf, ann_clf = load_author_models_from_hf(args.hf_repo, input_dim, device)

    # Get factuals (tensor format)
    print(f"Selecting negative instances from {len(test_X)} test samples...")
    xf_r = get_factuals(test_X, ann_clf, device, args.n_samples)
    print(f"Found {len(xf_r)} factuals that need recourse")

    # Convert factuals to DataFrame (adapt to repo interface)
    factuals_df = pd.DataFrame(xf_r.cpu().numpy())

    # Initialize GenRe with either cat_mask or data object
    print("Initializing GenRe...")
    
    if args.use_data_object:
        # Option 1: Use DataCatalog object (cat_mask calculated from data)
        dataset_mapping = {"adult-all": "adult", "compas-all": "compas", "heloc": "heloc"}
        repo_dataset = dataset_mapping.get(args.dataset, args.dataset)
        data = DataCatalog(repo_dataset, model_type="mlp", train_split=0.8)
        
        genre = GenRe(
            mlmodel=ann_clf,
            hyperparams={
                "data": data,  # Pass DataCatalog - cat_mask calculated automatically
                "temp": args.temp,
                "sigma": args.sigma,
                "best_k": args.best_k,
                "device": args.device,
            },
        )
    else:
        # Option 2: Use author's cat_mask directly (default)
        genre = GenRe(
            mlmodel=ann_clf,
            hyperparams={
                "cat_mask": cat_mask,  # Pass cat_mask directly
                "temp": args.temp,
                "sigma": args.sigma,
                "best_k": args.best_k,
                "device": args.device,
            },
        )
    
    # genre = GenRe(
    #         mlmodel=ann_clf,
    #         hyperparams={
    #             "cat_mask": cat_mask,  # Pass cat_mask directly
    #             "temp": args.temp,
    #             "sigma": args.sigma,
    #             "best_k": args.best_k,
    #             "device": args.device,
    #         },
    # )

    # Generate counterfactuals using standard interface
    print("Generating counterfactuals...")
    cfs_df = genre.get_counterfactuals(factuals_df)

    # Convert back to tensor for evaluation
    sample_xcf = torch.tensor(cfs_df.values).to(torch.float32)

    # Evaluate
    print("Evaluating...")
    results = evaluate(xf_r, sample_xcf, rf_clf, train_X, test_X)

    print("\nResults:")
    print(f"  Cost:     {results['cost']:.4f}")
    print(f"  Validity: {results['validity']:.4f}")
    print(f"  LOF:      {results['lof']:.4f}")
    print(f"  Score:    {results['score']:.4f}")

    # Verify results (Adult dataset expectations)
    if args.dataset == "adult-all":
        print("\nVerification (Adult Dataset - Paper Table 3):")

        EXPECTED_VAL = (0.90, 1.00)
        EXPECTED_LOF = (0.88, 1.00)
        EXPECTED_COST = (0.65, 0.75)
        EXPECTED_SCORE = (1.75, 2.00)

        val = results["validity"]
        lof = results["lof"]
        cost = results["cost"]
        score = results["score"]

        try:
            assert EXPECTED_VAL[0] <= val <= EXPECTED_VAL[1]
            print(f"PASS Val = {val:.4f} is within expected range {EXPECTED_VAL}")
        except AssertionError:
            print(f"FAIL Val = {val:.4f} outside expected range {EXPECTED_VAL}")

        try:
            assert EXPECTED_LOF[0] <= lof <= EXPECTED_LOF[1]
            print(f"PASS LOF = {lof:.4f} is within expected range {EXPECTED_LOF}")
        except AssertionError:
            print(f"FAIL LOF = {lof:.4f} outside expected range {EXPECTED_LOF}")

        try:
            assert EXPECTED_COST[0] <= cost <= EXPECTED_COST[1]
            print(f"PASS Cost = {cost:.4f} is within expected range {EXPECTED_COST}")
        except AssertionError:
            print(f"FAIL Cost = {cost:.4f} outside expected range {EXPECTED_COST}")

        try:
            assert EXPECTED_SCORE[0] <= score <= EXPECTED_SCORE[1]
            print(f"PASS Score = {score:.4f} is within expected range {EXPECTED_SCORE}")
        except AssertionError:
            print(f"FAIL Score = {score:.4f} outside expected range {EXPECTED_SCORE}")


def test_compatibility(dataset_name, model_type, backend):
    """Test GenRe compatibility with repo's DataCatalog and ModelCatalog"""
    dataset = DataCatalog(dataset_name, model_type=model_type, train_split=0.8)
    model = ModelCatalog(dataset, model_type, backend)
    
    factuals = dataset.df_train.drop(columns=[dataset.target]).sample(
        n=5, random_state=RANDOM_SEED
    )
    
    genre = GenRe(model, hyperparams={"data": dataset})
    
    # Generate counterfactual examples
    counterfactuals = genre.get_counterfactuals(factuals)
    print(counterfactuals)


if __name__ == "__main__":
    # reproduce_results()
    test_compatibility("genre_adult", "mlp", "pytorch") 