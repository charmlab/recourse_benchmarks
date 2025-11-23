"""
GenRe Reproduction Script

Reproduces GenRe paper results using the RecourseMethod interface.

Usage:
    python reproduce.py --dataset adult-all --device cpu
"""

import argparse
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("Install huggingface-hub: pip install huggingface-hub")

# Add GenRe library to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENRE_LIB_PATH = os.path.join(SCRIPT_DIR, 'library')
sys.path.insert(0, GENRE_LIB_PATH)

# Add repo root to path for imports
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../..'))
sys.path.insert(0, REPO_ROOT)

# Debug: Print paths
print(f"GENRE_LIB_PATH: {GENRE_LIB_PATH}")
print(f"Path exists: {os.path.exists(GENRE_LIB_PATH)}")
print(f"Contents: {os.listdir(GENRE_LIB_PATH) if os.path.exists(GENRE_LIB_PATH) else 'N/A'}")


# Import author's modules
import library.data.utils as dutils
import library.models.binnedpm as bpm
from library.models.classifiers.ann import BinaryClassifier
import utils as genre_utils

# Import our GenRe wrapper
from methods.catalog.genre import GenRe


def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce GenRe results')
    parser.add_argument('--dataset', type=str, default='adult-all',
                       choices=['adult-all', 'compas-all', 'heloc'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--temp', type=float, default=10.0)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--best_k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hf_repo', type=str, default='jamie250/genrereproduce')
    return parser.parse_args()


def load_author_data(dataset_name):
    """Load author's data with preprocessing"""
    train_y, train_X, test_y, test_X, cat_mask, immutable_mask = dutils.load_dataset(
        dataset_name, ret_tensor=True, min_max=True, ret_masks=True
    )
    return train_y, train_X, test_y, test_X, cat_mask, immutable_mask


def load_author_models_from_hf(hf_repo, input_dim, device):
    """Load author's pretrained models from HuggingFace"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genre_models")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download models
    rf_path = hf_hub_download(repo_id=hf_repo, filename="rf_state.pkl", cache_dir=cache_dir)
    ann_path = hf_hub_download(repo_id=hf_repo, filename="ann_state.pth", cache_dir=cache_dir)
    genre_path = hf_hub_download(repo_id=hf_repo, filename="genre_state.pth", cache_dir=cache_dir)
    
    # Load RF
    with open(rf_path, 'rb') as f:
        rf_clf = pickle.load(f)
    
    # Load ANN
    ann_state = torch.load(ann_path, map_location=device)
    ann_clf = BinaryClassifier(layer_sizes=[input_dim, 10, 10, 10])
    ann_clf.load_state_dict(ann_state['state_dict'])
    ann_clf.to(device)
    ann_clf.eval()
    
    # Load Transformer
    genre_state = torch.load(genre_path, map_location=device)
    transformer = bpm.PairedTransformerBinned(
        n_bins=50, num_inputs=input_dim, num_labels=1,
        num_encoder_layers=16, num_decoder_layers=16,
        emb_size=32, nhead=8, dim_feedforward=32, dropout=0.1
    )
    transformer.load_state_dict(genre_state['state_dict'])
    transformer.to(device)
    transformer.eval()
    
    return rf_clf, ann_clf, transformer


# ========== FUTURE: Repo format loading functions ==========

def load_repo_data(dataset_name):
    """Load data using repo's DataCatalog (FUTURE)"""
    from data.catalog import DataCatalog
    
    # Map author's dataset names to repo names
    dataset_mapping = {
        'adult-all': 'adult',
        'compas-all': 'compas',
        'heloc': 'heloc'
    }
    repo_dataset = dataset_mapping.get(dataset_name, dataset_name)
    
    data = DataCatalog(repo_dataset, train_split=0.8)
    return data


def load_repo_models(data, device):
    """Load models using repo's ModelCatalog (FUTURE)"""
    from models.catalog import ModelCatalog
    
    # Load models
    rf_clf = ModelCatalog(data, model_type='rf', backend='sklearn')
    ann_clf = ModelCatalog(data, model_type='ann', backend='pytorch')
    
    # Note: Transformer not available in repo yet
    # For now, still need to load from HuggingFace
    transformer = None  # TODO: Get from repo when available
    
    return rf_clf, ann_clf, transformer


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
    
    return {'cost': cost, 'validity': validity, 'lof': lof_score, 'score': score}


def main():
    args = parse_args()
    genre_utils.set_seed(args.seed)
    device = torch.device(args.device)
    
    print(f"GenRe Reproduction - {args.dataset}")
    
    # ========== CURRENT: Load author's data and models ==========
    # 1. Load author's data (tensor format)
    print("Loading author's data...")
    train_y, train_X, test_y, test_X, cat_mask, _ = load_author_data(args.dataset)
    input_dim = train_X.shape[1]
    
    # 2. Load author's models from HuggingFace
    print("Loading author's models from HuggingFace...")
    rf_clf, ann_clf, transformer = load_author_models_from_hf(
        args.hf_repo, input_dim, device
    )
    
    # ========== FUTURE: Load repo's data and models ==========
    # Uncomment these lines when switching to repo format:
    #
    # # 1. Load repo's data
    # print("Loading repo's data...")
    # data = load_repo_data(args.dataset)
    # train_X = torch.tensor(data.df_train.drop(columns=[data.target]).values)
    # test_X = torch.tensor(data.df_test.drop(columns=[data.target]).values)
    # cat_mask = torch.tensor([1 if f in data.categorical else 0 
    #                          for f in data.df_train.columns if f != data.target])
    #
    # # 2. Load repo's models
    # print("Loading repo's models...")
    # rf_clf, ann_clf, transformer = load_repo_models(data, device)
    # # Note: Still need to load transformer from HF until repo provides it
    # _, _, transformer = load_author_models_from_hf(args.hf_repo, train_X.shape[1], device)
    
    # 3. Get factuals (tensor format)
    print(f"Selecting negative instances from {len(test_X)} test samples...")
    xf_r = get_factuals(test_X, ann_clf, device, args.n_samples)
    print(f"Found {len(xf_r)} factuals that need recourse")
    
    # 4. Convert factuals to DataFrame (adapt to repo interface)
    factuals_df = pd.DataFrame(xf_r.cpu().numpy())
    
    # ========== CURRENT: Initialize GenRe with author's components ==========
    # 5. Initialize GenRe using our wrapper
    print("Initializing GenRe...")
    genre = GenRe(
        mlmodel=ann_clf,  # Pass ANN directly
        hyperparams={
            'transformer': transformer,  # Pass transformer
            'cat_mask': cat_mask,  # Pass cat_mask
            'temp': args.temp,
            'sigma': args.sigma,
            'best_k': args.best_k,
            'device': args.device
        }
    )
    
    # ========== FUTURE: Use repo's ModelCatalog ==========
    # When switching to repo models, model.py must also switch to FUTURE.
    # The repo's ModelCatalog should have .transformer and .data.cat_mask
    #
    # genre = GenRe(
    #     mlmodel=ann_clf,  # This is ModelCatalog from load_repo_models()
    #     hyperparams={
    #         'temp': args.temp,
    #         'sigma': args.sigma,
    #         'best_k': args.best_k,
    #         'device': args.device
    #     }
    # )
    # Note: transformer and cat_mask will be accessed from mlmodel directly
    
    # 6. Generate counterfactuals using standard interface
    print("Generating counterfactuals...")
    cfs_df = genre.get_counterfactuals(factuals_df)
    
    # 7. Convert back to tensor for evaluation
    sample_xcf = torch.tensor(cfs_df.values).to(torch.float32)
    
    # 8. Evaluate
    print("Evaluating...")
    results = evaluate(xf_r, sample_xcf, rf_clf, train_X, test_X)
    
    print("\nResults:")
    print(f"  Cost:     {results['cost']:.4f}")
    print(f"  Validity: {results['validity']:.4f}")
    print(f"  LOF:      {results['lof']:.4f}")
    print(f"  Score:    {results['score']:.4f}")
    
    # 9. Verify results (Adult dataset expectations)
    if args.dataset == 'adult-all':
        print("\nVerification (Adult Dataset - Paper Table 3):")
        
        EXPECTED_VAL = (0.95, 1.00)
        EXPECTED_LOF = (0.95, 1.00)
        EXPECTED_COST = (0.65, 0.75)
        EXPECTED_SCORE = (1.88, 2.00)
        
        val = results['validity']
        lof = results['lof']
        cost = results['cost']
        score = results['score']
        
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
    
    # # Save results
    # output_dir = os.path.join("results", "genre_reproduction", args.dataset)
    # os.makedirs(output_dir, exist_ok=True)
    # results_file = os.path.join(output_dir, 'metrics.txt')
    # with open(results_file, 'w') as f:
    #     f.write(f"GenRe Reproduction Results - {args.dataset}\n")
    #     for key, value in results.items():
    #         f.write(f"{key}: {value}\n")
    
    # print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()