"""
GenRe Reproduction Script for Adult Dataset

Models are automatically downloaded from HuggingFace Hub.
https://huggingface.co/jamie250/genrereproduce

"""
import argparse
import os
import sys
import pickle
import warnings
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# Import huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError(
        "huggingface_hub is required to download models. "
        "Install it with: pip install huggingface-hub"
    )

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add GenRe library to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENRE_LIB_PATH = os.path.join(SCRIPT_DIR, 'library')

# added this line to solve pytest error
sys.path.insert(0, GENRE_LIB_PATH)

# Import author's modules
import data.utils as dutils
import models.binnedpm as bpm
import recourse.utils as rutils
from recourse.genre import GenRe as GenReOriginal
from models.classifiers.ann import BinaryClassifier
import utils as genre_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce GenRe results')
    parser.add_argument('--dataset', type=str, default='adult-all',
                       choices=['adult-all', 'compas-all', 'heloc'],
                       help='Dataset to use')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda:0, mps)')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Number of samples to generate CFs for (None = all negative instances)')
    parser.add_argument('--temp', type=float, default=10.0,
                       help='Temperature for GenRe sampling')
    parser.add_argument('--sigma', type=float, default=0.0,
                       help='Noise level for GenRe sampling')
    parser.add_argument('--best_k', type=int, default=10,
                       help='Number of candidates to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--hf_repo', type=str,
                       default='jamie250/genrereproduce',
                       help='HuggingFace repository ID')
    return parser.parse_args()


def download_models_from_hf(hf_repo, dataset_name, cache_dir=None):
    """Download pre-trained models from HuggingFace Hub"""
    print(f"\n{'='*80}")
    print(f"Downloading models from HuggingFace: {hf_repo}")
    print(f"{'='*80}")
    
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "genre_models")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download files
    try:
        print("Downloading RF model...")
        rf_path = hf_hub_download(
            repo_id=hf_repo,
            filename="rf_state.pkl",
            cache_dir=cache_dir
        )
        print(f"  OK Downloaded to: {rf_path}")
        
        print("Downloading ANN model...")
        ann_path = hf_hub_download(
            repo_id=hf_repo,
            filename="ann_state.pth",
            cache_dir=cache_dir
        )
        print(f"  OK Downloaded to: {ann_path}")
        
        print("Downloading GenRe model...")
        genre_path = hf_hub_download(
            repo_id=hf_repo,
            filename="genre_state.pth",
            cache_dir=cache_dir
        )
        print(f"  OK Downloaded to: {genre_path}")
        
        print(f"{'='*80}\n")
        
        return rf_path, ann_path, genre_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to download models from HuggingFace: {e}")


def load_data(dataset_name):
    """Load dataset using author's data loader"""
    print(f"\n{'='*80}")
    print(f"Loading {dataset_name} dataset...")
    print(f"{'='*80}")
    
    train_y, train_X, test_y, test_X, cat_mask, immutable_mask = dutils.load_dataset(
        dataset_name,
        ret_tensor=True,
        min_max=True,
        ret_masks=True
    )
    
    print(f"OK Data loaded successfully")
    print(f"  Train: {train_X.shape[0]} samples, {train_X.shape[1]} features")
    print(f"  Test:  {test_X.shape[0]} samples, {test_X.shape[1]} features")
    print(f"  Value range: [{train_X.min():.4f}, {train_X.max():.4f}]")
    
    return train_y, train_X, test_y, test_X, cat_mask, immutable_mask


def load_models_hf(hf_repo, dataset_name, input_dim, device):
    """Load pre-trained models from HuggingFace Hub"""
    print(f"\n{'='*80}")
    print(f"Loading pre-trained models (HuggingFace)...")
    print(f"{'='*80}")
    
    # Download models
    rf_path, ann_path, genre_path = download_models_from_hf(hf_repo, dataset_name)
    
    # Load RF
    print(f"Loading RF from: {rf_path}")
    with open(rf_path, 'rb') as f:
        rf_clf = pickle.load(f)
    print(f"OK RF loaded")
    
    # Load ANN
    print(f"Loading ANN from: {ann_path}")
    ann_state = torch.load(ann_path, map_location=device)
    ann_clf = BinaryClassifier(layer_sizes=[input_dim, 10, 10, 10])
    ann_clf.load_state_dict(ann_state['state_dict'])
    ann_clf.to(device)
    ann_clf.eval()
    print(f"OK ANN loaded")
    
    # Load GenRe
    print(f"Loading GenRe from: {genre_path}")
    genre_state = torch.load(genre_path, map_location=device)
    pair_model = bpm.PairedTransformerBinned(
        n_bins=50,
        num_inputs=input_dim,
        num_labels=1,
        num_encoder_layers=16,
        num_decoder_layers=16,
        emb_size=32,
        nhead=8,
        dim_feedforward=32,
        dropout=0.1
    )
    pair_model.load_state_dict(genre_state['state_dict'])
    pair_model.to(device)
    pair_model.eval()
    print(f"OK GenRe Transformer loaded")
    
    return rf_clf, ann_clf, pair_model


def generate_counterfactuals(test_X, test_y, ann_clf, pair_model, cat_mask, 
                             device, temp, sigma, best_k, n_samples=None):
    """Generate counterfactuals using GenRe"""
    print(f"\n{'='*80}")
    print(f"Generating Counterfactuals...")
    print(f"{'='*80}")
    
    # Get negative instances from test set
    test_pred_prob = ann_clf(test_X.to(device)).cpu().detach()
    test_pred_binary = (test_pred_prob <= 0.5).squeeze()
    
    xf_r = test_X[test_pred_binary].to(torch.float32)
    
    if n_samples is not None and len(xf_r) > n_samples:
        xf_r = xf_r[:n_samples]
    
    xf_r = xf_r.to(device)
    print(f"OK Found {len(xf_r)} negative instances (factuals)")
    
    # Initialize GenRe recourse module
    rec_module = GenReOriginal(
        pair_model=pair_model,
        temp=temp,
        sigma=sigma,
        best_k=best_k,
        ann_clf=ann_clf,
        ystar=1.0,
        cat_mask=cat_mask
    )
    
    print(f"Generating counterfactuals with:")
    print(f"  Temperature: {temp}")
    print(f"  Sigma: {sigma}")
    print(f"  Best-k: {best_k}")
    
    # Generate counterfactuals
    with torch.no_grad():
        sample_xcf = rec_module(xf_r).squeeze()
        sample_xcf = sample_xcf.to(torch.float32)
    
    print(f"OK Generated {len(sample_xcf)} counterfactuals")
    
    return xf_r, sample_xcf


def evaluate_counterfactuals(xf_r, sample_xcf, rf_clf, train_X, test_X, device):
    """
    Evaluate counterfactual quality using metrics from the paper:
    1. Cost: L1 distance between x and x+ 
    2. Val (Validity): RF classifier assigns y+
    3. LOF (Plausibility): Local Outlier Factor score
    4. Score: Val + LOF - (Cost/d)
    """
    print(f"\n{'='*80}")
    print(f"Evaluation Results (Paper Metrics)")
    print(f"{'='*80}")
    
    xf_r_cpu = xf_r.cpu().to(torch.float32)
    sample_xcf_cpu = sample_xcf.cpu().to(torch.float32)
    n_features = xf_r_cpu.shape[1]
    
    # 1. Cost: L1 distance
    cost_per_instance = torch.abs(sample_xcf_cpu - xf_r_cpu).sum(dim=1)
    cost_mean = cost_per_instance.mean().item()
    print(f"\n1. Cost (L1 distance): {cost_mean:.4f}")
    
    # 2. Val (Validity)
    rf_pred_proba = rf_clf.predict_proba(sample_xcf_cpu.numpy())[:, 1]
    validity = (rf_pred_proba > 0.5).mean()
    print(f"\n2. Val (Validity with RF): {validity:.4f}")
    
    # 3. LOF (Plausibility)
    all_X = torch.cat((train_X, test_X)).cpu().numpy()
    all_pred_rf = (rf_clf.predict_proba(all_X)[:, 1] > 0.5) * 1.0
    positive_instances = all_X[all_pred_rf == 1.0]
    
    lof_clf = LocalOutlierFactor(n_neighbors=5, novelty=True)
    lof_clf.fit(positive_instances)
    
    lof_predictions = lof_clf.predict(sample_xcf_cpu.numpy())
    lof_score = (lof_predictions == 1).mean()
    print(f"\n3. LOF (Plausibility): {lof_score:.4f}")
    
    # 4. Score
    score = validity + lof_score - (cost_mean / n_features)
    print(f"\n4. Score (Val + LOF - Cost/d): {score:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Cost:     {cost_mean:.4f}")
    print(f"Val:      {validity:.4f}")
    print(f"LOF:      {lof_score:.4f}")
    print(f"Score:    {score:.4f}")
    
    return {
        'cost': cost_mean,
        'validity': validity,
        'lof': lof_score,
        'score': score,
        'n_features': n_features
    }


def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"GenRe Reproduction - {args.dataset}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device: {args.device}")
    print(f"  HF Repo: {args.hf_repo}")
    print(f"  Temperature: {args.temp}")
    print(f"  Sigma: {args.sigma}")
    print(f"  Best-k: {args.best_k}")
    print(f"  Seed: {args.seed}")
    
    # Set seed
    genre_utils.set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device)
    
    # 1. Load data
    train_y, train_X, test_y, test_X, cat_mask, immutable_mask = load_data(args.dataset)
    input_dim = train_X.shape[1]
    
    # 2. Load models from HuggingFace
    rf_clf, ann_clf, pair_model = load_models_hf(
        args.hf_repo, args.dataset, input_dim, device
    )
    
    # 3. Generate counterfactuals
    xf_r, sample_xcf = generate_counterfactuals(
        test_X, test_y, ann_clf, pair_model, cat_mask,
        device, args.temp, args.sigma, args.best_k, args.n_samples
    )
    
    # 4. Evaluate
    results = evaluate_counterfactuals(
        xf_r, sample_xcf, rf_clf, train_X, test_X, device
    )
    
    # 5. Verify results
    print(f"\n{'='*80}")
    print(f"Verification (Adult Dataset - Paper Table 3)")
    print(f"{'='*80}")
    
    EXPECTED_VAL = (0.95, 1.00)   
    EXPECTED_LOF = (0.95, 1.00)   
    EXPECTED_COST = (0.65, 0.75)    
    EXPECTED_SCORE = (1.88, 2.00)
    
    val = results['validity']
    lof = results['lof']
    cost = results['cost']
    score = results['score']
    
    # Check ranges
    try:
        assert EXPECTED_VAL[0] <= val <= EXPECTED_VAL[1]
        print(f"OK Val = {val:.4f} is within expected range {EXPECTED_VAL}")
    except AssertionError:
        print(f"FAIL Val = {val:.4f} outside expected range {EXPECTED_VAL}")
    
    try:
        assert EXPECTED_LOF[0] <= lof <= EXPECTED_LOF[1]
        print(f"OK LOF = {lof:.4f} is within expected range {EXPECTED_LOF}")
    except AssertionError:
        print(f"FAIL LOF = {lof:.4f} outside expected range {EXPECTED_LOF}")
    
    try:
        assert EXPECTED_COST[0] <= cost <= EXPECTED_COST[1]
        print(f"OK Cost = {cost:.4f} is within expected range {EXPECTED_COST}")
    except AssertionError:
        print(f"FAIL Cost = {cost:.4f} outside expected range {EXPECTED_COST}")
    
    try:
        assert EXPECTED_SCORE[0] <= score <= EXPECTED_SCORE[1]
        print(f"OK Score = {score:.4f} is within expected range {EXPECTED_SCORE}")
    except AssertionError:
        print(f"FAIL Score = {score:.4f} outside expected range {EXPECTED_SCORE}")
    
    print(f"\n{'='*80}")
    print(f"OK Reproduction Complete!")
    print(f"{'='*80}")
    
    # Save results
    output_dir = os.path.join("results", "genre_reproduction", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"GenRe Reproduction Results - {args.dataset}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"HF Repo: {args.hf_repo}\n")
        f.write(f"\nMetrics:\n")
        for key, value in results.items():
            if value is not None:
                f.write(f"{key}: {value}\n")
    
    print(f"\nOK Results saved to: {results_file}")


if __name__ == "__main__":
    main()