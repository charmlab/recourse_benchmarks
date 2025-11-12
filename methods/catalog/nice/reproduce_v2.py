"""
Test NICE on adult dataset to reproduce paper results

Tests all 4 NICE variants (none, sparsity, proximity, plausibility) on the
adult dataset with both Random Forest and MLP models.

Sample size: 200 instances (matching paper experiments)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest
import time
import os
import sys

from data.catalog import DataCatalog
from methods import NICE
from models.catalog import ModelCatalog
from models.negative_instances import predict_negative_instances

# Import NICE_experiments components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nice_experiments'))
from pmlb_fetcher import PmlbFetcher
from adapter import NICEExperimentsDataAdapter, NICEExperimentsModelAdapter

from methods import NICE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# EXPECTED RANGES 
EXPECTED_RANGES = {
    "forest": {
        "none": {
            "sparsity": (2.0, 3.0),
            "proximity": (0.6, 0.9),
            "plausibility": (0.01, 0.02),
        },
        "sparsity": {
            "sparsity": (1.25, 2.25),
            "proximity": (0.4, 0.7),
            "plausibility": (0.01, 0.02),
        },
        "proximity": {
            "sparsity": (1.25, 2.25),
            "proximity": (0.4, 0.7),
            "plausibility": (0.01, 0.02),
        },
        "plausibility": {
            "sparsity": (1.75, 2.75),
            "proximity": (0.6, 0.9),
            "plausibility": (0.01, 0.02),
        },
    },
    "mlp": {
        "none": {
            "sparsity": (3.5, 4.0),
            "proximity": (2.9, 3.4),
            "plausibility": (0.18, 0.23),
        },
        "sparsity": {
            "sparsity": (1.0, 1.5),
            "proximity": (0.8, 1.3),
            "plausibility": (0.18, 0.23),
        },
        "proximity": {
            "sparsity": (1.2, 1.7),
            "proximity": (0.8, 1.3),
            "plausibility": (0.18, 0.23),
        },
        "plausibility": {
            "sparsity": (2.1, 2.6),
            "proximity": (1.8, 2.3),
            "plausibility": (0.18, 0.23),
        },
    },
}

@pytest.mark.parametrize(
    "model_type,optimization",
    [
        ("forest", "none"),
        ("forest", "sparsity"),
        ("forest", "proximity"),
        ("forest", "plausibility"),
        # ("mlp", "none"),
        # ("mlp", "sparsity"),
        # ("mlp", "proximity"),
        # ("mlp", "plausibility"),
    ],
)

def setup_nice_experiments_data_and_model(model_type):
    """
    Load data and train model using NICE_experiments approach
    
    Returns data_adapted and model_adapted that are API-compatible with repo
    """
    # Load NICE_experiments data
    fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
    nice_data = fetcher.dataset
    
    # Create data adapter
    data_adapted = NICEExperimentsDataAdapter(nice_data)
    
    # Train model using NICE_experiments data
    X_train = data_adapted.df_train.drop(columns=['y']).values
    y_train = data_adapted.df_train['y'].values
    
    # use the best parameters give by grid search
    if model_type == "forest":
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            max_depth=25
        )
    elif model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            random_state=42,
            max_iter=300
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print(f"Training {model_type} model on NICE_experiments data...")
    model.fit(X_train, y_train)
    
    # Evaluate
    X_test = data_adapted.df_test.drop(columns=['y']).values
    y_test = data_adapted.df_test['y'].values
    acc = model.score(X_test, y_test)
    print(f"  Model accuracy: {acc:.4f}")
    
    # Create model adapter
    model_adapted = NICEExperimentsModelAdapter(model, data_adapted)
    
    return data_adapted, model_adapted


def get_negative_instances_nice_experiments(model_adapted, data_adapted, n=200):
    """
    Get negative instances from test set
    """
    df_test = data_adapted.df_test.copy()
    X_test = df_test.drop(columns=['y'])
    
    # Predict
    y_pred = model_adapted.predict(X_test)
    
    # Filter negative predictions
    negative_mask = (y_pred == 0)
    negative_instances = df_test[negative_mask].head(n)
    
    print(f"Found {len(negative_instances)} negative instances")
    
    return negative_instances

def test_nice_coverage(model_type, optimization):
    """
    Test that NICE achieves 100% coverage.
    
    Critical requirement: NICE should always find a counterfactual.
    """

    # data = DataCatalog("adult", model_type=model_type, train_split=0.7)
    # model = ModelCatalog(data, model_type=model_type, backend="sklearn")
    data, model = setup_nice_experiments_data_and_model(model_type)
    factuals = get_negative_instances_nice_experiments(model, data, n=200)
    
    nice = NICE(mlmodel=model, hyperparams={"optimization": optimization})
    
    factuals = predict_negative_instances(model, data).iloc[:200]
    counterfactuals = nice.get_counterfactuals(factuals)
    
    # Check coverage
    assert not counterfactuals.isna().any().any(), \
        f"NICE({optimization}) on {model_type} did not achieve 100% coverage"
    
    # Check predictions flip
    predictions = model.predict(counterfactuals)
    num_flipped = (predictions >= 0.5).sum()
    
    assert num_flipped == len(factuals), \
        f"NICE({optimization}) on {model_type} only flipped {num_flipped}/{len(factuals)}"

def test_nice_quality(model_type, optimization):
    """
    Test that NICE produces quality counterfactuals with all metrics in expected ranges.
    """

    # data = DataCatalog("adult", model_type=model_type, train_split=0.7)
    # model = ModelCatalog(data, model_type=model_type, backend="sklearn")
    data, model = setup_nice_experiments_data_and_model(model_type)
    factuals = get_negative_instances_nice_experiments(model, data, n=200)
    
    nice = NICE(mlmodel=model, hyperparams={"optimization": optimization})
    
    factuals = predict_negative_instances(model, data).iloc[:200]
    
    # Measure CPU time
    start_time = time.time()
    counterfactuals = nice.get_counterfactuals(factuals)
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) * 1000) / len(factuals)
    
    factuals_ordered = model.get_ordered_features(factuals)
    
    # Calculate sparsity
    sparsity = (factuals_ordered.values != counterfactuals.values).sum(axis=1)
    avg_sparsity = sparsity.mean()
    
    # Calculate proximity (HEOM)
    X_train = data.df_train.drop(columns=["y"]).values
    ranges = {}
    for idx in nice.num_feat_idx:
        ranges[idx] = X_train[:, idx].max() - X_train[:, idx].min()
    
    proximities = []
    for i in range(len(factuals)):
        distance = 0
        for idx in nice.cat_feat_idx:
            if factuals_ordered.values[i, idx] != counterfactuals.values[i, idx]:
                distance += 1
        for idx in nice.num_feat_idx:
            diff = abs(factuals_ordered.values[i, idx] - counterfactuals.values[i, idx])
            distance += diff / ranges[idx]
        proximities.append(distance)
    
    avg_proximity = np.mean(proximities)
    
    # Calculate plausibility (AE)
    from methods.catalog.nice.library.autoencoder import AutoEncoder
    
    feature_names = data.df_train.drop(columns=["y"]).columns.tolist()
    cat_feat_idx = [feature_names.index(f) for f in data.categorical]
    num_feat_idx = [feature_names.index(f) for f in data.continuous]
    
    ae = AutoEncoder(
        X_train=X_train,
        cat_feat_idx=cat_feat_idx,
        num_feat_idx=num_feat_idx,
        epochs=100,
        batch_size=32,
        verbose=0,
        early_stopping=True,
        patience=5,
    )
    
    ae_errors = ae(counterfactuals.values)
    avg_ae_error = ae_errors.mean()
    
    # Get expected ranges for this model type and optimization
    expected = EXPECTED_RANGES[model_type][optimization]
    
    # Assert all metrics are in expected ranges
    assert expected["sparsity"][0] <= avg_sparsity <= expected["sparsity"][1], \
        f"NICE({optimization}) on {model_type}: Sparsity {avg_sparsity:.2f} outside expected range {expected['sparsity']}"
    
    assert expected["proximity"][0] <= avg_proximity <= expected["proximity"][1], \
        f"NICE({optimization}) on {model_type}: Proximity {avg_proximity:.2f} outside expected range {expected['proximity']}"
    
    assert expected["plausibility"][0] <= avg_ae_error <= expected["plausibility"][1], \
        f"NICE({optimization}) on {model_type}: Plausibility {avg_ae_error:.4f} outside expected range {expected['plausibility']}"


@pytest.mark.parametrize("model_type", ["forest", "mlp"])
def test_nice_variants_comparison(model_type):
    """
    Compare all 4 NICE variants on the same instances.
    
    Reproduces Table 5 comparison with detailed metrics.
    """

    data = DataCatalog("adult", model_type=model_type, train_split=0.7)
    model = ModelCatalog(data, model_type=model_type, backend="sklearn")
    
    factuals = predict_negative_instances(model, data).iloc[:200]
    
    # Train autoencoder once
    from methods.catalog.nice.library.autoencoder import AutoEncoder
    
    X_train = data.df_train.drop(columns=["y"]).values
    feature_names = data.df_train.drop(columns=["y"]).columns.tolist()
    cat_feat_idx = [feature_names.index(f) for f in data.categorical]
    num_feat_idx = [feature_names.index(f) for f in data.continuous]
    
    ae = AutoEncoder(
        X_train=X_train,
        cat_feat_idx=cat_feat_idx,
        num_feat_idx=num_feat_idx,
        epochs=100,
        batch_size=32,
        verbose=0,
        early_stopping=True,
        patience=5,
    )
    
    # Calculate ranges once
    ranges = {}
    for idx in num_feat_idx:
        ranges[idx] = X_train[:, idx].max() - X_train[:, idx].min()
    
    results = {}
    
    for opt in ["none", "sparsity", "proximity", "plausibility"]:
        nice = NICE(mlmodel=model, hyperparams={"optimization": opt})
        
        start_time = time.time()
        cfs = nice.get_counterfactuals(factuals)
        end_time = time.time()
        
        avg_cpu_time_ms = ((end_time - start_time) * 1000) / len(factuals)
        
        factuals_ordered = model.get_ordered_features(factuals)
        
        # Sparsity
        sparsity = (factuals_ordered.values != cfs.values).sum(axis=1)
        
        # Proximity (HEOM)
        proximities = []
        for i in range(len(factuals)):
            distance = 0
            for idx in cat_feat_idx:
                if factuals_ordered.values[i, idx] != cfs.values[i, idx]:
                    distance += 1
            for idx in num_feat_idx:
                diff = abs(factuals_ordered.values[i, idx] - cfs.values[i, idx])
                distance += diff / ranges[idx]
            proximities.append(distance)
        
        proximity = np.array(proximities)
        
        # Plausibility (AE)
        ae_errors = ae(cfs.values)
        
        # Coverage
        coverage = (~cfs.isna().any(axis=1)).sum()
        
        results[opt] = {
            "coverage": coverage,
            "cpu_time_avg_ms": avg_cpu_time_ms,
            "avg_sparsity": sparsity.mean(),
            "std_sparsity": sparsity.std(),
            "avg_proximity": proximity.mean(),
            "std_proximity": proximity.std(),
            "avg_plausibility": ae_errors.mean(),
            "std_plausibility": ae_errors.std(),
        }
    
    # Print comparison table
    print(f"\n{'='*100}")
    print(f"Comparison of NICE Variants (reproducing Table 5 from paper)")
    print(f"{'='*100}")
    print(f"Dataset: adult, Model: {model_type.upper()}, n={len(factuals)}")
    print(f"{'-'*100}")
    print(f"{'Variant':<15} {'Coverage':<12} {'CPU (ms)':<12} {'Sparsity':<20} "
          f"{'Proximity (HEOM)':<20} {'Plausibility'}")
    print(f"{'-'*100}")
    
    for opt, metrics in results.items():
        print(f"{opt:<15} "
              f"{metrics['coverage']}/{len(factuals):<10} "
              f"{metrics['cpu_time_avg_ms']:>8.2f}    "
              f"{metrics['avg_sparsity']:>5.2f} ± {metrics['std_sparsity']:<4.2f}      "
              f"{metrics['avg_proximity']:>6.2f} ± {metrics['std_proximity']:<5.2f}    "
              f"{metrics['avg_plausibility']:>6.4f} ± {metrics['std_plausibility']:<6.4f}")
    
    print(f"{'='*100}")
    
    # Assertions: Verify expectations
    for opt, metrics in results.items():
        assert metrics["coverage"] == len(factuals), \
            f"NICE({opt}) on {model_type} failed coverage: {metrics['coverage']}/{len(factuals)}"
    
    # Sparsity optimization should have best sparsity
    spars_sparsity = results["sparsity"]["avg_sparsity"]
    for opt in ["none", "proximity", "plausibility"]:
        assert spars_sparsity <= results[opt]["avg_sparsity"] + 0.5, \
            f"NICE(sparsity) should have best sparsity on {model_type}, but NICE({opt}) is better"
    
    # Proximity optimization should have best proximity
    prox_proximity = results["proximity"]["avg_proximity"]
    for opt in ["none", "plausibility"]:
        assert prox_proximity <= results[opt]["avg_proximity"] + 0.3, \
            f"NICE(proximity) should have best proximity on {model_type}, but NICE({opt}) is better"
    
    # None should be very plausible
    # none_plausibility = results["none"]["avg_plausibility"]
    # assert none_plausibility <= 0.02, \
    #     f"NICE(none) should be very plausible on {model_type}: {none_plausibility:.4f}"
    
    # None should be fastest
    # none_time = results["none"]["cpu_time_avg_ms"]
    # for opt in ["sparsity", "proximity", "plausibility"]:
    #     assert none_time <= results[opt]["cpu_time_avg_ms"], \
    #         f"NICE(none) should be fastest on {model_type}, but NICE({opt}) is faster"


@pytest.mark.parametrize(
    "dataset_name,model_type",
    [("adult", "forest"), ("adult", "mlp")],
)
def test_nice_benchmark_compatibility(dataset_name, model_type):
    """
    Test that NICE integrates properly with the benchmark framework.
    """
    data = DataCatalog(dataset_name, model_type=model_type, train_split=0.7)
    model = ModelCatalog(data, model_type=model_type, backend="sklearn")
    
    nice = NICE(mlmodel=model)
    
    factuals = predict_negative_instances(model, data).iloc[:200]
    cfs = nice.get_counterfactuals(factuals)
    
    # Check output format
    assert isinstance(cfs, pd.DataFrame), "Output should be DataFrame"
    assert cfs.shape[0] == factuals.shape[0], "Should return same number of counterfactuals"
    assert not cfs.isna().all(axis=1).any(), "Should not have all-NaN rows"
    assert list(cfs.columns) == model.feature_input_order, \
        "Counterfactual features should match model's expected input order"


if __name__ == "__main__":
    print("=" * 100)
    print("NICE Integration Tests - Reproducing Paper Results")
    print("=" * 100)
    print("\nDataset: adult (NICE_experiments preprocessing)")
    print("Models: Random Forest, MLP")
    print("Sample size: 200 instances (matching paper experiments)")
    print("=" * 100)
    
    for model_type in ["forest"]: #, "mlp"]:  # 先只测 forest，之后取消注释
        print(f"\n{'#'*100}")
        print(f"# Testing with {model_type.upper()} model")
        print(f"{'#'*100}")
        
        # Test 1: Coverage
        print(f"\n[Test 1] Testing 100% Coverage Guarantee on {model_type.upper()}...")
        all_passed = True
        for opt in ["none", "sparsity", "proximity", "plausibility"]:
            try:
                test_nice_coverage(model_type, opt)
                print(f"  ✓ NICE({opt}) coverage test passed")
            except AssertionError as e:
                print(f"  ✗ NICE({opt}) failed: {e}")
                all_passed = False
        
        if all_passed:
            print(f"  ✓ All coverage tests passed for {model_type.upper()}")
        
        # Test 2: Quality metrics (取消注释来启用)
        print(f"\n[Test 2] Testing Quality Metrics on {model_type.upper()}...")
        all_passed = True
        for opt in ["none", "sparsity", "proximity", "plausibility"]:
            try:
                test_nice_quality(model_type, opt)
                print(f"  ✓ NICE({opt}) quality test passed")
            except AssertionError as e:
                print(f"  ✗ NICE({opt}) failed: {e}")
                all_passed = False
        
        if all_passed:
            print(f"  ✓ All quality tests passed for {model_type.upper()}")
        
        # Test 3: Variant comparison (取消注释来启用)
        print(f"\n[Test 3] Comparing All Variants on {model_type.upper()}...")
        try:
            test_nice_variants_comparison(model_type)
            print(f"  ✓ Variant comparison test passed for {model_type.upper()}")
        except AssertionError as e:
            print(f"  ✗ Variant comparison failed: {e}")
        
        # Test 4: Benchmark compatibility (取消注释来启用)
        print(f"\n[Test 4] Testing Benchmark Compatibility on {model_type.upper()}...")
        try:
            test_nice_benchmark_compatibility("adult", model_type)
            print(f"  ✓ Benchmark compatibility test passed for {model_type.upper()}")
        except AssertionError as e:
            print(f"  ✗ Benchmark compatibility failed: {e}")