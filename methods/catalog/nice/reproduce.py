"""
Test NICE on adult dataset to reproduce paper results

Tests all 4 NICE variants (none, sparsity, proximity, plausibility) on the
adult dataset to verify:
1. 100% coverage (always finds a counterfactual)
2. Valid counterfactuals (prediction flips to positive class)
3. Quality metrics (sparsity, proximity, plausibility) - ALL 4 VARIANTS
4. CPU time performance

These tests aim to reproduce results from:
Brughmans et al. (2024) Tables 5-6 on the adult dataset.

Sample size: 200 instances (matching paper experiments)
"""
import numpy as np
import pandas as pd
import pytest
import time

from data.catalog import DataCatalog
from methods import NICE
from models.catalog import ModelCatalog
from models.negative_instances import predict_negative_instances


@pytest.mark.parametrize(
    "optimization",
    ["none", "sparsity", "proximity", "plausibility"],
)
def test_nice_adult_coverage(optimization):
    """
    Test that NICE achieves 100% coverage on adult dataset.
    
    This is a critical requirement - NICE should always find a counterfactual.
    Uses 200 instances to match paper experiments.
    """
    # Load adult dataset with Random Forest
    # Use forest (not mlp/linear) to get non-one-hot encoded data
    data = DataCatalog("adult", model_type="forest", train_split=0.7)
    model = ModelCatalog(data, model_type="forest", backend="sklearn")
    
    # Initialize NICE with specified optimization
    nice = NICE(
        mlmodel=model,
        hyperparams={"optimization": optimization}
    )
    
    # Get 200 negative instances to explain (matching paper)
    factuals = predict_negative_instances(model, data).iloc[:200]
    
    # Generate counterfactuals
    counterfactuals = nice.get_counterfactuals(factuals)
    
    # Check coverage: all counterfactuals should be valid (not NaN)
    assert not counterfactuals.isna().any().any(), \
        f"NICE({optimization}) did not achieve 100% coverage - some counterfactuals are NaN"
    
    # Check predictions flip to positive class
    predictions = model.predict(counterfactuals)
    num_flipped = (predictions >= 0.5).sum()
    
    assert num_flipped == len(factuals), \
        f"NICE({optimization}) only flipped {num_flipped}/{len(factuals)} predictions"
    
    print(f"✓ NICE({optimization}): 100% coverage achieved ({len(factuals)}/{len(factuals)} valid)")


@pytest.mark.parametrize(
    "optimization",
    ["none", "sparsity", "proximity", "plausibility"],
)
def test_nice_adult_quality(optimization):
    """
    Test that NICE produces quality counterfactuals.
    
    Measures ALL FOUR metrics for ALL FOUR variants:
    - CPU time (milliseconds)
    - Sparsity (number of changes)
    - Proximity (L1 distance)
    - Plausibility (autoencoder reconstruction error)
    
    Uses 200 instances to match paper experiments.
    """
    data = DataCatalog("adult", model_type="forest", train_split=0.7)
    model = ModelCatalog(data, model_type="forest", backend="sklearn")
    
    nice = NICE(
        mlmodel=model,
        hyperparams={"optimization": optimization}
    )
    
    # Use 200 instances like the paper (not 20!)
    factuals = predict_negative_instances(model, data).iloc[:200]
    
    # ============================================
    # Metric 0: CPU TIME (milliseconds)
    # ============================================
    start_time = time.time()
    counterfactuals = nice.get_counterfactuals(factuals)
    end_time = time.time()
    
    total_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    avg_time_ms = total_time_ms / len(factuals)  # Average time per instance
    
    # Get ordered features for comparison
    factuals_ordered = model.get_ordered_features(factuals)
    
    # ============================================
    # Metric 1: SPARSITY (number of changes)
    # ============================================
    sparsity = (factuals_ordered.values != counterfactuals.values).sum(axis=1)
    avg_sparsity = sparsity.mean()
    min_sparsity = sparsity.min()
    max_sparsity = sparsity.max()
    std_sparsity = sparsity.std()
    
    # ============================================
    # Metric 2: PROXIMITY (L1 distance)
    # ============================================
    proximity_per_instance = np.abs(factuals_ordered.values - counterfactuals.values).sum(axis=1)
    avg_proximity = proximity_per_instance.mean()
    std_proximity = proximity_per_instance.std()
    
    # ============================================
    # Metric 3: PLAUSIBILITY (AE reconstruction error)
    # ============================================
    # Train autoencoder on training data (for all variants!)
    from methods.catalog.nice.library.autoencoder import AutoEncoder
    
    X_train = data.df_train.drop(columns=["y"]).values
    feature_names = data.df_train.drop(columns=["y"]).columns.tolist()
    cat_feat_idx = [feature_names.index(f) for f in data.categorical]
    num_feat_idx = [feature_names.index(f) for f in data.continuous]
    
    ae = AutoEncoder(
        X_train=X_train,
        cat_feat_idx=cat_feat_idx,
        num_feat_idx=num_feat_idx,
        encoding_dim=32,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Calculate AE error for counterfactuals
    ae_errors = ae(counterfactuals.values)
    avg_ae_error = ae_errors.mean()
    std_ae_error = ae_errors.std()
    
    # ============================================
    # PRINT ALL FOUR METRICS (like Table 5 in paper)
    # ============================================
    print(f"\n{'='*70}")
    print(f"NICE({optimization}) Quality Metrics (n={len(factuals)}):")
    print(f"{'='*70}")
    print(f"  CPU Time:")
    print(f"    Total:   {total_time_ms:.2f} ms")
    print(f"    Average: {avg_time_ms:.2f} ms per instance")
    print(f"  Sparsity:")
    print(f"    Average: {avg_sparsity:.2f} ± {std_sparsity:.2f} features changed")
    print(f"    Range:   [{min_sparsity}, {max_sparsity}]")
    print(f"  Proximity (L1):")
    print(f"    Average: {avg_proximity:.2f} ± {std_proximity:.2f}")
    print(f"  Plausibility (AE reconstruction error):")
    print(f"    Average: {avg_ae_error:.4f} ± {std_ae_error:.4f}")
    print(f"{'='*70}")
    
    # ============================================
    # SANITY CHECKS
    # ============================================
    assert avg_sparsity < factuals_ordered.shape[1], \
        f"Sparsity too high: changing all {factuals_ordered.shape[1]} features"
    assert avg_sparsity > 0, \
        "No changes made - counterfactuals identical to factuals"
    
    # Variant-specific checks
    if optimization == "sparsity":
        assert avg_sparsity <= 5, \
            f"NICE(sparsity) should have low sparsity, got: {avg_sparsity:.2f}"
    elif optimization == "proximity":
        # Proximity variant should have good proximity
        assert avg_proximity <= 2.0, \
            f"NICE(proximity) should have low proximity, got: {avg_proximity:.2f}"
    elif optimization == "none":
        # None should be very plausible (it's an actual instance!)
        # But we allow some tolerance since we measure on test set
        assert avg_ae_error <= 0.02, \
            f"NICE(none) should be very plausible, got: {avg_ae_error:.4f}"


def test_nice_variants_comparison():
    """
    Compare all 4 NICE variants on the same instances.
    
    Reproduces the comparison shown in Table 5 of the paper.
    Verifies:
    - All variants achieve 100% coverage
    - CPU time performance
    - Trade-offs between sparsity, proximity, and plausibility
    
    Uses 200 instances to match paper experiments.
    """
    data = DataCatalog("adult", model_type="forest", train_split=0.7)
    model = ModelCatalog(data, model_type="forest", backend="sklearn")
    
    # Get test instances (200 to match paper)
    factuals = predict_negative_instances(model, data).iloc[:200]
    
    # Train autoencoder once for plausibility measurement
    from methods.catalog.nice.library.autoencoder import AutoEncoder
    
    X_train = data.df_train.drop(columns=["y"]).values
    feature_names = data.df_train.drop(columns=["y"]).columns.tolist()
    cat_feat_idx = [feature_names.index(f) for f in data.categorical]
    num_feat_idx = [feature_names.index(f) for f in data.continuous]
    
    ae = AutoEncoder(
        X_train=X_train,
        cat_feat_idx=cat_feat_idx,
        num_feat_idx=num_feat_idx,
        encoding_dim=32,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    results = {}
    
    for opt in ["none", "sparsity", "proximity", "plausibility"]:
        nice = NICE(mlmodel=model, hyperparams={"optimization": opt})
        
        # Measure CPU time
        start_time = time.time()
        cfs = nice.get_counterfactuals(factuals)
        end_time = time.time()
        
        cpu_time_ms = (end_time - start_time) * 1000
        avg_cpu_time_ms = cpu_time_ms / len(factuals)
        
        # Calculate metrics
        factuals_ordered = model.get_ordered_features(factuals)
        
        # Sparsity
        sparsity = (factuals_ordered.values != cfs.values).sum(axis=1)
        
        # Proximity (L1)
        proximity = np.abs(factuals_ordered.values - cfs.values).sum(axis=1)
        
        # Plausibility (AE error)
        ae_errors = ae(cfs.values)
        
        # Coverage
        coverage = (~cfs.isna().any(axis=1)).sum()
        
        results[opt] = {
            "coverage": coverage,
            "cpu_time_total_ms": cpu_time_ms,
            "cpu_time_avg_ms": avg_cpu_time_ms,
            "avg_sparsity": sparsity.mean(),
            "std_sparsity": sparsity.std(),
            "avg_proximity": proximity.mean(),
            "std_proximity": proximity.std(),
            "avg_plausibility": ae_errors.mean(),
            "std_plausibility": ae_errors.std(),
        }
    
    # Print comparison table (like Table 5)
    print(f"\n{'='*100}")
    print("Comparison of NICE Variants (reproducing Table 5 from paper)")
    print(f"{'='*100}")
    print(f"Dataset: adult, Model: Random Forest, n={len(factuals)}")
    print(f"{'-'*100}")
    print(f"{'Variant':<15} {'Coverage':<12} {'CPU (ms)':<12} {'Sparsity':<20} {'Proximity':<20} {'Plausibility':<15}")
    print(f"{'-'*100}")
    
    for opt, metrics in results.items():
        print(f"{opt:<15} "
              f"{metrics['coverage']}/{len(factuals):<10} "
              f"{metrics['cpu_time_avg_ms']:>8.2f}    "
              f"{metrics['avg_sparsity']:>5.2f} ± {metrics['std_sparsity']:<4.2f}      "
              f"{metrics['avg_proximity']:>6.2f} ± {metrics['std_proximity']:<5.2f}    "
              f"{metrics['avg_plausibility']:>6.4f} ± {metrics['std_plausibility']:<6.4f}")
    
    print(f"{'='*100}")
    
    # Additional summary
    print(f"\nTotal CPU Time:")
    for opt, metrics in results.items():
        print(f"  NICE({opt:<12}): {metrics['cpu_time_total_ms']:>8.2f} ms total "
              f"({metrics['cpu_time_avg_ms']:>6.2f} ms per instance)")
    
    # Verify expectations
    # All should have 100% coverage
    for opt, metrics in results.items():
        assert metrics["coverage"] == len(factuals), \
            f"NICE({opt}) failed coverage: {metrics['coverage']}/{len(factuals)}"
    
    # NICE(sparsity) should have lowest or near-lowest sparsity
    spars_sparsity = results["sparsity"]["avg_sparsity"]
    for opt in ["none", "proximity", "plausibility"]:
        # Allow small tolerance for statistical variation
        assert spars_sparsity <= results[opt]["avg_sparsity"] + 0.5, \
            f"NICE(sparsity) should have best sparsity, but NICE({opt}) is better"
    
    # NICE(proximity) should have lowest or near-lowest proximity
    prox_proximity = results["proximity"]["avg_proximity"]
    for opt in ["none", "plausibility"]:
        assert prox_proximity <= results[opt]["avg_proximity"] + 0.3, \
            f"NICE(proximity) should have best proximity, but NICE({opt}) is better"
    
    # NICE(none) should have very good plausibility (it's actual instances!)
    none_plausibility = results["none"]["avg_plausibility"]
    assert none_plausibility <= 0.02, \
        f"NICE(none) should be very plausible: {none_plausibility:.4f}"
    
    # NICE(none) should be fastest
    none_time = results["none"]["cpu_time_avg_ms"]
    for opt in ["sparsity", "proximity", "plausibility"]:
        assert none_time <= results[opt]["cpu_time_avg_ms"], \
            f"NICE(none) should be fastest, but NICE({opt}) is faster"
    
    print("\n✓ All variant comparisons passed!")


@pytest.mark.parametrize(
    "dataset_name, model_type",
    [
        ("adult", "forest"),
    ],
)
def test_nice_benchmark_compatibility(dataset_name, model_type):
    """
    Test that NICE integrates properly with the benchmark framework.
    
    Verifies:
    - Correct data loading
    - Model compatibility
    - Output format matches benchmark expectations
    """
    # Load data
    data = DataCatalog(dataset_name, model_type=model_type, train_split=0.7)
    model = ModelCatalog(data, model_type=model_type, backend="sklearn")
    
    # Initialize NICE
    nice = NICE(mlmodel=model)
    
    # Get factuals
    factuals = predict_negative_instances(model, data).iloc[:20]
    
    # Generate counterfactuals
    cfs = nice.get_counterfactuals(factuals)
    
    # Check output format
    assert isinstance(cfs, pd.DataFrame), "Output should be DataFrame"
    assert cfs.shape[0] == factuals.shape[0], "Should return same number of counterfactuals"
    assert not cfs.isna().all(axis=1).any(), "Should not have all-NaN rows"
    
    # Check feature order matches model expectations
    assert list(cfs.columns) == model.feature_input_order, \
        "Counterfactual features should match model's expected input order"
    
    print(f"✓ NICE integrates correctly with {dataset_name} dataset")


if __name__ == "__main__":
    """
    Run tests manually (without pytest)
    
    Usage:
        python reproduce.py
    """
    print("=" * 100)
    print("NICE Integration Tests - Reproducing Paper Results")
    print("=" * 100)
    print("\nDataset: adult")
    print("Model: Random Forest")
    print("Sample size: 200 instances (matching paper experiments)")
    print("=" * 100)
    
    # Test 1: Coverage for all variants
    print("\n[Test 1] Testing 100% Coverage Guarantee...")
    print("-" * 100)
    for opt in ["none", "sparsity", "proximity", "plausibility"]:
        try:
            test_nice_adult_coverage(opt)
        except Exception as e:
            print(f"✗ NICE({opt}) failed: {e}")
    
    # Test 2: Quality metrics for all variants
    print("\n[Test 2] Testing Quality Metrics (CPU Time, Sparsity, Proximity, Plausibility)...")
    print("-" * 100)
    for opt in ["none", "sparsity", "proximity", "plausibility"]:
        try:
            test_nice_adult_quality(opt)
        except Exception as e:
            print(f"✗ NICE({opt}) quality test failed: {e}")
    
    # Test 3: Variant comparison (reproducing Table 5)
    print("\n[Test 3] Comparing All Variants (Reproducing Table 5 from paper)...")
    print("-" * 100)
    try:
        test_nice_variants_comparison()
    except Exception as e:
        print(f"✗ Variant comparison failed: {e}")
    
    # Test 4: Benchmark compatibility
    print("\n[Test 4] Testing Benchmark Compatibility...")
    print("-" * 100)
    try:
        test_nice_benchmark_compatibility("adult", "forest")
    except Exception as e:
        print(f"✗ Benchmark compatibility failed: {e}")
    
    print("\n" + "=" * 100)
    print("All tests completed!")
    print("=" * 100)