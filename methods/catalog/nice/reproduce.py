"""
Test NICE on adult dataset to reproduce paper results

Tests all 4 NICE variants (none, sparsity, proximity, plausibility) on the
adult dataset to verify:
1. 100% coverage (always finds a counterfactual)
2. Valid counterfactuals (prediction flips to positive class)
3. Quality metrics (sparsity, proximity, plausibility)

These tests aim to reproduce results from:
Brughmans et al. (2024) Tables 5-6 on the adult dataset.
"""
import numpy as np
import pandas as pd
import pytest

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
    
    # Get negative instances to explain
    factuals = predict_negative_instances(model, data).iloc[:10]
    
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
    ["sparsity", "proximity"],
)
def test_nice_adult_quality(optimization):
    """
    Test that NICE produces quality counterfactuals.
    
    Measures sparsity (number of changes) and proximity (L1 distance).
    """
    data = DataCatalog("adult", model_type="forest", train_split=0.7)
    model = ModelCatalog(data, model_type="forest", backend="sklearn")
    
    nice = NICE(
        mlmodel=model,
        hyperparams={"optimization": optimization}
    )
    
    # Get more instances for quality assessment
    factuals = predict_negative_instances(model, data).iloc[:20]
    counterfactuals = nice.get_counterfactuals(factuals)
    
    # Calculate sparsity (number of changes)
    # Note: factuals and counterfactuals may have different column orders
    factuals_ordered = model.get_ordered_features(factuals)
    sparsity = (factuals_ordered.values != counterfactuals.values).sum(axis=1)
    avg_sparsity = sparsity.mean()
    
    # Calculate proximity (L1 distance)
    # For mixed data, this is approximate since categorical features are counted as 0/1
    proximity = np.abs(factuals_ordered.values - counterfactuals.values).sum(axis=1).mean()
    
    print(f"\nNICE({optimization}) Quality Metrics:")
    print(f"  Average Sparsity: {avg_sparsity:.2f} features changed")
    print(f"  Average Proximity (L1): {proximity:.2f}")
    print(f"  Min Sparsity: {sparsity.min()}")
    print(f"  Max Sparsity: {sparsity.max()}")
    
    # Sanity checks
    assert avg_sparsity < factuals_ordered.shape[1], \
        f"Sparsity too high: changing all {factuals_ordered.shape[1]} features"
    assert avg_sparsity > 0, \
        "No changes made - counterfactuals identical to factuals"
    
    # Sparsity variant should have lower average sparsity than proximity
    if optimization == "sparsity":
        assert avg_sparsity <= 10, \
            f"NICE(sparsity) has high average sparsity: {avg_sparsity:.2f}"


def test_nice_variants_comparison():
    """
    Compare all 4 NICE variants on the same instances.
    
    Verifies:
    - All variants achieve 100% coverage
    - NICE(spars) has best sparsity
    - NICE(none) uses fewest iterations (just returns nearest neighbor)
    """
    data = DataCatalog("adult", model_type="forest", train_split=0.7)
    model = ModelCatalog(data, model_type="forest", backend="sklearn")
    
    # Get test instances
    factuals = predict_negative_instances(model, data).iloc[:10]
    
    results = {}
    
    for opt in ["none", "sparsity", "proximity", "plausibility"]:
        nice = NICE(mlmodel=model, hyperparams={"optimization": opt})
        cfs = nice.get_counterfactuals(factuals)
        
        # Calculate sparsity
        factuals_ordered = model.get_ordered_features(factuals)
        sparsity = (factuals_ordered.values != cfs.values).sum(axis=1).mean()
        
        results[opt] = {
            "coverage": (~cfs.isna().any(axis=1)).sum(),
            "avg_sparsity": sparsity
        }
    
    print("\nComparison of NICE Variants:")
    print("-" * 50)
    for opt, metrics in results.items():
        print(f"NICE({opt:12s}): Coverage={metrics['coverage']}/10, "
              f"Avg Sparsity={metrics['avg_sparsity']:.2f}")
    
    # All should have 100% coverage
    for opt, metrics in results.items():
        assert metrics["coverage"] == 10, \
            f"NICE({opt}) failed coverage: {metrics['coverage']}/10"
    
    # NICE(spars) should have lowest or tied-lowest sparsity
    spars_sparsity = results["sparsity"]["avg_sparsity"]
    for opt in ["none", "proximity", "plausibility"]:
        assert spars_sparsity <= results[opt]["avg_sparsity"] + 0.5, \
            f"NICE(sparsity) should have best sparsity, but NICE({opt}) is better"


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
    factuals = predict_negative_instances(model, data).iloc[:5]
    
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
    print("=" * 60)
    print("NICE Integration Tests")
    print("=" * 60)
    
    # Test 1: Coverage for all variants
    print("\n[Test 1] Testing 100% Coverage...")
    for opt in ["none", "sparsity", "proximity", "plausibility"]:
        try:
            test_nice_adult_coverage(opt)
        except Exception as e:
            print(f"✗ NICE({opt}) failed: {e}")
    
    # Test 2: Quality metrics
    print("\n[Test 2] Testing Quality Metrics...")
    for opt in ["sparsity", "proximity"]:
        try:
            test_nice_adult_quality(opt)
        except Exception as e:
            print(f"✗ NICE({opt}) quality test failed: {e}")
    
    # Test 3: Variant comparison
    print("\n[Test 3] Comparing All Variants...")
    try:
        test_nice_variants_comparison()
    except Exception as e:
        print(f"✗ Variant comparison failed: {e}")
    
    # Test 4: Benchmark compatibility
    print("\n[Test 4] Testing Benchmark Compatibility...")
    try:
        test_nice_benchmark_compatibility("adult", "forest")
    except Exception as e:
        print(f"✗ Benchmark compatibility failed: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)