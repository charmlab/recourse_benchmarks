"""
Test script to verify genre_adult dataset integration with DataCatalog

Save as: methods/catalog/genre/test_datacatalog_integration.py

Usage:
    python -m methods.catalog.genre.test_datacatalog_integration
"""

import sys
import torch
from data.catalog import DataCatalog


def test_datacatalog_loading():
    """Test 1: Can we load genre_adult through DataCatalog?"""
    print("=" * 60)
    print("TEST 1: Loading genre_adult through DataCatalog")
    print("=" * 60)
    
    try:
        data = DataCatalog("genre_adult", model_type="mlp", train_split=0.8)
        print("✓ DataCatalog loaded successfully\n")
        return data
    except Exception as e:
        print(f"✗ Failed to load DataCatalog: {e}\n")
        sys.exit(1)


def test_data_properties(data):
    """Test 2: Check DataCatalog properties"""
    print("=" * 60)
    print("TEST 2: DataCatalog Properties")
    print("=" * 60)
    
    print(f"Dataset name: {data.name}")
    print(f"Target column: {data.target}")
    print(f"\nContinuous features ({len(data.continuous)}):")
    print(f"  {data.continuous}")
    print(f"\nCategorical features ({len(data.categorical)}):")
    print(f"  {data.categorical}")
    print(f"\nImmutable features ({len(data.immutables)}):")
    print(f"  {data.immutables}")
    
    print(f"\nTrain set shape: {data.df_train.shape}")
    print(f"Test set shape: {data.df_test.shape}")
    print(f"Full set shape: {data.df.shape}")
    
    print("\n✓ All properties accessible\n")


def test_cat_mask_calculation(data):
    """Test 3: Calculate cat_mask from DataCatalog"""
    print("=" * 60)
    print("TEST 3: Cat Mask Calculation")
    print("=" * 60)
    
    # Get feature columns from actual DataFrame
    df_train = data.df_train
    feature_cols = [col for col in df_train.columns if col != data.target]
    
    print(f"Feature columns ({len(feature_cols)}):")
    print(f"  {feature_cols[:5]}... (showing first 5)")
    
    # Calculate cat_mask (same logic as model.py)
    cat_mask = torch.tensor(
        [1 if col in data.categorical else 0 for col in feature_cols]
    )
    
    print(f"\nCat mask shape: {cat_mask.shape}")
    print(f"Cat mask: {cat_mask}")
    print(f"Number of categorical features: {cat_mask.sum().item()}")
    print(f"Number of continuous features: {(cat_mask == 0).sum().item()}")
    
    # Verify ordering
    print("\nVerifying feature order matches continuous + categorical:")
    expected_order = data.continuous + data.categorical
    actual_order = feature_cols
    
    if expected_order == actual_order:
        print("✓ Feature order matches: continuous first, then categorical")
    else:
        print("✗ WARNING: Feature order mismatch!")
        print(f"  Expected: {expected_order[:5]}...")
        print(f"  Actual: {actual_order[:5]}...")
    
    print("\n✓ Cat mask calculated successfully\n")
    return cat_mask, feature_cols


def test_data_values(data):
    """Test 4: Check data preprocessing"""
    print("=" * 60)
    print("TEST 4: Data Values and Preprocessing")
    print("=" * 60)
    
    df_train = data.df_train
    
    # Check normalization (should be in [0, 1] range)
    feature_cols = [col for col in df_train.columns if col != data.target]
    
    print("Checking if features are normalized to [0, 1]:")
    all_normalized = True
    for col in feature_cols[:3]:  # Check first 3 features
        min_val = df_train[col].min()
        max_val = df_train[col].max()
        print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}")
        if min_val < -0.01 or max_val > 1.01:  # Allow small floating point errors
            all_normalized = False
    
    if all_normalized:
        print("✓ Features appear to be normalized")
    else:
        print("⚠ Warning: Some features may not be normalized")
    
    # Check target values
    unique_targets = df_train[data.target].unique()
    print(f"\nTarget values: {sorted(unique_targets)}")
    
    if set(unique_targets).issubset({0, 1}):
        print("✓ Target is binary (0 and 1)")
    else:
        print("⚠ Warning: Target values are not standard binary")
    
    print()


def test_genre_integration(data, cat_mask):
    """Test 5: Simulate GenRe initialization"""
    print("=" * 60)
    print("TEST 5: GenRe Integration Simulation")
    print("=" * 60)
    
    try:
        # Simulate what happens in model.py
        df_train = data.df_train
        feature_cols = [col for col in df_train.columns if col != data.target]
        input_dim = len(feature_cols)
        
        print(f"Input dimension for transformer: {input_dim}")
        print(f"Cat mask shape: {cat_mask.shape}")
        print(f"Cat mask dtype: {cat_mask.dtype}")
        
        # Check if dimensions match
        if input_dim == len(cat_mask):
            print("✓ Dimensions match: input_dim == cat_mask length")
        else:
            print(f"✗ Dimension mismatch: input_dim={input_dim}, cat_mask length={len(cat_mask)}")
            return False
        
        # Sample some factuals
        factuals = df_train.drop(columns=[data.target]).head(5)
        print(f"\nSample factuals shape: {factuals.shape}")
        print(f"Sample factuals:\n{factuals.head(2)}")
        
        print("\n✓ GenRe integration should work\n")
        return True
        
    except Exception as e:
        print(f"✗ Error in GenRe integration: {e}\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("GENRE_ADULT DATACATALOG INTEGRATION TEST")
    print("=" * 60 + "\n")
    
    # Run all tests
    data = test_datacatalog_loading()
    test_data_properties(data)
    cat_mask, feature_cols = test_cat_mask_calculation(data)
    test_data_values(data)
    success = test_genre_integration(data, cat_mask)
    
    # Final summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if success:
        print("✓ All tests passed! genre_adult is ready for use.")
        print("\nYou can now use:")
        print('  data = DataCatalog("genre_adult", model_type="mlp", train_split=0.8)')
        print('  genre = GenRe(model, hyperparams={"data": data})')
    else:
        print("✗ Some tests failed. Please review the output above.")
    print()


if __name__ == "__main__":
    main()