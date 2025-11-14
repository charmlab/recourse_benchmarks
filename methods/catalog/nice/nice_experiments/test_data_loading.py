"""
Test if PmlbFetcher works correctly
"""
from pmlb_fetcher import PmlbFetcher
import numpy as np

def test_pmlb_fetcher():
    """Test basic data loading"""
    print("="*80)
    print("Testing PmlbFetcher")
    print("="*80)
    
    # Load adult dataset
    print("\n[1] Loading adult dataset from PMLB...")
    fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
    
    # Check dataset structure
    print("\n[2] Checking dataset structure...")
    assert 'X_train' in fetcher.dataset
    assert 'y_train' in fetcher.dataset
    assert 'feature_names' in fetcher.dataset
    assert 'cat_feat' in fetcher.dataset
    assert 'con_feat' in fetcher.dataset
    print("  ✓ All required keys present")
    
    # Check data types
    print("\n[3] Checking data types...")
    assert isinstance(fetcher.dataset['X_train'], np.ndarray)
    assert isinstance(fetcher.dataset['y_train'], np.ndarray)
    assert isinstance(fetcher.dataset['feature_names'], list)
    print("  ✓ Correct data types")
    
    # Check shapes
    print("\n[4] Checking data shapes...")
    X_train = fetcher.dataset['X_train']
    y_train = fetcher.dataset['y_train']
    feature_names = fetcher.dataset['feature_names']
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Number of features: {len(feature_names)}")
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == len(feature_names)
    print("  ✓ Shapes are consistent")
    
    # Check categorical encoding
    print("\n[5] Checking categorical encoding...")
    cat_indices = fetcher.dataset['cat_feat']
    for idx in cat_indices:
        values = np.unique(X_train[:, idx])
        print(f"  Categorical feature {idx}: {len(values)} unique values")
        # Should be integer encoded starting from 0
        assert all(v == int(v) for v in values), f"Feature {idx} not integer encoded"
        assert values.min() >= 0, f"Feature {idx} has negative values"
    print("  ✓ Categorical features are label encoded")
    
    # Print summary
    print("\n[6] Data Summary")
    print("-"*80)
    print(f"  Dataset: adult")
    print(f"  Train samples: {X_train.shape[0]}")
    print(f"  Test samples: {fetcher.dataset['X_test'].shape[0]}")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Categorical: {len(cat_indices)}")
    print(f"  Continuous: {len(fetcher.dataset['con_feat'])}")
    print(f"  Feature names: {feature_names}")
    
    print("\n" + "="*80)
    print("✓ PmlbFetcher test PASSED")
    print("="*80)
    
    return fetcher

if __name__ == "__main__":
    fetcher = test_pmlb_fetcher()