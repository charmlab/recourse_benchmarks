"""
Test if adapters make data API-compatible
"""
from pmlb_fetcher import PmlbFetcher
from adapter import NICEExperimentsDataAdapter, NICEExperimentsModelAdapter
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def test_data_adapter():
    """Test that data adapter creates correct interface"""
    print("="*80)
    print("Testing NICEExperimentsDataAdapter")
    print("="*80)
    
    # Load data
    print("\n[1] Loading NICE_experiments data...")
    fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
    nice_data = fetcher.dataset
    
    # Create adapter
    print("\n[2] Creating data adapter...")
    data_adapter = NICEExperimentsDataAdapter(nice_data)
    
    # Check required attributes
    print("\n[3] Checking required attributes...")
    assert hasattr(data_adapter, 'df_train'), "Missing df_train"
    assert hasattr(data_adapter, 'df_test'), "Missing df_test"
    assert hasattr(data_adapter, 'categorical'), "Missing categorical"
    assert hasattr(data_adapter, 'continuous'), "Missing continuous"
    print("  ✓ All required attributes present")
    
    # Check df_train structure
    print("\n[4] Checking df_train structure...")
    assert isinstance(data_adapter.df_train, pd.DataFrame)
    assert 'y' in data_adapter.df_train.columns
    print(f"  df_train shape: {data_adapter.df_train.shape}")
    print(f"  df_train columns: {list(data_adapter.df_train.columns)}")
    print("  ✓ df_train is correct format")
    
    # Check feature lists
    print("\n[5] Checking feature lists...")
    assert isinstance(data_adapter.categorical, list)
    assert isinstance(data_adapter.continuous, list)
    assert all(isinstance(f, str) for f in data_adapter.categorical)
    assert all(isinstance(f, str) for f in data_adapter.continuous)
    print(f"  Categorical features ({len(data_adapter.categorical)}): {data_adapter.categorical}")
    print(f"  Continuous features ({len(data_adapter.continuous)}): {data_adapter.continuous}")
    print("  ✓ Feature lists are correct")
    
    # Check data content matches
    print("\n[6] Checking data content...")
    X_train_orig = nice_data['X_train']
    X_train_wrapped = data_adapter.df_train.drop(columns=['y']).values
    
    assert np.array_equal(X_train_orig, X_train_wrapped), "Data content doesn't match!"
    print("  ✓ Data content preserved correctly")
    
    print("\n" + "="*80)
    print("✓ Data Adapter test PASSED")
    print("="*80)
    
    return data_adapter

def test_model_adapter():
    """Test that model adapter creates correct interface"""
    print("\n" + "="*80)
    print("Testing NICEExperimentsModelAdapter")
    print("="*80)
    
    # Get data adapter
    print("\n[1] Setting up data...")
    fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
    data_adapter = NICEExperimentsDataAdapter(fetcher.dataset)
    
    # Train a simple model
    print("\n[2] Training model...")
    X_train = data_adapter.df_train.drop(columns=['y']).values
    y_train = data_adapter.df_train['y'].values
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    print("  ✓ Model trained")
    
    # Create model adapter
    print("\n[3] Creating model adapter...")
    model_adapter = NICEExperimentsModelAdapter(model, data_adapter)
    
    # Check required attributes
    print("\n[4] Checking required attributes...")
    assert hasattr(model_adapter, 'predict_proba'), "Missing predict_proba"
    assert hasattr(model_adapter, 'predict'), "Missing predict"
    assert hasattr(model_adapter, 'data'), "Missing data"
    assert hasattr(model_adapter, 'get_ordered_features'), "Missing get_ordered_features"
    print("  ✓ All required methods present")
    
    # Test predict_proba
    print("\n[5] Testing predict_proba...")
    test_df = data_adapter.df_test.head(10)
    proba = model_adapter.predict_proba(test_df)
    
    assert isinstance(proba, np.ndarray)
    assert proba.shape[0] == 10
    assert proba.shape[1] == 2  # Binary classification
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)
    print(f"  Proba shape: {proba.shape}")
    print(f"  Sample probabilities: {proba[0]}")
    print("  ✓ predict_proba works correctly")
    
    # Test predict
    print("\n[6] Testing predict...")
    preds = model_adapter.predict(test_df)
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 10
    assert all(p in [0, 1] for p in preds)
    print(f"  Predictions: {preds}")
    print("  ✓ predict works correctly")
    
    # Test get_ordered_features
    print("\n[7] Testing get_ordered_features...")
    ordered_df = model_adapter.get_ordered_features(test_df)
    
    assert isinstance(ordered_df, pd.DataFrame)
    assert list(ordered_df.columns) == data_adapter.feature_names + ['y']
    print(f"  Ordered columns: {list(ordered_df.columns)}")
    print("  ✓ get_ordered_features works correctly")
    
    print("\n" + "="*80)
    print("✓ Model Adapter test PASSED")
    print("="*80)
    
    return model_adapter

if __name__ == "__main__":
    # Test data adapter
    data_adapter = test_data_adapter()
    
    # Test model adapter
    model_adapter = test_model_adapter()
    
    print("\n" + "="*80)
    print("ALL ADAPTER TESTS PASSED ✓")
    print("="*80)
    print("\nData and Model are now API-compatible!")
    print("Next step: Test with NICE method")