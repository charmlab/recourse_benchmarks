"""
Test Autoencoder plausibility calculation independently
"""
import numpy as np
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nice_experiments.pmlb_fetcher import PmlbFetcher
from nice_experiments.adapter import NICEExperimentsDataAdapter
from library.autoencoder import AutoEncoder
from sklearn.preprocessing import MinMaxScaler

# Load data
print("="*80)
print("Testing Autoencoder")
print("="*80)

fetcher = PmlbFetcher('adult', test_size=0.2, explain_n=200)
nice_data = fetcher.dataset
data_adapted = NICEExperimentsDataAdapter(nice_data)

# Prepare data
X_train_raw = data_adapted.df_train.drop(columns=['y']).values
X_test_raw = data_adapted.df_test.drop(columns=['y']).values

print(f"\nData shapes:")
print(f"  X_train: {X_train_raw.shape}")
print(f"  X_test: {X_test_raw.shape}")

# Get feature indices
feature_names = data_adapted.feature_names
cat_feat_idx = [feature_names.index(f) for f in data_adapted.categorical]
num_feat_idx = [feature_names.index(f) for f in data_adapted.continuous]

print(f"\nFeature indices:")
print(f"  Categorical: {cat_feat_idx}")
print(f"  Continuous: {num_feat_idx}")

# ============================================================================
# NORMALIZE CONTINUOUS FEATURES ONLY
# ============================================================================
print(f"\n{'='*80}")
print("Normalizing continuous features...")
print(f"{'='*80}")

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

# Fit scaler on training data continuous features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Before normalization:")
print(f"  Train continuous - min: {X_train_raw[:, num_feat_idx].min():.2f}, max: {X_train_raw[:, num_feat_idx].max():.2f}")
print(f"After normalization:")
print(f"  Train continuous - mean: {X_train[:, num_feat_idx].mean():.4f}, std: {X_train[:, num_feat_idx].std():.4f}")
print(f"  Train continuous - min: {X_train[:, num_feat_idx].min():.4f}, max: {X_train[:, num_feat_idx].max():.4f}")

# Train autoencoder
print(f"\n{'='*80}")
print("Training Autoencoder...")
print(f"{'='*80}")

ae = AutoEncoder(
    X_train=X_train,  # 用归一化后的数据
    cat_feat_idx=cat_feat_idx,
    num_feat_idx=num_feat_idx,
    epochs=100,
    batch_size=32,
    verbose=1,
    early_stopping=True,
    patience=5
)

# Test on training data (归一化后的)
print(f"\n{'='*80}")
print("Testing on TRAINING data (should have LOW error)")
print(f"{'='*80}")

train_errors = ae(X_train[:1000])

train_errors = ae(X_train[:1000])  # Test on first 1000 samples
print(f"Train reconstruction errors:")
print(f"  Mean: {train_errors.mean():.6f}")
print(f"  Std: {train_errors.std():.6f}")
print(f"  Min: {train_errors.min():.6f}")
print(f"  Max: {train_errors.max():.6f}")
print(f"  Median: {np.median(train_errors):.6f}")

# Test on test data (should also have reasonably low error)
print(f"\n{'='*80}")
print("Testing on TEST data (should also be LOW)")
print(f"{'='*80}")

test_errors = ae(X_test)
print(f"Test reconstruction errors:")
print(f"  Mean: {test_errors.mean():.6f}")
print(f"  Std: {test_errors.std():.6f}")
print(f"  Min: {test_errors.min():.6f}")
print(f"  Max: {test_errors.max():.6f}")
print(f"  Median: {np.median(test_errors):.6f}")

# Create some random/adversarial examples (should have HIGH error)
print(f"\n{'='*80}")
print("Testing on RANDOM data (should have HIGH error)")
print(f"{'='*80}")

X_random = np.random.randn(*X_test[:100].shape)  # Random noise
random_errors = ae(X_random)
print(f"Random reconstruction errors:")
print(f"  Mean: {random_errors.mean():.6f}")
print(f"  Std: {random_errors.std():.6f}")
print(f"  Min: {random_errors.min():.6f}")
print(f"  Max: {random_errors.max():.6f}")
print(f"  Median: {np.median(random_errors):.6f}")

# Summary
print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")
print(f"Train error mean: {train_errors.mean():.6f}")
print(f"Test error mean:  {test_errors.mean():.6f}")
print(f"Random error mean: {random_errors.mean():.6f}")
print(f"\nIf working correctly:")
print(f"  - Train/Test errors should be similar and LOW")
print(f"  - Random errors should be much HIGHER")
print(f"  - Typical good values: < 0.1")
print("="*80)