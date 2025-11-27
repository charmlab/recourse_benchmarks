"""
Check genre_adult dataset dimensions

Save as: methods/catalog/genre/check_dimensions.py

Usage:
    python -m methods.catalog.genre.check_dimensions
"""

import pandas as pd
import pathlib


def check_csv_dimensions():
    """Check dimensions of CSV files"""
    print("=" * 60)
    print("CSV FILE DIMENSIONS")
    print("=" * 60)
    
    # Path to CSV files
    current_file = pathlib.Path(__file__).resolve()
    genre_root = current_file.parent
    data_dir = genre_root / "library" / "datasets" / "adult"
    
    print(f"Data directory: {data_dir}")
    print(f"Directory exists: {data_dir.exists()}\n")
    
    if not data_dir.exists():
        print("ERROR: Data directory not found!")
        return
    
    # Check train.csv
    train_path = data_dir / "train.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        print(f"train.csv:")
        print(f"  Shape: {train_df.shape}")
        print(f"  Columns ({len(train_df.columns)}): {list(train_df.columns)}")
        print(f"  Number of features (excluding target): {len(train_df.columns) - 1}")
        print()
    else:
        print("train.csv not found!\n")
    
    # Check test.csv
    test_path = data_dir / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        print(f"test.csv:")
        print(f"  Shape: {test_df.shape}")
        print(f"  Columns ({len(test_df.columns)}): {list(test_df.columns)}")
        print(f"  Number of features (excluding target): {len(test_df.columns) - 1}")
        print()
    else:
        print("test.csv not found!\n")


def check_datacatalog_dimensions():
    """Check dimensions after DataCatalog processing"""
    print("=" * 60)
    print("DATACATALOG DIMENSIONS")
    print("=" * 60)
    
    try:
        from data.catalog import DataCatalog
        
        data = DataCatalog("genre_adult", model_type="mlp", train_split=0.8)
        
        print(f"Dataset name: {data.name}")
        print(f"Target: {data.target}")
        print(f"\ndf_train shape: {data.df_train.shape}")
        print(f"df_train columns ({len(data.df_train.columns)}): {list(data.df_train.columns)}")
        
        # Count features (excluding target)
        feature_cols = [col for col in data.df_train.columns if col != data.target]
        print(f"\nNumber of features (excluding {data.target}): {len(feature_cols)}")
        print(f"Features: {feature_cols}")
        
        print(f"\nContinuous ({len(data.continuous)}): {data.continuous}")
        print(f"Categorical ({len(data.categorical)}): {data.categorical}")
        print(f"Total: {len(data.continuous) + len(data.categorical)}")
        
    except Exception as e:
        print(f"ERROR loading DataCatalog: {e}")
        import traceback
        traceback.print_exc()


def check_transformer_dimensions():
    """Check transformer expected dimensions"""
    print("=" * 60)
    print("TRANSFORMER DIMENSIONS")
    print("=" * 60)
    
    import torch
    
    current_file = pathlib.Path(__file__).resolve()
    genre_root = current_file.parent
    transformer_path = genre_root / "library" / "transformer" / "genre_transformer.pth"
    
    print(f"Transformer path: {transformer_path}")
    print(f"File exists: {transformer_path.exists()}")
    
    if transformer_path.exists():
        state_dict = torch.load(transformer_path, map_location="cpu")
        
        # Check if nested
        if "state_dict" in state_dict:
            actual_state = state_dict["state_dict"]
        else:
            actual_state = state_dict
        
        # Find input dimension from model parameters
        # Look for embedding or positional encoding parameters
        for key, value in actual_state.items():
            if "positional_encoding" in key or "embed" in key:
                print(f"\n{key}: {value.shape}")
                if "positional_encoding.pe" in key:
                    input_dim = value.shape[0]
                    print(f"\nTransformer expects input_dim = {input_dim}")
                    break
    else:
        print("Transformer file not found!")


def main():
    print("\n" + "=" * 60)
    print("GENRE DIMENSION CHECKER")
    print("=" * 60 + "\n")
    
    check_csv_dimensions()
    print()
    check_datacatalog_dimensions()
    print()
    check_transformer_dimensions()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Check if all dimensions match!")
    print()


if __name__ == "__main__":
    main()