import pathlib

import pandas as pd


class GenreAdultData:
    """Container for genre_adult dataset with metadata"""

    def __init__(
        self,
        df,
        categorical_features,
        continuous_features,
        immutable_features,
        target_column,
    ):
        self.df = df
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.immutable_features = immutable_features
        self.target_column = target_column


def load_genre_adult_data():
    """
    Load and preprocess author's adult-all dataset exactly as GenRe expects.
    """
    current_file = pathlib.Path(__file__).resolve()
    repo_root = current_file.parent.parent.parent.parent.parent
    data_dir = (
        repo_root / "methods" / "catalog" / "genre" / "library" / "datasets" / "adult"
    )

    if not data_dir.exists():
        raise FileNotFoundError(f"GenRe adult data not found at {data_dir}")

    # Load CSV files
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Target column
    target = "income"

    # Define categorical and immutable columns (from utils.py)
    cat_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    immutable_cols = ["race", "sex"]

    # Convert categorical columns to codes
    train_df[cat_cols] = train_df[cat_cols].astype("category")
    test_df[cat_cols] = test_df[cat_cols].astype("category")

    train_df[cat_cols] = train_df[cat_cols].apply(lambda x: x.cat.codes)
    test_df[cat_cols] = test_df[cat_cols].apply(lambda x: x.cat.codes)

    # Split target and features
    train_y = train_df[target]
    train_X = train_df.drop(columns=[target])
    test_y = test_df[target]
    test_X = test_df.drop(columns=[target])

    # REORDER: continuous first, then categorical
    all_feature_cols = list(train_X.columns)
    continuous_cols = [col for col in all_feature_cols if col not in cat_cols]
    categorical_cols = [col for col in all_feature_cols if col in cat_cols]

    # Reorder columns
    ordered_cols = continuous_cols + categorical_cols
    train_X = train_X[ordered_cols]
    test_X = test_X[ordered_cols]

    # Min-max normalization (after reordering)
    train_min = train_X.min()
    train_range = train_X.max() - train_X.min()
    train_X = (train_X - train_min) / train_range
    test_X = (test_X - train_min) / train_range

    # Add target back
    train_X[target] = train_y
    test_X[target] = test_y

    # Combine for DataCatalog
    full_df = pd.concat([train_X, test_X], ignore_index=True)

    # Return object with metadata
    return GenreAdultData(
        df=full_df,
        categorical_features=categorical_cols,
        continuous_features=continuous_cols,
        immutable_features=immutable_cols,
        target_column=target,
    )
