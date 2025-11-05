import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Transformer:
    def __init__(self, data, numerical, scale=True):
        self.data = data
        self.empty_df = pd.DataFrame(columns=data.columns)

        self.num_name = numerical
        self.cat_name = list(data.columns.difference(numerical))

        numeric_transformer = StandardScaler(with_std=scale, with_mean=scale)

        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.encs = []
        if len(self.num_name) > 0:
            self.encs.append(("num", numeric_transformer, self.num_name))
        if len(self.cat_name) > 0:
            self.encs.append(("cat", categorical_transformer, self.cat_name))

        self.transformer = ColumnTransformer(transformers=self.encs)

        self.transformer.fit(data)

        self.cat_indices = []
        self.n_num_features_out = len(self.num_name)

        if len(self.num_name) > 0:
            self.enc_num = self.transformer.named_transformers_["num"]
        if len(self.cat_name) > 0:
            self.enc_cat = self.transformer.named_transformers_["cat"]

            self.n_cat_features_out = len(self.enc_cat.get_feature_names())
            self.cat_indices = list(range(self.n_num_features_out, self.n_num_features_out + self.n_cat_features_out))
        else:
            self.n_cat_features_out = 0

    def transform(self, X):
        return self.transformer.transform(X)

    def inverse_transform(self, X):
        df = self.empty_df.copy()
        n_num = len(self.num_name)

        if n_num > 0:
            X_num = X[..., :n_num]
            inv_num = self.enc_num.inverse_transform(X_num)
            df[self.num_name] = inv_num

        if len(self.cat_name) > 0:
            X_cat = X[..., n_num:]
            inv_cat = self.enc_cat.inverse_transform(X_cat)
            df[self.cat_name] = inv_cat

        return df


def get_transformer(dataset_name, dataset):
    # dataset_name = dataset_name.replace("_modified", "")
    if "german" in dataset_name:
        numerical = [
            "age",
            "amount",
            "duration",
        ]
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized for transformer.")
    # dataset = pd.read_csv(f"methods/catalog/rbr/library/{dataset_name}.csv") # keep it in the same location for simplicity

    dataset = dataset.drop("y", axis=1)

    transformer = Transformer(dataset, numerical, ("synthesis" not in dataset_name))

    return transformer
