try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class DataLoader:
    """A data interface for public data."""

    def __init__(self, params):
        """Init method

        :param dataframe: Pandas DataFrame.
        :param continuous_features: List of names of continuous features. The remaining features are categorical features.
        :param outcome_name: Outcome feature name.
        :param permitted_range (optional): Dictionary with feature names as keys and permitted range as values. Defaults to the range inferred from training data.
        :param test_size (optional): Proportion of test set split. Defaults to 0.2.
        :param test_split_random_state (optional): Random state for train test split. Defaults to 17.

        """

        if isinstance(params["dataframe"], pd.DataFrame):
            self.data_df = params["dataframe"]
        else:
            raise ValueError("should provide a pandas dataframe")

        if type(params["continuous_features"]) is list:
            self.continuous_feature_names = params["continuous_features"]
        else:
            raise ValueError(
                "should provide the name(s) of continuous features in the data"
            )

        if type(params["outcome_name"]) is str:
            self.outcome_name = params["outcome_name"]
        else:
            raise ValueError("should provide the name of outcome feature")

        self.categorical_feature_names = [
            name
            for name in self.data_df.columns.tolist()
            if name not in self.continuous_feature_names + [self.outcome_name]
        ]

        self.feature_names = [
            name for name in self.data_df.columns.tolist() if name != self.outcome_name
        ]

        self.continuous_feature_indexes = [
            self.data_df.columns.get_loc(name)
            for name in self.continuous_feature_names
            if name in self.data_df
        ]

        self.categorical_feature_indexes = [
            self.data_df.columns.get_loc(name)
            for name in self.categorical_feature_names
            if name in self.data_df
        ]

        if "test_size" in params:
            self.test_size = params["test_size"]
        else:
            self.test_size = 0.2

        if "test_split_random_state" in params:
            self.test_split_random_state = params["test_split_random_state"]
        else:
            self.test_split_random_state = 17

        if len(self.categorical_feature_names) > 0:
            self.data_df[self.categorical_feature_names] = self.data_df[
                self.categorical_feature_names
            ].astype("category")
        if len(self.continuous_feature_names) > 0:
            print(self.data_df.head())
            print(self.data_df.head())

        if len(self.categorical_feature_names) > 0:
            self.one_hot_encoded_data = self.one_hot_encode_data(self.data_df)
            self.encoded_feature_names = [
                x
                for x in self.one_hot_encoded_data.columns.tolist()
                if x not in np.array([self.outcome_name])
            ]
        else:
            # one-hot-encoded data is same as orignial data if there is no categorical features.
            self.one_hot_encoded_data = self.data_df
            self.encoded_feature_names = self.feature_names

        self.train_df, self.test_df = self.split_data(self.data_df)
        if "permitted_range" in params:
            self.permitted_range = params["permitted_range"]
        else:
            self.permitted_range = self.get_features_range()

    def get_features_range(self):
        ranges = {}
        for feature_name in self.continuous_feature_names:
            ranges[feature_name] = [
                self.data_df[feature_name].min(),
                self.data_df[feature_name].max(),
            ]
        return ranges

    def get_data_type(self, col):
        """Infers data type of a feature from the training data."""
        for instance in col.tolist():
            if isinstance(instance, int):
                return "int"
            else:
                if float(str(instance).split(".")[1]) > 0:
                    return "float"
        return "int"

    def one_hot_encode_data(self, data):
        """One-hot-encodes the data."""
        return pd.get_dummies(
            data, drop_first=False, columns=self.categorical_feature_names
        )

    def normalize_data(self, df):
        """Normalizes continuous features to make them fall in the range [0,1]."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (
                max_value - min_value
            )
        return result

    def de_normalize_data(self, df):
        """De-normalizes continuous features from [0,1] range to original range."""
        result = df.copy()
        for feature_name in self.continuous_feature_names:
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()
            result[feature_name] = (
                df[feature_name] * (max_value - min_value)
            ) + min_value
        return result

    def get_minx_maxx(self, normalized=True):
        """Gets the min/max value of features in normalized or de-normalized form."""
        minx = np.array([[0.0] * len(self.encoded_feature_names)])
        maxx = np.array([[1.0] * len(self.encoded_feature_names)])

        for idx, feature_name in enumerate(self.continuous_feature_names):
            max_value = self.data_df[feature_name].max()
            min_value = self.data_df[feature_name].min()

            if normalized:
                minx[0][idx] = (self.permitted_range[feature_name][0] - min_value) / (
                    max_value - min_value
                )
                maxx[0][idx] = (self.permitted_range[feature_name][1] - min_value) / (
                    max_value - min_value
                )
            else:
                minx[0][idx] = self.permitted_range[feature_name][0]
                maxx[0][idx] = self.permitted_range[feature_name][1]
        return minx, maxx

    def split_data(self, data):
        train_df, test_df = train_test_split(
            data, test_size=self.test_size, random_state=self.test_split_random_state
        )
        return train_df, test_df

    def get_mads_from_training_data(self, normalized=False):
        """Computes Median Absolute Deviation of features."""

        mads = {}
        if normalized is False:
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(
                        self.data_df[feature].values
                        - np.median(self.data_df[feature].values)
                    )
                )
        else:
            normalized_train_df = self.normalize_data(self.train_df)
            for feature in self.continuous_feature_names:
                mads[feature] = np.median(
                    abs(
                        normalized_train_df[feature].values
                        - np.median(normalized_train_df[feature].values)
                    )
                )
        return mads

    def get_data_params(self):
        """Gets all data related params for DiCE."""

        minx, maxx = self.get_minx_maxx(normalized=True)

        # get the column indexes of categorical features after one-hot-encoding
        self.encoded_categorical_feature_indexes = (
            self.get_encoded_categorical_feature_indexes()
        )

        return minx, maxx, self.encoded_categorical_feature_indexes

    def get_encoded_categorical_feature_indexes(self):
        """Gets the column indexes categorical features after one-hot-encoding."""
        cols = []
        for col_parent in self.categorical_feature_names:
            temp = [
                self.encoded_feature_names.index(col)
                for col in self.encoded_feature_names
                if col.startswith(col_parent)
            ]
            cols.append(temp)
        return cols

    def get_indexes_of_features_to_vary(self, features_to_vary="all"):
        """Gets indexes from feature names of one-hot-encoded data."""
        if features_to_vary == "all":
            return [i for i in range(len(self.encoded_feature_names))]
        else:
            return [
                colidx
                for colidx, col in enumerate(self.encoded_feature_names)
                if col.startswith(tuple(features_to_vary))
            ]

    def from_dummies(self, data, prefix_sep="_"):
        """Gets the original data from dummy encoded data with k levels."""
        out = data.copy()
        for l in self.categorical_feature_names:  # noqa: E741
            cols, labs = [
                [c.replace(x, "") for c in data.columns if l + prefix_sep in c]
                for x in ["", l + prefix_sep]
            ]
            out[l] = pd.Categorical(
                np.array(labs)[np.argmax(data[cols].values, axis=1)]
            )
            out.drop(cols, axis=1, inplace=True)
        return out

    def get_decimal_precisions(self):
        """ "Gets the precision of continuous features in the data."""
        precisions = [0] * len(self.feature_names)
        for ix, col in enumerate(self.continuous_feature_names):
            precisions[ix] = 0
            for instance in self.data_df[col].tolist():
                if isinstance(instance, int):
                    precisions[ix] = 0
                    break
                else:
                    if float(str(instance).split(".")[1]) > 0:
                        precisions[ix] = len(str(instance).split(".")[1])
                        break
        return precisions

    def get_decoded_data(self, data):
        """Gets the original data from dummy encoded data."""
        if isinstance(data, np.ndarray):
            index = [i for i in range(0, len(data))]
            data = pd.DataFrame(
                data=data, index=index, columns=self.encoded_feature_names
            )
        return self.from_dummies(data)

    def prepare_df_for_encoding(self):
        """Facilitates prepare_query_instance() function."""
        levels = []
        colnames = self.categorical_feature_names
        for cat_feature in colnames:
            levels.append(self.data_df[cat_feature].cat.categories.tolist())

        df = pd.DataFrame({colnames[0]: levels[0]})
        for col in range(1, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: levels[col]})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        colnames = self.continuous_feature_names
        for col in range(0, len(colnames)):
            temp_df = pd.DataFrame({colnames[col]: []})
            df = pd.concat([df, temp_df], axis=1, sort=False)

        return df

    def prepare_query_instance(self, query_instance, encode):
        """Prepares user defined test input for DiCE."""

        if isinstance(query_instance, list):
            query_instance = {"row1": query_instance}
            test = pd.DataFrame.from_dict(
                query_instance, orient="index", columns=self.feature_names
            )

        elif isinstance(query_instance, dict):
            query_instance = dict(
                zip(query_instance.keys(), [[q] for q in query_instance.values()])
            )
            test = pd.DataFrame(query_instance, columns=self.feature_names)

        test = test.reset_index(drop=True)

        if encode is False:
            return self.normalize_data(test)
        else:
            temp = self.prepare_df_for_encoding()

            temp = temp.append(test, ignore_index=True, sort=False)
            temp = self.one_hot_encode_data(temp)
            temp = self.normalize_data(temp)

            return temp.tail(test.shape[0]).reset_index(drop=True)


def load_adult_income_dataset(save_intermediate=False):
    """
    Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :param: save_intermediate: save the transformed dataset. Do not save by default.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="adult_ds_"))
    ds = np.DataSource(destpath=str(tmpdir))
    with ds.open(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "rb",
    ) as fh:
        raw_data = np.genfromtxt(fh, delimiter=", ", dtype=str)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    adult_data = pd.DataFrame(raw_data, columns=column_names)

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype(
        {"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64}
    )

    adult_data = adult_data.replace(
        {"workclass": {"Without-pay": "Other/Unknown", "Never-worked": "Other/Unknown"}}
    )
    adult_data = adult_data.replace(
        {
            "workclass": {
                "Federal-gov": "Government",
                "State-gov": "Government",
                "Local-gov": "Government",
            }
        }
    )
    adult_data = adult_data.replace(
        {
            "workclass": {
                "Self-emp-not-inc": "Self-Employed",
                "Self-emp-inc": "Self-Employed",
            }
        }
    )
    adult_data = adult_data.replace(
        {"workclass": {"Never-worked": "Self-Employed", "Without-pay": "Self-Employed"}}
    )
    adult_data = adult_data.replace({"workclass": {"?": "Other/Unknown"}})

    adult_data = adult_data.replace(
        {
            "occupation": {
                "Adm-clerical": "White-Collar",
                "Craft-repair": "Blue-Collar",
                "Exec-managerial": "White-Collar",
                "Farming-fishing": "Blue-Collar",
                "Handlers-cleaners": "Blue-Collar",
                "Machine-op-inspct": "Blue-Collar",
                "Other-service": "Service",
                "Priv-house-serv": "Service",
                "Prof-specialty": "Professional",
                "Protective-serv": "Service",
                "Tech-support": "Service",
                "Transport-moving": "Blue-Collar",
                "Unknown": "Other/Unknown",
                "Armed-Forces": "Other/Unknown",
                "?": "Other/Unknown",
            }
        }
    )

    adult_data = adult_data.replace(
        {
            "marital-status": {
                "Married-civ-spouse": "Married",
                "Married-AF-spouse": "Married",
                "Married-spouse-absent": "Married",
                "Never-married": "Single",
            }
        }
    )

    adult_data = adult_data.replace(
        {
            "race": {
                "Black": "Other",
                "Asian-Pac-Islander": "Other",
                "Amer-Indian-Eskimo": "Other",
            }
        }
    )

    adult_data = adult_data[
        [
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "race",
            "gender",
            "hours-per-week",
            "income",
        ]
    ]

    adult_data = adult_data.replace({"income": {"<=50K": 0, ">50K": 1}})

    adult_data = adult_data.replace(
        {
            "education": {
                "Assoc-voc": "Assoc",
                "Assoc-acdm": "Assoc",
                "11th": "School",
                "10th": "School",
                "7th-8th": "School",
                "9th": "School",
                "12th": "School",
                "5th-6th": "School",
                "1st-4th": "School",
                "Preschool": "School",
            }
        }
    )

    adult_data = adult_data.rename(
        columns={"marital-status": "marital_status", "hours-per-week": "hours_per_week"}
    )

    if save_intermediate:
        adult_data.to_csv("adult.csv", index=False)

    return adult_data


def get_adult_data_info():
    feature_description = {
        "age": "age",
        "workclass": "type of industry (Government, Other/Unknown, Private, Self-Employed)",
        "education": "education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)",
        "marital_status": "marital status (Divorced, Married, Separated, Single, Widowed)",
        "occupation": "occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)",
        "race": "white or other race?",
        "gender": "male or female?",
        "hours_per_week": "total work hours per week",
        "income": "0 (<=50K) vs 1 (>50K)",
    }
    return feature_description


def load_adult_preference_dataset(preference_dataset_path: str):
    dataset = load_adult_income_dataset()
    params = {
        "dataframe": dataset.copy(),
        "continuous_features": ["age", "hours_per_week"],
        "outcome_name": "income",
    }
    dataloader = DataLoader(params)
    dataset_cloned = dataloader.train_df.copy()
    dataset_cloned.drop("income", axis=1, inplace=True)

    # Loading the fine tune dataset
    columns = dataset_cloned.columns
    x_prefer = pd.DataFrame(columns=columns)
    train_x = []
    y_prefer = []
    counter = 0
    with open(preference_dataset_path, "r") as f:
        oracle_dataset = json.load(f)
    for key in oracle_dataset.keys():
        x_prefer.loc[counter] = pd.read_json(oracle_dataset[key][0]).loc[0]
        train_x.append(oracle_dataset[key][1][0])
        y_prefer.append(oracle_dataset[key][2])
        counter += 1

    x_prefer = x_prefer
    train_x = np.array(train_x)
    y_prefer = np.array(y_prefer)

    # For generating proper one hot encodings: Merge with the whole dataset to get all the categories
    x_prefer = pd.concat([x_prefer, dataset_cloned], keys=[0, 1])
    x_prefer = dataloader.one_hot_encode_data(x_prefer)

    x_prefer = x_prefer.xs(0)
    x_prefer = dataloader.normalize_data(x_prefer)
    x_prefer = x_prefer.to_numpy()

    train_x = torch.tensor(train_x).float()
    x_prefer = torch.tensor(x_prefer).float()
    y_prefer = torch.tensor(y_prefer).long()

    train_dataset = train_x
    preference_dataset = {"x_prefer": {}, "y_prefer": {}}
    for x, xp, yp in train_x, x_prefer, y_prefer:
        preference_dataset["x_prefer"][x] = xp
        preference_dataset["y_prefer"][x] = yp

    return train_dataset, preference_dataset


def load_pretrained_binaries(filename: str):
    resource = files("methods.catalog.cfvae.resources.data") / filename
    target_dir = Path(tempfile.mkdtemp(prefix="cfvae_resources_"))
    target_path = target_dir / filename

    with resource.open("rb") as src, target_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    return str(target_path)
