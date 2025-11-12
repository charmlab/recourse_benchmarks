import os
from random import seed

import pandas as pd

RANDOM_SEED = 54321
seed(
    RANDOM_SEED
)  # set the random seed so that the random permutations can be reproduced again


def get_feat_types(df):
    cat_feat = []
    num_feat = []
    for key in list(df):
        if df[key].dtype == object:
            cat_feat.append(key)
        elif len(set(df[key])) > 2:
            num_feat.append(key)
    return cat_feat, num_feat


def load_sba_data(modified=False):
    # Define attributes of interest
    attrs = [
        "Zip",
        "NAICS",
        "ApprovalDate",
        "ApprovalFY",
        "Term",
        "NoEmp",
        "NewExist",
        "CreateJob",
        "RetainedJob",
        "FranchiseCode",
        "UrbanRural",
        "RevLineCr",
        "ChgOffDate",
        "DisbursementDate",
        "DisbursementGross",
        "ChgOffPrinGr",
        "GrAppv",
        "SBA_Appv",
        "New",
        "RealEstate",
        "Portion",
        "Recession",
        "daysterm",
        "xx",
    ]
    sensitive_attrs = []  # just an example, pick what matters for fairness
    attrs_to_ignore = []  # IDs or very sparse high-cardinality

    # Path to raw SBA file
    this_files_directory = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(
        this_files_directory, "..", "raw_data", "SBAcase.11.13.17.csv"
    )

    # Load file
    df = pd.read_csv(file_name)
    df = df.fillna(-1)  # replace NaNs with sentinel
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # print(df['RevLineCr'].value_counts())

    # Define target
    y = 1 - df["Default"].values

    # Dicts for storage
    x_control = {}
    attrs_to_vals = {}

    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = df[k].tolist()
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = df[k].tolist()

    # Combine
    all_attrs_to_vals = attrs_to_vals
    for k in sensitive_attrs:
        all_attrs_to_vals[k] = x_control[k]
    all_attrs_to_vals["label"] = y

    df_all = pd.DataFrame.from_dict(all_attrs_to_vals)

    _, num_feat = get_feat_types(df_all)

    # for key in num_feat:
    #     scaler = StandardScaler()
    #     df_all[key] = scaler.fit_transform(df_all[key].values.reshape(-1,1))

    # ---- Create processed dataframe with integer encodings ----
    processed_df = pd.DataFrame()

    # Numeric attributes: keep directly
    num_attrs = [
        "Zip",
        "NAICS",
        "ApprovalDate",
        "ApprovalFY",
        "Term",
        "NoEmp",
        "NewExist",
        "CreateJob",
        "RetainedJob",
        "FranchiseCode",
        "UrbanRural",
    ]
    for a in num_attrs:
        processed_df[a] = df_all[a]

    # RevLineCr ("Y"/"N"/other) → 1,2,3
    processed_df.loc[df_all["RevLineCr"] == "Y", "RevLineCr"] = 1
    processed_df.loc[df_all["RevLineCr"] == "N", "RevLineCr"] = 2
    processed_df.loc[df_all["RevLineCr"] == "T", "RevLineCr"] = 3
    processed_df.loc[df_all["RevLineCr"] == "0", "RevLineCr"] = 4
    # processed_df.loc[df_all["RevLineCr"] == -1, "RevLineCr"] = 5

    # print(processed_df['RevLineCr'].value_counts())
    # cant think of what to do, can just drop the Nas actaully.

    # processed_df['RevLineCr'] = pd.Categorical(processed_df['RevLineCr'])

    # Add recession, real estate, portion, etc. directly
    for a in [
        "ChgOffDate",
        "DisbursementDate",
        "DisbursementGross",
        "ChgOffPrinGr",
        "GrAppv",
        "SBA_Appv",
        "New",
        "RealEstate",
        "Portion",
        "Recession",
        "daysterm",
        "xx",
    ]:
        processed_df[a] = df_all[a]

    processed_df["Label"] = df_all["label"]

    if modified is False:
        processed_df = processed_df[processed_df["ApprovalFY"] < 2006]

    processed_df = processed_df[processed_df["RevLineCr"].notna()]

    return processed_df.astype("float64")
