import os

import pandas as pd

"""
URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.
"""

# Original UCI column semantics
COLUMN_NAME_MAP = {
    "limit_bal": "LimitBalance",  # X1
    "sex": "Sex",  # X2: 1=male, 2=female
    "education": "EducationLevel",  # X3
    "marriage": "MaritalStatus",  # X4
    "age": "Age",  # X5
    "pay_0": "RepayStatusSep",  # X6
    "pay_2": "RepayStatusAug",  # X7
    "pay_3": "RepayStatusJul",  # X8
    "pay_4": "RepayStatusJun",  # X9
    "pay_5": "RepayStatusMay",  # X10
    "pay_6": "RepayStatusApr",  # X11
    "bill_amt1": "BillAmountSep",  # X12
    "bill_amt2": "BillAmountAug",  # X13
    "bill_amt3": "BillAmountJul",  # X14
    "bill_amt4": "BillAmountJun",  # X15
    "bill_amt5": "BillAmountMay",  # X16
    "bill_amt6": "BillAmountApr",  # X17
    "pay_amt1": "PaymentAmountSep",  # X18
    "pay_amt2": "PaymentAmountAug",  # X19
    "pay_amt3": "PaymentAmountJul",  # X20
    "pay_amt4": "PaymentAmountJun",  # X21
    "pay_amt5": "PaymentAmountMay",  # X22
    "pay_amt6": "PaymentAmountApr",  # X23
}

TARGET_NAME_MAP = {"default payment next month": "DefaultNextMonth(label)"}

STATUS_COLUMNS = [
    "RepayStatusSep",
    "RepayStatusAug",
    "RepayStatusJul",
    "RepayStatusJun",
    "RepayStatusMay",
    "RepayStatusApr",
]

BILL_COLUMNS = [
    "BillAmountSep",
    "BillAmountAug",
    "BillAmountJul",
    "BillAmountJun",
    "BillAmountMay",
    "BillAmountApr",
]

PAYMENT_COLUMNS = [
    "PaymentAmountSep",
    "PaymentAmountAug",
    "PaymentAmountJul",
    "PaymentAmountJun",
    "PaymentAmountMay",
    "PaymentAmountApr",
]

# Ordered to match UCI description (X1-X23) with label first
ORDERED_COLUMNS = [
    "DefaultNextMonth(label)",
    "LimitBalance",
    "Sex",
    "EducationLevel",
    "MaritalStatus",
    "Age",
    *STATUS_COLUMNS,
    *BILL_COLUMNS,
    *PAYMENT_COLUMNS,
]


def _rename_columns(df, mapping):
    rename_dict = {}
    for column in df.columns:
        column_lower = column.lower()
        if column_lower in mapping:
            rename_dict[column] = mapping[column_lower]
    return df.rename(columns=rename_dict)


def _read_raw_uci_credit():
    raw_data_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "raw_data"
    )
    raw_data_file = os.path.join(raw_data_dir, "uci_credit.csv")

    # Official CSV puts the real header in the second row
    df = pd.read_csv(raw_data_file, header=1)
    return df


def _standardize_raw_columns(df):
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])
    df = _rename_columns(df, COLUMN_NAME_MAP)
    df = _rename_columns(df, TARGET_NAME_MAP)
    return df


def _enforce_types(df):
    df = df.copy()
    # Categorical / ordinal / label
    int_columns = [
        "Sex",
        "EducationLevel",
        "MaritalStatus",
        "Age",
        *STATUS_COLUMNS,
        "DefaultNextMonth(label)",
    ]
    for col in int_columns:
        df[col] = df[col].astype(int)

    # Monetary amounts
    amount_columns = BILL_COLUMNS + PAYMENT_COLUMNS + ["LimitBalance"]
    df[amount_columns] = df[amount_columns].astype(float)
    return df


def load_uci_credit_data():
    """
    Load dataset following the official UCI description (X1-X23 + binary label).
    """
    raw_df = _read_raw_uci_credit()
    processed_df = _standardize_raw_columns(raw_df)

    missing_columns = set(ORDERED_COLUMNS) - set(processed_df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # fix: map unknown education codes (0,5,6) -> "others" (4); unknown marital (0) -> "others" (3); unknown status (0) -> "pay duly" (-1)
    processed_df["EducationLevel"] = processed_df["EducationLevel"].replace(
        {0: 4, 5: 4, 6: 4}
    )
    processed_df["MaritalStatus"] = processed_df["MaritalStatus"].replace({0: 3})
    for status in STATUS_COLUMNS:
        processed_df[status] = processed_df[status].replace({0: -1})

    processed_df = _enforce_types(processed_df)
    processed_df = processed_df.loc[:, ORDERED_COLUMNS].reset_index(drop=True)
    return processed_df
