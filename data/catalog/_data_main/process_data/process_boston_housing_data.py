import os

import pandas as pd


def load_boston_housing_data():
    file_path = os.path.join(
        os.path.dirname(__file__), "..", "raw_data", "boston_housing.csv"
    )
    data_frame = pd.read_csv(file_path)

    return data_frame
