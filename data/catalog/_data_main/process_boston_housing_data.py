import pandas as pd

def load_boston_housing_data():
    file_path = "boston_housing.csv"
    data_frame = pd.read_csv(file_path)

    return data_frame
