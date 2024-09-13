import pandas as pd

def load_breast_cancer_data():
    file_path = "breast_cancer.csv"
    data_frame = pd.read_csv(file_path)
    
    return data_frame
