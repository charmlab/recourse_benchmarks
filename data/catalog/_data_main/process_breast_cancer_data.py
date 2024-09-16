import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_breast_cancer_data():
    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target (label)'] = data.target
    
    return df
