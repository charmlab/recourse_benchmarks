import pandas as pd
import numpy as np

def load_online_news_popularity_data():
    file_path = "data\catalog\_data_main\online_news_popularity.csv"
    data_frame = pd.read_csv(file_path)
    
    data_frame.columns = data_frame.columns.str.strip()
    data_frame['shares'] = np.where(data_frame['shares'] > 1400, 1, 0)
    data_frame.rename(columns={'shares': 'shares (label)'}, inplace=True)
    
    data_frame.drop(columns=['url'], inplace=True)

    return data_frame