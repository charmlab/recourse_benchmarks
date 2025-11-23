"""
Preprocessing module matching NICE_experiments
One-Hot Encoding for categorical + MinMax scaling for continuous
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class OHE_minmax:
    """
    Preprocessor matching NICE_experiments exactly
    
    - Categorical features: One-Hot Encoding
    - Continuous features: MinMaxScaler to [-1, 1]
    """
    
    def __init__(self, cat_feat, con_feat):
        """
        Parameters
        ----------
        cat_feat : list of int
            Indices of categorical features
        con_feat : list of int
            Indices of continuous features
        """
        self.cat_feat = cat_feat
        self.con_feat = con_feat
        
    def fit(self, X):
        """Fit encoders on training data"""
        if self.cat_feat != []:
            self.OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.OHE.fit(X[:, self.cat_feat])
            self.nb_OHE = self.OHE.transform(X[0:1, self.cat_feat]).shape[1]
        
        if self.con_feat != []:
            self.minmax = MinMaxScaler(feature_range=(-1, 1))# -1,1? 0,1?
            self.minmax.fit(X[:, self.con_feat])
    
    def transform(self, X):
        """Transform data: OHE + MinMax"""
        if self.cat_feat == []:
            return self.minmax.transform(X[:, self.con_feat])
        elif self.con_feat == []:
            return self.OHE.transform(X[:, self.cat_feat])
        else:
            X_minmax = self.minmax.transform(X[:, self.con_feat])
            X_ohe = self.OHE.transform(X[:, self.cat_feat])
            return np.c_[X_ohe, X_minmax]
    
    def inverse_transform(self, X):
        """Inverse transform: back to label-encoded + original scale"""
        if self.cat_feat == []:
            return self.minmax.inverse_transform(X)
        elif self.con_feat == []:
            return self.OHE.inverse_transform(X[:, :self.nb_OHE])
        else:
            X_con = self.minmax.inverse_transform(X[:, self.nb_OHE:])
            X_cat = self.OHE.inverse_transform(X[:, :self.nb_OHE])
            return np.c_[X_cat, X_con]