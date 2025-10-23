"""
Data handling for NICE algorithm

Adapted from nice/utils/data.py in NICE repository:
https://github.com/DBrughmans/NICE
"""
import numpy as np
from scipy.stats import mode


class data_NICE:
    """
    Data wrapper for NICE algorithm.
    
    Handles training data, predictions, and candidate filtering for
    nearest unlike neighbor search.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data features
    y_train : np.ndarray or None
        Training data labels (for justified_cf)
    cat_feat : list
        Indices of categorical features
    num_feat : list or 'auto'
        Indices of numerical features. If 'auto', inferred from cat_feat
    predict_fn : callable
        Function that takes X and returns prediction probabilities
    justified_cf : bool
        If True, only use correctly classified training instances
    eps : float
        Small epsilon for numerical stability
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cat_feat: list,
        num_feat='auto',
        predict_fn=None,
        justified_cf: bool = True,
        eps: float = 1e-10
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.cat_feat = cat_feat
        self.num_feat = num_feat
        self.predict_fn = predict_fn
        self.justified_cf = justified_cf
        self.eps = eps

        # Auto-detect numerical features if not provided
        if self.num_feat == 'auto':
            self.num_feat = [
                feat for feat in range(self.X_train.shape[1]) 
                if feat not in self.cat_feat
            ]

        # Ensure numerical features are float
        self.X_train = self.num_as_float(self.X_train)

        # Get predictions for training data
        self.train_proba = predict_fn(X_train)
        self.n_classes = self.train_proba.shape[1]
        self.X_train_class = np.argmax(self.train_proba, axis=1)

        # Create candidate mask
        if self.justified_cf:
            # Only use correctly classified instances
            self.candidates_mask = self.y_train == self.X_train_class
        else:
            # Use all training instances
            self.candidates_mask = np.ones(self.X_train.shape[0], dtype=bool)

    def num_as_float(self, X: np.ndarray) -> np.ndarray:
        """Convert numerical features to float64"""
        X = X.copy()
        X[:, self.num_feat] = X[:, self.num_feat].astype(np.float64)
        return X

    def fit_to_X(self, X: np.ndarray, target_class='other'):
        """
        Prepare data object for explaining instance X.
        
        Parameters
        ----------
        X : np.ndarray
            Instance to explain (1 x M)
        target_class : 'other' or list
            Target class(es) for counterfactual
        """
        self.X = self.num_as_float(X)
        self.X_score = self.predict_fn(self.X)
        self.X_class = self.X_score.argmax()
        
        # Determine target class(es)
        if target_class == 'other':
            self.target_class = [
                i for i in range(self.n_classes) 
                if i != self.X_class
            ]
        else:
            self.target_class = target_class
        
        # Create mask for opposite class candidates
        self.class_mask = np.array([
            i in self.target_class 
            for i in self.X_train_class
        ])
        
        # Combine class mask with candidates mask
        self.mask = self.class_mask & self.candidates_mask
        
        # Create view of valid candidates
        self.candidates_view = self.X_train[self.mask, :].view()