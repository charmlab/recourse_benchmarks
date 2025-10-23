"""
Simple Autoencoder for plausibility measurement
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class AutoEncoder:
    """
    Autoencoder to measure plausibility via reconstruction error
    """
    
    def __init__(self, X_train, cat_feat_idx, num_feat_idx, 
                 encoding_dim=32, epochs=50):
        """
        Train autoencoder on training data
        
        Parameters
        ----------
        X_train : np.ndarray
            Training data
        cat_feat_idx : list
            Indices of categorical features
        num_feat_idx : list
            Indices of numerical features
        encoding_dim : int
            Dimension of encoded representation
        epochs : int
            Training epochs
        """
        self.cat_feat_idx = cat_feat_idx
        self.num_feat_idx = num_feat_idx
        
        input_dim = X_train.shape[1]
        
        # Build autoencoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            shuffle=True
        )
    
    def __call__(self, X):
        """
        Calculate reconstruction error for instances
        
        Parameters
        ----------
        X : np.ndarray
            Instances to evaluate
        
        Returns
        -------
        np.ndarray
            Reconstruction errors (one per instance)
        """
        # Get reconstructions
        X_reconstructed = self.autoencoder.predict(X, verbose=0)
        
        # Calculate MSE per instance
        errors = np.mean(np.square(X - X_reconstructed), axis=1)
        
        return errors