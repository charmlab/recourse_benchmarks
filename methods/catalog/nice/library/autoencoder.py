"""
Simple Autoencoder for plausibility measurement in NICE(plaus)

Based on standard architecture used in counterfactual literature
(CFproto, similar methods) and the benchmark's existing test patterns.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AutoEncoder:
    """
    Autoencoder to measure plausibility via reconstruction error.
    
    Trains a simple feedforward autoencoder on training data.
    Lower reconstruction error = more plausible (closer to data manifold).
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data (N x M)
    cat_feat_idx : list
        Indices of categorical features (not used in current implementation)
    num_feat_idx : list
        Indices of numerical features (not used in current implementation)
    encoding_dim : int, default=32
        Dimension of encoded (latent) representation
    epochs : int, default=50
        Number of training epochs
    batch_size : int, default=32
        Batch size for training
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    
    Attributes
    ----------
    autoencoder : keras.Model
        Trained autoencoder model
        
    Notes
    -----
    Architecture: input → 64 (ReLU) → encoding_dim (ReLU) → 64 (ReLU) → output (Sigmoid)
    Loss: Mean Squared Error (MSE)
    Optimizer: Adam
    
    This is a standard architecture used in counterfactual literature for
    measuring plausibility through reconstruction error.
    """
    
    def __init__(
        self, 
        X_train: np.ndarray,
        cat_feat_idx: list = None,
        num_feat_idx: list = None,
        encoding_dim: int = 32,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0
    ):
        self.cat_feat_idx = cat_feat_idx if cat_feat_idx is not None else []
        self.num_feat_idx = num_feat_idx if num_feat_idx is not None else []
        self.encoding_dim = encoding_dim
        self.input_dim = X_train.shape[1]  # ✅ FIXED! Number of features, not instances
        self.batch_size = batch_size
        
        # Build autoencoder architecture
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Create model
        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder on training data
        self.autoencoder.fit(
            X_train, 
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
            validation_split=0.1
        )
        
        if verbose > 0:
            print(f"Autoencoder trained on {X_train.shape[0]} instances")
            print(f"Architecture: {self.input_dim} → 64 → {encoding_dim} → 64 → {self.input_dim}")
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for instances.
        
        This is the main interface used by PlausibilityReward.
        
        Parameters
        ----------
        X : np.ndarray
            Instances to evaluate (N x M)
        
        Returns
        -------
        np.ndarray
            Reconstruction errors, one per instance (N,)
            Higher error = less plausible (farther from data manifold)
        """
        # Get reconstructions
        X_reconstructed = self.autoencoder.predict(X, verbose=0)
        
        # Calculate Mean Squared Error per instance
        errors = np.mean(np.square(X - X_reconstructed), axis=1)
        
        return errors
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct instances through the autoencoder.
        
        Parameters
        ----------
        X : np.ndarray
            Instances to reconstruct (N x M)
        
        Returns
        -------
        np.ndarray
            Reconstructed instances (N x M)
        """
        return self.autoencoder.predict(X, verbose=0)